// #include <ATen/native/TensorIterator.h> // TODO: fix this include
#include <bitset>

#include <torch/script.h>
#include <tpc_kernels/include/perf_lib_layer_params.h>

#include "habana_device/HPUCheck.h"
#include "habana_device/HPUContext.h"
#include "habana_device/hpu_cached_devices.h"
#include "habana_helpers/tensor_utils.h"
#include "habana_kernels/kernel_utils.h"

using namespace torch;
// TODO: DimMask = TensorIterator::DimMask
using DimMask = std::bitset<64>;

// Copy paste from PT
namespace {
inline int64_t maybe_wrap_dim(
    int64_t dim,
    int64_t dim_post_expr,
    bool wrap_scalar = true) {
  return c10::maybe_wrap_dim(dim, dim_post_expr, wrap_scalar);
}

DimMask make_dim_mask(IntArrayRef dims, int64_t ndim) {
  auto mask = DimMask();
  if (dims.empty()) {
    mask.flip();
  } else {
    for (int64_t dim : dims) {
      mask.set(maybe_wrap_dim(dim, ndim));
    }
  }
  return mask;
}

void allocate_reduction_result(
    Tensor& result,
    const Tensor& self,
    DimMask mask,
    bool keepdim,
    ScalarType dtype) {
  auto shape = DimVector(self.sizes());
  for (int dim = shape.size() - 1; dim >= 0; dim--) {
    if (mask[dim]) {
      if (keepdim) {
        shape[dim] = 1;
      } else {
        shape.erase(shape.begin() + dim);
      }
    }
  }
  if (result.defined()) {
    result.resize_(shape);
  } else {
    result = at::empty(shape, self.options().dtype(dtype));
  }
}

ScalarType get_dtype(
    Tensor& result,
    const Tensor& self,
    optional<ScalarType> dtype,
    bool promote_integers = false) {
  if (dtype.has_value()) {
    return dtype.value();

  } else if (result.defined()) {
    return result.scalar_type();
  }
  ScalarType src_type = self.scalar_type();
  if (promote_integers && at::isIntegralType(src_type, /*includeBool=*/true)) {
    return kLong;
  }
  return src_type;
}

Tensor review_reduce_result(
    const Tensor& result,
    int ndim,
    DimMask mask,
    bool keepdim) {
  if (keepdim) {
    return result;
  }
  auto shape = DimVector(result.sizes());
  auto stride = DimVector(result.strides());
  for (int dim = 0; dim < ndim; dim++) {
    if (mask[dim]) {
      shape.insert(shape.begin() + dim, 1);
      stride.insert(stride.begin() + dim, 0);
    }
  }
  return result.as_strided(shape, stride);
}
} // namespace

void synapse_reduce_sum(
    const Tensor& output,
    const Tensor& input,
    unsigned dim) {
  std::cout << "Reduction axis " << dim << std::endl;
  auto& device =
      synapse_helpers::HPURegistrar::get_device(input.device().index());
  const auto device_id = device.id();

  // graph_handle scope
  synGraphHandle graph_handle;
  TORCH_HABANA_CHECK(
      synGraphCreate(&graph_handle, synDeviceType::synDeviceGaudi),
      "synGraphCreate failed");

  { // tensors scope
    std::vector<synapse_helpers::tensor> syn_helper_inputs, syn_helper_outputs;
    std::vector<synTensor> syn_inputs, syn_outputs;

    std::tie(syn_helper_inputs, syn_inputs) = habana_helpers::create_tensors(
        std::vector<const at::Tensor*>{&input},
        {"input"},
        graph_handle,
        {true});
    std::tie(syn_helper_outputs, syn_outputs) = habana_helpers::create_tensors(
        std::vector<const at::Tensor*>{&output},
        {"output"},
        graph_handle,
        {true});

    {
      const std::string node_type = "reduce_sum_fwd_" +
          habana_helpers::name_suffix_from_type(input.scalar_type());
      ns_Reduction::Params params{};
      params.reductionDimension = dim;
      { // add node
        TORCH_HABANA_CHECK(
            synNodeCreate(
                graph_handle,
                syn_inputs.data(),
                syn_outputs.data(),
                syn_inputs.size(),
                syn_outputs.size(),
                &params,
                sizeof(params),
                node_type.c_str(),
                "",
                nullptr,
                nullptr),
            "synNodeCreate failed");
      }

      habana_helpers::compile_and_run(
          node_type,
          graph_handle,
          habana_helpers::names(syn_helper_inputs),
          habana_helpers::names(syn_helper_outputs),
          {input.data_ptr()},
          {output.data_ptr()},
          device_id);
    }
  }
  TORCH_HABANA_CHECK(synGraphDestroy(graph_handle), "synGraphDestroy failed");
}

Tensor sum_dim_IntList_habana(
    const Tensor& self,
    IntArrayRef dim,
    bool keepdim,
    c10::optional<ScalarType> dtype) {
  std::cout << "sum_dim_IntList_habana called\n"; // TODO: remove

  Tensor output;
  auto ndim = self.dim();
  auto mask = make_dim_mask(dim, ndim);
  allocate_reduction_result(
      output, self, mask, keepdim, get_dtype(output, self, dtype, false));
  auto viewed_result = review_reduce_result(output, ndim, mask, keepdim);
  TORCH_CHECK(
      viewed_result.scalar_type() == self.scalar_type(),
      "Habana reduction ops don't support casts yet");
  TORCH_CHECK(dim.size() == 1, "Habana support only single dim reduction");
  TORCH_CHECK(keepdim, "Habana reduction keepdim must be turned on");
  synapse_reduce_sum(viewed_result, self, ndim - dim[0] - 1);
  // TODO: implement support for keepdim = false and multiple dims to reduce
  // One way of implementing it is calling multiple times kernel with single
  // reduction but it will be slower. I don't know if synapse support multi axis
  // reduction. Keep dim can be implemented just by modifing metadata of PT
  // tensor, synapse requires to always keep them

  return viewed_result;
}

static auto registry = torch::RegisterOperators().op(
    torch::RegisterOperators::options()
        .schema(
            "aten::sum.dim_IntList(Tensor self, int[1] dim, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor")
        .impl_unboxedOnlyKernel<
            decltype(sum_dim_IntList_habana),
            &sum_dim_IntList_habana>(TensorTypeId::HABANATensorId)
        .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA));
