/******************************************************************************
 * Copyright (C) 2020 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */
// #include <ATen/native/TensorIterator.h> // TODO: fix this include
#include <bitset>

#include <perf_lib_layer_params.h>
#include <torch/script.h>

#include "habana_device/HPUCheck.h"
#include "habana_device/hpu_cached_devices.h"
#include "habana_helpers/tensor_utils.h"
#include "habana_kernels/kernel_utils.h"
#include "habana_kernels/resize.h"
#include "habana_kernels/simple_generic_kernel.h"

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

  // Following code is required to convert Pytorch 0d tensor
  // to a 1d tensor. This is required because synapse_helpers
  // tensor_builder does not support 0d tensors
  if (shape.size() == 0) {
    shape.push_back(1);
  }

  if (result.defined()) {
    auto tht_result = result.unsafeGetTensorImpl();
    THHTensor_resizeNd(tht_result, shape.size(), shape.data(), nullptr);
    // result.resize_(shape);
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

/**
 * @brief CastKernel params structure
 */
ns_CastKernel::Params synapse_cast_params_builder(){

  ns_CastKernel::Params cast_params{};
  cast_params.round_mode = CAST_ROUND_HALF_NE;

  return cast_params;
}

/*Tensor review_reduce_result(
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
}*/
} // namespace

void synapse_reduce_generic(
    const Tensor& output,
    const Tensor& input,
    const IntArrayRef& dim,
    bool keepdim,
    std::string nodetype) {
  std::cout << "Reduction axis " << dim << std::endl;
  auto& device =
      synapse_helpers::HPURegistrar::get_device(input.device().index());
  const auto device_id = device.id();
  auto ndim = input.dim();

  // graph_handle scope
  synGraphHandle graph_handle;
  TORCH_HABANA_CHECK(
      synGraphCreate(&graph_handle, synDeviceType::synDeviceGaudi),
      "synGraphCreate failed");

  { // tensors scope
    std::vector<synapse_helpers::tensor> syn_helper_inputs, syn_helper_outputs,
        syn_helper_intermediate;
    std::vector<synTensor> syn_inputs, syn_outputs, syn_intermediate;

    std::tie(syn_helper_inputs, syn_inputs) = habana_helpers::create_tensors(
        std::vector<const at::Tensor*>{&input}, graph_handle, true);
    syn_intermediate.push_back(syn_helper_inputs[0].get());
    std::tie(syn_helper_outputs, syn_outputs) = habana_helpers::create_tensors(
        std::vector<const at::Tensor*>{&output}, graph_handle, true);

    unsigned loopend = keepdim ? dim.size() - 1 : dim.size();
    std::vector<int64_t> dims = input.sizes().vec();
    for (unsigned i = 0; i < loopend; i++) {
      dims[dim[i]] = 1;
      c10::IntArrayRef shape(dims.data(), input.dim());
      syn_helper_intermediate.push_back(habana_helpers::create_tensor(
          shape,
          graph_handle,
          false,
          input.device().index(),
          input.scalar_type()));
      syn_intermediate.push_back(syn_helper_intermediate[i].get());
    }

    syn_intermediate.push_back(syn_helper_outputs[0].get());

    {
      for (unsigned i = 0; i < dim.size(); i++) {
        const std::string node_type = nodetype;
        ns_Reduction::Params params{};
        params.reductionDimension = ndim - dim[i] - 1;
        { // add node
          TORCH_HABANA_CHECK(
              synNodeCreate(
                  graph_handle,
                  syn_intermediate.data() + i,
                  syn_intermediate.data() + i + 1,
                  1,
                  1,
                  &params,
                  sizeof(params),
                  node_type.c_str(),
                  "",
                  nullptr,
                  nullptr),
              "synNodeCreate failed");
        }
      }

      if (!keepdim) {
        const std::string node_type = "reshape";
        { // add node
          TORCH_HABANA_CHECK(
              synNodeCreate(
                  graph_handle,
                  syn_intermediate.data() + dim.size(),
                  syn_intermediate.data() + dim.size() + 1,
                  1,
                  1,
                  nullptr,
                  0,
                  node_type.c_str(),
                  "",
                  nullptr,
                  nullptr),
              "synNodeCreate failed");
        }
      }

      habana_helpers::compile_and_run(
          nodetype,
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

Tensor sum_dim_IntList_hpu(
    const Tensor& self,
    IntArrayRef dim,
    bool keepdim,
    c10::optional<ScalarType> dtype) {
  PT_KERNEL_BEGIN;

  Tensor output;
  auto ndim = self.dim();
  auto mask = make_dim_mask(dim, ndim);
  allocate_reduction_result(
      output, self, mask, keepdim, get_dtype(output, self, dtype, false));
  // auto viewed_result = review_reduce_result(output, ndim, mask, keepdim);
  TORCH_CHECK(
      output.scalar_type() == self.scalar_type(),
      "Habana reduction ops don't support casts yet");
  TORCH_CHECK(
      keepdim || static_cast<int64_t>(dim.size()) != ndim,
      "Reduction to 0d tensor not supported yet");

  std::string nodetype = "reduce_sum_fwd_" +
      habana_helpers::name_suffix_from_type(self.scalar_type());
  synapse_reduce_generic(output, self, dim, keepdim, nodetype);

  PT_KERNEL_END;
  return output;
}

Tensor& sum_IntList_out_hpu(
    Tensor& output,
    const Tensor& self,
    IntArrayRef dim,
    bool keepdim,
    c10::optional<ScalarType> dtype) {
  PT_KERNEL_BEGIN;

  auto ndim = self.dim();
  auto mask = make_dim_mask(dim, ndim);
  allocate_reduction_result(
      output, self, mask, keepdim, get_dtype(output, self, dtype, false));
  // auto viewed_result = review_reduce_result(output, ndim, mask, keepdim);
  TORCH_CHECK(
      output.scalar_type() == self.scalar_type(),
      "Habana reduction ops don't support casts yet");
  TORCH_CHECK(
      keepdim || static_cast<int64_t>(dim.size()) != ndim,
      "Reduction to 0d tensor not supported yet");

  std::string nodetype = "reduce_sum_fwd_" +
      habana_helpers::name_suffix_from_type(self.scalar_type());
  synapse_reduce_generic(output, self, dim, keepdim, nodetype);

  PT_KERNEL_END;
  return output;
}

Tensor mean_dim_hpu(
    const Tensor& self,
    IntArrayRef dim,
    bool keepdim,
    c10::optional<ScalarType> dtype) {
  PT_KERNEL_BEGIN;

  Tensor output;
  auto ndim = self.dim();
  auto mask = make_dim_mask(dim, ndim);
  allocate_reduction_result(
      output, self, mask, keepdim, get_dtype(output, self, dtype, false));
  // auto viewed_result = review_reduce_result(output, ndim, mask, keepdim);
  TORCH_CHECK(
      output.scalar_type() == self.scalar_type(),
      "Habana reduction ops don't support casts yet");
  TORCH_CHECK(
      keepdim || static_cast<int64_t>(dim.size()) != ndim,
      "Reduction to 0d tensor not supported yet");

  std::string nodetype = "reduce_mean_fwd_" +
      habana_helpers::name_suffix_from_type(self.scalar_type());
  synapse_reduce_generic(output, self, dim, keepdim, nodetype);

  PT_KERNEL_END;
  return output;
}

Tensor& mean_dim_out_hpu(
    Tensor& output,
    const Tensor& self,
    IntArrayRef dim,
    bool keepdim,
    c10::optional<ScalarType> dtype) {
  PT_KERNEL_BEGIN;

  auto ndim = self.dim();
  auto mask = make_dim_mask(dim, ndim);
  allocate_reduction_result(
      output, self, mask, keepdim, get_dtype(output, self, dtype, false));
  // auto viewed_result = review_reduce_result(output, ndim, mask, keepdim);
  TORCH_CHECK(
      output.scalar_type() == self.scalar_type(),
      "Habana reduction ops don't support casts yet");
  TORCH_CHECK(
      keepdim || static_cast<int64_t>(dim.size()) != ndim,
      "Reduction to 0d tensor not supported yet");

  std::string nodetype = "reduce_mean_fwd_" +
      habana_helpers::name_suffix_from_type(self.scalar_type());
  synapse_reduce_generic(output, self, dim, keepdim, nodetype);

  PT_KERNEL_END;
  return output;
}

Tensor sum_hpu(const Tensor& self, c10::optional<ScalarType> dtype) {
  PT_KERNEL_BEGIN;

  Tensor output;
  auto ndim = self.dim();
  int64_t data[4];
  for (int i = 0; i < ndim; i++) {
    data[i] = i;
  }
  IntArrayRef dim(data, ndim);
  auto mask = make_dim_mask(dim, ndim);
  allocate_reduction_result(
      output, self, mask, 0, get_dtype(output, self, dtype, false));
  // auto viewed_result = review_reduce_result(output, ndim, mask, keepdim);
  TORCH_CHECK(
      output.scalar_type() == self.scalar_type(),
      "Habana reduction ops don't support casts yet");

  std::string nodetype = "reduce_sum_fwd_" +
      habana_helpers::name_suffix_from_type(self.scalar_type());
  synapse_reduce_generic(output, self, dim, 0, nodetype);

  PT_KERNEL_END;
  return output[0];
}

Tensor mean_hpu(const Tensor& self, c10::optional<ScalarType> dtype) {
  PT_KERNEL_BEGIN;

  Tensor output;
  auto ndim = self.dim();
  int64_t data[4];
  for (int i = 0; i < ndim; i++) {
    data[i] = i;
  }
  IntArrayRef dim(data, ndim);
  auto mask = make_dim_mask(dim, ndim);
  allocate_reduction_result(
      output, self, mask, 0, get_dtype(output, self, dtype, false));
  // auto viewed_result = review_reduce_result(output, ndim, mask, keepdim);
  TORCH_CHECK(
      output.scalar_type() == self.scalar_type(),
      "Habana reduction ops don't support casts yet");

  std::string nodetype = "reduce_mean_fwd_" +
      habana_helpers::name_suffix_from_type(self.scalar_type());
  synapse_reduce_generic(output, self, dim, 0, nodetype);

  PT_KERNEL_END;
  return output[0];
}

/*************************************************************************
 * @brief Kernel implementation for reduction kernel torch.any(output, self, dim, keepdim)
 * @param [out] output - output tensor, bool
 * @param [in] self - input tensor, bool
 * @param [in] dim - along which dimension to reduce, int64_t
 * @param [in] keepdim - output tensor has dim retained or not, bool, default = false
 ************************************************************************/
Tensor& any_dim_out_hpu(
    Tensor& output,
    const Tensor& self,
    int64_t dim,
    bool keepdim) {
  PT_KERNEL_BEGIN;

  self.to(c10::ScalarType::Char);

  auto self_float =
      at::empty(self.sizes(), self.options().dtype(c10::ScalarType::Float));

  std::vector<const at::Tensor*> pt_outputs{&self_float};
  std::vector<const at::Tensor*> pt_inputs{&self};

  auto syn_cast_params =
      synapse_cast_params_builder();

  synapse_simple_generic_kernel(
      pt_outputs, pt_inputs, "cast_i8_to_f32", &syn_cast_params,
      sizeof(syn_cast_params), SynapsePassType::NO_PASS);

  pt_inputs.clear();
  pt_outputs.clear();

  int64_t data[1];
  data[0] = dim;
  IntArrayRef dim_arr(data, 1);

  Tensor output_reduce =
      sum_dim_IntList_hpu( self_float,dim_arr,keepdim, self_float.scalar_type());

  output.to(c10::ScalarType::Char);
  pt_outputs.push_back(&output);
  pt_inputs.push_back(&output_reduce);

  synapse_simple_generic_kernel(
      pt_outputs, pt_inputs, "cast_f32_to_i8", &syn_cast_params,
      sizeof(syn_cast_params), SynapsePassType::NO_PASS);

  output.to(c10::ScalarType::Bool);

  PT_KERNEL_END;
  return output;
}

/*************************************************************************
 * @brief Kernel implementation for reduction kernel output = torch.any(self, dim, keepdim)
 * @param [out] output - output tensor, bool
 * @param [in] self - input tensor, bool
 * @param [in] dim - along which dimension to reduce, int64_t
 * @param [in] keepdim - output tensor has dim retained or not, bool, default = false
 ************************************************************************/
Tensor any_dim_hpu(
    const Tensor& self,
    int64_t dim,
    bool keepdim) {
  PT_KERNEL_BEGIN;

  Tensor output ;
  int64_t data[1];
  data[0] = dim;
  IntArrayRef dim_arr(data, 1);

  auto ndim = self.dim();
  auto mask = make_dim_mask(dim_arr, ndim);
  allocate_reduction_result(
      output, self, mask, keepdim, self.scalar_type());

  any_dim_out_hpu(output,self,dim,keepdim);

  PT_KERNEL_END;
  return output;
}


/*************************************************************************
 * @brief Kernel implementation for reduction kernel output = torch.any(self)
 * @param [out] output - output tensor, bool
 * @param [in] self - input tensor, bool
 ************************************************************************/
Tensor any_hpu(const Tensor& self) {
  PT_KERNEL_BEGIN;

  self.to(c10::ScalarType::Char);

  auto self_float =
      at::empty(self.sizes(), self.options().dtype(c10::ScalarType::Float));

  std::vector<const at::Tensor*> pt_outputs{&self_float};
  std::vector<const at::Tensor*> pt_inputs{&self};

  auto syn_cast_params =
      synapse_cast_params_builder();

  synapse_simple_generic_kernel(
      pt_outputs, pt_inputs, "cast_i8_to_f32", &syn_cast_params,
      sizeof(syn_cast_params), SynapsePassType::NO_PASS);

  pt_inputs.clear();
  pt_outputs.clear();

  Tensor output_reduce = sum_hpu(self_float,self_float.scalar_type());

  //coverting 0d tensor to 1d tensor
  output_reduce.unsafeGetTensorImpl()->set_sizes_and_strides({1}, {1});

  auto output =
      at::empty(output_reduce.sizes(), output_reduce.options().dtype(c10::ScalarType::Char));

  pt_outputs.push_back(&output);
  pt_inputs.push_back(&output_reduce);

  synapse_simple_generic_kernel(
      pt_outputs, pt_inputs, "cast_f32_to_i8", &syn_cast_params,
      sizeof(syn_cast_params), SynapsePassType::NO_PASS);

  PT_KERNEL_END;
  return output[0].to(c10::ScalarType::Bool);
}

static auto registry =
    torch::RegisterOperators()
        .op(torch::RegisterOperators::options()
                .schema(
                    "aten::sum.dim_IntList(Tensor self, int[1] dim, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor")
                .impl_unboxedOnlyKernel<
                    decltype(sum_dim_IntList_hpu),
                    &sum_dim_IntList_hpu>(DispatchKey::HABANATensorId)
                .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
        .op(torch::RegisterOperators::options()
                .schema(
                    "aten::sum.IntList_out(Tensor self, int[1] dim, bool keepdim=False, *, ScalarType? dtype=None, Tensor(a!) out) -> Tensor(a!)")
                .impl_unboxedOnlyKernel<
                    decltype(sum_IntList_out_hpu),
                    &sum_IntList_out_hpu>(DispatchKey::HABANATensorId)
                .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
        .op(torch::RegisterOperators::options()
                .schema(
                    "aten::mean.dim(Tensor self, int[1] dim, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor")
                .impl_unboxedOnlyKernel<decltype(mean_dim_hpu), &mean_dim_hpu>(
                    DispatchKey::HABANATensorId)
                .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
        .op(torch::RegisterOperators::options()
                .schema(
                    "aten::mean.out(Tensor self, int[1] dim, bool keepdim=False, *, ScalarType? dtype=None, Tensor(a!) out) -> Tensor(a!)")
                .impl_unboxedOnlyKernel<
                    decltype(mean_dim_out_hpu),
                    &mean_dim_out_hpu>(DispatchKey::HABANATensorId)
                .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
        .op(torch::RegisterOperators::options()
                .schema(
                    "aten::sum(Tensor self, *, ScalarType? dtype=None) -> Tensor")
                .impl_unboxedOnlyKernel<decltype(sum_hpu), &sum_hpu>(
                    DispatchKey::HABANATensorId)
                .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
        .op(torch::RegisterOperators::options()
                .schema(
                    "aten::mean(Tensor self, *, ScalarType? dtype=None) -> Tensor")
                .impl_unboxedOnlyKernel<decltype(mean_hpu), &mean_hpu>(
                    DispatchKey::HABANATensorId)
                .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
        .op(torch::RegisterOperators::options()
                .schema(
                    "aten::any.dim(Tensor self, int dim, bool keepdim=False) -> Tensor")
                .impl_unboxedOnlyKernel<
                    decltype(any_dim_hpu),
                    &any_dim_hpu>(DispatchKey::HABANATensorId)
                .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
        .op(torch::RegisterOperators::options()
                .schema(
                    "aten::any(Tensor self) -> Tensor")
                .impl_unboxedOnlyKernel<decltype(any_hpu), &any_hpu>(
                    DispatchKey::HABANATensorId)
                .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
        .op(torch::RegisterOperators::options()
                .schema(
                    "aten::any.out(Tensor self, int dim, bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!)")
                .impl_unboxedOnlyKernel<decltype(any_dim_out_hpu), &any_dim_out_hpu>(
                    DispatchKey::HABANATensorId)
                .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA));
