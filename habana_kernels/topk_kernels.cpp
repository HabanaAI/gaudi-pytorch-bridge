/******************************************************************************
 * Copyright (C) 2020 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */
#include <ATen/WrapDimUtils.h>
#include <perf_lib_layer_params.h>
#include <torch/script.h>

#include "habana_device/HPUCheck.h"
#include "habana_device/HPUContext.h"
#include "habana_device/hpu_cached_devices.h"
#include "habana_helpers/tensor_utils.h"
#include "habana_kernels/kernel_utils.h"
#include "habana_kernels/resize.h"

using namespace torch;

// ensure we get good values and indices for topk
inline void _allocate_or_resize_output_with_indices(
    Tensor& values,
    Tensor& indices,
    const Tensor& self,
    int64_t dim,
    int64_t k) {
  auto result_sizes = self.sizes().vec();
  if (result_sizes.size() > 0) {
    result_sizes[dim] = k;
  }
  if (values.defined()) {
    TORCH_CHECK(
        self.type() == values.type(),
        "output values must be of same type as input");
    auto tht_values = values.unsafeGetTensorImpl();
    THHTensor_resizeNd(tht_values, self.dim(), result_sizes.data(), nullptr);
  } else {
    values = at::empty(result_sizes, self.options());
  }
  if (indices.defined()) {
    TORCH_CHECK(
        indices.dtype() == c10::ScalarType::Int,
        "output indices must be of scalar type Int");
    TORCH_CHECK(
        indices.device() == self.device(),
        "output indices must be on same device as input");
    auto tht_indices = indices.unsafeGetTensorImpl();
    THHTensor_resizeNd(tht_indices, self.dim(), result_sizes.data(), nullptr);
  } else {
    indices =
        at::empty(result_sizes, self.options().dtype(c10::ScalarType::Int));
  }
}

void topk_impl(
    const Tensor& values,
    const Tensor& indices,
    const Tensor& self,
    int64_t k,
    int64_t dim_,
    bool largest,
    bool sorted) {
  std::cout << "TopK axis " << dim_ << std::endl;
  std::cout << "largest " << largest << std::endl;
  std::cout << "sorted " << sorted << std::endl;
  auto& device =
      synapse_helpers::HPURegistrar::get_device(self.device().index());
  const auto device_id = device.id();

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
        std::vector<const at::Tensor*>{&self}, graph_handle, true);
    std::tie(syn_helper_outputs, syn_outputs) = habana_helpers::create_tensors(
        std::vector<const at::Tensor*>{&values, &indices}, graph_handle, true);

    // Choose chunkSize to be min(inputSize/8,k)
    // chunkSize must be a multiple of 4 (TPC kernel requirement)
    auto chunkSize = (std::ceil(self.size(dim_) / 8) < k)
        ? k
        : std::ceil(self.size(dim_) / 8);
    chunkSize = std::ceil(chunkSize / 4) * 4;

    // Output tensors must of same size as input tensors
    // except dim in give axis should be ceil(inputSize/chunkSize)*k
    std::vector<int64_t> dims = self.sizes().vec();
    dims[dim_] = std::ceil(self.size(dim_) / chunkSize) * k;
    c10::IntArrayRef shape(dims.data(), self.dim());
    // Create intermediate chunk output & chunk indices tensors
    syn_helper_intermediate.push_back(habana_helpers::create_tensor(
        shape,
        graph_handle,
        false,
        values.device().index(),
        values.scalar_type()));
    syn_intermediate.push_back(syn_helper_intermediate[0].get());
    syn_helper_intermediate.push_back(habana_helpers::create_tensor(
        shape,
        graph_handle,
        false,
        indices.device().index(),
        indices.scalar_type()));
    syn_intermediate.push_back(syn_helper_intermediate[1].get());

    {
      ns_TopK::Params params{};
      params.kSize = k;
      params.axis = self.dim() - dim_ - 1;
      params.chunkSize = chunkSize;
      {
        std::string nodetype = "top_k_st1_fwd_" +
            habana_helpers::name_suffix_from_type(self.scalar_type());
        TORCH_HABANA_CHECK(
            synNodeCreate(
                graph_handle,
                syn_inputs.data(),
                syn_intermediate.data(),
                syn_inputs.size(),
                syn_intermediate.size(),
                &params,
                sizeof(params),
                nodetype.c_str(),
                "",
                nullptr,
                nullptr),
            "synNodeCreate failed");

        nodetype = "top_k_st2_fwd_" +
            habana_helpers::name_suffix_from_type(self.scalar_type());
        TORCH_HABANA_CHECK(
            synNodeCreate(
                graph_handle,
                syn_intermediate.data(),
                syn_outputs.data(),
                syn_intermediate.size(),
                syn_outputs.size(),
                &params,
                sizeof(params),
                nodetype.c_str(),
                "",
                nullptr,
                nullptr),
            "synNodeCreate failed");
      }

      habana_helpers::compile_and_run(
          "top_k",
          graph_handle,
          habana_helpers::names(syn_helper_inputs),
          habana_helpers::names(syn_helper_outputs),
          {self.data_ptr()},
          {values.data_ptr(), indices.data_ptr()},
          device_id);
    }
  }
}

std::tuple<Tensor&, Tensor&> topk_out_hpu(
    Tensor& values,
    Tensor& indices,
    const Tensor& self,
    int64_t k,
    int64_t dim_,
    bool largest,
    bool sorted) {
  int64_t dim = at::maybe_wrap_dim(dim_, self.dim(), /*wrap_scalar=*/true);
  TORCH_CHECK(
      k >= 0 && k <= (self.dim() > 0 ? self.size(dim) : 1),
      "selected index k out of range");
  TORCH_CHECK(largest == true, "smallest k element not supported")
  TORCH_CHECK(sorted == true, "unsorted output not supported")

  _allocate_or_resize_output_with_indices(values, indices, self, dim, k);
  if (self.dim() == 0 && self.numel() == 1) {
    values.copy_(self);
    indices.zero_();
    return std::forward_as_tuple(values, indices);
  }

  topk_impl(values, indices, self, k, dim, largest, sorted);

  return std::forward_as_tuple(values, indices);
}

std::tuple<Tensor, Tensor> topk_hpu(
    const Tensor& self,
    int64_t k,
    int64_t dim,
    bool largest,
    bool sorted) {
  Tensor values = at::empty({0}, self.options());
  Tensor indices = at::empty({0}, self.options().dtype(c10::ScalarType::Int));
  topk_out_hpu(values, indices, self, k, dim, largest, sorted);
  return std::make_tuple(values, indices);
}

static auto registry =
    torch::RegisterOperators()
        .op(torch::RegisterOperators::options()
                .schema(
                    "aten::topk(Tensor self, int k, int dim=-1, bool largest=True, bool sorted=True) -> (Tensor values, Tensor indices)")
                .impl_unboxedOnlyKernel<decltype(topk_hpu), &topk_hpu>(
                    TensorTypeId::HABANATensorId)
                .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
        .op(torch::RegisterOperators::options()
                .schema(
                    "aten::topk.values(Tensor self, int k, int dim=-1, bool largest=True, bool sorted=True, *, Tensor(a!) values, Tensor(b!) indices) ->(Tensor(a!) values, Tensor(b!) indices)")
                .impl_unboxedOnlyKernel<decltype(topk_out_hpu), &topk_out_hpu>(
                    TensorTypeId::HABANATensorId)
                .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA));
