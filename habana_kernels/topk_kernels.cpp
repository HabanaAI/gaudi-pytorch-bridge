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
#include "habana_device/hpu_cached_devices.h"
#include "habana_helpers/tensor_utils.h"
#include "habana_kernels/kernel_utils.h"
#include "habana_kernels/resize.h"
#include "habana_kernels/simple_generic_kernel.h"

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
        self.options().type_equal(values.options()),
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

std::tuple<Tensor&, Tensor&> topk_out_hpu(
    Tensor& values,
    Tensor& indices,
    const Tensor& self,
    int64_t k,
    int64_t dim_,
    bool largest,
    bool sorted) {
  PT_KERNEL_BEGIN;

  int64_t dim = at::maybe_wrap_dim(dim_, self.dim(), /*wrap_scalar=*/true);
  TORCH_CHECK(dim == self.dim()-1, "topk supports sort along fastest changing dim only")
  TORCH_CHECK(self.dim() == 2, "topk supports 2D input tensors only")
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

  ns_TopK::Params params{};
  params.kSize = k;
  params.axis = self.dim() - dim - 1;
  std::vector<const at::Tensor*> pt_inputs{&self};
  std::vector<const at::Tensor*> pt_outputs{&values, &indices};
  synapse_simple_generic_kernel(
      pt_outputs,
      pt_inputs,
      "topk",
      &params,
      sizeof(params),
      SynapsePassType::NO_PASS);

  PT_KERNEL_END;
  return std::forward_as_tuple(values, indices);
}

std::tuple<Tensor, Tensor> topk_hpu(
    const Tensor& self,
    int64_t k,
    int64_t dim,
    bool largest,
    bool sorted) {
  PT_KERNEL_BEGIN;

  Tensor values = at::empty({0}, self.options());
  Tensor indices = at::empty({0}, self.options().dtype(c10::ScalarType::Int));
  topk_out_hpu(values, indices, self, k, dim, largest, sorted);

  PT_KERNEL_END;
  return std::make_tuple(values, indices);
}

/*************************************************************************
 * @brief Kernel implementation for sort OP
 *        out_sorted, out_indices = torch.sort(self, dim, descending)
 * @param [out] sorted - output tensor, 1-4D, FP32
 * @param [out] indices - output tensor, 1-4D, I32
 * @param [in] self - input tensor, 1-4D, FP32
 * @param [in] dim - along which dimension to sort, int64_t, default = -1
 * @param [in] descending - sorting order (ascending or descending), bool,
 *default = false
 ************************************************************************/
std::tuple<Tensor, Tensor> sort_hpu(
    const Tensor& self,
    int64_t dim,
    bool descending) {
  PT_KERNEL_BEGIN;

  int64_t dim_ = at::maybe_wrap_dim(dim, self.dim(), /*wrap_scalar=*/true);
  TORCH_CHECK(
      descending == true,
      "sort in descending order is only supported currently")
  TORCH_CHECK(dim_ == self.dim()-1,
      "sort is supported along fastest changing dim only")
  TORCH_CHECK(self.dim() == 2,
      "sort supports 2D input tensors only")

  Tensor values, indices;
  std::tie(values, indices) =
      at::topk(self, self.size(dim_), dim_, descending, true);

  PT_KERNEL_END;
  return std::make_tuple(values, indices);
}

static auto registry =
    torch::RegisterOperators()
        .op(torch::RegisterOperators::options()
                .schema(
                    "aten::topk(Tensor self, int k, int dim=-1, bool largest=True, bool sorted=True) -> (Tensor values, Tensor indices)")
                .impl_unboxedOnlyKernel<decltype(topk_hpu), &topk_hpu>(
                    DispatchKey::HABANATensorId)
                .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
        .op(torch::RegisterOperators::options()
                .schema(
                    "aten::topk.values(Tensor self, int k, int dim=-1, bool largest=True, bool sorted=True, *, Tensor(a!) values, Tensor(b!) indices) ->(Tensor(a!) values, Tensor(b!) indices)")
                .impl_unboxedOnlyKernel<decltype(topk_out_hpu), &topk_out_hpu>(
                    DispatchKey::HABANATensorId)
                .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
        .op(torch::RegisterOperators::options()
                .schema(
                    "aten::sort(Tensor self, int dim=-1, bool descending=False) -> (Tensor values, Tensor indices)")
                .impl_unboxedOnlyKernel<decltype(sort_hpu), &sort_hpu>(
                    DispatchKey::HABANATensorId)
                .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA));