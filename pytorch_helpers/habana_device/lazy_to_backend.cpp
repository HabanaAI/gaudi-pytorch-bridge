/*******************************************************************************
 * Copyright (C) 2023 Habana Labs, Ltd. an Intel Company
 * All Rights Reserved.
 *
 * Unauthorized copying of this file or any element(s) within it, via any medium
 * is strictly prohibited.
 * This file contains Habana Labs, Ltd. proprietary and confidential information
 * and is subject to the confidentiality and license agreements under which it
 * was provided.
 *
 *******************************************************************************
 */

#include "backend/lazy_to_backend.h"
#include "habana_lazy/aten_lazy_bridge.h"
#include "habana_lazy/lazy_executor.h"

bool lazy_to_backend::is_const_tensor(const at::Tensor& tensor) {
  const auto& hb_tensor = habana_lazy::GetHbInternalTensorImpl(tensor);
  return hb_tensor->IsConstTensor();
}

void* lazy_to_backend::host_ptr_for_const_tensor(const at::Tensor& tensor) {
  const auto& hb_tensor = habana_lazy::GetHbInternalTensorImpl(tensor);
  return hb_tensor->get_host_ptr();
}

std::tuple<synapse_helpers::layouts::MemoryPermutation, bool> lazy_to_backend::
    get_memory_permutation(const at::Tensor& tensor) {
  // It should be handled in SW-122018
  if (GET_ENV_FLAG_NEW(PT_HPU_EAGER_OPS)) {
    PT_EAGER_DEBUG(
        "Skipping permutations for EagerOp with duplicate inputs...");
    return {synapse_helpers::layouts::MemoryPermutation{}, false};
  }
  auto hb_weight_impl = habana_lazy::GetHbInternalTensorImpl(tensor);
  if (hb_weight_impl)
    return {
        hb_weight_impl->GetMemoryPermutation(),
        hb_weight_impl->GetDontAllowPermutation()};
  return {synapse_helpers::layouts::MemoryPermutation{}, false};
}

bool lazy_to_backend::is_shape_tensor(const at::Tensor& tensor) {
  if (GET_ENV_FLAG_NEW(PT_HPU_EAGER_OPS)) {
    return false;
  }
  auto hb_weight_impl = habana_lazy::GetHbInternalTensorImpl(tensor);
  return hb_weight_impl->isShapeTensor();
}

bool lazy_to_backend::is_lazy_inference_call_context() {
  if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) != 0) {
    if (!habana_lazy::isDeviceInLoweringMode()) {
      // Lazy mode shape inference call, early return without execution
      return true;
    }
  }
  return false;
}

at::Tensor lazy_to_backend::create_empty_tensor(const PtTensorInfo& ti) {
  auto pt_tensor = at::empty(ti.get_shape(), ti.get_topts(), ti.get_mf());
  if (GET_ENV_FLAG_NEW(PT_HPU_EAGER_OPS)) {
    return pt_tensor;
  }
  auto hb_internal_tensor = habana_lazy::GetHbInternalTensorImpl(pt_tensor);
  PT_BRIDGE_DEBUG(
      "Cache created a BE tensor, HbInternal address: ", hb_internal_tensor);
  TORCH_CHECK(
      hb_internal_tensor != nullptr,
      "Tensor for ",
      ti.get_ir_name(),
      " does not have HbInternalTensor");
  auto internal_lf = hb_internal_tensor->GetTensorLayout();
  auto internal_lf_new = ti.getHbInternalLayoutFormat();
  if (internal_lf != internal_lf_new) {
    PT_BRIDGE_DEBUG(
        "For ",
        ti.get_ir_name(),
        " updating HbInternalTensorImpl layout from ",
        internal_lf,
        " to ",
        internal_lf_new);
    hb_internal_tensor->SetTensorLayout(internal_lf_new);
  }
  if (GET_ENV_FLAG_NEW(PT_HPU_ENABLE_SYNAPSE_LAYOUT_HANDLING)) {
    PT_BRIDGE_DEBUG(
        "Setting synapse permutation as saved in the cache to the output tensor id: ",
        ti.get_tensor_id(),
        " permutation: ",
        VecToString(ti.getHbInternalPermute()));
    hb_internal_tensor->SetMemoryPermutation(ti.getHbInternalPermute());
  }
  return pt_tensor;
}

void lazy_to_backend::set_memory_permutations(
    at::Tensor& tensor,
    synapse_helpers::layouts::MemoryPermutation permutation,
    const synRetrievedLaunchTensorInfoExt* info) {
  // It should be handled in SW-122018
  if (GET_ENV_FLAG_NEW(PT_HPU_EAGER_OPS)) {
    PT_EAGER_DEBUG("Skipping permutations for EagerOp...");
    return;
  }
  auto impl = habana_lazy::GetHbInternalTensorImpl(tensor);
  TORCH_CHECK(
      impl,
      "Failed to set the permutation because the BE tensor has no internal impl");

  PT_BRIDGE_DEBUG(
      "Updating the PT tensor HbInternalTensorImpl address: ",
      impl,
      " storage address : ",
      impl->data(),
      " with permutation: ",
      VecToString(permutation),
      " old permutation was: ",
      VecToString(impl->GetMemoryPermutation()));

  if (permutation.size() != tensor.sizes().size()) {
    if (!permutation.empty()) {
      if (info)
        PT_BRIDGE_WARN(
            "wrong permute size - info.tensorId=",
            info->tensorId,
            " tensor name: ",
            info->tensorName,
            "  permute_vec.size = ",
            permutation.size(),
            "  PT tensor shape.dims =",
            tensor.sizes().size(),
            " PT shape: ",
            VecToString(tensor.sizes().vec()),
            " synapse returned tensor dims: ",
            info->tensorDims,
            " synapse returned tensor shape: ",
            VecToString(std::vector<uint64_t>(
                info->tensorMaxSize, info->tensorMaxSize + info->tensorDims)));
      HABANA_ASSERT(false);
    }
  }
  impl->SetMemoryPermutation(permutation);
}

void lazy_to_backend::set_tensor_layout_format(
    at::Tensor& tensor,
    habana::LayoutFormat layout) {
  // It should be handled in SW-122018
  if (GET_ENV_FLAG_NEW(PT_HPU_EAGER_OPS)) {
    PT_EAGER_DEBUG("Skipping setting layout for EagerOp...");
    return;
  }
  auto impl = habana_lazy::GetHbInternalTensorImpl(tensor);
  TORCH_CHECK(
      impl,
      "Failed to set tensor layout because the BE tensor has no internal impl");
  impl->SetTensorLayout(layout);
}

habana::LayoutFormat lazy_to_backend::get_tensor_layout_format(
    const at::Tensor& tensor) {
  // It should be handled in SW-122018
  if (GET_ENV_FLAG_NEW(PT_HPU_EAGER_OPS)) {
    PT_EAGER_DEBUG("Returning fixed NCHW layout for EagerOp...");
    return habana::LayoutFormat::NCHW;
  }
  auto impl = habana_lazy::GetHbInternalTensorImpl(tensor);
  TORCH_CHECK(
      impl,
      "Failed to get tensor layout because the BE tensor has no internal impl");
  return impl->GetTensorLayout();
}

void lazy_to_backend::set_host_ptr(const at::Tensor& tensor, void* host_ptr) {
  if (GET_ENV_FLAG_NEW(PT_HPU_EAGER_OPS)) {
    PT_EAGER_DEBUG("Skipping host ptr for EagerOp...");
    return;
  }
  auto impl = habana_lazy::GetHbInternalTensorImpl(tensor);
  TORCH_CHECK(impl, "not a lazy tensor");
  return impl->set_host_ptr(host_ptr);
}

void* lazy_to_backend::get_host_ptr(const at::Tensor& tensor) {
  if (GET_ENV_FLAG_NEW(PT_HPU_EAGER_OPS)) {
    PT_EAGER_DEBUG("returning null host ptr for EagerOp...");
    return nullptr;
  }
  auto impl = habana_lazy::GetHbInternalTensorImpl(tensor);
  TORCH_CHECK(impl, "not a lazy tensor");
  return impl->get_host_ptr();
};

std::string lazy_to_backend::detail::
    InternalFormatter<lazy_to_backend::FormatTokens>::format(
        const at::Tensor& tensor,
        lazy_to_backend::FormatTokens token) {
  if (GET_ENV_FLAG_NEW(PT_HPU_EAGER_OPS)) {
    PT_EAGER_DEBUG("Skipping permutations for EagerOp...");
    return "<EAGER>";
  }
  auto impl = habana_lazy::GetHbInternalTensorImpl(tensor);
  if (!impl) {
    return "<NOT_A_LAZY_TENSOR>";
  }
  switch (token) {
    case lazy_to_backend::FormatTokens::Permutations:
      return VecToString(impl->GetMemoryPermutation());
    case lazy_to_backend::FormatTokens::Layout:
      return habana::DebugString(impl->GetTensorLayout());
    case lazy_to_backend::FormatTokens::ImplPtr:
      return absl::StrCat(absl::Hex(impl, absl::kZeroPad8));
    case lazy_to_backend::FormatTokens::DataPtr:
      return absl::StrCat(absl::Hex(impl->data(), absl::kZeroPad8));
  }
  return "";
}
