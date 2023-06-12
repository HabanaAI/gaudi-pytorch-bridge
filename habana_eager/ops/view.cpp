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

#include "habana_eager/ops/view.h"
#include <ATen/InferSize.h>
#include <ATen/TensorUtils.h>
#include "backend/backend_meta.h"
#include "backend/habana_device/hpu_cached_devices.h"
#include "backend/helpers/eager_pipeline.h"
#include "backend/helpers/get_n_bytes.h"
#include "habana_eager/eager_context.h"
#include "habana_kernels/kernel_utils.h"

namespace habana {
namespace eager {
at::Tensor view_hpu(const at::Tensor& self, c10::SymIntArrayRef size) {
  PT_EAGER_TRACE;
  auto inferred_size = at::infer_size_dv(size, self.numel());
  auto stride = at::detail::computeStride(
      self.sym_sizes(), self.sym_strides(), inferred_size);
  TORCH_CHECK(
      stride.has_value(),
      "view size is "
      "not compatible with input tensor's size and stride (at least one dimension"
      " spans across two contiguous subspaces). Use .reshape(...) instead.");
  auto out = alias_with_sizes_and_strides(self, inferred_size, *stride);

  view_propagate_permutation(self, out);
  return out;
}

void view_propagate_permutation_task(at::Tensor base_t, at::Tensor view_t) {
  PT_EAGER_TRACE;

  auto input_tmeta{habana::get_tensor_extra_meta(base_t)};
  auto output_tmeta{habana::get_tensor_extra_meta(view_t)};

  // once we set view tensor, JIT IR pass will get invoked.
  // We need JIT IT pass under the following cases
  // base has permutation or the view is non-contiguous
  auto base_permute = input_tmeta->get_memory_permutation();
  if (base_permute.size() != 0) {
    output_tmeta->set_memory_permutation(base_permute);
  }

  output_tmeta->set_view_lowering(
      (base_permute.size() != 0) || (!view_t.is_contiguous()));
}

void view_propagate_permutation(at::Tensor base_t, at::Tensor view_t) {
  PT_EAGER_TRACE;
  auto input_tmeta{habana::get_tensor_extra_meta(base_t)};
  auto output_tmeta{habana::get_tensor_extra_meta(view_t)};

  // propagate the base size unconditionally.
  // This is important in multilevel views. Example: the first view can be
  // contiguous whereas the second one can be non-contiguous
  auto base_sizes = input_tmeta->is_view_tensor()
      ? input_tmeta->get_base_tensor_size()
      : base_t.sizes();
  output_tmeta->set_base_tensor_size(base_sizes.vec());

  output_tmeta->set_view_tensor();

  if (GET_ENV_FLAG_NEW(PT_HPU_EAGER_PIPELINE_ENABLE)) {
    SingleTonEagerContext::getInstance().m_lowering_thread_handle =
        habana_helpers::SingleTonLoweringThreadPool::getInstance().enqueue(
            view_propagate_permutation_task,
            std::move(base_t),
            std::move(view_t));
  } else {
    view_propagate_permutation_task(std::move(base_t), std::move(view_t));
  }
}

at::Tensor alias(const at::Tensor& self) {
  PT_EAGER_TRACE;
  auto out = alias_with_sizes_and_strides(self, self.sizes(), self.strides());
  view_propagate_permutation(self, out);
  return out;
}

at::Tensor unfold(
    const at::Tensor& self,
    int64_t d,
    int64_t size,
    int64_t step) {
  PT_EAGER_TRACE;
  return at::native::unfold(self, d, size, step);
}

at::Tensor create_base(const at::Tensor& self) {
  auto self_impl = self.unsafeGetTensorImpl();
  auto self_tmeta{habana::get_tensor_extra_meta(self)};
  at::Tensor base;
  if (self_tmeta->get_memory_permutation().size()) {
    base = at::empty(
        self_tmeta->get_base_tensor_size(),
        self.options(),
        c10::MemoryFormat::Contiguous);
  } else {
    auto base_size = (int64_t)(
        habana_helpers::GetNBytes(self_impl) /
        c10::elementSize(habana_helpers::getInternalDtype(self.scalar_type())));
    base = at::empty(base_size, self.options(), c10::MemoryFormat::Contiguous);
  }

  base.unsafeGetTensorImpl()->set_storage_keep_dtype(self.storage());

  // propagate permutation to base
  auto base_tmeta{habana::get_tensor_extra_meta(base)};
  base_tmeta->set_memory_permutation(self_tmeta->get_memory_permutation());
  return base;
}

} // namespace eager
} // namespace habana

TORCH_LIBRARY_IMPL(aten, HPU, m) {
  m.impl(
      "alias",
      static_cast<at::Tensor (*)(const at::Tensor&)>(&habana::eager::alias));
  m.impl(
      "unfold",
      static_cast<at::Tensor (*)(
          const at::Tensor& self, int64_t d, int64_t size, int64_t step)>(
          &habana::eager::unfold));
}
