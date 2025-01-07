/**
* Copyright (c) 2021-2024 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*/

#include "habana_eager/ops/view.h"
#include <ATen/InferSize.h>
#include <ATen/TensorUtils.h>
#include "backend/backend_meta.h"
#include "backend/habana_device/hpu_cached_devices.h"
#include "backend/helpers/get_n_bytes.h"
#include "habana_eager/eager_context.h"
#include "habana_eager/eager_pipeline_utils.h"
#include "habana_eager/ops/eager_op.h"
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
  auto result = alias_with_sizes_and_strides(self, inferred_size, *stride);
  // Pipeline_Or_Direct_view(self, out);
  auto src_backend = habana::eager::HbEagerTensorPool::get_backend_tensor(self);
  auto dst_backend =
      habana::eager::HbEagerTensorPool::get_backend_tensor(result);
  auto dst_hb_tmeta{habana::get_tensor_extra_meta(dst_backend)};
  dst_hb_tmeta->set_tensor_pipelined();
  auto pipeline_or_direct_view = [](const at::Tensor& self,
                                    const at::Tensor& result) {
    habana::eager::view_propagate_permutation(self, result);
  };
  pipeline_or_direct_generic(
      pipeline_or_direct_view, std::move(src_backend), std::move(dst_backend));

  return result;
}

void view_propagate_permutation(at::Tensor base_t, at::Tensor view_t) {
  PT_EAGER_TRACE;
  auto input_tmeta{habana::get_tensor_extra_meta(base_t)};
  auto input_smeta{habana::get_storage_extra_meta(base_t)};
  auto output_tmeta{habana::get_tensor_extra_meta(view_t)};
  auto output_smeta{habana::get_storage_extra_meta(view_t)};

  if (input_smeta == nullptr)
    return;

  HABANA_ASSERT(output_smeta);

  TORCH_CHECK(
      !input_tmeta->is_maybe_grad_view(),
      " Multilevel views on bucket grad view neither expected,  nor supported");

  // propagate the base size unconditionally.
  // This is important in multilevel views. Example: the first view can be
  // contiguous whereas the second one can be non-contiguous
  auto base_sizes = input_tmeta->is_view_tensor()
      ? input_smeta->get_base_tensor_size()
      : base_t.sizes();
  output_smeta->set_base_tensor_size(base_sizes.vec());
  output_tmeta->set_view_tensor();
}

at::Tensor alias(const at::Tensor& self) {
  PT_EAGER_TRACE;
  auto result =
      alias_with_sizes_and_strides(self, self.sizes(), self.strides());
  auto src_backend = habana::eager::HbEagerTensorPool::get_backend_tensor(self);
  auto dst_backend =
      habana::eager::HbEagerTensorPool::get_backend_tensor(result);
  auto dst_hb_tmeta{habana::get_tensor_extra_meta(dst_backend)};
  dst_hb_tmeta->set_tensor_pipelined();
  auto pipeline_or_direct_alias = [](const at::Tensor& self,
                                     const at::Tensor& result) {
    habana::eager::view_propagate_permutation(self, result);
  };
  pipeline_or_direct_generic(
      pipeline_or_direct_alias, std::move(src_backend), std::move(dst_backend));
  return result;
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
  auto base = at::empty(
      habana::get_base_tensor_size(self),
      self.options(),
      c10::MemoryFormat::Contiguous);

  base.unsafeGetTensorImpl()->set_storage_keep_dtype(self.storage());
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
