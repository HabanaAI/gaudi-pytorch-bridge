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

#include "habana_eager/ops/as_strided.h"
#include <ATen/native/Resize.h>
#include "habana_eager/eager_context.h"
#include "habana_eager/eager_pipeline_utils.h"
#include "habana_eager/ops/eager_op.h"
#include "habana_eager/ops/view.h"
#include "pytorch_helpers/habana_helpers/misc_utils.h"
namespace habana {
namespace eager {
at::Tensor as_strided_hpu(
    const at::Tensor& self,
    c10::SymIntArrayRef size,
    c10::SymIntArrayRef stride,
    c10::optional<c10::SymInt> storage_offset_) {
  auto storage_offset = storage_offset_.value_or(self.storage_offset());
  at::Tensor result = at::detail::make_tensor<at::TensorImpl>(
      c10::TensorImpl::VIEW,
      c10::Storage(self.storage()),
      self.key_set(),
      self.dtype());
  at::native::setStrided(result, size, stride, storage_offset);
  auto src_backend = habana::eager::HbEagerTensorPool::get_backend_tensor(self);
  auto dst_backend =
      habana::eager::HbEagerTensorPool::get_backend_tensor(result);
  auto dst_hb_tmeta{habana::get_tensor_extra_meta(dst_backend)};
  dst_hb_tmeta->set_tensor_pipelined();
  auto pipeline_or_direct_as_strided = [](const at::Tensor& self,
                                          const at::Tensor& result) {
    habana::eager::view_propagate_permutation(self, result);
  };
  habana::eager::pipeline_or_direct_generic(
      pipeline_or_direct_as_strided,
      std::move(src_backend),
      std::move(dst_backend));

  return result;
}

} // namespace eager
} // namespace habana
