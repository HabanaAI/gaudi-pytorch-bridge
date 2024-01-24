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

#include "habana_eager/ops/as_strided.h"
#include <ATen/native/Resize.h>
#include "habana_eager/ops/view.h"
#include "habana_kernels/kernel_utils.h"
#include "pytorch_helpers/habana_helpers/misc_utils.h"
namespace habana {
namespace eager {
at::Tensor as_strided_hpu(
    const at::Tensor& self,
    c10::SymIntArrayRef size,
    c10::SymIntArrayRef stride,
    c10::optional<c10::SymInt> storage_offset_) {
  auto storage_offset = storage_offset_.value_or(self.storage_offset());
  auto result = at::detail::make_tensor<at::TensorImpl>(
      c10::TensorImpl::VIEW,
      c10::Storage(self.storage()),
      self.key_set(),
      self.dtype());
  at::native::setStrided(result, size, stride, storage_offset);
  if (auto backend_meta =
          (const_cast<c10::TensorImpl*>(self.unsafeGetTensorImpl()))
              ->get_backend_meta()) {
    auto hb_backend_meta = dynamic_cast<habana::TensorExtraMeta*>(backend_meta);
    if (hb_backend_meta->is_tensor_pipelined()) {
      habana::TryJoinPendingEagerPipelineThreads();
    }
  }

  habana::eager::view_propagate_permutation(self, result);
  habana_helpers::set_output_hw_scaling_meta(self, result);
  return result;
}
} // namespace eager
} // namespace habana
