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
#include "backend/helpers/eager_pipeline.h"
#include "habana_eager/eager_context.h"
#include "habana_eager/ops/eager_op.h"
#include "habana_eager/ops/view.h"
#include "habana_kernels/kernel_utils.h"
#include "pytorch_helpers/habana_helpers/misc_utils.h"
namespace habana {
namespace eager {
// TODO: move to generic lowering code for StorageExtraMeta
void as_strided_Execute_Empty_Task() {}

void as_strided_Compile_Empty_Task() {
  habana_helpers::Singleton_ExecThreadPool::getInstance().Enqueue(
      as_strided_Execute_Empty_Task);

  if (not GET_ENV_FLAG_NEW(PT_HPU_EAGER_4_STAGE_PIPELINE_ENABLE)) {
    habana_helpers::Singleton_ExecThreadPool::getInstance().JoinPendingThread();
  }
}

void as_strided_Lowering_Task(const at::Tensor& src, const at::Tensor& dst) {
  habana::eager::view_propagate_permutation(src, dst);
  habana_helpers::set_output_hw_scaling_meta(src, dst);
  habana_helpers::Singleton_CompileThreadPool::getInstance().Enqueue(
      as_strided_Compile_Empty_Task);
  if (not GET_ENV_FLAG_NEW(PT_HPU_EAGER_4_STAGE_PIPELINE_ENABLE)) {
    habana_helpers::Singleton_CompileThreadPool::getInstance()
        .JoinPendingThread();
  }
}

void Pipeline_Or_Direct_as_strided(
    const at::Tensor& self,
    const at::Tensor& result) {
  bool pipeline_flag = GET_ENV_FLAG_NEW(PT_HPU_EAGER_PIPELINE_ENABLE);
  if (pipeline_flag) {
    auto src_backend = HbEagerTensorPool::get_backend_tensor(self);
    auto dst_backend = HbEagerTensorPool::get_backend_tensor(result);
    // Set pipeline metadata on the dst hpu tensor
    auto dst_hb_tmeta{habana::get_tensor_extra_meta(dst_backend)};
    dst_hb_tmeta->set_tensor_pipelined();
    habana::eager::SingleTonEagerContext::getInstance()
        .ScheduleWorkAndUpdateLoweringThreadHandle(
            as_strided_Lowering_Task,
            std::move(src_backend),
            std::move(dst_backend));
  } else {
    habana::eager::JoinPendingPipelineThreads();
    habana::eager::view_propagate_permutation(self, result);
    habana_helpers::set_output_hw_scaling_meta(self, result);
  }
}

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

  Pipeline_Or_Direct_as_strided(self, result);
  return result;
}

} // namespace eager
} // namespace habana
