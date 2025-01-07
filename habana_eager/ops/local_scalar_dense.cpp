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
#include "local_scalar_dense.h"
#include <ATen/Dispatch.h>
#include <ATen/core/TensorBody.h>
#include <ATen/ops/empty_like.h>
#include <pybind11/pybind11.h>
#include "backend/habana_device/HPUStream.h"
#include "common/utils.h"
#include "habana_eager/eager_context.h"
#include "habana_eager/eager_tensor.h"
#include "habana_eager/helpers.h"
#include "habana_helpers/frontend_utils.h"

void Copy_Scalar_To_Host_Execute_Task(
    const at::Tensor& src,
    const at::Tensor& dst,
    uint32_t size,
    c10::hpu::HPUStream stream) {
  habana_helpers::copy_scalar_to_host(src, dst.data_ptr(), size, stream);
}

void Copy_Scalar_To_Host_Empty_Compile_Task(
    const at::Tensor& src,
    const at::Tensor& dst,
    uint32_t size,
    c10::hpu::HPUStream stream) {
  habana::HPUDeviceContext::execute_thread().enqueue(
      Copy_Scalar_To_Host_Execute_Task,
      std::move(src),
      std::move(dst),
      size,
      std::move(stream));
  if (not GET_ENV_FLAG_NEW(PT_HPU_EAGER_4_STAGE_PIPELINE_ENABLE)) {
    habana::HPUDeviceContext::execute_thread().waitWorkComplete();
  }
}

void Copy_Scalar_To_Host_Empty_Lowering_Task(
    const at::Tensor& src,
    const at::Tensor& dst,
    uint32_t size,
    c10::hpu::HPUStream stream) {
  habana::HPUDeviceContext::compile_thread().enqueue(
      Copy_Scalar_To_Host_Empty_Compile_Task,
      std::move(src),
      std::move(dst),
      size,
      std::move(stream));
  if (not GET_ENV_FLAG_NEW(PT_HPU_EAGER_4_STAGE_PIPELINE_ENABLE)) {
    habana::HPUDeviceContext::compile_thread().waitWorkComplete();
  }
}

namespace habana {
namespace eager {
at::Scalar _local_scalar_dense_hpu(const at::Tensor& self) {
  c10::Scalar r;

  // Note:
  // 1. This macro expands to more types than HPU supports,
  //   but that should not be an issue issue.
  // 2. Pytorch uses this function to check a specific emement of a tensor
  //   eg. embedding_bag validates the first value offsets to be 0 using this
  //   function
  // 3. A TORCH_CHECK is added to ensure that the size at source
  //   matches with the destination.

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpragmas"
#pragma GCC diagnostic ignored "-Wunknown-warning-option"
// Due to runtime check cases detected by compiler won't appear.
// (clang does not have this check at all)
#pragma GCC diagnostic ignored "-Wstrict-aliasing"
#pragma GCC diagnostic ignored "-Warray-bounds"

  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(
      at::ScalarType::Bool,
      at::ScalarType::BFloat16,
      at::ScalarType::Half,
      self.scalar_type(),
      "_local_scalar_dense",
      [&] {
        scalar_t val;
        TORCH_CHECK(
            elementSize(self.scalar_type()) == sizeof(val),
            " source and destination size mismatch");
        auto dst =
            at::empty_like(self, self.options().device(c10::DeviceType::CPU));
        auto src_backend = HbEagerTensorPool::get_backend_tensor(self);
        auto dst_backend = HbEagerTensorPool::get_backend_tensor(dst);
        habana::eager::ScheduleWorkAndUpdateLoweringThreadHandle(
            Copy_Scalar_To_Host_Empty_Lowering_Task,
            std::move(src_backend),
            std::move(dst_backend),
            sizeof(val),
            c10::hpu::getCurrentHPUStream());
        /*
         * eager::Joinpending to ensure the copy_scalar_to_host is completed in
         * the execute thread i.e. copy op is queued on the copy d2h stream.
         * If src tensor data is not ready example pending in either h2d or
         * compute etc. Copy d2h stream on the device side will wait for the
         * event.
         * On the host side, copy_scalar_to_host further waits for the
         * completion of the copy.
         */
        habana::eager::JoinPendingPipelineThreads();
        val = *reinterpret_cast<scalar_t*>(dst.data_ptr());
        //  copy_from_ operator is doing implicit down/upcasting
        //  for Long and Double. _local_scalar_dense_hpu needs to preserve it
        if (self.scalar_type() == c10::ScalarType::Long &&
            !common::IsInt64Supported()) {
          val = *reinterpret_cast<int32_t*>(&val);
        } else if (self.scalar_type() == c10::ScalarType::Double) {
          val = *reinterpret_cast<float*>(&val);
        }
        r = c10::Scalar(val);
      });

#pragma GCC diagnostic pop
  return r;
}
} // namespace eager
} // namespace habana
