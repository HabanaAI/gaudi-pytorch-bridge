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
#include "habana_eager/ops/eager_op.h"

#include <torch/csrc/jit/ir/ir.h>

#include "backend/habana_device/hpu_cached_devices.h"
#include "backend/synapse_helpers/env_flags.h"
#include "habana_eager/eager_context.h"
#include "pytorch_helpers/habana_helpers/thread_pool/thread_pool.h"

namespace habana {
namespace eager {
void EagerLoweringTask(
    at::Symbol symbol,
    std::vector<at::IValue>&& inputs,
    OutputSpecsOrTensors&& out_spec_or_tensors,
    EagerOpMetaData&& eager_op_meta_data) {
  habana::eager::EagerExec hlexec{
      std::move(symbol),
      std::move(inputs),
      std::move(out_spec_or_tensors),
      true};

  hlexec.set_eager_op_info(std::move(eager_op_meta_data));
  // Launch the execution
  hlexec.launch();
}

void EagerOpBase::validate_inputs(
    const std::vector<at::IValue>& inputs,
    const std::string& qualstring) {
  for (size_t idx = 0; idx < inputs.size(); ++idx) {
    auto& t = inputs[idx];
    if (!t.isTensor()) {
      continue;
    }

    auto tensor = t.toTensor();
    if (!tensor.defined()) {
      continue;
    }

    /* The 3rd input for the masked fill, if placed on the CPU, should be
     * converted to Scalar. This is an exception for the masked_fill operation.
     * Pytorch accepts the 3rd input on the CPU when the operation is performed
     * on cuda/xpu */
    std::string maskedFillPrefix = "aten::masked_fill";
    if ((tensor.device().type() == c10::DeviceType::HPU) ||
        (idx == 2 &&
         std::equal(
             std::begin(maskedFillPrefix),
             std::end(maskedFillPrefix),
             std::begin(qualstring)))) {
      continue;
    }

    if (tensor.unsafeGetTensorImpl()->is_wrapped_number()) {
      continue;
    }

    HABANA_ASSERT(
        0,
        "Expected all tensors to be on the HPU device, but found at least one input[idx=",
        idx,
        "] on ",
        tensor.device(),
        " (details: ",
        tensor.toString(),
        ")");
  }
}

std::mutex EagerOpBase::m_mutex;

void EagerOpBase::run(OutputSpecsOrTensors&& out_spec_or_tensors) {
  auto stack = convert_ivalues_to_backend_tensors(m_inputs, m_symbol);
  if (GET_ENV_FLAG_NEW(PT_HPU_EAGER_PIPELINE_ENABLE)) {
    for (const at::IValue& ivalue : stack) {
      if (ivalue.isTensor()) {
        auto hb_tmeta{habana::get_tensor_extra_meta(ivalue.toTensor())};
        hb_tmeta->set_tensor_pipelined();
      }
    }
    SingleTonEagerContext::getInstance()
        .ScheduleWorkAndUpdateLoweringThreadHandle(
            EagerLoweringTask,
            m_symbol,
            std::move(stack),
            std::move(out_spec_or_tensors),
            std::move(m_eager_op_meta_data));

  } else {
    // To maintain the order for launch, ensure that all pending tasks in
    // pipeline are completed
    habana::eager::JoinPendingPipelineThreads();

    habana::eager::EagerExec hlexec{
        m_symbol, std::move(stack), std::move(out_spec_or_tensors), false};

    hlexec.set_eager_op_info(std::move(m_eager_op_meta_data));
    std::lock_guard lock(m_mutex);
    hlexec.launch();
  }
}

} // namespace eager
} // namespace habana
