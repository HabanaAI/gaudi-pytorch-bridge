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
#include "backend/helpers/eager_pipeline.h"
#include "backend/synapse_helpers/env_flags.h"
#include "habana_eager/eager_context.h"
#include "pytorch_helpers/habana_device/hpu_cached_devices.h"

#include <torch/csrc/jit/ir/ir.h>

// PT_HPU_EAGER_FRONTEND is used to setup backend to work with Eager Flow
auto eager_frontend_enabled = []() {
  SET_ENV_FLAG_NEW(PT_HPU_EAGER_FRONTEND, true, 1);
  return 0;
}();

namespace habana {
namespace eager {
void EagerLoweringTask(
    at::Symbol symbol,
    std::vector<at::IValue> inputs,
    std::vector<OutputSpec> out_spec,
    EagerOpMetaData eager_op_meta_data) {
  habana::eager::EagerExec hlexec{
      std::move(symbol), std::move(inputs), std::move(out_spec)};

  hlexec.set_eager_op_info(std::move(eager_op_meta_data));

  // Launch the execution
  try {
    hlexec.launch();
  } catch (const std::exception& e) {
    PT_BRIDGE_WARN(
        "Exception caught in Lowering thread (will be rethrown in main thread)...\n",
        e.what());
    SingleTonEagerContext::getInstance().StoreLoweringThreadException(
        std::current_exception());

  } catch (...) {
    PT_BRIDGE_WARN(
        "Exception caught in Lowering thread (will be rethrown in main thread)...\n");
    SingleTonEagerContext::getInstance().StoreLoweringThreadException(
        std::current_exception());
  }
}

torch::jit::Stack EagerOpBase::run(std::vector<OutputSpec>&& out_spec) {
  auto stack = convert_inputs_to_backend_tensors(m_inputs);

  if (GET_ENV_FLAG_NEW(PT_HPU_EAGER_PIPELINE_ENABLE) &&
      m_is_pipeline_supported) {
    SingleTonEagerContext::getInstance().m_lowering_thread_handle =
        habana_helpers::SingleTonLoweringThreadPool::getInstance().enqueue(
            EagerLoweringTask,
            m_symbol,
            std::move(stack),
            std::move(out_spec),
            std::move(m_eager_op_meta_data));

    return {torch::jit::IValue()};

  } else {
    // To maintain the order for launch, ensure that all pending tasks in
    // pipeline are completed
    SingleTonEagerContext::getInstance().JoinPendingLoweringThread();

    habana::eager::EagerExec hlexec{
        m_symbol, std::move(stack), std::move(out_spec)};

    hlexec.set_eager_op_info(std::move(m_eager_op_meta_data));

    return hlexec.launch();
  }
}

} // namespace eager
} // namespace habana
