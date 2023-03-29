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
#include "backend/synapse_helpers/env_flags.h"
#include "pytorch_helpers/habana_device/hpu_cached_devices.h"

#include <torch/csrc/jit/ir/ir.h>

// PT_HPU_EAGER_FRONTEND is used to setup backend to work with Eager Flow
auto eager_frontend_enabled = []() {
  SET_ENV_FLAG_NEW(PT_HPU_EAGER_FRONTEND, true, 1);
  return 0;
}();

namespace habana {
namespace eager {
torch::jit::Stack EagerOpBase::run(std::vector<OutputSpec>&& out_spec) {
  auto input_tensors = convert_inputs_to_backend_tensors(m_inputs);
  habana::eager::EagerExec hlexec{
      m_symbol,
      std::move(input_tensors),
      std::move(m_inputs),
      std::move(out_spec)};

  hlexec.set_eager_op_info(m_eager_op_meta_data);

  // Launch the execution
  return hlexec.launch();
}

} // namespace eager
} // namespace habana
