/******************************************************************************
 * Copyright (C) 2022 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */
#pragma once
#include "habana_lazy/ir.h"

// The Side-By-Side (SBS) Debug Tool is a debug capability for comparing
// between tensors that are calculated by HPU to tensors that are calculated
// by CPU.
// Run it by adding the env var PT_SBS with one of the enum values described
// here: sbs_runner.h :: SBSModes
// See more here:
// https://confluence.habana-labs.com/display/SYN/Side-By-Side+Debug+Tool

namespace habana_lazy {

enum SBSModes : unsigned {
  SBS_MODE_DISABLED = 0,
  SBS_MODE_STANDALONE = 1,
  SBS_MODE_USE_CPU_INPUT = 2,
  SBS_MODE_USE_HPU_INPUT = 3
};

class SBSRunner {
 public:
  static void populateInputForCPUOp(
      const std::vector<at::IValue>& inputs,
      const ir::MetaData& metadata,
      std::vector<at::IValue>& stack);

  static void run(
      at::TensorList results,
      const std::vector<at::IValue>& inputs,
      const std::vector<at::IValue>& prealloc_stack =
          std::vector<at::IValue>());
};

} // namespace habana_lazy