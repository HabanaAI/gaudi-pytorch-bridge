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
#pragma once

#include "backend/helpers/tensor_utils.h"
#include "backend/kernel/hpu_habana_launch_op_pt.h"

namespace habana {
class HabanaCompile {
 public:
  explicit HabanaCompile() {}
  virtual ~HabanaCompile() {}
  static void CompileSynapse(
      bool is_shape_agnostic_cache_miss,
      std::shared_ptr<HabanaLaunchOpPT> hb_launch_op,
      size_t graph_key_with_perm,
      bool do_nothing_compile,
      bool do_nothing_execute,
      bool dry_run);
};
} // namespace habana
