/******************************************************************************
 * Copyright (C) 2022 Habana Labs, Ltd. an Intel Company
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

#include "hpu_ops/hpu_op_helper.h"
#include "hpu_ops/op_backend.h"

namespace habana {
typedef struct NonZeroParams {
  c10::ScalarType dtype;
  std::vector<int64_t> sizes;
  int64_t numel;
  bool force_long;
} NonZeroParams_t;
std::vector<int64_t> compute_nonzero_output_shape(
    NonZeroParams_t self_params,
    bool use_tpc_impl = false);
struct NonZeroEager : OpBackend {
  NonZeroEager(int device_id, c10::ScalarType scalar_type);
  void AddNode(synapse_helpers::graph&, const at::Stack&) override;
};

} // namespace habana
