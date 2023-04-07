/******************************************************************************
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

#include "hpu_ops/hpu_op_helper.h"
#include "hpu_ops/op_backend.h"

namespace habana {

struct OptimizerFusedLambNorm : OpBackend {
  OptimizerFusedLambNorm(int device_id, c10::ScalarType scalar_type);

  void AddNode(synapse_helpers::graph&, const at::Stack&) override;
  void CustomHandler(synapse_helpers::graph&, at::Stack&) override;
};

OUTMETA_DECL(ComputeLambOutputMetadata)

HPU_OP_FRONTEND(habana_lazy::LazyOp, LazyOptimizerLambFusedNorm);
HPU_OP_FRONTEND(eager::EagerOp, EagerOptimizerLambFusedNorm)

} // namespace habana