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

#include "habana_eager/ops/eager_op.h"
#include "habana_kernels/lazy_kernels.h"
#include "hpu_ops/hpu_op_helper.h"
#include "hpu_ops/op_backend.h"

namespace habana {

struct OptimizerLambNorm : OpBackend {
  OptimizerLambNorm(int device_id, c10::ScalarType scalar_type);

  void AddNode(synapse_helpers::graph&, const at::Stack&) override;
};

OUTMETA_DECL(ComputeLambOutputMetadata)

HPU_OP_FRONTEND(habana_lazy::LazyOp, LazyOptimizerLambNorm);
HPU_OP_FRONTEND(eager::EagerOp, EagerOptimizerLambNorm)

struct OptimizerLambPhase1 : OpBackend {
  OptimizerLambPhase1(int device_id, c10::ScalarType scalar_type);

  void AddNode(synapse_helpers::graph&, const at::Stack&) override;
};

struct OptimizerLambPhase2 : OpBackend {
  OptimizerLambPhase2(int device_id, c10::ScalarType scalar_type);

  void AddNode(synapse_helpers::graph&, const at::Stack&) override;
};

} // namespace habana
