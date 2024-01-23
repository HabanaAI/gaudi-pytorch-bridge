/******************************************************************************
 * Copyright (C) 2023-2024 Habana Labs, Ltd. an Intel Company
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

#define DEFINE_OP(op)                                                 \
  struct op : OpBackend {                                             \
    op(int device_id, c10::ScalarType scalar_type);                   \
    void AddNode(synapse_helpers::graph&, const at::Stack&) override; \
  };

namespace habana {

DEFINE_OP(HabanaBernoulli)
DEFINE_OP(HabanaRand)
DEFINE_OP(HabanaRandn)
DEFINE_OP(HabanaRandint)
DEFINE_OP(HabanaSeedGenerator)
DEFINE_OP(HabanaRandPermOp)

struct HabanaMultinomial : OpBackend {
  HabanaMultinomial(int device_id, c10::ScalarType scalar_type);
  void CustomHandler(synapse_helpers::graph&, at::Stack&) override;
};

OUTMETA_DECL(HabanaRandOutputMeta);
OUTMETA_DECL(HabanaRandintOutputMeta);
OUTMETA_DECL(HabanaMultinomialOutputMeta);
OUTMETA_DECL(HabanaSeedGeneratorOutputMeta);
OUTMETA_DECL(HabanaRandPermMeta);
} // namespace habana
