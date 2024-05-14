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

#define DEFINE_RANDOM_OP(op)                        \
  struct op : HabanaRandBase {                      \
    op(int device_id, c10::ScalarType scalar_type); \
  };

namespace habana {

DEFINE_OP(HabanaBernoulli)
DEFINE_OP(HabanaRandPermOp)
DEFINE_OP(HabanaNativeDropoutOp)
DEFINE_OP(HabanaRandPermOpDS)

struct HabanaRandBase : OpBackend {
  HabanaRandBase(
      int device_id,
      c10::ScalarType scalar_type,
      std::string_view kernel_name);
  void AddNode(synapse_helpers::graph&, const at::Stack&) override;
};

DEFINE_RANDOM_OP(HabanaRand)
DEFINE_RANDOM_OP(HabanaRandn)
DEFINE_RANDOM_OP(HabanaRandint)
DEFINE_RANDOM_OP(HabanaUniform)
DEFINE_RANDOM_OP(HabanaSeedGenerator)

struct HabanaMultinomial : OpBackend {
  HabanaMultinomial(int device_id, c10::ScalarType scalar_type);
  void CustomHandler(synapse_helpers::graph&, at::Stack&) override;
};

} // namespace habana
