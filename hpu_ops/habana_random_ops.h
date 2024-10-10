/*******************************************************************************
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

#define DEFINE_RANDOM_CHECKPOINT_OP(op)             \
  struct op : HabanaRandCheckpointBase {            \
    op(int device_id, c10::ScalarType scalar_type); \
  };

#define REGISTER_RANDOM_OP(name, Name)                              \
  add("hpu::habana_" #name, KERNEL_FN_GLOBAL(habana::Habana##Name)) \
      .add(                                                         \
          "hpu::habana_" #name "_checkpoint",                       \
          KERNEL_FN_GLOBAL(habana::Habana##Name##Checkpoint))

namespace habana {

OutputMetaData SeedOutputMeta();

struct HabanaRandomBase : OpBackend {
  HabanaRandomBase(
      int device_id,
      std::string_view kernel_name,
      c10::ScalarType scalar_type,
      std::vector<int> res_ids);
  void AddNodeCommon(synapse_helpers::graph&, const at::Stack&, bool);
};

struct HabanaRandBase : HabanaRandomBase {
  HabanaRandBase(
      int device_id,
      std::string_view kernel_name,
      c10::ScalarType scalar_type);
  void AddNode(synapse_helpers::graph&, const at::Stack&) override;
};

struct HabanaRandCheckpointBase : HabanaRandomBase {
  HabanaRandCheckpointBase(
      int device_id,
      std::string_view kernel_name,
      c10::ScalarType scalar_type);
  void AddNode(synapse_helpers::graph&, const at::Stack&) override;
};

struct HabanaRandint : HabanaRandBase {
  HabanaRandint(int device_id, c10::ScalarType scalar_type);
  void AddNode(synapse_helpers::graph&, const at::Stack&) override;
};

struct HabanaRandintCheckpoint : HabanaRandCheckpointBase {
  HabanaRandintCheckpoint(int device_id, c10::ScalarType scalar_type);
  void AddNode(synapse_helpers::graph&, const at::Stack&) override;
};

struct HabanaMultinomial : OpBackend {
  HabanaMultinomial(int device_id, c10::ScalarType scalar_type);
  void CustomHandler(synapse_helpers::graph&, at::Stack&) override;
};

DEFINE_OP(HabanaBernoulli)
DEFINE_OP(HabanaBernoulliCheckpoint)
DEFINE_OP(HabanaBernoulliP)
DEFINE_OP(HabanaBernoulliTensor)
DEFINE_OP(HabanaBernoulliSize)
DEFINE_OP(HabanaPoisson)
DEFINE_OP(HabanaPoissonCheckpoint)
DEFINE_OP(HabanaRandPermOp)
DEFINE_OP(HabanaRandPermOpCheckpoint)
DEFINE_OP(HabanaNativeDropoutOp)
DEFINE_OP(HabanaNativeDropoutOpCheckpoint)
DEFINE_OP(HabanaRandPermOpDS)
DEFINE_OP(HabanaMultinomialCheckpoint)

DEFINE_RANDOM_OP(HabanaRand)
DEFINE_RANDOM_OP(HabanaRandn)
DEFINE_RANDOM_OP(HabanaUniform)
DEFINE_RANDOM_OP(HabanaSeedGenerator)

DEFINE_RANDOM_CHECKPOINT_OP(HabanaRandCheckpoint)
DEFINE_RANDOM_CHECKPOINT_OP(HabanaRandnCheckpoint)
DEFINE_RANDOM_CHECKPOINT_OP(HabanaUniformCheckpoint)

} // namespace habana
