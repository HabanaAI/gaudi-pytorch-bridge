/**
* Copyright (c) 2021-2024 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
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
