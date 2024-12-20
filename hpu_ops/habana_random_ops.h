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

#define DEFINE_BASE_RANDOM_OP(op)                   \
  struct op : HabanaRandomBase {                    \
    op(int device_id, c10::ScalarType scalar_type); \
  };

#define DEFINE_RANDOM_OP(op)                                          \
  struct op : HabanaRandomBase {                                      \
    op(int device_id, c10::ScalarType scalar_type);                   \
    void AddNode(synapse_helpers::graph&, const at::Stack&) override; \
  };

#define DEFINE_BASE_RANDOM_CHECKPOINT_FORWARD_OP(op)            \
  struct op##Checkpoint : HabanaRandCheckpointBase {            \
    op##Checkpoint(int device_id, c10::ScalarType scalar_type); \
  };

#define DEFINE_RANDOM_CHECKPOINT_FORWARD_OP(op)                       \
  struct op##Checkpoint : HabanaRandCheckpointBase {                  \
    op##Checkpoint(int device_id, c10::ScalarType scalar_type);       \
    void AddNode(synapse_helpers::graph&, const at::Stack&) override; \
  };

#define DEFINE_RANDOM_CHECKPOINT_BASE_OP(op)                          \
  struct op##Base : HabanaRandomBase {                                \
    op##Base(                                                         \
        int device_id,                                                \
        c10::ScalarType scalar_type,                                  \
        bool is_deterministic);                                       \
    void AddNode(synapse_helpers::graph&, const at::Stack&) override; \
  };

#define DEFINE_BASE_RANDOM_CHECKPOINT_BASE_OP(op) \
  struct op##Base : HabanaRandomBase {            \
    op##Base(                                     \
        int device_id,                            \
        c10::ScalarType scalar_type,              \
        bool is_deterministic);                   \
  };

#define DEFINE_RANDOM_CHECKPOINT_BACKWARD_OP(op)                  \
  struct op : op##Base {                                          \
    op(int device_id, c10::ScalarType scalar_type)                \
        : op##Base(device_id, scalar_type, false){};              \
  };                                                              \
                                                                  \
  struct op##CheckpointBwd : op##Base {                           \
    op##CheckpointBwd(int device_id, c10::ScalarType scalar_type) \
        : op##Base(device_id, scalar_type, true){};               \
  };

#define DEFINE_BASE_CHECKPOINT_OP(op)          \
  DEFINE_BASE_RANDOM_CHECKPOINT_FORWARD_OP(op) \
  DEFINE_BASE_RANDOM_CHECKPOINT_BASE_OP(op)    \
  DEFINE_RANDOM_CHECKPOINT_BACKWARD_OP(op)

#define DEFINE_CHECKPOINT_OP(op)          \
  DEFINE_RANDOM_CHECKPOINT_FORWARD_OP(op) \
  DEFINE_RANDOM_CHECKPOINT_BASE_OP(op)    \
  DEFINE_RANDOM_CHECKPOINT_BACKWARD_OP(op)

#define REGISTER_RANDOM_CHECKPOINT_OP(name, Name)                   \
  add("hpu::habana_" #name, KERNEL_FN_GLOBAL(habana::Habana##Name)) \
      .add(                                                         \
          "hpu::habana_" #name "_checkpoint",                       \
          KERNEL_FN_GLOBAL(habana::Habana##Name##Checkpoint))       \
      .add(                                                         \
          "hpu::habana_" #name "_checkpoint_backward",              \
          KERNEL_FN_GLOBAL(habana::Habana##Name##CheckpointBwd))

namespace habana {

OutputMetaData SeedOutputMeta();

struct HabanaRandomBase : OpBackend {
  HabanaRandomBase(
      int device_id,
      std::string_view kernel_name,
      c10::ScalarType scalar_type,
      std::vector<int> res_ids,
      bool is_deterministic);

  void AddNodeCommon(synapse_helpers::graph&, const at::Stack&, bool);
  void AddNode(synapse_helpers::graph&, const at::Stack&) override;
  void CustomHandler(synapse_helpers::graph&, at::Stack&) override;

 private:
  const bool is_deterministic;
};

struct HabanaRandCheckpointBase : HabanaRandomBase {
  HabanaRandCheckpointBase(
      int device_id,
      std::string_view kernel_name,
      c10::ScalarType scalar_type,
      std::vector<int> res_ids = {0});
  void AddNode(synapse_helpers::graph&, const at::Stack&) override;
};

DEFINE_BASE_RANDOM_OP(HabanaSeedGenerator)
DEFINE_RANDOM_OP(HabanaRandPermDS)

DEFINE_BASE_CHECKPOINT_OP(HabanaRand)
DEFINE_BASE_CHECKPOINT_OP(HabanaRandn)
DEFINE_BASE_CHECKPOINT_OP(HabanaUniform)

DEFINE_CHECKPOINT_OP(HabanaNativeDropout)
DEFINE_CHECKPOINT_OP(HabanaMultinomial)
DEFINE_CHECKPOINT_OP(HabanaRandint)
DEFINE_CHECKPOINT_OP(HabanaBernoulli)
DEFINE_CHECKPOINT_OP(HabanaPoisson)
DEFINE_CHECKPOINT_OP(HabanaRandPerm)

} // namespace habana
