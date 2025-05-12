/**
 * Copyright (c) 2021-2025 Intel Corporation
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

#define DEFINE_BASIC_RANDOM_OP(op)                  \
  struct op : HabanaRandomBase {                    \
    op(int device_id, c10::ScalarType scalar_type); \
  };

#define DEFINE_COMPLEX_RANDOM_OP(op)                                  \
  struct op : HabanaRandomBase {                                      \
    op(int device_id, c10::ScalarType scalar_type);                   \
    void AddNode(synapse_helpers::graph&, const at::Stack&) override; \
  };

#define REGISTER_HABANA_RANDOM_OP(name, Name) \
  REGISTER_HPU_BACKEND("hpu::habana_" #name, habana::Habana##Name)

namespace habana {

struct HabanaRandomBase : OpBackend {
  HabanaRandomBase(
      int device_id,
      std::string_view kernel_name,
      c10::ScalarType scalar_type,
      std::vector<int> res_ids);

  void AddNode(synapse_helpers::graph&, const at::Stack&) override;
};

DEFINE_BASIC_RANDOM_OP(HabanaSeedGenerator)

DEFINE_BASIC_RANDOM_OP(HabanaExponential)
DEFINE_BASIC_RANDOM_OP(HabanaRand)
DEFINE_BASIC_RANDOM_OP(HabanaRandn)
DEFINE_BASIC_RANDOM_OP(HabanaUniform)

DEFINE_COMPLEX_RANDOM_OP(HabanaBernoulli)
DEFINE_COMPLEX_RANDOM_OP(HabanaMultinomial)
DEFINE_COMPLEX_RANDOM_OP(HabanaNativeDropout)
DEFINE_COMPLEX_RANDOM_OP(HabanaPoisson)
DEFINE_COMPLEX_RANDOM_OP(HabanaRandint)
DEFINE_COMPLEX_RANDOM_OP(HabanaRandPermDS)
DEFINE_COMPLEX_RANDOM_OP(HabanaRandPerm)

} // namespace habana
