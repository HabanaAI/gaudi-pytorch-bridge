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
