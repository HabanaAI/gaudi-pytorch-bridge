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

namespace habana {

ns_CastKernel::Params GetCastParams(
    const bool stochastic,
    const at::ScalarType& from_dtype,
    const at::ScalarType& to_dtype);

// Originally all below ops are out-of-place with preallocated output.
// Unfortunately, such ops' outputs are marked as persistent in graphs,
// which leads to OOM.
//
// V2 ops are out-of-place ops that allocate outputs by themselves in
// order to deal with that problem.
DEFINE_OP(CastToFp8)
DEFINE_OP(CastToFp8V2)
DEFINE_OP(CastFromFp8)
DEFINE_OP(Fp8Gemm)
DEFINE_OP(Fp8GemmV2)
HPU_OP_BACKEND(InPlaceInterleaveCommon)

OUTSHAPE_DECL(CastToFp8V2OutputShape)
OUTSHAPE_DECL(Fp8GemmV2OutputShape)

// Determines if STOCHASTIC_FLUSH_TO_ZERO should be used instead
// of STORCHASTIC_ROUNDING.
const bool is_sr_sftz = GET_ENV_FLAG_NEW(PT_HPU_STOCHASTIC_ROUNDING_MODE) == 1;

} // namespace habana