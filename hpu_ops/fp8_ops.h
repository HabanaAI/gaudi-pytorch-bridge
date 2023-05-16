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

#define DEFINE_OP(op)                                                 \
  struct op : OpBackend {                                             \
    op(int device_id, c10::ScalarType scalar_type);                   \
    void AddNode(synapse_helpers::graph&, const at::Stack&) override; \
  };

namespace habana {

// Originally all below ops are out-of-place with preallocated output.
// Unfortunately, such ops' outputs are marked as persistent in graphs,
// which leads to OOM.
//
// V2 ops are out-of-place ops that allocate outputs by themselves in
// order to deal with that problem.
DEFINE_OP(CastToFp8)
DEFINE_OP(CastToFp8V2)
DEFINE_OP(Fp8CastTranspose)
DEFINE_OP(Fp8CastTransposeBgrad)
DEFINE_OP(Fp8CastTransposeBgradDgelu)
DEFINE_OP(CastFromFp8)
DEFINE_OP(Fp8Dropout)
DEFINE_OP(Fp8Gelu)
DEFINE_OP(Fp8BgradDgelu)
DEFINE_OP(Fp8Layernorm)
DEFINE_OP(Fp8Gemm)
DEFINE_OP(Fp8GemmV2)
DEFINE_OP(Fp8Transpose)
DEFINE_OP(Fp8Permute)
DEFINE_OP(Fp8Reshape)

OUTSHAPE_DECL(Fp8DropoutOutputShape)
OUTSHAPE_DECL(Fp8BgradDgeluOutputShape)
OUTSHAPE_DECL(Fp8ReshapeOutputShape)
OUTSHAPE_DECL(CastToFp8V2OutputShape)
OUTSHAPE_DECL(Fp8GemmV2OutputShape)

} // namespace habana