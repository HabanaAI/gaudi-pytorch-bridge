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

DEFINE_OP(CastToFp8)
DEFINE_OP(Fp8CastTranspose)
DEFINE_OP(Fp8CastTransposeBgrad)
DEFINE_OP(Fp8CastTransposeBgradDgelu)
DEFINE_OP(CastFromFp8)
DEFINE_OP(Fp8Gelu)
DEFINE_OP(Fp8Layernorm)
DEFINE_OP(Fp8Gemm)
DEFINE_OP(Fp8Transpose)

} // namespace habana