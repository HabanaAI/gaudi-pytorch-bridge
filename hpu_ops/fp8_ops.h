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
DEFINE_OP(Fp8CastTranspose)
DEFINE_OP(Fp8CastTransposeBgrad)
DEFINE_OP(Fp8CastTransposeBgradDgelu)
DEFINE_OP(CastFromFp8)
DEFINE_OP(Fp8Dropout)
DEFINE_OP(Fp8Gelu)
DEFINE_OP(Fp8GeluV2)
DEFINE_OP(Fp8BgradDgelu)
DEFINE_OP(Fp8FastSoftmax)
DEFINE_OP(Fp8Layernorm)
DEFINE_OP(Fp8Gemm)
DEFINE_OP(Fp8GemmV2)
DEFINE_OP(Fp8Transpose)
DEFINE_OP(Fp8Permute)
DEFINE_OP(Fp8Reshape)
DEFINE_OP(Fp8Copy_)
DEFINE_OP(Fp8KvReorder)
DEFINE_OP(Fp8IndexCopy_)
DEFINE_OP(Fp8RepeatV2)
DEFINE_OP(Fp8IndexSelectV2)
HPU_OP_BACKEND(InPlaceInterleaveCommon)
DEFINE_OP(Conv2dFp8)
DEFINE_OP(SumFp8)

OUTSHAPE_DECL(CastToFp8V2OutputShape)
OUTSHAPE_DECL(Fp8DropoutOutputShape)
OUTSHAPE_DECL(Fp8GeluV2OutputShape)
OUTSHAPE_DECL(Fp8GemmV2OutputShape)
OUTSHAPE_DECL(Fp8BgradDgeluOutputShape)
OUTSHAPE_DECL(Fp8FastSoftmaxOutputShape)
OUTSHAPE_DECL(Fp8ReshapeOutputShape)
OUTSHAPE_DECL(Fp8RepeatV2OutputShape)
OUTSHAPE_DECL(Fp8IndexSelectV2OutputShape)
OUTSHAPE_DECL(Conv2dFp8OutputShape)
OUTSHAPE_DECL(SumFp8OutputShape)

const synDataType fp8_syn_type =
    GET_ENV_FLAG_NEW(PT_USE_FP8_143) ? syn_type_fp8_143 : syn_type_fp8_152;
const bool is_sr_sftz = GET_ENV_FLAG_NEW(PT_HPU_STOCHASTIC_ROUNDING_MODE) == 1;

} // namespace habana