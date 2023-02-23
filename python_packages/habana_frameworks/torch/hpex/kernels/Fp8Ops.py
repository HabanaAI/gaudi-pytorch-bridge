###############################################################################
# Copyright (C) 2023 Habana Labs, Ltd. an Intel Company
# All Rights Reserved.
#
# Unauthorized copying of this file or any element(s) within it, via any medium
# is strictly prohibited.
# This file contains Habana Labs, Ltd. proprietary and confidential information
# and is subject to the confidentiality and license agreements under which it
# was provided.
#
###############################################################################
import torch
from typing import Optional, Tuple, Union
import habana_frameworks.torch.core

def cast_to_fp8(input: torch.Tensor,
                scale: torch.Tensor,
                amax: torch.Tensor,
                stochastic = False) -> torch.Tensor:
    # Error checking
    dtype = input.dtype
    if dtype != torch.bfloat16 and dtype != torch.float32:
        raise TypeError(f"Only float32 and bfloat16 can be casted to fp8, got: {dtype}")

    out = torch.empty(
            input.shape,
            dtype=torch.int8,
            device=input.device,
        )
    amax_temp = torch.tensor(0, dtype=torch.float).to("hpu")

    torch.ops.hpu.cast_to_fp8(input, scale, stochastic, out, amax_temp)
    amax.copy_(amax_temp)

    return out

def fp8_cast_transpose_fused(input: torch.Tensor,
    scale: torch.Tensor,
    amax: torch.Tensor,
    stochastic = False,
    cast_out: Optional[torch.Tensor] = None,
    transpose_out: Optional[torch.Tensor] = None
) -> Union[Tuple[torch.Tensor, torch.Tensor], None]:
    # Error checking
    dtype = input.dtype
    assert dtype in (torch.bfloat16, torch.float32), f"Only float32 and bfloat16 can be casted to fp8, got: {dtype}."

    input_shape = input.shape
    assert len(input_shape) == 2, f"fp8_cast_transpose_fused supports only 2D tensors, got {len(input_shape)}D."

    return_outputs = False
    if cast_out is None or transpose_out is None:
        cast_out = torch.empty(
                input_shape,
                dtype=torch.int8,
                device="hpu",
            )
        transpose_out = torch.empty(
                (input_shape[1], input_shape[0]),
                dtype=torch.int8,
                device="hpu",
            )
        return_outputs = True
    amax_temp = torch.tensor(0, dtype=torch.float).to("hpu")

    torch.ops.hpu.fp8_cast_transpose(input, scale, stochastic, cast_out, amax_temp, transpose_out)
    amax.copy_(amax_temp)

    if return_outputs:
        return cast_out, transpose_out
    return None

def fp8_cast_transpose_bgrad_fused(input: torch.Tensor,
    scale: torch.Tensor,
    amax: torch.Tensor,
    stochastic = False
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Error checking
    dtype = input.dtype
    assert dtype in (torch.bfloat16, torch.float32), f"Only float32 and bfloat16 can be casted to fp8, got: {dtype}."

    input_shape = input.shape
    assert len(input_shape) == 2, f"fp8_cast_transpose_bgrad_fused supports only 2D tensors, got {len(input_shape)}D."

    cast_out = torch.empty(
            input_shape,
            dtype=torch.int8,
            device="hpu",
        )
    transpose_out = torch.empty(
            (input_shape[1], input_shape[0]),
            dtype=torch.int8,
            device="hpu",
        )
    bgrad_out = torch.empty(
            (input_shape[1],),
            dtype=dtype,
            device="hpu",
        )
    amax_temp = torch.tensor(0, dtype=torch.float).to("hpu")

    torch.ops.hpu.fp8_cast_transpose_bgrad(input, scale, stochastic, cast_out, amax_temp, transpose_out, bgrad_out)
    amax.copy_(amax_temp)

    return bgrad_out, cast_out, transpose_out

def fp8_cast_transpose_bgrad_dgelu_fused(grad: torch.Tensor,
    input: torch.Tensor,
    scale: torch.Tensor,
    amax: torch.Tensor,
    stochastic = False,
    retain: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Error checking
    dtype = input.dtype
    assert dtype in (torch.bfloat16, torch.float32), f"Only float32 and bfloat16 can be casted to fp8, got: {dtype}."

    input_shape = input.shape
    assert len(input_shape) == 2, f"fp8_cast_transpose_bgrad_dgelu_fused supports only 2D tensors, got {len(input_shape)}D."

    cast_out = torch.empty(
            input_shape,
            dtype=torch.int8,
            device="hpu",
        )
    transpose_out = torch.empty(
            (input_shape[1], input_shape[0]),
            dtype=torch.int8,
            device="hpu",
        )
    bgrad_out = torch.empty(
            (input_shape[1],),
            dtype=dtype,
            device="hpu",
        )
    amax_temp = torch.tensor(0, dtype=torch.float).to("hpu")

    torch.ops.hpu.fp8_cast_transpose_bgrad_dgelu(grad, input, scale, retain, stochastic, cast_out, amax_temp, transpose_out, bgrad_out)
    amax.copy_(amax_temp)

    return bgrad_out, cast_out, transpose_out

def cast_from_fp8(input: torch.Tensor, scale: torch.Tensor, out_dtype: torch.dtype) -> torch.Tensor:
    # Error checking
    if out_dtype != torch.bfloat16 and out_dtype != torch.float32:
        raise TypeError(f"fp8 can be casted only to float32 and bfloat16, got: {out_dtype}")

    return torch.ops.hpu.cast_from_fp8(input, scale, out_dtype)

def fp8_gelu(input: torch.Tensor, scale: torch.Tensor, amax: torch.Tensor, stochastic = False, retain: torch.Tensor = None) -> torch.Tensor:
    # Error checking
    dtype = input.dtype
    if dtype != torch.bfloat16 and dtype != torch.float32:
        raise TypeError(f"Only float32 and bfloat16 can be casted to fp8, got: {dtype}")

    out = torch.empty(
            input.shape,
            dtype=torch.int8,
            device="hpu",
        )

    if retain == None:
        retain = torch.empty(
                input.shape,
                dtype=dtype,
                device="hpu",
            )
    amax_temp = torch.tensor(0, dtype=torch.float).to("hpu")

    torch.ops.hpu.fp8_gelu(input, scale, stochastic, out, amax_temp, retain)
    amax.copy_(amax_temp)

    return out

def layernorm_fwd_fp8(input: torch.Tensor,
                      weight: torch.Tensor,
                      bias: torch.Tensor,
                      eps: float,
                      scale: torch.Tensor,
                      amax: torch.Tensor,
                      stochastic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Error checking
    dtype = input.dtype
    if dtype != torch.bfloat16 and dtype != torch.float32:
        raise TypeError(f"Only float32 and bfloat16 can be casted to fp8, got: {dtype}")

    out = torch.empty(
            input.shape,
            dtype=torch.int8,
            device="hpu",
        )
    mean = torch.empty(
            (input.shape[0],),
            dtype=torch.float,
            device="hpu",
        )
    istd = torch.empty(
            (input.shape[0],),
            dtype=torch.float,
            device="hpu",
        )
    amax_temp = torch.tensor(0, dtype=torch.float).to("hpu")

    torch.ops.hpu.fp8_layernorm(input, weight, bias, eps, scale, stochastic, out, amax_temp, mean, istd)
    amax.copy_(amax_temp)

    return out, mean, istd

def fp8_gemm(A: torch.Tensor,
             A_scale_inv: torch.Tensor,
             B: torch.Tensor,
             B_scale_inv: torch.Tensor,
             out_dtype: torch.dtype,
             accumulate: bool = False,
             out: Optional[torch.Tensor] = None,
             bias: Optional[torch.Tensor] = None,
             use_bias: bool = False) -> torch.Tensor:
    A_dtype = A.dtype
    B_dtype = B.dtype
    if A_dtype != torch.int8 or B_dtype != torch.int8:
        raise TypeError(f"Input tensors must have torch.uint8 dtype, got {A_dtype} and {B_dtype}")

    if out_dtype not in (torch.float, torch.bfloat16):
        raise TypeError(f"Output tensor must have torch.float or torch.bfloat16 dtype, got {out_dtype}")

    return_output = False
    if out is None:
        out = torch.empty(
            A.shape[-1],
            B.shape[-1],
            dtype=out_dtype,
            device=A.device,
        )
        return_output = True

    torch.ops.hpu.fp8_gemm(A, A_scale_inv, True, B, B_scale_inv, False, out, out_dtype, bias if use_bias else None, accumulate, out)

    if return_output:
        return out

def fp8_transpose(input: torch.Tensor, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    # Error checking
    dtype = input.dtype
    if dtype != torch.int8:
        raise TypeError(f"fp8_transpose support only torch.int8 dtype, got: {dtype}")

    return_output = False
    if out is None:
        out = torch.empty(
            input.shape[1],
            input.shape[0],
            dtype=torch.int8,
            device=input.device,
        )
        return_output = True

    torch.ops.hpu.fp8_transpose(input, out)
    if return_output:
        return out
