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
from typing import Optional
from habana_frameworks.torch import _hpex_C

def cast_to_fp8_te(x: torch.tensor, scale: torch.tensor, amax: torch.tensor, stochastic = False) -> torch.tensor:
    # Error checking
    dtype = x.dtype
    if dtype != torch.bfloat16 and dtype != torch.float32:
        raise TypeError(f"Only float32 and bfloat16 can be casted to fp8, got: {dtype}")

    out = torch.empty(
            x.shape,
            dtype=torch.int8,
            device="hpu",
        )

    _hpex_C.cast_to_fp8_te(x, scale, stochastic, out, amax)
    return out

def cast_from_fp8(x: torch.tensor, scale: torch.tensor, out_dtype: torch.dtype) -> torch.tensor:
    # Error checking
    if out_dtype != torch.bfloat16 and out_dtype != torch.float32:
        raise TypeError(f"fp8 can be casted only to float32 and bfloat16, got: {out_dtype}")

    return _hpex_C.cast_from_fp8(x, scale, out_dtype)

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
            device="hpu",
        )
        return_output = True

    _hpex_C.fp8_gemm(A, A_scale_inv, True, B, B_scale_inv, False, out, out_dtype, bias if use_bias else None, accumulate)

    if return_output:
        return out

def fp8_transpose(x: torch.tensor, out: Optional[torch.Tensor] = None) -> torch.tensor:
    # Error checking
    dtype = x.dtype
    if dtype != torch.int8:
        raise TypeError(f"fp8_transpose support only torch.int8 dtype, got: {dtype}")

    return_output = False
    if out is None:
        out = torch.empty(
            x.shape[1],
            x.shape[0],
            dtype=torch.int8,
            device="hpu",
        )
        return_output = True

    _hpex_C.fp8_transpose(x, out)
    if return_output:
        return out
