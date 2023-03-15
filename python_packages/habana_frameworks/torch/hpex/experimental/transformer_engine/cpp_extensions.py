# Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
#
# Copyright (C) 2023 Habana Labs, Ltd. an Intel Company
# All Rights Reserved.
#
# Unauthorized copying of this file or any element(s) within it, via any medium
# is strictly prohibited.
# This file contains Habana Labs, Ltd. proprietary and confidential information
# and is subject to the confidentiality and license agreements under which it
# was provided.
#
# Changes:
# - Adapted and modified some interfaces with optional HPU specific parameters

"""TE FP8 extensions and GEMMs"""
from typing import Optional, Tuple, Union
import torch
from habana_frameworks.torch import _hpex_C as tex
from .constants import Torch_DType


def fp8_gemm(
    A: torch.Tensor,
    A_scale_inv: torch.Tensor,
    A_dtype: tex.DType,
    B: torch.Tensor,
    B_scale_inv: torch.Tensor,
    B_dtype: tex.DType,
    out_dtype: torch.dtype,
    workspace: torch.Tensor = None,
    accumulate: bool = False,
    out: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    use_bias: bool = False,
    fp32_output: bool = False,
    use_split_accumulator: bool = False,
    transa: bool = True,
    transb: bool = False,
) -> torch.Tensor:
    """TN layout GEMM with fp8 inputs."""

    return_output = False
    if out is None:
        rank = len(B.shape)
        dimb = B.shape[-2] if not transb else B.shape[-1]
        dima = A.shape[-1] if not transa else A.shape[-2]
        out_shape = B.shape[0:(rank-2)] + (dimb,) + (dima,)
        out = torch.empty(
            out_shape,
            dtype=torch.float32 if fp32_output else out_dtype,
            device="hpu",
        )
        return_output = True

    torch.ops.hpu.fp8_gemm(
        B,
        B_scale_inv,
        transb,
        A,
        A_scale_inv,
        transa,
        out,
        out_dtype,
        bias if use_bias else None,
        accumulate,
        out)

    if return_output:
        return out
    return None


def fp8_cast_transpose_fused(
    inp: torch.Tensor,
    fp8_meta_tensor: tex.FP8TensorMeta,
    fp8_tensor: Union[tex.FP8FwdTensors, tex.FP8BwdTensors],
    otype: tex.DType,
    cast_out: Optional[torch.Tensor] = None,
    transpose_out: Optional[torch.Tensor] = None,
    stochastic_rounding = False
) -> Union[Tuple[torch.Tensor, torch.Tensor], None]:
    """Cast + Transpose with FP8 output"""

    return_outputs = False
    if cast_out is None or transpose_out is None:
        cast_out = torch.empty_like(inp, dtype=torch.int8)
        transpose_out = torch.empty(
            inp.shape[1], inp.shape[0], device="hpu", dtype=torch.int8
        )
        return_outputs = True

    fp8_meta_tensor.scale_inv[fp8_tensor] = torch.reciprocal(fp8_meta_tensor.scale[fp8_tensor])
    #TODO SW-124456 replace with native fp8_cast_transpose_fused call
    _cast_to_fp8(
        inp,
        fp8_meta_tensor,
        fp8_tensor,
        stochastic_rounding,
        cast_out)
    torch.ops.hpu.fp8_transpose(cast_out, transpose_out)

    if return_outputs:
        return cast_out, transpose_out
    return None


def fp8_cast_transpose_bgrad_fused(
    inp: torch.Tensor,
    fp8_meta_tensor: tex.FP8TensorMeta,
    fp8_tensor: Union[tex.FP8FwdTensors, tex.FP8BwdTensors],
    otype: tex.DType,
    stochastic_rounding = False
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Cast + Transpose + BGRAD with FP8 output"""
    cast_out = torch.empty_like(inp, dtype=torch.int8)
    fp8_meta_tensor.scale_inv[fp8_tensor] = torch.reciprocal(fp8_meta_tensor.scale[fp8_tensor])
    transpose_out = torch.empty(inp.shape[1], inp.shape[0], dtype=torch.int8, device="hpu")
    #TODO SW-124458 replace with native fp8_cast_transpose_bgrad_fused call
    _cast_to_fp8(
        inp,
        fp8_meta_tensor,
        fp8_tensor,
        stochastic_rounding,
        cast_out)
    torch.ops.hpu.fp8_transpose(cast_out, transpose_out)
    bgrad_out = inp.sum(dim=0)

    return bgrad_out, cast_out, transpose_out


#TODO SW-124459 implement using native fp8_cast_transpose_bgrad_dgelu_fused call
# def fp8_cast_transpose_bgrad_dgelu_fused(
#     grad_output: torch.Tensor,
#     gelu_input: torch.Tensor,
#     fp8_meta_tensor: tex.FP8TensorMeta,
#     fp8_tensor: Union[tex.FP8FwdTensors, tex.FP8BwdTensors],
#     otype: tex.DType,
# ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
#     """Cast + Transpose + BGRAD + DGELU with FP8 output"""
#     return tex.fused_cast_transpose_bgrad_dgelu(
#         grad_output,
#         gelu_input,
#         fp8_meta_tensor.scale[fp8_tensor],
#         fp8_meta_tensor.amax_history[0][fp8_tensor],
#         fp8_meta_tensor.scale_inv[fp8_tensor],
#         otype,
#     )

#TODO SW-124460 implement using native fp8_gelu call
# def fp8_gelu(
#     inp: torch.Tensor,
#     fp8_meta_tensor: tex.FP8TensorMeta,
#     fp8_tensor: Union[tex.FP8FwdTensors, tex.FP8BwdTensors],
#     otype: tex.DType,
# ) -> torch.Tensor:
#     """GeLU with FP8 output"""
#     return torch.ops.hpu.fp8_gelu(
#         inp,
#         fp8_meta_tensor.scale[fp8_tensor],
#         fp8_meta_tensor.amax_history[0][fp8_tensor],
#         fp8_meta_tensor.scale_inv[fp8_tensor],
#         otype,
#     )

#TODO SW-124462 implement using native layernorm_fwd_fp8 call
# def layernorm_fwd_fp8(
#     inp: torch.Tensor,
#     weight: torch.Tensor,
#     bias: torch.Tensor,
#     eps: float,
#     fp8_meta_tensor: tex.FP8TensorMeta,
#     fp8_tensor: Union[tex.FP8FwdTensors, tex.FP8BwdTensors],
#     otype: tex.DType,
# ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
#     """LayerNorm with FP8 output"""
#     return tex.layernorm_fwd_fp8(
#         inp,
#         weight,
#         bias,
#         eps,
#         fp8_meta_tensor.scale[fp8_tensor],
#         fp8_meta_tensor.amax_history[0][fp8_tensor],
#         fp8_meta_tensor.scale_inv[fp8_tensor],
#         otype,
#     )


def cast_to_fp8(
    inp: torch.Tensor,
    fp8_meta_tensor: tex.FP8TensorMeta,
    fp8_tensor: Union[tex.FP8FwdTensors, tex.FP8BwdTensors],
    otype: tex.DType,
    stochastic_rounding = False
) -> torch.Tensor:
    """Cast input to FP8"""
    cast_out = torch.empty_like(inp, dtype=torch.int8)
    fp8_meta_tensor.scale_inv[fp8_tensor] = torch.reciprocal(fp8_meta_tensor.scale[fp8_tensor])
    _cast_to_fp8(
        inp,
        fp8_meta_tensor,
        fp8_tensor,
        stochastic_rounding,
        cast_out)
    return cast_out

def cast_from_fp8(
    inp: torch.Tensor,
    fp8_meta_tensor: tex.FP8TensorMeta,
    fp8_tensor: Union[tex.FP8FwdTensors, tex.FP8BwdTensors],
    itype: tex.DType,
    otype: tex.DType,
) -> torch.Tensor:
    """Cast input from FP8"""
    return torch.ops.hpu.cast_from_fp8(
        inp,
        fp8_meta_tensor.scale_inv[fp8_tensor],
        Torch_DType[otype],
    )


def _cast_to_fp8(
    inp: torch.Tensor,
    fp8_meta_tensor: tex.FP8TensorMeta,
    fp8_tensor: Union[tex.FP8FwdTensors, tex.FP8BwdTensors],
    stochastic_rounding: bool,
    cast_out: torch.Tensor
):
    if fp8_meta_tensor.amax_history.shape[0] > 1:
        # amax_history length > 1
        # NOTE: This path is functional, but performance could be improved by removing the temporary tensor
        tmp = torch.index_select(fp8_meta_tensor.amax_history, dim=0, index=fp8_meta_tensor.amax_history_index)
        amax_tmp = torch.empty_like(tmp[0][fp8_tensor])
        torch.ops.hpu.cast_to_fp8(
            inp,
            fp8_meta_tensor.scale[fp8_tensor],
            stochastic_rounding,
            cast_out,
            amax_tmp)
        tmp[0][fp8_tensor].copy_(amax_tmp)
        fp8_meta_tensor.amax_history[fp8_meta_tensor.amax_history_index] = tmp
    else:
        # In case amax_history length = 1, we don't need to use amax_history_index - it simplifies the graph
        amax_tmp = torch.empty_like(fp8_meta_tensor.amax_history[0][fp8_tensor])
        torch.ops.hpu.cast_to_fp8(
            inp,
            fp8_meta_tensor.scale[fp8_tensor],
            stochastic_rounding,
            cast_out,
            amax_tmp)
        fp8_meta_tensor.amax_history[0][fp8_tensor].copy_(amax_tmp)
