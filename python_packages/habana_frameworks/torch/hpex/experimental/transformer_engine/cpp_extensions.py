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
# - Removed unused functions

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

    if out is None:
        out = torch.ops.hpu.fp8_gemm_v2(
            B,
            transb,
            A,
            transa,
            None,
            out_dtype,
            B_scale_inv,
            A_scale_inv,
            bias if use_bias else None,
            accumulate)
    else:
        torch.ops.hpu.fp8_gemm(
            B,
            transb,
            A,
            transa,
            out,
            out_dtype,
            B_scale_inv,
            A_scale_inv,
            bias if use_bias else None,
            accumulate,
            out)

    return out


def fp8_gelu(
    inp: torch.Tensor,
    fp8_meta_tensor: tex.FP8TensorMeta,
    fp8_tensor: Union[tex.FP8FwdTensors, tex.FP8BwdTensors],
    otype: tex.DType,
    retain: torch.Tensor = None,
    stochastic_rounding = False,
    measure_amax = True
) -> torch.Tensor:
    """GeLU with FP8 output"""

    fp8_meta_tensor.scale_inv[fp8_tensor] = torch.reciprocal(fp8_meta_tensor.scale[fp8_tensor])
    def operator():
        return torch.ops.hpu.fp8_gelu_v2(inp, fp8_meta_tensor.scale[fp8_tensor], stochastic_rounding, measure_amax)
    out, retain, amax = _select_amax_and_exec(
        fp8_meta_tensor,
        fp8_tensor,
        operator,
        measure_amax=measure_amax
        )

    return out, retain


def cast_to_fp8(
    inp: torch.Tensor,
    fp8_meta_tensor: tex.FP8TensorMeta,
    fp8_tensor: Union[tex.FP8FwdTensors, tex.FP8BwdTensors],
    otype: tex.DType,
    stochastic_rounding = False,
    measure_amax=True
) -> torch.Tensor:
    """Cast input to FP8"""
    def operator():
        return torch.ops.hpu.cast_to_fp8_v2(inp, fp8_meta_tensor.scale[fp8_tensor], stochastic_rounding, measure_amax)
    cast_out, amax = _select_amax_and_exec(
        fp8_meta_tensor,
        fp8_tensor,
        operator,
        measure_amax=measure_amax
        )

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


def _select_amax_and_exec(
    fp8_meta_tensor: tex.FP8TensorMeta,
    fp8_tensor: Union[tex.FP8FwdTensors, tex.FP8BwdTensors],
    operator,
    measure_amax=True
    ):
    outputs = operator()
    if measure_amax and fp8_meta_tensor.amax_history.shape[0] > 1:
        # amax_history length > 1
        # NOTE: This path is functional, but performance could be improved by removing the temporary tensor
        tmp = torch.index_select(fp8_meta_tensor.amax_history, dim=0, index=fp8_meta_tensor.amax_history_index)
        tmp[0][fp8_tensor].copy_(outputs[-1])
        fp8_meta_tensor.amax_history[fp8_meta_tensor.amax_history_index] = tmp
    elif measure_amax:
        # In case amax_history length = 1, we don't need to use amax_history_index - it simplifies the graph
        fp8_meta_tensor.amax_history[0][fp8_tensor].copy_(outputs[-1])
    return outputs
