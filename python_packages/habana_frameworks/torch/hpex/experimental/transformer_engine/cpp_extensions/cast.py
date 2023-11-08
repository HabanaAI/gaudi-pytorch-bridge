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

"""Python interface for cast extensions"""
from typing import Union
import torch
from habana_frameworks.torch import _hpex_C as tex
from ._utils import select_amax_and_exec


def cast_to_fp8(
    inp: torch.Tensor,
    fp8_meta_tensor: tex.FP8TensorMeta,
    fp8_tensor: Union[tex.FP8FwdTensors, tex.FP8BwdTensors],
    otype: torch.dtype,
    measure_amax = True
) -> torch.Tensor:
    """Cast input to FP8"""
    def operator():
        return torch.ops.hpu.cast_to_fp8_v2(inp, fp8_meta_tensor.scale[fp8_tensor], False, measure_amax, dtype=otype)
    cast_out, = select_amax_and_exec(
        fp8_meta_tensor,
        fp8_tensor,
        operator,
        measure_amax=measure_amax,
        )

    return cast_out

def cast_from_fp8(
    inp: torch.Tensor,
    fp8_meta_tensor: tex.FP8TensorMeta,
    fp8_tensor: Union[tex.FP8FwdTensors, tex.FP8BwdTensors],
    otype: torch.dtype,
) -> torch.Tensor:
    """Cast input from FP8"""
    return torch.ops.hpu.cast_from_fp8(
        inp,
        fp8_meta_tensor.scale_inv[fp8_tensor],
        otype,
    )

