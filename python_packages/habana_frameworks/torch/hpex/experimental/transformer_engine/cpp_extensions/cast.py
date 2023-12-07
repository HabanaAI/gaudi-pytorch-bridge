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

from ..utils import FP8BwdTensors, FP8FwdTensors, FP8TensorMeta
from ._utils import select_amax_and_exec


def cast_to_fp8(
    inp: torch.Tensor,
    fp8_meta_tensor: FP8TensorMeta,
    fp8_tensor: Union[FP8FwdTensors, FP8BwdTensors],
    otype: torch.dtype,
    stochastic_rounding=False,
    measure_amax=True,
) -> torch.Tensor:
    """Cast input to FP8"""

    def operator():
        return torch.ops.hpu.cast_to_fp8_v2(
            inp, fp8_meta_tensor.scale[fp8_tensor], stochastic_rounding, measure_amax, dtype=otype
        )

    (cast_out,) = select_amax_and_exec(
        operator,
        fp8_meta_tensor,
        fp8_tensor,
        measure_amax=measure_amax,
    )
    return cast_out


def cast_to_fp8_hybrid(
    inp: torch.Tensor,
    e5m2_meta_tensor: FP8TensorMeta,
    e4m3_meta_tensor: FP8TensorMeta,
    fp8_tensor: Union[FP8FwdTensors, FP8BwdTensors],
    measure_amax=True,
) -> torch.Tensor:
    """Cast input to both fp8 formats"""

    def operator():
        return torch.ops.hpu.cast_to_fp8_hybrid(
            inp, e5m2_meta_tensor.scale[fp8_tensor], e4m3_meta_tensor.scale[fp8_tensor], False, measure_amax
        )

    out_e5m2, out_e4m3 = select_amax_and_exec(
        operator,
        fp8_meta_tensor=e5m2_meta_tensor,
        fp8_tensor=fp8_tensor,
        fp8_meta_tensor2=e4m3_meta_tensor,
        measure_amax=measure_amax,
    )
    return out_e5m2, out_e4m3


def cast_from_fp8(
    inp: torch.Tensor,
    fp8_meta_tensor: FP8TensorMeta,
    fp8_tensor: Union[FP8FwdTensors, FP8BwdTensors],
    otype: torch.dtype,
) -> torch.Tensor:
    """Cast input from FP8"""
    return torch.ops.hpu.cast_from_fp8(
        inp,
        fp8_meta_tensor.scale_inv[fp8_tensor],
        otype,
    )
