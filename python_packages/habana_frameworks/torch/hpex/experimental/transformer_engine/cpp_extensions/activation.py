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
# - Removed unused functions

"""Python interface for activation extensions"""
from typing import Union
import torch
from habana_frameworks.torch import _hpex_C as tex
from ._utils import select_amax_and_exec


def fp8_gelu(
    inp: torch.Tensor,
    fp8_meta_tensor: tex.FP8TensorMeta,
    fp8_tensor: Union[tex.FP8FwdTensors, tex.FP8BwdTensors],
    otype: torch.dtype,
    retain: torch.Tensor = None,
    measure_amax = True
) -> torch.Tensor:
    """GeLU with FP8 output"""

    fp8_meta_tensor.scale_inv[fp8_tensor] = torch.reciprocal(fp8_meta_tensor.scale[fp8_tensor])
    def operator():
        return torch.ops.hpu.fp8_gelu_v2(inp, fp8_meta_tensor.scale[fp8_tensor], False, measure_amax)
    out, retain = select_amax_and_exec(
        operator,
        fp8_meta_tensor,
        fp8_tensor,
        measure_amax=measure_amax
        )

    return out, retain
