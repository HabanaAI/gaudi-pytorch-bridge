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

"""Python interface for GEMM extensions"""
from typing import Optional

import torch


def fp8_gemm(
    A: torch.Tensor,
    A_scale_inv: torch.Tensor,
    B: torch.Tensor,
    B_scale_inv: torch.Tensor,
    out_dtype: torch.dtype,
    workspace: torch.Tensor = None,
    accumulate: bool = False,
    out: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    use_bias: bool = False,
    use_split_accumulator: bool = False,
    transa: bool = True,
    transb: bool = False,
) -> torch.Tensor:
    """GEMM with fp8 inputs."""

    # TODO: Remove these params if not needed
    assert not use_split_accumulator

    if out is None:
        out = torch.ops.hpu.fp8_gemm_v2(
            B, transb, A, transa, None, out_dtype, B_scale_inv, A_scale_inv, bias if use_bias else None, accumulate
        )
    else:
        torch.ops.hpu.fp8_gemm(
            B, transb, A, transa, out, out_dtype, B_scale_inv, A_scale_inv, bias if use_bias else None, accumulate, out
        )

    return out
