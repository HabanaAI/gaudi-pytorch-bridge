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
# - Removed unused constants

"""Enums for e2e transformer"""
import torch
from habana_frameworks.torch import _hpex_C as tex

"""
This is a map: torch.dtype -> int
Used for passing dtypes into cuda
extension. Has one to one mapping
with enum in transformer_engine.h
"""
TE_DType = {
    torch.int8: tex.DType.kByte,
    torch.int32: tex.DType.kInt32,
    torch.float32: tex.DType.kFloat32,
    torch.half: tex.DType.kFloat16,
    torch.bfloat16: tex.DType.kBFloat16,
}

Torch_DType = {
    tex.DType.kByte: torch.int8,
    tex.DType.kInt32: torch.int32,
    tex.DType.kFloat32: torch.float32,
    tex.DType.kFloat16: torch.half,
    tex.DType.kBFloat16: torch.bfloat16,
    tex.DType.kFloat8E4M3: torch.int8,
    tex.DType.kFloat8E5M2: torch.int8
}

GemmParallelModes = ("row", "column", None)

dist_group_type = torch._C._distributed_c10d.ProcessGroup
