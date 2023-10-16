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
# - Removed Torch_DType and TE_DType enums

"""Enums for e2e transformer"""
import torch
import torch.distributed

GemmParallelModes = ("row", "column", None)

dist_group_type = torch.distributed.ProcessGroup
