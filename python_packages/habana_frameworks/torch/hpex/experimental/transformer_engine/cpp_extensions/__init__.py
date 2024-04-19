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

"""Python interface for C++ extensions"""

from ._utils import _update_amax_history
from .activation import fp8_gelu
from .cast import cast_from_fp8, cast_to_fp8, cast_to_fp8_hybrid
from .gemm import fp8_gemm
