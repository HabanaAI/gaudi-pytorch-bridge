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
# - Added MatMul layer
# - Added SelfAttention layer

"""Transformer Engine bindings for pyTorch"""
#from .module import LayerNormLinear TODO SW-124446
from .module import Linear
#from .module import LayerNormMLP TODO SW-119025
from .module import LayerNorm
from .module import MatMul
from .module import SelfAttentionScoresAndValue
from .module import SelfAttentionContext
#from .transformer import TransformerLayer TODO SW-124447
from .fp8 import fp8_autocast
from .distributed import checkpoint
