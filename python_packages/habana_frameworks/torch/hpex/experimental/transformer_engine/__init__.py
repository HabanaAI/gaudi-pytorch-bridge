# Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE.txt for license information.
#
# Copyright (C) 2023 Habana Labs, Ltd. an Intel Company
# All Rights Reserved.
#
# Unauthorized copying of this file or any element(s) within it, via any medium
# is strictly prohibited.
# This file contains Habana Labs, Ltd. proprietary and confidential information
# and is subject to the confidentiality and license agreements under which it
# was provided.

"""Transformer Engine import for pyTorch"""

import importlib
import sys

te = importlib.import_module("intel_transformer_engine")

for name in dir(te):
    if not name.startswith("__"):
        attr = getattr(te, name)
        if isinstance(attr, type(sys)):
            sys.modules[f"habana_frameworks.torch.hpex.experimental.transformer_engine.{name}"] = attr
        globals()[name] = attr

__all__ = dir(te)
