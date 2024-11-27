# Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE.txt for license information.
#
###############################################################################
#
#  Copyright (c) 2021-2024 Intel Corporation
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
###############################################################################


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
