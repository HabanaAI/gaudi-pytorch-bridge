###############################################################################
#
#  Copyright (c) 2021-2025 Intel Corporation
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

import torch


def random_seed(x: torch.tensor) -> torch.tensor:
    # Error checking
    dtype = x.dtype
    if dtype != torch.int32:
        raise TypeError(f"Only int32 seed is accepted, got: {dtype}")
    device = x.device

    if device == torch.device("cpu"):
        raise ValueError("HPU RandomSeed is only supported on hpu device")
    else:
        try:
            from habana_frameworks.torch import _hpex_C

            return _hpex_C.random_seed(x)
        except ImportError:
            raise ImportError("Please install habana_torch.")
