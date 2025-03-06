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

###############################################################################
# Copyright (C) 2024 Habana Labs, Ltd. an Intel Company
# All Rights Reserved.
#
# Unauthorized copying of this file or any element(s) within it, via any medium
# is strictly prohibited.
# This file contains Habana Labs, Ltd. proprietary and confidential information
# and is subject to the confidentiality and license agreements under which it
# was provided.
#
###############################################################################
import habana_frameworks.torch.hpu
import pytest
import torch

dtypes = [
    torch.float32,
    torch.bfloat16,
    torch.int,
]  # or dtypes = [torch.float, torch.bfloat16, torch.long, torch.int, torch.short, torch.uint8, torch.int8]


@pytest.mark.parametrize("dtype", dtypes)
def test_hpu_pin_memory(dtype):
    ifm = torch.tensor([[1, 2, 3, 4], [4, 5, 6, 8]], dtype=dtype)
    ifm = ifm.pin_memory()
    assert ifm.is_pinned()
