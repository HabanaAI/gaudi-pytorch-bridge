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

# Test for SW-179074

import pytest
import torch

device = "hpu"
if device == "hpu":
    import habana_frameworks.torch as htorch
    import habana_frameworks.torch.core as htcore


def test_wrong_dimensions_linear():
    with pytest.raises(Exception) as e_info:
        input_tensor = torch.rand(8, 16, 384).to(device)
        # (8x16x384) x (128x64)
        # The innermost dimensions are expected to be same
        # Test will fail if error is not raised
        bad_layer = torch.nn.Linear(in_features=128, out_features=64, bias=True).to(device)
        bad_out = bad_layer(input_tensor)


if device == "hpu":
    htorch.hpu.synchronize()
