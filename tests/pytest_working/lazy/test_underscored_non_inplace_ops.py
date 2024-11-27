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

import pytest
import torch
from test_utils import format_tc


# Test checking if following operation didn't return Graph compile failed
@pytest.mark.parametrize(
    "op, kwargs",
    [
        (torch.Tensor.bernoulli_, {}),
        (torch.Tensor.exponential_, {}),
        (
            torch.Tensor.geometric_,
            {"p": 0.2},
        ),
        (torch.Tensor.log_normal_, {}),
        (torch.Tensor.normal_, {}),
        (torch.Tensor.random_, {}),
        (torch.Tensor.uniform_, {}),
    ],
    ids=format_tc,
)
def test_underscored_non_inplace_op(op, kwargs):
    try:
        x = torch.randn((10)).to("hpu")
        x_clone = x.clone()
        op(x_clone, **kwargs).cpu()
    except RuntimeError:
        assert False, "Test shouldn't throw any exception"
