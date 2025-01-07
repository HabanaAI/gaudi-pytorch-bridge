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
from test_utils import cpu


@pytest.mark.skip(reason="wrong dimensions")
def test_hpu_cat_permute():
    a = torch.randn(2, 1, 4, 4).to("hpu")
    d = torch.randn(2, 1, 4, 4).to("hpu")
    wt = torch.randn(1, 1, 3, 3).to("hpu")
    b = torch.nn.functional.conv2d(a, wt)
    e = torch.nn.functional.conv2d(d, wt)
    c = torch.cat([b, e], dim=1)
    c.to(cpu)
