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


import habana_frameworks.torch.core as htcore
import torch
from test_utils import compare_tensors


def test_multilevel_view_dtype():
    a = torch.randn(8)
    b = a.view(torch.bool)

    b_hpu = b.to("hpu")
    c_hpu = b_hpu.view(torch.float)
    d_hpu = c_hpu.view(-1)
    compare_tensors(d_hpu.cpu(), a, 0.001, 0.001)
