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

import habana_frameworks.torch._core_C as htcore
import habana_frameworks.torch.hpu as hpu
import pytest
import torch

torch.manual_seed(2)


def test_graph():
    input = [(2, 3, 4, 4), (2, 3, 6, 6), (2, 3, 8, 8), (2, 3, 10, 10), (2, 3, 2, 2)]

    def raw_function(input_tensor):
        out1 = torch.mul(input_tensor, 2)
        out2 = torch.add(input_tensor, out1)
        return out2

    for s in input:
        t = torch.randn(s, requires_grad=False)
        t_hpu = t.to("hpu")
        result = raw_function(t_hpu)
        htcore._mark_step()
        print(result.to("cpu"))


test_graph()
