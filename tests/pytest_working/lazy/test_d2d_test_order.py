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


def test_d2d_order():
    device = "hpu"
    weight = torch.tensor([3.0], device=device)
    t = torch.tensor([1.0], device=device)
    temp = torch.tensor([4.0], device=device)

    htcore.mark_step()

    t1 = torch.add(weight, t)  # t1 = weight + 1 = 3 + 1 = 4
    weight.copy_(temp)  # weight = 4

    htcore.mark_step()
    print(weight)
    print(t1)
    assert weight[0] == 4.0, "Data mismatch in weight"
    assert t1[0] == 4.0, "Data mismatch in t1 (result of add)"
