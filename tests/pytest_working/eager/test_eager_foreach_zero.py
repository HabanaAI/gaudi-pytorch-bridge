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

import torch


def test_foreach_zero_with_view():
    l1_hpu = []
    l1 = []
    t1_hpu = torch.arange(-3, 15).int().to("hpu").as_strided((3, 3), (6, 2))
    t2_hpu = torch.arange(-3, 15).int().to("hpu").as_strided((3, 3), (3, 1))
    t1 = torch.arange(-3, 15).int().as_strided((3, 3), (6, 2))
    t2 = torch.arange(-3, 15).int().as_strided((3, 3), (3, 1))

    l1_hpu.append(t1_hpu)
    l1_hpu.append(t2_hpu)
    l1.append(t1)
    l1.append(t2)
    torch._foreach_zero_(l1_hpu)
    torch._foreach_zero_(l1)
    for i in range(len(l1)):
        assert torch.allclose(l1_hpu[i].cpu(), l1[i], atol=0.01, rtol=0.01)
