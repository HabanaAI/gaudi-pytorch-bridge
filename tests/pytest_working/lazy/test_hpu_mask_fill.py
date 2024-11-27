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

# Based on SW-186390

import habana_frameworks.torch as ht
import torch


def func(dev):
    i = torch.empty(2, 4, device=dev)
    m = torch.ones(2, 4, device=dev, dtype=torch.bool)
    s = torch.nn.Softmax(-1)
    a = i.sqrt()
    a /= 2
    a.masked_fill_(m, 1)
    b1 = s(a)
    a.masked_fill_(m, 2)
    b2 = s(a)
    a.masked_fill_(m, 3)
    b3 = s(a)
    c = b1 + b2 + b3
    return c.cpu()


def test_misc_mask_filled():
    cpu_out = func("cpu")
    hpu_out = func("hpu")
    assert torch.allclose(cpu_out, hpu_out)
