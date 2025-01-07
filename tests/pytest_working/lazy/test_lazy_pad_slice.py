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
import torch.nn.functional as F
from test_utils import compare_tensors

try:
    import habana_frameworks.torch.core as htcore
except ImportError:
    raise AssertionError("Could Not import habana_frameworks.torch.core")

test_case_list = [
    # True for CL else False,
    True,
    False,
]

dev_hpu = torch.device("hpu")
dev_cpu = torch.device("cpu")


def _hpu_lazy_pad(dev, t):
    pad = (1, 1, 1, 1)
    if dev == torch.device("hpu"):
        t1 = t.to(dev, non_blocking=False)
    else:
        t1 = t

    t1 = F.pad(t1, pad, "constant", 1)
    if dev == torch.device("hpu"):
        htcore.mark_step()
    return t1


def _hpu_lazy_slice(dev, t):
    t1 = t.to(dev, non_blocking=False)
    slice_val = [
        [
            slice(0, 1, None),
            slice(None, None, None),
            slice(0, 128, None),
            slice(0, 128, None),
            slice(0, 128, None),
        ],
        [
            slice(0, 1, None),
            slice(None, None, None),
            slice(0, 128, None),
            slice(0, 128, None),
            slice(8, 136, None),
        ],
    ]
    list_out = [t1[win_slice] for win_slice in slice_val]
    if dev == torch.device("hpu"):
        htcore.mark_step()
    return list_out


@pytest.mark.parametrize("cl", test_case_list)
def test_hpu_lazy_pad_slice(cl):

    t = torch.randn([1, 4, 132, 176, 136])
    if cl is True:
        t = t.contiguous(memory_format=torch.channels_last_3d)

    cpu_pad = _hpu_lazy_pad(dev_cpu, t)
    hpu_pad = _hpu_lazy_pad(dev_hpu, t)

    # Pad
    compare_tensors(hpu_pad.detach().to("cpu"), cpu_pad.detach(), rtol=1e-3, atol=1e-3)

    cpu_slice = _hpu_lazy_slice(dev_cpu, cpu_pad)
    hpu_slice = _hpu_lazy_slice(dev_hpu, hpu_pad)

    # Slice
    for hpu, cpu in zip(hpu_slice, cpu_slice):
        cpu_hpu = hpu.to("cpu")
        compare_tensors(cpu_hpu, cpu, rtol=1e-3, atol=1e-3)
