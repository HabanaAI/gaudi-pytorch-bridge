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
from test_utils import compare_tensors

sizeList = [
    # size
    (4, 5),
    (2, 3, 4),
    (2, 3, 2, 3),
]

broadcast_list = [
    # size1, size2
    ((3), (1)),
    ((3, 2), (1, 1)),
    ((3, 5, 4), (1, 1, 1)),
    ((3, 5, 4, 3), (3, 1, 4, 3)),
    ((3, 2, 4, 3, 4), (1, 2, 1, 3, 1)),
    ((1, 3), (3, 3, 3)),
    ((), (3)),
    ((), (12, 19)),
    ((), (2, 3, 13)),
    ((), (6, 3, 2, 4)),
    ((), (2, 3, 2, 3, 2)),
]

toggle_sizes = [True, False]


@pytest.mark.parametrize("keep_original_order", toggle_sizes)
@pytest.mark.parametrize("size1, size2", broadcast_list)
def test_hpu_lazy_floor_divide_broadcast(size1, size2, keep_original_order):
    t1 = torch.randn(size1 if keep_original_order else size2, requires_grad=False)
    t2 = torch.randn(size2 if keep_original_order else size1, requires_grad=False)

    hpu = torch.device("hpu")

    t1_h = t1.to(hpu)
    t2_h = t2.to(hpu)

    out_h = t1_h // t2_h
    out = t1 // t2

    compare_tensors(out_h, out, atol=0.001, rtol=0.001)


@pytest.mark.parametrize("size ", sizeList)
def test_hpu_lazy_floor_divide_size_variation(size):
    t1 = torch.randn(size, requires_grad=False)
    t2 = torch.randn(size, requires_grad=False)

    hpu = torch.device("hpu")

    t1_h = t1.to(hpu)
    t2_h = t2.to(hpu)

    out_h = t1_h // t2_h
    out = t1 // t2

    compare_tensors(out_h, out, atol=0.001, rtol=0.001)


def test_hpu_lazy_floor_divid_int_int():
    size = (5, 4)
    min_int_tested = 1
    max_int_tested = 300

    t1 = torch.randint(min_int_tested, max_int_tested, size, requires_grad=False)
    t2 = torch.randint(min_int_tested, max_int_tested, size, requires_grad=False)

    hpu = torch.device("hpu")

    t1_h = t1.to(hpu)
    t2_h = t2.to(hpu)

    out_h = t1_h // t2_h
    out = t1 // t2
    compare_tensors(out_h, out, atol=1, rtol=1)
