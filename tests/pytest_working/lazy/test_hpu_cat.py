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
from test_utils import evaluate_fwd_bwd_kernel, evaluate_fwd_kernel

cat_op_list = [
    # op, op params dict
    (
        torch.cat,
        {"tensors": (torch.randn(8, 3, 24, 24), torch.randn(8, 3, 24, 12)), "dim": 3},
    ),
    (
        torch.cat,
        {"tensors": (torch.randn(8, 3, 24, 24), torch.randn(8, 3, 24, 24)), "dim": -1},
    ),
    (
        torch.cat,
        {"tensors": (torch.empty(0), torch.randn(8, 3, 24, 24)), "dim": 1},
    ),
    (
        torch.cat,
        {"tensors": (torch.randn(8, 3, 24, 24), torch.empty(0)), "dim": 2},
    ),
    (
        torch.cat,
        {"tensors": (torch.randn(8, 0, 24, 24), torch.randn(8, 0, 24, 24)), "dim": 0},
    ),
    (
        torch.cat,
        {"tensors": (torch.empty((0, 0, 0)),), "dim": 0},
    ),
    (
        torch.cat,
        {"tensors": (torch.empty(0), torch.empty(0)), "dim": 0},
    ),
    (
        torch.cat,
        {"tensors": (torch.empty((0, 0, 0)), torch.empty((0, 0, 0))), "dim": 0},
    ),
    (
        torch.cat,
        {
            "tensors": (
                torch.randn(8, 3, 24, 24),
                torch.randn(8, 3, 24, 12),
                torch.randn(8, 3, 24, 4),
            ),
            "out": torch.empty(0),
            "dim": 3,
        },
    ),
    (
        torch.cat,
        {
            "tensors": (torch.randn(8, 3, 24, 24), torch.randn(8, 3, 24, 24)),
            "out": torch.empty(0),
            "dim": 3,
        },
    ),
    pytest.param(
        torch.cat,
        {
            "tensors": (torch.empty(0), torch.randn(8, 3, 24, 24)),
            "out": torch.empty(0),
            "dim": 3,
        },
        marks=[pytest.mark.skip(reason="segv, Dimension out of range (expected to be in range of [-1, 0], but got 3)")],
    ),
    (
        torch.cat,
        {
            "tensors": (torch.randn(8, 3, 0, 24), torch.randn(8, 3, 0, 24)),
            "out": torch.empty(0),
            "dim": 2,
        },
    ),
    (
        torch.cat,
        {
            "tensors": (torch.empty(0), torch.randn(0)),
            "out": torch.empty(0),
            "dim": 0,
        },
    ),
]

split_op_list = [
    # op, op params dict
    (
        torch.split_with_sizes,
        {"input": torch.randn(8, 3, 24, 12), "split_sizes": [2, 4, 2], "dim": 0},
    ),
    (
        torch.split_with_sizes,
        {"input": torch.randn(8, 3, 24, 12), "split_sizes": [12, 6, 6], "dim": 2},
    ),
]

cat_op_list_fwd_bwd = [
    # op, op params dict
    (
        torch.cat,
        {
            "tensors": (
                torch.randn(8, 3, 24, 24, requires_grad=True),
                torch.randn(8, 3, 24, 12, requires_grad=True),
            ),
            "dim": 3,
        },
    ),
]


@pytest.mark.parametrize("cat_op, kernel_params_fwd", cat_op_list)
def test_hpu_cat(cat_op, kernel_params_fwd):
    # import pudb; pudb.set_trace()
    [print(t.shape) for t in kernel_params_fwd["tensors"]]
    print(kernel_params_fwd["dim"])
    evaluate_fwd_kernel(kernel=cat_op, kernel_params=kernel_params_fwd)


@pytest.mark.parametrize("cat_op, kernel_params_fwd", cat_op_list_fwd_bwd)
def test_hpu_cat_fwd_bwd(cat_op, kernel_params_fwd):
    dim = kernel_params_fwd["dim"]
    tensors = kernel_params_fwd["tensors"]
    shape = list(tensors[0].size())
    shape[dim] = 0
    for i in range(0, len(tensors)):
        shape[dim] += tensors[i].size()[dim]
    bwd_tensors = [torch.randn(tuple(shape))]
    evaluate_fwd_bwd_kernel(kernel=cat_op, tensor_list_bwd=bwd_tensors, kernel_params_fwd=kernel_params_fwd)


@pytest.mark.parametrize("split_op, kernel_params_fwd", split_op_list)
def test_hpu_split_with_sizes(split_op, kernel_params_fwd):
    evaluate_fwd_kernel(kernel=split_op, kernel_params=kernel_params_fwd)
