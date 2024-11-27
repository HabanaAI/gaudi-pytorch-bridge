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
from habana_frameworks.torch.utils.debug.dynamo_utils import FxGraphAnalyzer
from test_utils import fga_assert_helper


@torch.compile(backend="hpu_backend")
def scalar_fn(x, y):
    x = x + x
    y = y * y
    res = y * x.item()
    res = res * x
    return res


@torch.compile(backend="hpu_backend")
def dynamic_output_shape_ops_fn(x, y):
    x = x * x
    y = y + y
    res = torch.nonzero(x)
    res2 = torch.unique(y)
    return res, res2


def test_capture_scalar_outputs():
    with FxGraphAnalyzer(capture_non_hpu_output=True) as fga, torch._dynamo.config.patch(capture_scalar_outputs=True):
        input1 = torch.tensor([3], device="hpu")
        input2 = torch.tensor([2, 2], device="hpu")
        res = scalar_fn(input1, input2)
        fga_assert_helper(fga.get_ops_summary(), "torch.ops.aten._local_scalar_dense.default", [{0, 1}])


def test_capture_dynamic_output_shape_ops():
    with FxGraphAnalyzer(capture_non_hpu_output=True) as fga, torch._dynamo.config.patch(
        capture_dynamic_output_shape_ops=True
    ):
        input1 = torch.tensor([1, 1, 1, 0, 1], device="hpu")
        input2 = torch.tensor([1, 1, 1, 1], device="hpu")
        res = dynamic_output_shape_ops_fn(input1, input2)
        fga_assert_helper(fga.get_ops_summary(), "torch.ops.aten.nonzero.default", [{0, 1}])
        fga_assert_helper(fga.get_ops_summary(), "torch.ops.aten._unique2.default", [{0, 1}])
