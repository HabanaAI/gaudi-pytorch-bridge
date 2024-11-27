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
from enum import Enum

import habana_frameworks.torch.core as htcore
import numpy as np
import pytest
import torch
from test_utils import compare_tensors, is_gaudi1


class Mode(Enum):
    SCALAR = 1
    TENSOR_SINGLE = 2
    TENSOR_FULL = 3
    NONE = 4


modes = [
    (Mode.SCALAR, Mode.SCALAR),
    (Mode.SCALAR, Mode.NONE),
    (Mode.TENSOR_SINGLE, Mode.TENSOR_SINGLE),
    (Mode.TENSOR_SINGLE, Mode.TENSOR_FULL),
    (Mode.TENSOR_SINGLE, Mode.NONE),
    (Mode.TENSOR_FULL, Mode.TENSOR_SINGLE),
    (Mode.TENSOR_FULL, Mode.TENSOR_FULL),
    (Mode.TENSOR_FULL, Mode.NONE),
    (Mode.NONE, Mode.SCALAR),
    (Mode.NONE, Mode.TENSOR_SINGLE),
    (Mode.NONE, Mode.TENSOR_FULL),
]

dtypes = [torch.float32, torch.bfloat16, torch.int, torch.long]
if not is_gaudi1():
    dtypes += [torch.half, torch.float8_e5m2, torch.float8_e4m3fn]


LIMIT_SCALE = 10


def generate_limits(mode, dtype, shape):
    if mode == Mode.SCALAR:
        if dtype in [torch.float8_e5m2, torch.float8_e4m3fn]:
            # make sure that limit is representable in fp8 formats
            limit = np.random.choice([-8.0, -4.0, -2.0, 0.0, 2.0, 4.0, 8.0])
        else:
            limit = int(np.random.randn() * LIMIT_SCALE)
        return (limit, limit)
    if mode == Mode.TENSOR_SINGLE:
        limit = (torch.randn(()) * LIMIT_SCALE).to(dtype)
        return (limit, limit.to("hpu"))
    if mode == Mode.TENSOR_FULL:
        limit = (torch.randn(shape) * LIMIT_SCALE).to(dtype)
        return (limit, limit.to("hpu"))
    return (None, None)


@pytest.mark.parametrize("shape", [(4, 5), (2, 4, 6)])
@pytest.mark.parametrize("min_mode, max_mode", modes)
@pytest.mark.parametrize("dtype", dtypes)
def test_clamp(shape, min_mode, max_mode, dtype):
    if pytest.mode == "compile":
        pytest.skip(reason="https://jira.habana-labs.com/browse/SW-167770")
    if (
        pytest.mode == "compile"
        and dtype in [torch.float8_e5m2, torch.float8_e4m3fn]
        # Below configuration is caused by https://jira.habana-labs.com/browse/SW-163439
        # but it's overriden by the SW-163692.
        # and min_mode == Mode.NONE
        # and max_mode == Mode.SCALAR
    ):
        pytest.skip("https://jira.habana-labs.com/browse/SW-163692")

    input = (torch.randn(shape) * LIMIT_SCALE).to(dtype)
    input_h = input.to("hpu")
    min, min_h = generate_limits(min_mode, dtype, shape)
    max, max_h = generate_limits(max_mode, dtype, shape)

    if dtype in [torch.float8_e5m2, torch.float8_e4m3fn]:
        input = input.float()
        if type(min) == torch.Tensor:
            min = min.float()
        if type(max) == torch.Tensor:
            max = max.float()

    def fn(input, min, max):
        return torch.clamp(input, min, max)

    if pytest.mode == "compile":
        torch._dynamo.reset()
        fn = torch.compile(fn, backend="hpu_backend")

    result_cpu = torch.clamp(input, min, max)
    result_hpu = fn(input_h, min_h, max_h)

    compare_tensors(result_hpu, result_cpu, atol=0.0, rtol=0.0)
