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
import pytest
import torch


@pytest.mark.parametrize("dtype", [None, torch.float, torch.bfloat16, torch.int8, torch.int32, torch.long])
@pytest.mark.parametrize("layout", [None, torch.strided])
@pytest.mark.parametrize("start", [None, 0, 10])
@pytest.mark.parametrize("step", [None, 1, 20])
@pytest.mark.parametrize("end", [40, 100])
def test_arange(dtype, layout, start, step, end):
    if step is not None and start is None:
        pytest.skip("Invalid case")

    def fn(start, layout, step, end, device):
        if step is not None:
            return torch.arange(start=start, step=step, end=end, device=device, dtype=dtype, layout=layout)
        elif start is not None:
            return torch.arange(start=start, end=end, device=device, dtype=dtype, layout=layout)
        else:
            return torch.arange(end=end, device=device, dtype=dtype, layout=layout)

    torch._dynamo.reset()
    compiled_fn = torch.compile(fn, backend="hpu_backend")

    expected = fn(start, layout, step, end, "cpu")
    result = compiled_fn(start, layout, step, end, "hpu").cpu()
    assert torch.equal(result, expected)


# Test for rounding issues in arange op
# SW-179498 (fixed)
@pytest.mark.parametrize("dtype", [torch.int32])
@pytest.mark.parametrize("layout", [torch.strided])
@pytest.mark.parametrize("start", [2.01, 2.2999999999999998])
@pytest.mark.parametrize("step", [3])
@pytest.mark.parametrize("end", [130, 134.5, 133.5, 135.5])
def test_arange_rounding_issue(dtype, layout, start, step, end):
    if step is not None and start is None:
        pytest.skip("Invalid case")

    def fn(start, layout, step, end, device):
        if step is not None:
            return torch.arange(start=start, step=step, end=end, device=device, dtype=dtype, layout=layout)
        elif start is not None:
            return torch.arange(start=start, end=end, device=device, dtype=dtype, layout=layout)
        else:
            return torch.arange(end=end, device=device, dtype=dtype, layout=layout)

    compiled_fn = torch.compile(fn, backend="hpu_backend")

    expected = fn(start, layout, step, end, "cpu")
    result = compiled_fn(start, layout, step, end, "hpu").cpu()
    assert torch.equal(result, expected)
