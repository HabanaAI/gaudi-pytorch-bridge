###############################################################################
# Copyright (c) 2021-2025 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
###############################################################################

from functools import reduce

import habana_frameworks.torch.dynamo.compile_backend  # noqa: F401
import pytest
import torch
from test_utils import compile_function_if_compile_mode


@pytest.mark.skip(reason="https://jira.habana-labs.com/browse/SW-167770")
@pytest.mark.parametrize("inputs_shape", [(6,), (4, 6), (3, 5, 2)])
@pytest.mark.parametrize("accumulate", [False, True])
def test_index_put_bool_mask_only(inputs_shape, accumulate):
    def fn(tensor, bool_mask, value, accumulate):
        return tensor.index_put([bool_mask], value, accumulate)

    self_numel = reduce(lambda x, y: x * y, list(inputs_shape))
    indices_numel = reduce(lambda x, y: x * y, list(inputs_shape))
    tensor = torch.arange(self_numel).view(inputs_shape)
    mask_in = torch.arange(indices_numel).view(inputs_shape)
    bool_mask = mask_in > indices_numel / 3
    values = torch.tensor([-100])

    torch._dynamo.reset()
    cpu_res = fn(tensor, bool_mask, values, accumulate)

    compiled_hpu = compile_function_if_compile_mode(fn)
    hpu_res = compiled_hpu(tensor.to("hpu"), bool_mask.to("hpu"), values.to("hpu"), accumulate)

    assert torch.allclose(cpu_res, hpu_res.to("cpu"), rtol=1e-3, atol=1e-3)


@pytest.mark.skip(reason="https://jira.habana-labs.com/browse/SW-167770")
@pytest.mark.parametrize("inputs_shape", [(3, 5, 2)])
@pytest.mark.parametrize("ind_shape", [(3, 5)])
@pytest.mark.parametrize("accumulate", [False])
def test_index_put_bool_adv_indexing(inputs_shape, ind_shape, accumulate):
    def fn(tensor, bool_mask, value, accumulate):
        # tensor.index_put([bool_mask], value, accumulate)
        tensor[bool_mask, :] = value
        return tensor

    self_numel = reduce(lambda x, y: x * y, list(inputs_shape))
    indices_numel = reduce(lambda x, y: x * y, list(ind_shape))
    tensor = torch.arange(self_numel).view(inputs_shape)
    mask_in = torch.arange(indices_numel).view(ind_shape)
    bool_mask = mask_in > indices_numel / 3
    values = torch.tensor([-100])

    torch._dynamo.reset()
    cpu_res = fn(tensor, bool_mask, values, accumulate)

    compiled_hpu = compile_function_if_compile_mode(fn)
    hpu_res = compiled_hpu(tensor.to("hpu"), bool_mask.to("hpu"), values.to("hpu"), accumulate)

    assert torch.allclose(cpu_res, hpu_res.to("cpu"), rtol=1e-3, atol=1e-3)
