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
from test_utils import clear_t_compile_logs, compare_tensors, hpu, is_lazy, is_pytest_mode_compile


@pytest.mark.skipif(is_lazy(), reason="aten::unfold is not implemented for lazy mode")
@pytest.mark.parametrize("modify_view", [True, False], ids=lambda val: f"modify_view={val}")
@pytest.mark.parametrize("step", [3], ids=lambda val: f"step={val}")
@pytest.mark.parametrize("size", [2], ids=lambda val: f"size={val}")
@pytest.mark.parametrize("dimension", [0, 2], ids=lambda val: f"dim={val}")
@pytest.mark.parametrize("shape", [(2, 3, 8), (6, 4, 3, 9)], ids=lambda val: f"shape={val}")
def test_unfold_view(shape, dimension, size, step, modify_view):
    def tensor_unfold(x, dimension, size, step):
        return x.unfold(dimension, size, step)

    def tensor_unfold_modified(x, dimension, size, step):
        y = x.unfold(dimension, size, step)
        return y + 0.278

    input_cpu = torch.randn(shape)
    input_hpu = input_cpu.to(hpu)
    unfold_fn = tensor_unfold_modified if modify_view else tensor_unfold
    result_cpu = unfold_fn(input_cpu, dimension, size, step)

    if is_pytest_mode_compile():
        torch._dynamo.reset()
        clear_t_compile_logs()
        unfold_fn = torch.compile(unfold_fn, backend="hpu_backend", dynamic=False)

    result_hpu = unfold_fn(input_hpu, dimension, size, step)
    compare_tensors(result_hpu, result_cpu, rtol=0, atol=0)
