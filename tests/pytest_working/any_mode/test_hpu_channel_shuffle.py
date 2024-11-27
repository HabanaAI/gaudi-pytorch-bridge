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

import habana_frameworks.torch.dynamo.compile_backend
import pytest
import torch
from test_utils import (
    check_ops_executed_in_jit_ir,
    clear_t_compile_logs,
    format_tc,
    is_gaudi1,
    is_gaudi3,
    is_pytest_mode_compile,
    setup_teardown_env_fixture,
)


@pytest.mark.parametrize(
    "shape_and_groups",
    [((3, 4, 2), 2), ((2, 9, 3, 4), 3), ((2, 6, 8, 4, 3), 2)],
    ids=format_tc,
)
@pytest.mark.parametrize("dynamic", [False, True])
@pytest.mark.parametrize(
    "dtype", [torch.float, torch.bfloat16, torch.float16, torch.int, torch.int8, torch.short, torch.bool], ids=format_tc
)
@pytest.mark.parametrize(
    "setup_teardown_env_fixture",
    [{"PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES": 1}],
    indirect=True,
)
def test_hpu_channel_shuffle(shape_and_groups, dynamic, dtype, setup_teardown_env_fixture):
    if dynamic and (is_gaudi3() or not pytest.mode == "compile"):
        pytest.skip("Not supported test configuration with dynamic shapes enabled")

    if dtype == torch.float16 and is_gaudi1():
        pytest.skip("Half is not supported on Gaudi")

    def fn(input, model):
        return model(input)

    shape, groups = shape_and_groups
    cpu_model = torch.nn.ChannelShuffle(groups)
    hpu_model = cpu_model.to("hpu")
    hpu_wrapped_fn = fn
    if is_pytest_mode_compile():
        hpu_wrapped_fn = torch.compile(fn, backend="hpu_backend")
        clear_t_compile_logs()
        torch._dynamo.reset()

    iters = [1, 3, 2] if dynamic else [1]
    for i in iters:
        modified_shape = [dim * i for dim in shape]
        cpu_input = (
            torch.rand(modified_shape, dtype=dtype)
            if dtype.is_floating_point
            else (
                torch.randint(2, size=modified_shape, dtype=dtype)
                if dtype == torch.bool
                else torch.randint(low=1, high=127, size=modified_shape, dtype=dtype)
            )
        )
        hpu_input = cpu_input.to("hpu")
        cpu_output = fn(cpu_input, cpu_model)
        hpu_output = hpu_wrapped_fn(hpu_input, hpu_model).cpu()
        assert torch.equal(cpu_output, hpu_output)

    if is_pytest_mode_compile():
        check_ops_executed_in_jit_ir("channel_shuffle")
