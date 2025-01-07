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
from test_utils import (
    check_op_in_fuser_fused_ops,
    check_ops_executed_in_jit_ir,
    clear_fuser_debug_logs,
    clear_t_compile_logs,
    get_fuser_debug_logs_path,
    is_pytest_mode_compile,
    is_pytest_mode_eager,
    setup_teardown_env_fixture,
)


@pytest.fixture
def enable_determinism():
    ### Enable determinism before test starts, and restore the flag once it is finished
    previously_deterministic = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    yield
    torch.use_deterministic_algorithms(previously_deterministic)


@pytest.mark.skip(
    "Cannot reliably enable logging selectively for single test - required for verification. Need to revisit once a way found."
)
@pytest.mark.usefixtures("setup_teardown_env_fixture")
@pytest.mark.usefixtures("enable_determinism")
@pytest.mark.usefixtures("clear_fuser_debug_logs")
@pytest.mark.parametrize("input_shape", [[10, 10, 10, 10]])
@pytest.mark.parametrize(
    "setup_teardown_env_fixture",
    [
        {
            "ENABLE_EXPERIMENTAL_FLAGS": "true",
            "FUSER_DEBUG_DATA": "1",
            "FUSER_DEBUG_PATH": get_fuser_debug_logs_path(),
        }
    ],
    indirect=True,
)
@pytest.mark.skipif(
    is_pytest_mode_eager(),
    reason="Fuser does not support deterministic BN for eager mode",
)
def test_batch_norm_deterministic(input_shape):
    assert torch.are_deterministic_algorithms_enabled()

    def fn(input, running_mean, running_var, training):
        return torch.nn.functional.batch_norm(
            input,
            running_mean,
            running_var,
            training=training,
        )

    if is_pytest_mode_compile():
        clear_t_compile_logs()
        torch._dynamo.reset()
        fn = torch.compile(fn, backend="hpu_backend")

    assert len(input_shape) >= 2
    input = torch.rand(input_shape)
    running_mean = torch.zeros(input_shape[1]).to("hpu")
    running_var = torch.ones(input_shape[1]).to("hpu")

    result = fn(
        input.to("hpu"),
        running_mean,
        running_var,
        training=True,
    ).to("cpu")

    if is_pytest_mode_compile():
        check_ops_executed_in_jit_ir("_native_batch_norm_legit_functional")

    # TODO: Find a way to reliably obtain GC post graphs for single test and check information in it

    assert not check_op_in_fuser_fused_ops(
        ["batch_norm_fwd_f32", "batch_norm_stage1_fwd_f32", "batch_norm_stage2_fwd_f32"]
    )
