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
from habana_frameworks.torch.dynamo.compile_backend.config import configuration_flags
from test_utils import (
    check_ops_executed_in_jit_ir,
    clear_t_compile_logs,
    compare_tensors,
    format_tc,
    is_gaudi1,
    is_gaudi3,
    is_pytest_mode_compile,
    setup_teardown_env_fixture,
)


# This test checks if the masked_fill op will accept a value tensor on the CPU while the input tensor is on the HPU
@pytest.mark.parametrize("shape", [(2, 7)], ids=format_tc)
@pytest.mark.parametrize("value", [2])
@pytest.mark.parametrize("scalar_value", [True, False])
@pytest.mark.parametrize("dynamic", [False, True])
@pytest.mark.parametrize("dtype", [torch.float], ids=format_tc)
@pytest.mark.parametrize(
    "setup_teardown_env_fixture",
    [{"PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES": 1}],
    indirect=True,
)
class TestHpuMaskedMixedDevices:
    @classmethod
    def setup_class(self):
        # For scalar tensor there is a fallback to eager
        self.original_configuration = configuration_flags["use_eager_fallback"]
        configuration_flags["use_eager_fallback"] = True

    @classmethod
    def teardown_class(self):
        configuration_flags["use_eager_fallback"] = self.original_configuration

    @staticmethod
    def test_hpu_masked_mixed_devices(shape, value, scalar_value, dynamic, dtype, setup_teardown_env_fixture):
        if dynamic and (is_gaudi3() or not pytest.mode == "compile"):
            pytest.skip("Not supported test configuration")

        def fn(input, mask, value):
            input.masked_fill_(mask, value)

        wrapped_fn = torch.compile(fn, backend="hpu_backend") if pytest.mode == "compile" else fn
        iters = 3 if dynamic else 1
        for i in range(iters):
            modified_shape = [(dim * (i + 1)) for dim in shape]
            mask = torch.randint(low=0, high=2, size=modified_shape, dtype=torch.bool, device="cpu").to("hpu")
            input = torch.rand(modified_shape, dtype=dtype, device="hpu")
            value = value if scalar_value else torch.tensor(value, dtype=dtype, device="cpu")
            wrapped_fn(input, mask, value)
            assert input.device.type == "hpu"


@pytest.mark.parametrize(
    "self_shape, mask_shape",
    [((12, 16), (12, 16)), ((8, 10, 12), (8, 1, 12)), ((1, 4, 8), (6, 4, 8)), ((4, 1, 8, 12), (4, 6, 1, 12))],
    ids=format_tc,
)
@pytest.mark.parametrize("value", [-5])
@pytest.mark.parametrize("scalar_value", [True, False])
@pytest.mark.parametrize("dtype", [torch.float, torch.bfloat16, torch.int], ids=format_tc)
@pytest.mark.parametrize(
    "setup_teardown_env_fixture",
    [{"PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES": 0}],
    indirect=True,
)
def test_masked_fill(self_shape, mask_shape, value, scalar_value, dtype, setup_teardown_env_fixture):
    def fn(self, mask, value):
        return self.masked_fill(mask, value)

    if is_pytest_mode_compile():
        torch._dynamo.reset()
        clear_t_compile_logs()
        fn = torch.compile(fn, backend="hpu_backend")

    self = torch.randint(low=-50, high=50, size=self_shape).to(dtype)
    mask = torch.randint(low=0, high=2, size=mask_shape, dtype=torch.bool)
    value = value if scalar_value else torch.tensor(value, dtype=dtype)

    self_hpu = self.to("hpu")
    mask_hpu = mask.to("hpu")
    value_hpu = value if scalar_value else value.to("hpu")

    expected = self.masked_fill(mask, value)
    result = fn(self_hpu, mask_hpu, value_hpu)

    compare_tensors(result, expected, atol=0.0, rtol=0.0)
    if is_pytest_mode_compile():
        ops = {"masked_fill"}
        check_ops_executed_in_jit_ir(ops)


@pytest.mark.skipif(is_gaudi1(), reason="Not supported by Gaudi")
@pytest.mark.parametrize("dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
@pytest.mark.parametrize(
    "setup_teardown_env_fixture",
    [{"PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES": 0}],
    indirect=True,
)
def test_masked_fill_float8(dtype, setup_teardown_env_fixture):
    def fn(inp, mask, max_val):
        output = inp.masked_fill_(mask, max_val)
        return output

    # CPU
    mask_val = -100
    input_c = torch.randn((1000, 1000), dtype=torch.bfloat16)
    mask = torch.randint(0, 2, (1000, 1000)).to(torch.bool)
    result = fn(input_c, mask, mask_val).to(dtype).to(torch.bfloat16)

    if pytest.mode == "compile":
        torch._dynamo.reset()
        clear_t_compile_logs()
        fn = torch.compile(fn, backend="hpu_backend")

    # HPU
    input_hpu = input_c.to("hpu").to(dtype)
    mask_hpu = mask.to("hpu")
    hresult = fn(input_hpu, mask_hpu, mask_val).cpu().to(torch.bfloat16)

    # we are comparing the result in bfloat16 as float8 comparison is giving
    # some issue , getting
    # RuntimeError: "mul_cpu_reduced_float" not implemented for 'Float8_e5m2'
    assert torch.allclose(result, hresult, atol=0.01, rtol=0.01)
    if is_pytest_mode_compile():
        ops = {"masked_fill"}
        check_ops_executed_in_jit_ir(ops)
