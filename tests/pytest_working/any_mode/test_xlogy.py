###############################################################################
# Copyright (C) 2024 Habana Labs, Ltd. an Intel Company
# All Rights Reserved.
#
# Unauthorized copying of this file or any element(s) within it, via any medium
# is strictly prohibited.
# This file contains Habana Labs, Ltd. proprietary and confidential information
# and is subject to the confidentiality and license agreements under which it
# was provided.
#
###############################################################################
import pytest
import torch
from habana_frameworks.torch.dynamo.compile_backend.config import configuration_flags
from test_utils import format_tc, is_gaudi1, is_pytest_mode_compile

dtypes = [torch.float, torch.bfloat16, torch.int, torch.long]

if not is_gaudi1():
    dtypes.append(torch.float16)


@pytest.mark.parametrize("input_shape, other_shape", [([2, 2], []), ([], [2, 2]), ([2, 2], [2, 2])], ids=format_tc)
@pytest.mark.parametrize("input_dtype", dtypes, ids=format_tc)
@pytest.mark.parametrize("other_dtype", dtypes, ids=format_tc)
class TestHpuXlogY:
    @classmethod
    def setup_class(self):
        # For scalar_tensor (coming from decomposition) there is expected fallback to eager
        self.original_configuration = configuration_flags["use_eager_fallback"]
        configuration_flags["use_eager_fallback"] = True

    @classmethod
    def teardown_class(self):
        configuration_flags["use_eager_fallback"] = self.original_configuration

    @staticmethod
    def test_hpu_xlogy(input_shape, other_shape, input_dtype, other_dtype):
        cpu_input = torch.rand(input_shape).to(dtype=input_dtype)
        hpu_input = cpu_input.to("hpu")
        cpu_other = torch.rand(other_shape).to(dtype=other_dtype)
        hpu_other = cpu_other.to("hpu")

        fn_hpu = torch.xlogy
        if is_pytest_mode_compile():
            fn_hpu = torch.compile(fn_hpu, backend="hpu_backend")
        torch._dynamo.reset()

        cpu_output = torch.xlogy(cpu_input, cpu_other)
        hpu_output = fn_hpu(hpu_input, hpu_other)

        torch.allclose(cpu_output, hpu_output.cpu())

    @staticmethod
    def test_hpu_xlogy_outplace(input_shape, other_shape, input_dtype, other_dtype):
        cpu_input = torch.rand(input_shape).to(dtype=input_dtype)
        hpu_input = cpu_input.to("hpu")
        cpu_other = torch.rand(other_shape).to(dtype=other_dtype)
        hpu_other = cpu_other.to("hpu")

        out_dtype = torch.promote_types(input_dtype, other_dtype)
        if not out_dtype.is_floating_point:
            out_dtype = torch.float32
        out_shape = torch.broadcast_shapes(input_shape, other_shape)

        cpu_out = torch.empty(size=out_shape, dtype=out_dtype, device="cpu")
        hpu_out = torch.empty(size=out_shape, dtype=out_dtype, device="hpu")

        fn_hpu = torch.xlogy
        if is_pytest_mode_compile():
            fn_hpu = torch.compile(fn_hpu, backend="hpu_backend")
        torch._dynamo.reset()

        torch.xlogy(cpu_input, cpu_other, out=cpu_out)
        fn_hpu(hpu_input, hpu_other, out=hpu_out)

        torch.allclose(cpu_out, hpu_out.cpu())

    @staticmethod
    def test_hpu_xlogy_inplace(input_shape, other_shape, input_dtype, other_dtype):
        if not input_dtype.is_floating_point or not input_dtype.is_floating_point:
            pytest.skip("Configuration unsupported: cannot cast Float to Int")
        if len(input_shape) == 0:
            pytest.skip("Configuration unsupported: Input shape doesn't match the broadcast shape")
        cpu_input = torch.rand(input_shape).to(dtype=input_dtype)
        hpu_input = cpu_input.to("hpu")
        cpu_other = torch.rand(other_shape).to(dtype=other_dtype)
        hpu_other = cpu_other.to("hpu")

        fn_hpu = torch.xlogy_
        if is_pytest_mode_compile():
            fn_hpu = torch.compile(fn_hpu, backend="hpu_backend")
        torch._dynamo.reset()

        torch.xlogy_(cpu_input, cpu_other)
        fn_hpu(hpu_input, hpu_other)

        torch.allclose(cpu_input, hpu_input.cpu())
