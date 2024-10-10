###############################################################################
# Copyright (C) 2023 Habana Labs, Ltd. an Intel Company
# All Rights Reserved.
#
# Unauthorized copying of this file or any element(s) within it, via any medium
# is strictly prohibited.
# This file contains Habana Labs, Ltd. proprietary and confidential information
# and is subject to the confidentiality and license agreements under which it
# was provided.
#
###############################################################################
import habana_frameworks.torch.dynamo.compile_backend
import pytest
import torch

SUPPORTED_DTYPES = [torch.float, torch.bfloat16, torch.int, torch.long]
TENSOR_SHAPE = [2, 3, 4]
SCALAR_VALUES = [2.3, float("nan"), 0]


@pytest.mark.parametrize("dtype", SUPPORTED_DTYPES, ids=[str(dtype) for dtype in SUPPORTED_DTYPES])
@pytest.mark.parametrize("op_code", [torch.xlogy, torch.special.xlog1py])
class TestHpuXlogy:
    @staticmethod
    def _compile(fn):
        torch._dynamo.reset()
        return torch.compile(fn), torch.compile(fn, backend="hpu_backend")

    @staticmethod
    def _compare(expected_cpu, actual_hpu):
        assert torch.allclose(expected_cpu, actual_hpu.cpu(), equal_nan=True)

    @staticmethod
    def _create_cpu_hpu_tensors(creator, shape, dtype):
        if creator == torch.randint:
            cpu_tensor = creator(low=0, high=10, size=shape, dtype=dtype)
        else:
            cpu_tensor = creator(shape, dtype=dtype)

        hpu_tensor = cpu_tensor.to("hpu")
        return cpu_tensor, hpu_tensor

    @classmethod
    def _create_rand_tensors(cls, shape, dtype):
        if dtype in (torch.int, torch.long):
            return cls._create_cpu_hpu_tensors(torch.randint, shape, dtype=dtype)
        else:
            return cls._create_cpu_hpu_tensors(torch.rand, shape, dtype=dtype)

    @classmethod
    def _create_empty_tensors(cls, shape, dtype):
        return cls._create_cpu_hpu_tensors(torch.empty, shape, dtype=dtype)

    @pytest.mark.skip(reason="https://jira.habana-labs.com/browse/SW-167770")
    def test_no_scalars_outplace(self, op_code, dtype):
        def fn(input, other):
            return op_code(input=input, other=other)

        cpu_input, hpu_input = TestHpuXlogy._create_rand_tensors(TENSOR_SHAPE, dtype)
        cpu_other, hpu_other = TestHpuXlogy._create_rand_tensors(TENSOR_SHAPE, dtype)

        cpu_compiled_fn, hpu_compiled_fn = TestHpuXlogy._compile(fn)

        cpu_output = cpu_compiled_fn(cpu_input, cpu_other)
        hpu_output = hpu_compiled_fn(hpu_input, hpu_other)

        TestHpuXlogy._compare(cpu_output, hpu_output)

    @pytest.mark.skip(reason="https://jira.habana-labs.com/browse/SW-167770")
    def test_no_scalars_out(self, op_code, dtype):
        def fn(input, other, out):
            op_code(input=input, other=other, out=out)

        cpu_input, hpu_input = TestHpuXlogy._create_rand_tensors(TENSOR_SHAPE, dtype)
        cpu_other, hpu_other = TestHpuXlogy._create_rand_tensors(TENSOR_SHAPE, dtype)
        cpu_out, hpu_out = TestHpuXlogy._create_empty_tensors(TENSOR_SHAPE, dtype)

        cpu_compiled_fn, hpu_compiled_fn = TestHpuXlogy._compile(fn)

        cpu_compiled_fn(cpu_input, cpu_other, cpu_out)
        hpu_compiled_fn(hpu_input, hpu_other, hpu_out)

        TestHpuXlogy._compare(cpu_out, hpu_out)

    def test_no_scalars_inplace(self, op_code, dtype):
        def fn(input, other):
            op_code(input=input, other=other)

        cpu_input, hpu_input = TestHpuXlogy._create_rand_tensors(TENSOR_SHAPE, dtype)
        cpu_other, hpu_other = TestHpuXlogy._create_rand_tensors(TENSOR_SHAPE, dtype)

        cpu_compiled_fn, hpu_compiled_fn = TestHpuXlogy._compile(fn)

        cpu_compiled_fn(cpu_input, cpu_other)
        hpu_compiled_fn(hpu_input, hpu_other)

        TestHpuXlogy._compare(cpu_input, hpu_input)

    @pytest.mark.skip(reason="https://jira.habana-labs.com/browse/SW-167770")
    @pytest.mark.parametrize("scalar_value", SCALAR_VALUES)
    def test_input_scalar_outplace(self, op_code, dtype, scalar_value):
        def fn(input, other):
            return op_code(self=input, other=other)

        cpu_other, hpu_other = TestHpuXlogy._create_rand_tensors(TENSOR_SHAPE, dtype)

        cpu_compiled_fn, hpu_compiled_fn = TestHpuXlogy._compile(fn)

        cpu_output = cpu_compiled_fn(scalar_value, cpu_other)
        hpu_output = hpu_compiled_fn(scalar_value, hpu_other)

        TestHpuXlogy._compare(cpu_output, hpu_output)

    @pytest.mark.skip(reason="https://jira.habana-labs.com/browse/SW-167770")
    @pytest.mark.parametrize("scalar_value", SCALAR_VALUES)
    def test_input_scalar_out(self, op_code, dtype, scalar_value):
        def fn(input, other, out):
            op_code(self=input, other=other, out=out)

        cpu_other, hpu_other = TestHpuXlogy._create_rand_tensors(TENSOR_SHAPE, dtype)
        cpu_out, hpu_out = TestHpuXlogy._create_empty_tensors(TENSOR_SHAPE, dtype)

        cpu_compiled_fn, hpu_compiled_fn = TestHpuXlogy._compile(fn)

        cpu_compiled_fn(scalar_value, cpu_other, cpu_out)
        hpu_compiled_fn(scalar_value, hpu_other, hpu_out)

        TestHpuXlogy._compare(cpu_out, hpu_out)

    @pytest.mark.skip(reason="https://jira.habana-labs.com/browse/SW-167770")
    @pytest.mark.parametrize("scalar_value", SCALAR_VALUES)
    def test_other_scalar_outplace(self, op_code, dtype, scalar_value):
        def fn(input, other):
            return op_code(input=input, other=other)

        cpu_input, hpu_input = TestHpuXlogy._create_rand_tensors(TENSOR_SHAPE, dtype)

        cpu_compiled_fn, hpu_compiled_fn = TestHpuXlogy._compile(fn)

        cpu_output = cpu_compiled_fn(cpu_input, scalar_value)
        hpu_output = hpu_compiled_fn(hpu_input, scalar_value)

        TestHpuXlogy._compare(cpu_output, hpu_output)

    @pytest.mark.skip(reason="https://jira.habana-labs.com/browse/SW-167770")
    @pytest.mark.parametrize("scalar_value", SCALAR_VALUES)
    def test_other_scalar_out(self, op_code, dtype, scalar_value):
        def fn(input, other, out):
            op_code(input=input, other=other, out=out)

        cpu_input, hpu_input = TestHpuXlogy._create_rand_tensors(TENSOR_SHAPE, dtype)
        cpu_out, hpu_out = TestHpuXlogy._create_empty_tensors(TENSOR_SHAPE, dtype)

        cpu_compiled_fn, hpu_compiled_fn = TestHpuXlogy._compile(fn)

        cpu_compiled_fn(cpu_input, scalar_value, cpu_out)
        hpu_compiled_fn(hpu_input, scalar_value, hpu_out)

        TestHpuXlogy._compare(cpu_out, hpu_out)

    @pytest.mark.parametrize("scalar_value", SCALAR_VALUES)
    def test_other_scalar_inplace(self, op_code, dtype, scalar_value):
        def fn(input, other):
            op_code(input=input, other=other)

        cpu_input, hpu_input = TestHpuXlogy._create_rand_tensors(TENSOR_SHAPE, dtype)

        cpu_compiled_fn, hpu_compiled_fn = TestHpuXlogy._compile(fn)

        cpu_compiled_fn(cpu_input, scalar_value)
        hpu_compiled_fn(hpu_input, scalar_value)

        TestHpuXlogy._compare(cpu_input, hpu_input)
