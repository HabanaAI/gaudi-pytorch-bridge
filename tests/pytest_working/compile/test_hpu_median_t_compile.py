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
import torch
import pytest
import habana_frameworks.torch.dynamo.compile_backend

SUPPORTED_DTYPES = [torch.float, torch.int]


@pytest.mark.parametrize("dtype", SUPPORTED_DTYPES, ids=[str(dtype) for dtype in SUPPORTED_DTYPES])
@pytest.mark.parametrize("shape", [(2, 4), (2, 4, 6, 8)])
class TestHpuMedian:
    @staticmethod
    def _compile(fn):
        torch._dynamo.reset()
        return torch.compile(fn), torch.compile(fn, backend="aot_hpu_training_backend")

    @classmethod
    def _create_rand_tensors(cls, shape, dtype):
        if dtype.is_floating_point:
            cpu_tensor = torch.randn(shape, dtype=dtype)
        else:
            cpu_tensor = torch.randint(low=-127, high=127, size=shape, dtype=dtype)
        return cpu_tensor, cpu_tensor.to("hpu")

    @classmethod
    def _create_empty_tensors(cls, shape, dtype):
        cpu_tensor = torch.empty(shape, dtype=dtype)
        return cpu_tensor, cpu_tensor.to("hpu")

    def test_median(self, dtype, shape):
        def fn(input):
            return torch.median(input=input)

        cpu_input, hpu_input = TestHpuMedian._create_rand_tensors(shape, dtype)

        cpu_compiled_fn, hpu_compiled_fn = TestHpuMedian._compile(fn)

        cpu_output = cpu_compiled_fn(cpu_input)
        hpu_output = hpu_compiled_fn(hpu_input)

        assert torch.equal(cpu_output, hpu_output.cpu())

    @pytest.mark.parametrize("dim", [0, -1])
    @pytest.mark.parametrize("keepdim", [True, False])
    def test_median_dim(self, dtype, shape, dim, keepdim):
        def fn(input, dim, keepdim):
            return torch.median(input=input, dim=dim, keepdim=keepdim)

        cpu_input, hpu_input = TestHpuMedian._create_rand_tensors(shape, dtype)

        cpu_compiled_fn, hpu_compiled_fn = TestHpuMedian._compile(fn)

        cpu_output = cpu_compiled_fn(cpu_input, dim, keepdim)
        hpu_output = hpu_compiled_fn(hpu_input, dim, keepdim)

        assert torch.equal(cpu_output.values, hpu_output.values.cpu())
        assert torch.equal(cpu_output.indices, hpu_output.indices.cpu())

    @pytest.mark.parametrize("dim", [0, -1])
    @pytest.mark.parametrize("keepdim", [True, False])
    def test_median_dim_out(self, dtype, shape, dim, keepdim):
        def fn(input, dim, keepdim, out):
            torch.median(input, dim=dim, keepdim=keepdim, out=out)

        expected_shape = list(shape)
        if keepdim:
            expected_shape[dim] = 1
        else:
            expected_shape.pop(dim)

        cpu_input, hpu_input = TestHpuMedian._create_rand_tensors(shape, dtype)
        cpu_value, hpu_value = TestHpuMedian._create_empty_tensors(expected_shape, dtype)
        cpu_index, hpu_index = TestHpuMedian._create_empty_tensors(expected_shape, torch.int64)
        cpu_out = cpu_value, cpu_index
        hpu_out = hpu_value, hpu_index

        cpu_compiled_fn, hpu_compiled_fn = TestHpuMedian._compile(fn)

        cpu_compiled_fn(cpu_input, dim, keepdim, cpu_out)
        hpu_compiled_fn(hpu_input, dim, keepdim, hpu_out)

        assert torch.equal(cpu_out[0], hpu_out[0].cpu())
        assert torch.equal(cpu_out[1], hpu_out[1].cpu())
