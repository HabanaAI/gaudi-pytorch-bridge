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


@pytest.mark.parametrize("shape", [(2, 4, 6), (8, 8, 4)])
@pytest.mark.parametrize("dim", [0, 1, 2, -1])
@pytest.mark.parametrize("keepdim", [True, False])
class TestHpuArgMinMax:

    @staticmethod
    def _common_test_argmin_max(shape, dim, keepdim, op_code, dtype):
        def fn(input, dim, keepdim):
            return op_code(input, dim, keepdim)

        if dtype.is_floating_point:
            cpu_input = torch.randn(shape, dtype=dtype)
        else:
            low = 0 if dtype == torch.uint8 else -127
            cpu_input = torch.randint(low=low, high=127, size=shape, dtype=dtype)
        hpu_input = cpu_input.to("hpu")

        torch._dynamo.reset()
        cpu_compiled_fn = torch.compile(fn)
        hpu_compiled_fn = torch.compile(fn, backend="aot_hpu_training_backend")

        cpu_output = cpu_compiled_fn(cpu_input, dim, keepdim)
        hpu_output = hpu_compiled_fn(hpu_input, dim, keepdim).cpu()

        assert torch.equal(hpu_output, cpu_output)

    @pytest.mark.parametrize("dtype", [torch.float, torch.bfloat16])
    def test_argmin(self, shape, dim, keepdim, dtype):
        TestHpuArgMinMax._common_test_argmin_max(shape, dim, keepdim, torch.argmin, dtype)

    @pytest.mark.parametrize("dtype", [torch.float, torch.bfloat16, torch.int32, torch.int8, torch.uint8])
    def test_argmax(self, shape, dim, keepdim, dtype):
        TestHpuArgMinMax._common_test_argmin_max(shape, dim, keepdim, torch.argmax, dtype)
