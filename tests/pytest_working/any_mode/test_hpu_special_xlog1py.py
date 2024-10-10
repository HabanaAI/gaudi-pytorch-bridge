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
import habana_frameworks.torch.dynamo.compile_backend
import pytest
import torch
from habana_frameworks.torch.dynamo.compile_backend.config import configuration_flags
from test_utils import format_tc, is_pytest_mode_compile


@pytest.mark.parametrize("shapes", [([2, 2], []), ([], [2, 2]), ([2, 2], [2, 2])], ids=format_tc)
@pytest.mark.parametrize("in_place_out", [True, False])
@pytest.mark.parametrize("dtype", [torch.float, torch.bfloat16], ids=format_tc)
class TestHpuSpecialXlog1py:
    @classmethod
    def setup_class(self):
        # For scalar_tensor (coming from decomposition) there is expected fallback to eager
        self.original_configuration = configuration_flags["use_eager_fallback"]
        configuration_flags["use_eager_fallback"] = True

    @classmethod
    def teardown_class(self):
        configuration_flags["use_eager_fallback"] = self.original_configuration

    @staticmethod
    def test_hpu_special_xlog1py(shapes, in_place_out, dtype):
        def fn(input, other, out=None):
            if out == None:
                return torch.special.xlog1py(input, other)
            else:
                torch.special.xlog1py(input, other, out=out)

        input_shape, other_shape = shapes
        input_dtype = dtype
        other_dtype = dtype
        check_mixed_dtypes = input_shape == other_shape
        if check_mixed_dtypes:
            if input_dtype == torch.float:
                other_dtype = torch.bfloat16
            else:
                other_dtype = torch.float

        cpu_input = torch.rand(input_shape, dtype=input_dtype)
        hpu_input = cpu_input.to("hpu")
        cpu_other = torch.rand(other_shape, dtype=other_dtype)
        hpu_other = cpu_other.to("hpu")
        cpu_out, hpu_out = None, None
        if in_place_out:
            if len(input_shape) == 0:
                cpu_out = torch.empty_like(cpu_other)
                hpu_out = torch.empty_like(hpu_other)
            else:
                cpu_out = torch.empty_like(cpu_input)
                hpu_out = torch.empty_like(hpu_input)
        hpu_wrapped_fn = torch.compile(fn, backend="hpu_backend") if is_pytest_mode_compile() else fn
        torch._dynamo.reset()

        cpu_output = fn(cpu_input, cpu_other, cpu_out)
        hpu_output = hpu_wrapped_fn(hpu_input, hpu_other, hpu_out)
        if in_place_out:
            torch.allclose(cpu_out, hpu_out.cpu())
        else:
            torch.allclose(cpu_output, hpu_output.cpu())
