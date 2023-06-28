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
import numpy as np
import torch
import pytest
from test_utils import env_var_in_scope

@pytest.mark.skip(reason="Tests in this file are changing environment variables")
@pytest.mark.xfail(
    reason="torch._dynamo.exc.TorchRuntimeError. Remove xfail when SW-150162 is done."
)
@pytest.mark.parametrize(
    "dtype", [torch.float, torch.bfloat16, torch.int32, torch.long]
)
@pytest.mark.parametrize("out_int32", [True, False])
@pytest.mark.parametrize("right", [True, False])
def test_searchsorted(dtype, out_int32, right):
    with env_var_in_scope(PT_HPU_LAZY_MODE="0"):
        torch.manual_seed(0)
        import habana_frameworks.torch.core as htcore

        def fn(sorted_sequence, values):
            return torch.searchsorted(
                sorted_sequence, values, out_int32=out_int32, right=right
            )

        hpu_compiled_function = torch.compile(fn, backend="aot_hpu_training_backend")
        cpu_compiled_function = torch.compile(fn)

        cpu_sorted_sequence, _ = torch.sort(torch.randn((10, 10)).to(dtype))
        cpu_inputs = torch.randn((10, 5)).to(dtype)
        hpu_sorted_sequence = cpu_sorted_sequence.to("hpu")
        hpu_inputs = cpu_inputs.to("hpu")

        hpu_result = hpu_compiled_function(hpu_sorted_sequence, hpu_inputs)
        cpu_result = cpu_compiled_function(cpu_sorted_sequence, cpu_inputs)

        np.testing.assert_array_equal(hpu_result.cpu().numpy(), cpu_result.numpy())
