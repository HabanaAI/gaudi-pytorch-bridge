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
import numpy as np
import habana_frameworks.torch.dynamo.compile_backend
from test_utils import format_tc, is_pytest_mode_compile, is_gaudi1

test_shapes_dtypes = [
    ((2, 2, 2, 2, 2), (2, 2, 1, 2), torch.float32),
    ((2, 2, 2, 2), (2, 2, 2, 2), torch.bfloat16),
    ((2, 2, 2, 2), (2, 2), torch.half),
    ((2, 2, 2), (1, 1, 2), torch.int32),
    ((2, 2, 2), (2), torch.long),
    ((2, 2), (1), torch.bool),
    ((2), (1), torch.int8),
]

@pytest.mark.parametrize("self_shape, mask_shape, dtype", test_shapes_dtypes, ids=format_tc)
class TestHpuMaskedSelect:
  @staticmethod
  def test_hpu_masked_select(self_shape, mask_shape, dtype):
    if pytest.mode == "compile" or pytest.mode == "lazy":
      pytest.skip(reason="Masked_select is implemented via hpu::index which can't run in compile mode, lazy not tested here")
    if is_gaudi1() and dtype == torch.half:
        pytest.skip("Half is not supported on Gaudi.")
    def fn(input, mask):
      return torch.masked_select(input, mask)

    cpu_input = torch.zeros(self_shape, dtype=dtype).random_()
    cpu_mask = torch.zeros(mask_shape, dtype=torch.bool).random_()

    hpu_input = cpu_input.to('hpu')
    hpu_mask = cpu_mask.to('hpu')

    hpu_wrapped_fn = (
        torch.compile(fn, backend="aot_hpu_training_backend")
        if is_pytest_mode_compile()
        else fn
    )
    torch._dynamo.reset()

    cpu_result = fn(cpu_input, cpu_mask)
    hpu_result = hpu_wrapped_fn(hpu_input, hpu_mask)

    torch.allclose(cpu_result, hpu_result.cpu())