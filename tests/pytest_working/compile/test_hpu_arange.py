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
import random
from utils import env_var_in_scope

@pytest.mark.xfail(reason="CI problem: undefined symbol: _ZN6habana5graph12GraphStorage3getEv [SW-150162]")
@pytest.mark.parametrize("dtype", [None, torch.float, torch.bfloat16, torch.int8, torch.int32, torch.long])
@pytest.mark.parametrize("layout", [None, torch.strided])
@pytest.mark.parametrize("start", [None, 0, 10])
@pytest.mark.parametrize("step", [None, 1, 20])
@pytest.mark.parametrize("end", [40, 100])
def test_arange(dtype, layout, start, step, end):
    if step is not None and start is None:
        pytest.skip('Invalid case')
    with env_var_in_scope(PT_HPU_LAZY_MODE="0", PT_HPU_DETERMINISTIC_ENABLE="0", PT_HPU_COMPILE_USE_RECIPES=True):
        import habana_frameworks.torch.core as htcore
        def fn(start, layout, step, end, device):
            if step is not None:
                return torch.arange(start=start, step=step, end=end, device=device, dtype=dtype, layout=layout)
            elif start is not None:
                return torch.arange(start=start, end=end, device=device, dtype=dtype, layout=layout)
            else:
                return torch.arange(end=end, device=device, dtype=dtype, layout=layout)
        compiled_fn = torch.compile(fn, backend="aot_hpu_training_backend")

        target_dtype = dtype if dtype is not None else torch.get_default_dtype()

        expected = fn(start, layout, step, end, "cpu")
        result = compiled_fn(start, layout, step, end, "hpu").cpu()
        assert torch.equal(result, expected)