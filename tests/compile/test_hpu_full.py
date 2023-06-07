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

@pytest.mark.parametrize("dtype", [None, torch.float, torch.bfloat16, torch.int8, torch.int32])
@pytest.mark.parametrize("shape_type", ["size", "tuple", "list"])
def test_full(dtype, shape_type):
    with env_var_in_scope(PT_HPU_LAZY_MODE="0", PT_HPU_DETERMINISTIC_ENABLE="0", PT_HPU_COMPILE_USE_RECIPES=True):
        import habana_frameworks.torch.core as htcore
        if shape_type == "size":
            shape = torch.Size([2,4])
        elif shape_type == "tuple":
            shape = (2,4)
        else:
            shape = [2,4]

        def fn(fill_value, device):
            return torch.full(shape, fill_value, device=device, dtype=dtype)
        compiled_fn = torch.compile(fn, backend="aot_hpu_training_backend")

        target_dtype = dtype if dtype is not None else torch.get_default_dtype()
        if (target_dtype.is_floating_point):
            val = random.random()
        else:
            val = random.randint(-128, 127)

        expected = fn(val, "cpu")
        result = compiled_fn(val, "hpu").cpu()
        assert torch.equal(result, expected)