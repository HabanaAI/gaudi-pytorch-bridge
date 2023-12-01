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

import pytest
import torch
from test_utils import (
    compare_tensors,
    is_pytest_mode_compile,
)

@pytest.mark.parametrize("start", [0.1664, 0.6964, 4.124])
@pytest.mark.parametrize("end", [1.2032, 2.0438, 2.5345])
@pytest.mark.parametrize("steps", [0, 1, 6, 13])
@pytest.mark.parametrize("op_attr", [(False, 0), (True, 1), (True, 2), (True, 10)])
def test_hpu_linspace_logspace(start, end, steps, op_attr):
    def fn(is_logspace):
        def linspace(start, end, steps, device='cpu'):
            return torch.linspace(start, end, steps, device=device)

        def logspace(start, end, steps, base, device='cpu'):
            return torch.logspace(start, end, steps, base, device=device)

        if is_logspace:
            return logspace
        else:
            return linspace

    is_logspace, base = op_attr
    ref_fn = fn(is_logspace)
    hpu_fn = fn(is_logspace)
    args=[start, end, steps]

    if is_logspace:
        args.append(base)

    expected_result = ref_fn(*args)

    if is_pytest_mode_compile():
        torch._dynamo.reset()
        hpu_fn = torch.compile(hpu_fn, backend="aot_hpu_training_backend")

    real_result = hpu_fn(*args, device="hpu")

    compare_tensors([real_result], [expected_result], atol=1e-8, rtol=1e-5)
