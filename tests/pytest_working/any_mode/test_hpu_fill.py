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
from test_utils import compare_tensors, hpu, is_pytest_mode_compile

test_case_list = [
    # N, C, fill_val,
    (2, 10, 2.2),
    (2, 10, 0.0),
]


@pytest.mark.parametrize("N, C, fill_val", test_case_list)
@pytest.mark.parametrize("is_masked", [True, False])
@pytest.mark.parametrize("is_inplace", [True, False])
@pytest.mark.parametrize("fill_with_scalar", [True, False])
def test_hpu_fill(N, C, fill_val, is_masked, is_inplace, fill_with_scalar):
    def fn(is_masked, is_inplace):
        def fill(self, value):
            return torch.fill(self, value)

        def fill_(self, value):
            return self.fill_(value)

        def masked_fill(self, mask, value):
            return self.masked_fill(mask, value)

        def masked_fill_(self, mask, value):
            return self.masked_fill_(mask, value)

        if is_masked:
            return masked_fill_ if is_inplace else masked_fill
        else:
            return fill_ if is_inplace else fill

    cpu_tensor = torch.randn(N, C)
    hpu_tensor = cpu_tensor.to(hpu)
    ref_fn = fn(is_masked, is_inplace)
    hpu_fn = fn(is_masked, is_inplace)
    cpu_args = [cpu_tensor]
    hpu_args = [hpu_tensor]

    if is_masked:
        mask = torch.randn(C) < 0
        cpu_args.append(mask)
        hpu_args.append(mask.to(hpu))

    if fill_with_scalar:
        cpu_args.append(fill_val)
        hpu_args.append(fill_val)
    else:
        fill_tensor = torch.tensor(fill_val)
        cpu_args.append(fill_tensor)
        hpu_args.append(fill_tensor.to(hpu))

    expected_result = ref_fn(*cpu_args)

    if is_pytest_mode_compile():
        torch._dynamo.reset()
        hpu_fn = torch.compile(hpu_fn, backend="hpu_backend")

    real_result = hpu_fn(*hpu_args)

    compare_tensors([real_result], [expected_result], atol=0, rtol=0)
