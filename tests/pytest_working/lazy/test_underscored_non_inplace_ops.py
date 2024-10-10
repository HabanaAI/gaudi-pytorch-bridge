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
from test_utils import format_tc


# Test checking if following operation didn't return Graph compile failed
@pytest.mark.parametrize(
    "op, kwargs",
    [
        (torch.Tensor.bernoulli_, {}),
        (torch.Tensor.exponential_, {}),
        (
            torch.Tensor.geometric_,
            {"p": 0.2},
        ),
        (torch.Tensor.log_normal_, {}),
        (torch.Tensor.normal_, {}),
        (torch.Tensor.random_, {}),
        (torch.Tensor.uniform_, {}),
    ],
    ids=format_tc,
)
def test_underscored_non_inplace_op(op, kwargs):
    try:
        x = torch.randn((10)).to("hpu")
        x_clone = x.clone()
        op(x_clone, **kwargs).cpu()
    except RuntimeError:
        assert False, "Test shouldn't throw any exception"
