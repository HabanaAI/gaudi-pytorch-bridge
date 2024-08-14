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
from habana_frameworks.torch.utils.debug.dynamo_utils import FxGraphAnalyzer
from test_utils import fga_assert_helper


@torch.compile(backend="hpu_backend")
def scalar_fn(x, y):
    x = x + x
    y = y * y
    res = y * x.item()
    res = res * x
    return res


def test_capture_scalar_outputs():
    with FxGraphAnalyzer(capture_non_hpu_output=True) as fga, torch._dynamo.config.patch(capture_scalar_outputs=True):
        input1 = torch.tensor([3], device="hpu")
        input2 = torch.tensor([2, 2], device="hpu")
        res = scalar_fn(input1, input2)
        fga_assert_helper(fga.get_ops_summary(), "torch.ops.aten._local_scalar_dense.default", [{0, 1}])
