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
from test_utils import cpu


@pytest.mark.skip(reason="wrong dimensions")
def test_hpu_cat_permute():
    a = torch.randn(2, 1, 4, 4).to("hpu")
    d = torch.randn(2, 1, 4, 4).to("hpu")
    wt = torch.randn(1, 1, 3, 3).to("hpu")
    b = torch.nn.functional.conv2d(a, wt)
    e = torch.nn.functional.conv2d(d, wt)
    c = torch.cat([b, e], dim=1)
    c.to(cpu)
