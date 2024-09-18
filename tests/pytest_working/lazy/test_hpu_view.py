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

import habana_frameworks.torch.core as htcore
import torch
from test_utils import compare_tensors


def test_multilevel_view_dtype():
    a = torch.randn(8)
    b = a.view(torch.bool)

    b_hpu = b.to("hpu")
    c_hpu = b_hpu.view(torch.float)
    d_hpu = c_hpu.view(-1)
    compare_tensors(d_hpu.cpu(), a, 0.001, 0.001)
