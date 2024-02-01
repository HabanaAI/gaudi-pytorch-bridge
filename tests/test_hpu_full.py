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

import torch

try:
    import habana_frameworks.torch.core as htcore
except ImportError:
    raise AssertionError("Could Not import habana_frameworks.torch.core")


def test_full_nographinput():
    a = torch.full([2], 0.0)
    a.fill_(1.0)

    # hpu
    ha = torch.full([2], 0.0, device="hpu")
    ha.fill_(1.0)

    hacpu = ha.cpu()
    assert torch.allclose(hacpu, a, atol=0.001, rtol=0.001)
