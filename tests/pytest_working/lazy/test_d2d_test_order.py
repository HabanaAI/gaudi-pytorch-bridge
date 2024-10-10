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

import habana_frameworks.torch.core as htcore
import torch


def test_d2d_order():
    device = "hpu"
    weight = torch.tensor([3.0], device=device)
    t = torch.tensor([1.0], device=device)
    temp = torch.tensor([4.0], device=device)

    htcore.mark_step()

    t1 = torch.add(weight, t)  # t1 = weight + 1 = 3 + 1 = 4
    weight.copy_(temp)  # weight = 4

    htcore.mark_step()
    print(weight)
    print(t1)
    assert weight[0] == 4.0, "Data mismatch in weight"
    assert t1[0] == 4.0, "Data mismatch in t1 (result of add)"
