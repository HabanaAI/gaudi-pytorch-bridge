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


def test_foreach_zero_with_view():
    l1_hpu = []
    l1 = []
    t1_hpu = torch.arange(-3, 15).int().to("hpu").as_strided((3, 3), (6, 2))
    t2_hpu = torch.arange(-3, 15).int().to("hpu").as_strided((3, 3), (3, 1))
    t1 = torch.arange(-3, 15).int().as_strided((3, 3), (6, 2))
    t2 = torch.arange(-3, 15).int().as_strided((3, 3), (3, 1))

    l1_hpu.append(t1_hpu)
    l1_hpu.append(t2_hpu)
    l1.append(t1)
    l1.append(t2)
    torch._foreach_zero_(l1_hpu)
    torch._foreach_zero_(l1)
    for i in range(len(l1)):
        assert torch.allclose(l1_hpu[i].cpu(), l1[i], atol=0.01, rtol=0.01)
