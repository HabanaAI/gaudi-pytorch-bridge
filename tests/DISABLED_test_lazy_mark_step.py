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
import torch.nn as nn
from test_utils import compare_tensors


class model(nn.Module):
    def __init__(self):
        super(model, self).__init__()

    def forward(self, x, y):
        z = torch.add(x, y)
        a = torch.relu(z)
        return a


def test_lazy_mark_step_basic():
    hpu = torch.device("hpu")
    cpu = torch.device("cpu")
    m = model()

    a = torch.randn(2, 2)
    b = torch.randn(2, 2)
    out_cpu = m(a, b)

    a_hpu = a.to(hpu)
    b_hpu = b.to(hpu)
    d_hpu = m(a_hpu, b_hpu)
    htcore.mark_step()
    d_hpu = m(a_hpu, b_hpu)
    htcore.mark_step()
    out_hpu = d_hpu.to(cpu)
    compare_tensors(out_hpu, out_cpu, atol=0, rtol=0)


if __name__ == "__main__":
    test_lazy_mark_step_basic()
