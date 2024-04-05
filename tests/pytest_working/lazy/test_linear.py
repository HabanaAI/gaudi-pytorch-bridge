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
from test_utils import compare_tensors


@pytest.mark.skip(reason="Results mismatch")
def test_linear():
    out_features = 16
    in_features = 8
    s1 = 5
    bsz = 4
    seqlen = 3
    """
    m = nn.Linear(in_features, out_features).to('hpu')
    input = torch.randn(seqlen, bsz, s1, in_features).to('hpu')
    output = m(input)
    print('..............................output size = ',output.size())
    """
    a = torch.randn((seqlen, bsz, s1, in_features), requires_grad=True).to(torch.float)
    ah = a.detach().to("hpu")
    ah.requires_grad = True
    b = torch.randn((out_features, in_features), requires_grad=True).to(torch.float)
    bh = b.detach().to("hpu")
    bh.requires_grad = True
    c = torch.randn((out_features), requires_grad=True).to(torch.float)
    ch = c.detach().to("hpu")
    ch.requires_grad = True
    dh = torch.nn.functional.linear(ah, bh, ch)
    d = torch.nn.functional.linear(a, b, c)
    compare_tensors(dh.to("cpu"), d, rtol=1e-3, atol=1e-3)
    lcpu = d.sum()
    lcpu.backward()
    lhpu = dh.sum()
    lhpu.backward()
    compare_tensors(ah.grad.to("cpu"), a.grad, rtol=1e-3, atol=1e-3)
    compare_tensors(bh.grad.to("cpu"), b.grad, rtol=1e-3, atol=1e-3)
    compare_tensors(ch.grad.to("cpu"), c.grad, rtol=1e-3, atol=1e-3)
