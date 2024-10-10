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
from test_utils import compare_tensors


def test_linear_2d():
    """
    m = nn.Linear(in_features, out_features).to('hpu')
    input = torch.randn(seqlen, bsz, s1, in_features).to('hpu')
    output = m(input)
    print('..............................output size = ',output.size())
    """
    out_features = 16
    in_features = 8
    bsz = 4

    residual = torch.randn((bsz, out_features), requires_grad=True).to(torch.float)
    rh = residual.detach().to("hpu")
    a = torch.randn((bsz, in_features), requires_grad=True).to(torch.float)
    ah = a.detach().to("hpu")
    ah.requires_grad = True
    b = torch.randn((out_features, in_features), requires_grad=True).to(torch.float)
    bh = b.detach().to("hpu")
    bh.requires_grad = True
    c = torch.randn((out_features), requires_grad=True).to(torch.float)
    ch = c.detach().to("hpu")
    ch.requires_grad = True
    dh = torch.nn.functional.linear(ah, bh, ch)
    # print(dh.to('cpu'))
    print(dh.shape)
    res = rh + dh
    # print(res.to('cpu'))
    # print(res.shape)
    d = torch.nn.functional.linear(a, b, c)
    comp_res = compare_tensors(dh.to("cpu"), d, rtol=1e-3, atol=1e-3)
    print("cpu and hpu result match = ", comp_res)
    lcpu = d.sum()
    print(lcpu)
    lcpu.backward()
    lhpu = dh.sum()
    print(lhpu, dh.requires_grad)
    lhpu.backward()
    # print(ah.grad)
    bwdres1 = compare_tensors(ah.grad.to("cpu"), a.grad, rtol=1e-3, atol=1e-3)
    print("t1: cpu and hpu result match = ", bwdres1)
    bwdres2 = compare_tensors(bh.grad.to("cpu"), b.grad, rtol=1e-3, atol=1e-3)
    print("t2: cpu and hpu result match = ", bwdres2)
    print("b = ", b.grad)
    print("bh = ", bh.grad.to("cpu"))
    bwdres3 = compare_tensors(ch.grad.to("cpu"), c.grad, rtol=1e-3, atol=1e-3)
    print("t3: cpu and hpu result match = ", bwdres3)
    print("c = ", c.grad)
    print("ch = ", ch.grad.to("cpu"))
