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

import numpy as np
import torch


def test_hpu_lazy_slice_fwd_bwd():
    t1 = torch.randn((5, 5), requires_grad=True)

    hpu = torch.device("hpu")

    t1_h = t1.detach().to(hpu)
    t1_h.requires_grad = True
    t1_h.retain_grad()

    out = t1[0:5:2, 0:2]
    out.sum().backward()
    grad_t1_cpu = t1.grad.clone().detach()
    out_h = t1_h[0:5:2, 0:2]
    out_h.sum().backward()
    grad_t1_h = t1_h.grad.cpu()

    assert np.allclose(grad_t1_cpu, grad_t1_h, atol=0, rtol=0), "Data mismatch"
