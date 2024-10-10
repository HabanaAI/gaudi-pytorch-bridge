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
import numpy as np
import pytest
import torch
from test_utils import compare_tensors, hpu

shapes = [
    # size1, size2
    ((2), (2)),
    ((3), (3, 2)),
    ((4), (3, 4, 2)),
    ((5), (2, 3, 5, 1)),
    ((6), (2, 3, 2, 6, 2)),
    ((2, 3), (3)),
    ((3, 2, 4), (4)),
    ((2, 3, 1, 5), (5)),
    ((2, 3, 2, 2, 6), (6)),
    ((2, 1, 3, 4, 2), (1, 4, 1, 2, 3)),
    ((2, 3), (3, 4)),
    ((2, 3, 4), (4, 5)),
    ((2, 3, 4), (2, 4, 5)),
    ((2, 3, 4), (4)),
    ((2, 2, 3, 4), (2, 4, 3)),
    ((1, 8, 16), (10, 16, 12)),
    ((1, 10, 8, 16), (2, 10, 16, 12)),
]


@pytest.mark.parametrize("size1, size2", shapes)
def test_hpu_matmul_fwd_bwd(size1, size2):
    t1 = torch.randn(size1, requires_grad=True)
    t2 = torch.randn(size2, requires_grad=True)

    t1_h = t1.to(hpu)
    t1_h.retain_grad()
    t2_h = t2.to(hpu)
    t2_h.retain_grad()

    out = torch.matmul(t1, t2)
    loss = out.sum()
    loss.backward()
    grad_t1_cpu = t1.grad.clone().detach()
    grad_t2_cpu = t2.grad.clone().detach()

    out_h = torch.matmul(t1_h, t2_h)
    loss_h = out_h.sum()
    loss_h.backward()

    grad_t1_h = t1_h.grad.cpu()
    grad_t2_h = t2_h.grad.cpu()

    assert np.allclose(grad_t1_cpu, grad_t1_h, atol=0.001, rtol=1.0e-3)
    assert np.allclose(grad_t2_cpu, grad_t2_h, atol=0.001, rtol=1.0e-3)


@pytest.mark.parametrize("size1, size2", shapes)
def test_hpu_matmul_fwd(size1, size2):
    t1 = torch.randn(size1)
    t2 = torch.randn(size2)
    t1_h = t1.to(hpu)
    t2_h = t2.to(hpu)

    out = torch.matmul(t1, t2)
    out_h = torch.matmul(t1_h, t2_h)

    compare_tensors([out_h], [out], atol=1.0e-3, rtol=1.0e-3, assert_enable=True)
