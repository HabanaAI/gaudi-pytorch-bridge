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
import pytest
import torch
from habana_frameworks.torch.hpex.kernels.Softmax import triu_masked_softmax

device = torch.device("hpu")


@pytest.mark.xfail(reason="synNodeCreateWithId failed for node: ragged_softmax_fwd__f32")
def test_ragged_softmax():
    # Reference calculations
    a = torch.randn((4, 4), requires_grad=True)
    b = torch.tensor([[1, 0, 0, 0], [1, 1, 0, 0], [1, 1, 1, 0], [1, 1, 1, 1]], dtype=torch.int32)
    ref = torch.softmax(a.masked_fill(torch.logical_not(b.to(torch.bool)), -10000), dim=-1)
    ref.backward(torch.eye(4))

    # Masking + Softmax using ragged softmax
    a2 = torch.clone(a).detach().to(device)
    a2.requires_grad = True
    out = triu_masked_softmax(a2)
    out.backward(torch.eye(4).to(device))

    assert np.allclose(out.cpu().detach().numpy(), ref.detach().numpy(), atol=1e-05)
    assert np.allclose(a2.grad.cpu().numpy(), a.grad.numpy(), atol=1e-05)


if __name__ == "__main__":
    test_ragged_softmax()
