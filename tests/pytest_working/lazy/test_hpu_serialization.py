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


def test_save_with_grad():
    device = "hpu"

    weights = torch.ones([10, 10])
    weights = weights.to(device)

    # Now, start to record operations done to weights
    weights.requires_grad_()
    out = weights.pow(2).sum()
    out.backward()

    assert weights.grad is not None, "No grad generated for tensor."

    # save should succeed without exception
    torch.save(weights, "model.th")
