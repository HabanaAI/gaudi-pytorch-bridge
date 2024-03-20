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
# Test for SW-179074

import pytest
import torch

device = "hpu"
if device == "hpu":
    import habana_frameworks.torch as htorch
    import habana_frameworks.torch.core as htcore


def test_wrong_dimensions_linear():
    with pytest.raises(Exception) as e_info:
        input_tensor = torch.rand(8, 16, 384).to(device)
        # (8x16x384) x (128x64)
        # The innermost dimensions are expected to be same
        # Test will fail if error is not raised
        bad_layer = torch.nn.Linear(in_features=128, out_features=64, bias=True).to(device)
        bad_out = bad_layer(input_tensor)


if device == "hpu":
    htorch.hpu.synchronize()
