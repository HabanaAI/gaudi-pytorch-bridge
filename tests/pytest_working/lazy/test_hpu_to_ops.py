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
import torch.nn.functional as F
from test_utils import compare_tensors, evaluate_fwd_bwd_kernel, evaluate_fwd_kernel


def test_to_hpu():
    cpu_tensor = torch.randn(2, 3, 4)
    hpu_tensor = cpu_tensor.to("hpu")
    assert torch.equal(cpu_tensor, hpu_tensor.to("cpu"))


def test_to_hpu0():
    cpu_tensor = torch.randn(2, 3, 4)
    hpu_tensor = cpu_tensor.to("hpu:0")
    assert torch.equal(cpu_tensor, hpu_tensor.to("cpu"))


@pytest.mark.xfail(reason="Guadi doesn't support hpu:X notition")
def test_to_hpux():
    cpu_tensor = torch.randn(2, 3, 4)
    hpu_tensor = cpu_tensor.to("hpu:1")
    assert torch.equal(cpu_tensor, hpu_tensor.to("cpu"))
