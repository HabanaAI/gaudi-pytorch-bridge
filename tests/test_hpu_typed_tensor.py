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
import pytest
from test_utils import is_gaudi1

@pytest.mark.skipif(is_gaudi1(), reason="G1 unsupported dtype")
def test_hpu_typed_tensors_creation():
    hpu = torch.device("hpu", 0)
    ht, ct = torch.hpu.BFloat16Tensor([2, 2]), torch.BFloat16Tensor([2, 2])
    assert ht.device == hpu
    assert ht.dtype == ct.dtype
    ht, ct = torch.hpu.BoolTensor([2, 2]), torch.BoolTensor([2, 2])
    assert ht.device == hpu
    assert ht.dtype == ct.dtype
    ht, ct = torch.hpu.ByteTensor([2, 2]), torch.ByteTensor([2, 2])
    assert ht.device == hpu
    assert ht.dtype == ct.dtype
    ht, ct = torch.hpu.CharTensor([2, 2]), torch.CharTensor([2, 2])
    assert ht.device == hpu
    assert ht.dtype == ct.dtype
    ht, ct = torch.hpu.DoubleTensor([2, 2]), torch.DoubleTensor([2, 2])
    assert ht.device == hpu
    assert ht.dtype == ct.dtype
    ht, ct = torch.hpu.FloatTensor([2, 2]), torch.FloatTensor([2, 2])
    assert ht.device == hpu
    assert ht.dtype == ct.dtype
    ht, ct = torch.hpu.HalfTensor([2, 2]), torch.HalfTensor([2, 2])
    assert ht.device == hpu
    assert ht.dtype == ct.dtype
    ht, ct = torch.hpu.IntTensor([2, 2]), torch.IntTensor([2, 2])
    assert ht.device == hpu
    assert ht.dtype == ct.dtype
    ht, ct = torch.hpu.LongTensor([2, 2]), torch.LongTensor([2, 2])
    assert ht.device == hpu
    assert ht.dtype == ct.dtype
    ht, ct = torch.hpu.ShortTensor([2, 2]), torch.ShortTensor([2, 2])
    assert ht.device == hpu
    assert ht.dtype == ct.dtype
