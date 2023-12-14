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

@torch.compile(backend="hpu_backend")
def fn(x):
    return x + x

def test_hpu_backend():
  x = torch.tensor(2.).to("hpu")
  y = fn(x)
  expected = torch.tensor(4.)
  assert torch.allclose( expected, y.to('cpu') )