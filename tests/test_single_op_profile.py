###########################################################################
# Copyright (C) 2020 HabanaLabs, Ltd.
# All Rights Reserved.
#
# Unauthorized copying of this file, via any medium is strictly prohibited.
# Proprietary and confidential.
###########################################################################

import os
import torch
import numpy as np
import pytest
from test_utils import compare_tensors
import habana_frameworks.torch.core as htcore


def test_hpu(t1_shape, t2_shape):
    t1 = torch.randint(0, 2, t1_shape)
    t1_hpu = t1.to("hpu")
    t2 = torch.ones(t2_shape)
    t2_hpu = t2.to("hpu")
    while 1:
        for k in range(1, 1000):
            t3_hpu = t1_hpu.div(t2_hpu)
            t3_hpu = t3_hpu.div(1.0)
            t3_hpu = t3_hpu.div(2.0)
        htcore.mark_step()


if __name__ == "__main__":
    torch.manual_seed(0)
    test_hpu([1023], [1])
