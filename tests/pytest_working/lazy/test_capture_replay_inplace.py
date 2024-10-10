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

import habana_frameworks.torch as ht
import numpy as np
import torch
from torch.autograd import Variable


def testCaptureInplay():
    g = ht.hpu.HPUGraph()
    s = ht.hpu.Stream()

    expected_result = torch.full((1, 1), 1)
    x = Variable(torch.full((1, 1), 1, device="hpu"))
    print("expected_result - ", expected_result)
    print(x)
    assert np.allclose(x.detach().to("cpu"), expected_result, atol=0, rtol=0), "Data mismatch"

    with ht.hpu.stream(s):
        g.capture_begin()
        x.data += 1  # x += 1 = 2
        g.capture_end()
    expected_result += 1
    print(x)

    g.replay()  # expected to be x += 1 = 3
    print(x)
    expected_result += 1
    assert np.allclose(x.detach().to("cpu"), expected_result, atol=0, rtol=0), "Data mismatch"


if __name__ == "__main__":
    testCaptureInplay()
