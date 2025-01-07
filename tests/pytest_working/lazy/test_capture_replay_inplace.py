###############################################################################
#
#  Copyright (c) 2021-2024 Intel Corporation
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
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
