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
import habana_frameworks.torch.core as htcore
import numpy as np
import pytest
import torch
from test_utils import hpu


@pytest.mark.skip
def testCapture():
    steps = torch.ones(3, 3)
    multiplier = 2  # torch.Tensor([2])

    # after capture, steps = 2
    # in replay, steps *= 2 is done for 20 times
    # this is equivalent to power(steps, 21)
    expected_result = torch.pow(steps * multiplier, 21)
    print("expected_result = ", expected_result)
    steps = steps.to(hpu)
    # multiplier = multiplier.to(hpu)

    htcore.mark_step()
    steps_new = steps
    print(steps_new)

    g = ht.hpu.HPUGraph()
    s = ht.hpu.Stream()

    with ht.hpu.stream(s):
        g.capture_begin()
        print("Here in capture")
        steps_new = steps * multiplier
        g.capture_end()

    print("Capture done")
    print(steps_new)
    num_w_batches = 10
    for i in range(num_w_batches):
        # steps.copy_(steps_new)
        # g.replay()
        # g.replayV2((steps_new, multiplier))
        g.replayV2((steps,), (steps_new,))
        print("replay number: ", i)
        print(steps_new)

    print(steps_new)

    num_w_batches = 20
    for i in range(10, num_w_batches):
        # steps.copy_(steps_new)
        # g.replay()
        # g.replayV2((steps_new, multiplier))
        g.replayV2((steps,), (steps_new,))
        print("replay number: ", i)
        print(steps_new)

    print(steps_new)
    assert np.allclose(steps_new.detach().to("cpu"), expected_result, atol=0, rtol=0), "Data mismatch"
