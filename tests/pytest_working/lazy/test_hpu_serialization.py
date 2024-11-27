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
