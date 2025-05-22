###############################################################################
#
#  Copyright (c) 2025 Intel Corporation
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
import pytest
import torch


@pytest.mark.skip(reason="Uses too much memory for simulator. Run on HW.")
def test_index_put_no_oom():
    def fun():
        x = torch.ones((1000, 1000, 1000, 20), device="hpu", dtype=torch.float32)
        x.index_put_((torch.tensor([0]).to("hpu"), torch.tensor([0]).to("hpu")), torch.zeros((1, 1000, 20)).to("hpu"))
        return x

    x = fun().sum((1, 2)).to(torch.int32).sum().cpu()
    if x < 0:
        x += 2**32

    assert x == ((1000 * 1000 * 1000 * 20) - (1 * 1000 * 20)) % 2**32
