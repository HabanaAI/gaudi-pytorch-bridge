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

# Test code for SW-179625

import habana_frameworks.torch
import torch
import torch.nn as nn


# Model with non-aten op Conv1d (torch.nn)
# Output of this model is passed to flip and topk op
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.score = nn.Conv1d(96, 192, 1)

    def forward(self, x):

        return self.score(x)


from habana_frameworks.torch.hpu import random as hpu_random

seed = 42
torch.manual_seed(seed)
hpu_random.manual_seed_all(seed)


# Initially, flip op gave wrong output in this setup
# as the dimension of input tensor was in TPC order
# and flip op attempted to convert it to TPC order
# assuming CPU order of dimensions.
# This issue was not observed if the non-aten op in Model
# was replaced by an aten op.
def test_flip_op():
    model = Model().to("hpu")
    x = torch.randn([1, 96, 198]).to("hpu")
    logits = model(x)
    for i in [-3, -2, -1, 0, 1, 2]:
        output_hpu = torch.flip(logits, [i])
        output_cpu = torch.flip(logits.cpu(), [i])
        assert torch.allclose(output_hpu.to("cpu"), output_cpu, atol=0.001, rtol=0.001)


# Initially, TopK op gave runtime error due to the same issue of
# unnecessary TPC order conversion.
def test_topk_op():
    model = Model().to("hpu")
    x = torch.randn([1, 96, 198]).to("hpu")
    logits = model(x)
    # Index -3, 0 results in a runtime error for both CPU and HPU
    # Expected range is [-3, 2]
    for i in [-2, -1, 1, 2]:
        output_hpu = (torch.topk(logits, 4, i))[1]
        output_cpu = (torch.topk(logits.cpu(), 4, i))[1]
        assert torch.allclose(output_hpu.to("cpu"), output_cpu, atol=0.001, rtol=0.001)
