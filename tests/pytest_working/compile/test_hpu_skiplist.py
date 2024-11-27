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


import habana_frameworks.torch as htorch
import torch
import torch._dynamo.testing


def test_graph_break():
    def fn(rand):
        if htorch.hpu.is_available():
            m = htorch.hpu._utils._get_device_index(htorch.hpu.current_device())
            return m * rand
        else:
            return -1

    cnts = torch._dynamo.testing.CompileCounter()
    ones = torch.ones(1, device="hpu")
    orig_out = fn(ones)
    opt_fn = torch._dynamo.optimize(cnts, nopython=False)(fn)
    optim_out = opt_fn(ones)
    assert cnts.frame_count == 1, "Frame Count not equal to 1, check for graph breaks"
    assert orig_out == optim_out, "Output mismatch"
