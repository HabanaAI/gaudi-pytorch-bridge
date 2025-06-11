###############################################################################
#
#  Copyright (c) 2021-2025 Intel Corporation
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

import torch


def test_hpu_double_backward():
    def fn(t1):
        t1 = t1 * 1
        t1.backward(retain_graph=True)
        return t1

    t1 = torch.ones([], requires_grad=True, device="hpu")
    compiled_fn = torch.compile(fn, backend="hpu_backend")
    result = compiled_fn(t1)

    with torch._dynamo.compiled_autograd._enable(torch.compile(backend="hpu_backend")):
        result.backward()

    assert result is not None and t1.grad is not None
