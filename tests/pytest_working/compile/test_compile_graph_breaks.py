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
import pytest
import torch


# Test pass if there is no graph break (fullgraph=True casue an exception when there is graph break) and
# When result of execution does not match expected result. Graph break was due to reoccuring execution of getnev() which is now executed once
def test_no_graph_break_is_lazy():
    def mymodel(t):
        htorch.utils.internal.is_lazy()
        out = torch.abs(t)
        return out

    htorch.utils.internal.is_lazy()
    t = torch.tensor([-1], device="hpu")
    mycompiledmodel = torch.compile(mymodel, backend="hpu_backend", fullgraph=True)
    computed_result = mycompiledmodel(t).to("cpu")
    expected_result = torch.tensor([1])
    assert torch.equal(computed_result, expected_result)


def test_no_graph_break_lazy_only():
    @htorch.utils.internal.lazy_only
    def execution(y):
        out = torch.abs(t)
        return out

    def mymodel(t):
        execution()
        out = torch.abs(t)
        return out

    t = torch.tensor([-1], device="hpu")
    htorch.utils.internal.is_lazy()
    mycompiledmodel = torch.compile(mymodel, backend="hpu_backend", fullgraph=True)
    computed_result = mycompiledmodel(t).to("cpu")
    expected_result = torch.tensor([1])
    assert torch.equal(computed_result, expected_result)


# Graph break from SW-199713 still happens, but device_count result is cached, so func call is faster.
@pytest.mark.skip(reason="SW-199713")
def test_no_graph_break_device_count():
    def mymodel(t):
        if htorch.hpu.device_count() > 0:
            out = torch.abs(t)
        else:
            out = t
        return out

    htorch.hpu.device_count()
    t = torch.tensor([-1], device="hpu")
    mycompiledmodel = torch.compile(mymodel, backend="hpu_backend", fullgraph=True)
    computed_result = mycompiledmodel(t).to("cpu")
    expected_result = torch.tensor([1])
    assert torch.equal(computed_result, expected_result)
