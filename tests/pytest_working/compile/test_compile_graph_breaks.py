###############################################################################
# Copyright (C) 2024 Habana Labs, Ltd. an Intel Company
# All Rights Reserved.
#
# Unauthorized copying of this file or any element(s) within it, via any medium
# is strictly prohibited.
# This file contains Habana Labs, Ltd. proprietary and confidential information
# and is subject to the confidentiality and license agreements under which it
# was provided.
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
