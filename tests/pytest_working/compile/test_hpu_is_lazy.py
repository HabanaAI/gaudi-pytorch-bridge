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
import torch


def mymodel(t):
    htorch.utils.internal.is_lazy()
    out = torch.abs(t)
    return out


# Test pass if there is no graph break (fullgraph=True casue an exception when there is graph break) and
# When result of execution does not match expected result. Graph break was due to reoccuring execution of getnev() which is now executed once
def test_graph_break():
    htorch.utils.internal.is_lazy()
    t = torch.tensor([-1], device="hpu")
    mycompiledmodel = torch.compile(mymodel, backend="hpu_backend", fullgraph=True)
    computed_result = mycompiledmodel(t).to("cpu")
    expected_result = torch.tensor([1])
    assert torch.equal(computed_result, expected_result)
