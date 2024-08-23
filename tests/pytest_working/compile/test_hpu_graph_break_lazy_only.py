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


@htorch.utils.internal.lazy_only
def execution(y):
    out = torch.abs(t)
    return out


def mymodel(t):
    execution()
    out = torch.abs(t)
    return out


def test_graph_break_lazy_only():
    t = torch.tensor([-1], device="hpu")
    htorch.utils.internal.is_lazy()
    mycompiledmodel = torch.compile(mymodel, backend="hpu_backend", fullgraph=True)
    computed_result = mycompiledmodel(t).to("cpu")
    expected_result = torch.tensor([1])
    assert torch.equal(computed_result, expected_result)
