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
import copy

import torch
from habana_frameworks.torch.utils.debug.dynamo_utils import FxGraphAnalyzer
from test_dynamo_utils import assert_helper


class MyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.param = torch.nn.Parameter(torch.rand(4, 5))
        self.linear = torch.nn.Linear(4, 5)

    def forward(self, x):
        param = self.param
        add = torch.ops.aten.add.Tensor(x, param.t())
        return torch.topk(torch.sum(self.linear(add).relu(), dim=-1), 3)


def func(x: torch.Tensor, m: torch.nn.Module, device: str, freeze: bool = False):
    m.eval()
    if device == "hpu":
        if freeze:
            m = torch.compile(m, backend="hpu_backend", options={"use_graph_freezing": True})
        else:
            m = torch.compile(m, backend="hpu_backend")
        m = m.to(torch.device(device))
    else:
        m = torch.compile(m, backend="eager")

    with torch.no_grad(), torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=True):
        x = x.to(device=torch.device(device))
        output = m(x)
        return output


"""
aot_autograd will lower aten.linear to t + mm/addmm
the following test checks if the pass to fuse t + mm/addmm sub-graphs
back to linear is working as expected and generating correct output on HPU
"""


def test_linear():
    torch.manual_seed(123)
    x = torch.randn((5, 4), dtype=torch.float, device=torch.device("cpu"))
    x_c = x.clone().detach()
    m = MyModule()
    m_c = copy.deepcopy(m)

    with FxGraphAnalyzer(reset_dynamo=False) as fga:
        out_hpu = func(x=x, m=m, device="hpu")

    ops_summary = fga.get_ops_summary()
    assert_helper(ops_summary=ops_summary, op="torch.ops.aten.linear", count_list=[(1, 0)])

    out_cpu = func(x=x_c, m=m_c, device="cpu")
    assert torch.allclose(out_cpu[0].float(), out_hpu[0].to(device=torch.device("cpu")), rtol=1e-3, atol=1e-3)


"""
graph freezing when enabled with torch.compile will try to constant fold all
operations done on constant parameters in the FX graph
the following test checks if the freezing pass is eliminating the cast and transpose
operations on the param input to the FX graph
"""


def test_graph_freeze():
    torch.manual_seed(123)
    x = torch.randn((5, 4), dtype=torch.float, device=torch.device("cpu"))
    x_c = x.clone().detach()
    m = MyModule()
    m_c = copy.deepcopy(m)

    with FxGraphAnalyzer(reset_dynamo=False) as fga:
        out_hpu_no_freeze = func(x=x, m=m, device="hpu", freeze=False)

    ops_summary = fga.get_ops_summary()
    assert_helper(ops_summary=ops_summary, op="torch.ops.aten._to_copy.default", count_list=[(4, 0)])
    assert_helper(ops_summary=ops_summary, op="torch.ops.aten.transpose.int", count_list=[(1, 0)])

    with FxGraphAnalyzer(reset_dynamo=False) as fga:
        out_hpu = func(x=x, m=m, device="hpu", freeze=True)

    ops_summary = fga.get_ops_summary()
    assert_helper(ops_summary=ops_summary, op="torch.ops.aten._to_copy.default", count_list=[(2, 0)])

    out_cpu = func(x=x_c, m=m_c, device="cpu")
    assert torch.allclose(out_cpu[0].float(), out_hpu[0].to(device=torch.device("cpu")), rtol=1e-3, atol=1e-3)
