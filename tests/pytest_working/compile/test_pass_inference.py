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

import pytest
import torch
import torch.nn.functional as F
from habana_frameworks.torch.utils.debug.dynamo_utils import FxGraphAnalyzer
from test_utils import fga_assert_helper


class MyModule(torch.nn.Module):
    def __init__(self, use_silu=False):
        super().__init__()
        self.param = torch.nn.Parameter(torch.rand(4, 5))
        self.linear = torch.nn.Linear(4, 5)
        self.use_silu = use_silu

    def forward(self, x):
        param = self.param
        add = torch.ops.aten.add.Tensor(x, param.t())
        if self.use_silu:
            return torch.topk(torch.sum(F.silu(self.linear(add)), dim=-1), 3)
        else:
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
    fga_assert_helper(ops_summary=ops_summary, op="torch.ops.aten.linear", count_list=[(1, 0)])

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
    fga_assert_helper(ops_summary=ops_summary, op="torch.ops.aten._to_copy.default", count_list=[(4, 0)])
    fga_assert_helper(ops_summary=ops_summary, op="torch.ops.aten.transpose.int", count_list=[(1, 0)])

    with FxGraphAnalyzer(reset_dynamo=True) as fga:
        out_hpu = func(x=x, m=m, device="hpu", freeze=True)

    ops_summary = fga.get_ops_summary()
    fga_assert_helper(ops_summary=ops_summary, op="torch.ops.aten._to_copy.default", count_list=[(2, 0)])

    out_cpu = func(x=x_c, m=m_c, device="cpu")
    assert torch.allclose(out_cpu[0].float(), out_hpu[0].to(device=torch.device("cpu")), rtol=1e-3, atol=1e-3)


"""
aten.silu.default decomposition is disabled for performance optimization in LLaMA inference
the following test checks if the FX graph contains the silu op if the input graph uses silu
"""


def test_silu():
    torch.manual_seed(123)
    x = torch.randn((5, 4), dtype=torch.float, device=torch.device("cpu"))
    x_c = x.clone().detach()
    m = MyModule(use_silu=True)
    m_c = copy.deepcopy(m)

    with FxGraphAnalyzer(reset_dynamo=False) as fga:
        out_hpu = func(x=x, m=m, device="hpu", freeze=True)

    ops_summary = fga.get_ops_summary()
    fga_assert_helper(ops_summary=ops_summary, op="torch.ops.aten.silu.default", count_list=[(1, 0)])

    out_cpu = func(x=x_c, m=m_c, device="cpu")
    # changed the tolerance value due to some differences seen between CPU and HPU accuracy for silu
    assert torch.allclose(out_cpu[0].float(), out_hpu[0].to(device=torch.device("cpu")), rtol=1e-2, atol=1e-2)


"""
The following test checks if pass_inference_fuse_linear works as expected and generates the correct result on HPU
When transpose is the first argument mm/addmm, then fuse should not be applied
"""


class TransposeMatmulModel(torch.nn.Module):
    def __init__(self, fuse_linear=True):
        super().__init__()
        self.param = torch.nn.Parameter(torch.rand(5, 4))
        self.fuse_linear = fuse_linear

    def forward(self, x):
        x = torch.ops.aten.mul(x, 2.0)
        param = torch.ops.aten.mul(self.param, 2.0)
        if self.fuse_linear:
            return torch.ops.aten.matmul.default(x, param.t())
        else:
            return torch.ops.aten.matmul.default(param.t(), x)


@pytest.mark.parametrize("fuse_linear", [False, True])
def test_pass_inference_fuse_linear(fuse_linear):
    torch.manual_seed(123)
    x = torch.randn((5, 4), dtype=torch.float, device=torch.device("cpu"))
    x_c = x.clone().detach()
    m = TransposeMatmulModel(fuse_linear)
    m_c = copy.deepcopy(m)
    with FxGraphAnalyzer(reset_dynamo=False) as fga:
        out_hpu = func(x=x, m=m, device="hpu")

    ops_summary = fga.get_ops_summary()
    op_name = "torch.ops.aten.linear" if fuse_linear is True else "torch.ops.aten.mm.default"
    fga_assert_helper(ops_summary=ops_summary, op=op_name, count_list=[(1, 0)])

    out_cpu = func(x=x_c, m=m_c, device="cpu")
    assert torch.allclose(out_cpu[0].float(), out_hpu[0].to(device=torch.device("cpu")).float(), rtol=1e-3, atol=1e-3)
