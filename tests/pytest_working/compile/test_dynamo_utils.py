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
from contextlib import contextmanager

import torch
from habana_frameworks.torch.dynamo.compile_backend.config import configuration_flags
from habana_frameworks.torch.dynamo.compile_backend.shared_layer import hpu_fallback_op_list
from habana_frameworks.torch.utils.debug.dynamo_utils import FxGraphAnalyzer


@contextmanager
def use_eager_fallback():
    original = configuration_flags["use_eager_fallback"]
    configuration_flags["use_eager_fallback"] = True
    try:
        yield
    finally:
        configuration_flags["use_eager_fallback"] = original


@contextmanager
def use_randint_eager_fallback():
    revert = False
    if "randint" not in hpu_fallback_op_list:
        revert = True
        hpu_fallback_op_list.add("randint")
    yield
    if revert:
        hpu_fallback_op_list.remove("randint")


@torch.compile(backend="hpu_backend")
def fn(x, y, device):
    res = x + y
    eager_fallback_res = torch.randint(high=100, size=[1], device=device, dtype=torch.int32)
    return res + eager_fallback_res


@torch.compile(backend="hpu_backend")
def fn2(x, y):
    res = x + y
    res = res * x
    return res * res


def assert_helper(ops_summary, op, count_list):
    assert len(ops_summary) == len(count_list)
    for single_graph_summary, graph_eager_count in zip(ops_summary, count_list):
        if graph_eager_count is None:
            assert op not in single_graph_summary
        else:
            graph_count, eager_count = graph_eager_count
            if graph_count != 0 or eager_count != 0:
                assert op in single_graph_summary
                assert single_graph_summary[op].graph_count == graph_count
                assert single_graph_summary[op].eager_count == eager_count


def test_simple():
    with use_eager_fallback():
        with use_randint_eager_fallback():
            with FxGraphAnalyzer(reset_dynamo=True) as fga:
                t1 = torch.tensor([6], device="hpu")
                t2 = torch.tensor([2], device="hpu")
                fn(t1, t2, "hpu")

    ops_summary = fga.get_ops_summary()
    assert_helper(ops_summary, "torch.ops.aten.randint.low", [(0, 1)])
    assert_helper(ops_summary, "torch.ops.aten.add.Tensor", [(2, 0)])


def test_cpu():
    with FxGraphAnalyzer(reset_dynamo=True) as fga:
        t1 = torch.tensor([6], device="cpu")
        t2 = torch.tensor([2], device="cpu")
        fn(t1, t2, "cpu")

    assert len(fga.get_ops_summary()) == 1
    assert not fga.get_ops_summary()[0]


def test_multiple():
    with use_eager_fallback():
        with FxGraphAnalyzer(reset_dynamo=True) as fga:
            t1 = torch.tensor([6], device="hpu")
            t2 = torch.tensor([2], device="hpu")
            with FxGraphAnalyzer() as fga2:
                fn2(t1, t2)
            with FxGraphAnalyzer() as fga3:
                fn(t1, t2, "hpu")
            fn(t1.to("cpu"), t2.to("cpu"), "cpu")

    ops_summary = fga.get_ops_summary()
    assert_helper(ops_summary, "torch.ops.aten.randint.low", [None, (1, 0), None])
    assert_helper(ops_summary, "torch.ops.aten.add.Tensor", [(1, 0), (2, 0), None])
    assert_helper(ops_summary, "torch.ops.aten.mul.Tensor", [(2, 0), None, None])

    ops_summary2 = fga2.get_ops_summary()
    assert_helper(ops_summary2, "torch.ops.aten.add.Tensor", [(1, 0)])
    assert_helper(ops_summary2, "torch.ops.aten.mul.Tensor", [(2, 0)])

    ops_summary3 = fga3.get_ops_summary()
    assert_helper(ops_summary3, "torch.ops.aten.randint.low", [(1, 0)])
    assert_helper(ops_summary3, "torch.ops.aten.add.Tensor", [(2, 0)])


def test_bulitin():
    @torch.compile(backend="hpu_backend")
    def clone_fn(x):
        return x.add_(x)

    t = torch.tensor([1337], device="hpu")
    with FxGraphAnalyzer(reset_dynamo=True) as fga:
        clone_fn(t)

    ops_summary = fga.get_ops_summary()
    assert_helper(ops_summary, "torch.ops.aten.add.Tensor", [(1, 0)])
