###############################################################################
# Copyright (c) 2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
###############################################################################

import operator

import torch

from habana_frameworks.torch.dynamo.compile_backend._passes.utils import (
    OptimizationPassPlacement,
    OptimizerContext,
)
from habana_frameworks.torch.dynamo.compile_backend.passes import (
    pass_reinplace_triton_gaudi_gdn_decode,
)


def _make_functionalized_gdn_graph(*, extra_state_user: bool):
    graph = torch.fx.Graph()
    state = graph.placeholder("state")
    packed = graph.placeholder("packed")
    gate_a = graph.placeholder("gate_a")
    gate_b = graph.placeholder("gate_b")
    a_log = graph.placeholder("a_log")
    dt_bias = graph.placeholder("dt_bias")
    indices = graph.placeholder("indices")
    gdn_op = torch.ops.triton_gaudi.gdn_decode_packed.default
    functionalized = graph.call_function(
        torch.ops.higher_order.auto_functionalized_v2,
        args=(gdn_op,),
        kwargs={
            "packed_qkv": packed,
            "gate_a": gate_a,
            "gate_b": gate_b,
            "a_log": a_log,
            "dt_bias": dt_bias,
            "state_indices": indices,
            "artifact_hash": "0" * 64,
            "value_tile": 16,
            "_state_cache_base_index": 0,
            "_all_bases": [state],
        },
    )
    output = graph.call_function(operator.getitem, args=(functionalized, 0))
    updated_state = graph.call_function(operator.getitem, args=(functionalized, 1))
    graph.call_function(torch.ops.aten.copy_.default, args=(state, updated_state))
    if extra_state_user:
        alias = graph.call_function(torch.ops.aten.alias.default, args=(state,))
        graph.output((output, alias))
    else:
        graph.output(output)
    return torch.fx.GraphModule({}, graph)


def _make_functionalized_fused_gdn_graph(*, extra_conv_user: bool):
    graph = torch.fx.Graph()
    conv_state = graph.placeholder("conv_state")
    state = graph.placeholder("state")
    packed = graph.placeholder("packed")
    gate_a = graph.placeholder("gate_a")
    gate_b = graph.placeholder("gate_b")
    a_log = graph.placeholder("a_log")
    dt_bias = graph.placeholder("dt_bias")
    indices = graph.placeholder("indices")
    conv_weight_t = graph.placeholder("conv_weight_t")
    gdn_op = torch.ops.triton_gaudi.gdn_decode_conv_packed.default
    functionalized = graph.call_function(
        torch.ops.higher_order.auto_functionalized_v2,
        args=(gdn_op,),
        kwargs={
            "packed_qkv": packed,
            "gate_a": gate_a,
            "gate_b": gate_b,
            "a_log": a_log,
            "dt_bias": dt_bias,
            "state_indices": indices,
            "conv_weight_t": conv_weight_t,
            "artifact_hash": "0" * 64,
            "_conv_state_base_index": 0,
            "_state_cache_base_index": 1,
            "_all_bases": [conv_state, state],
        },
    )
    output = graph.call_function(operator.getitem, args=(functionalized, 0))
    updated_conv = graph.call_function(operator.getitem, args=(functionalized, 1))
    updated_state = graph.call_function(operator.getitem, args=(functionalized, 2))
    graph.call_function(
        torch.ops.aten.copy_.default,
        args=(conv_state, updated_conv),
    )
    graph.call_function(torch.ops.aten.copy_.default, args=(state, updated_state))
    if extra_conv_user:
        alias = graph.call_function(torch.ops.aten.alias.default, args=(conv_state,))
        graph.output((output, alias))
    else:
        graph.output(output)
    return torch.fx.GraphModule({}, graph)


def _context(graph_module):
    return OptimizerContext(
        graph_module,
        "test_triton_gaudi_gdn",
        [],
        False,
        False,
        False,
        OptimizationPassPlacement.PRE_PLACEMENT,
        [],
        [],
    )


def test_reinplace_triton_gaudi_gdn_removes_full_state_copy():
    graph_module = _make_functionalized_gdn_graph(extra_state_user=False)

    assert pass_reinplace_triton_gaudi_gdn_decode(_context(graph_module))

    targets = [node.target for node in graph_module.graph.nodes]
    assert torch.ops.higher_order.auto_functionalized_v2 not in targets
    assert torch.ops.aten.copy_.default not in targets
    assert targets.count(torch.ops.triton_gaudi.gdn_decode_packed.default) == 1


def test_reinplace_triton_gaudi_gdn_fails_closed_with_extra_state_user():
    graph_module = _make_functionalized_gdn_graph(extra_state_user=True)

    assert not pass_reinplace_triton_gaudi_gdn_decode(_context(graph_module))

    targets = [node.target for node in graph_module.graph.nodes]
    assert torch.ops.higher_order.auto_functionalized_v2 in targets
    assert torch.ops.aten.copy_.default in targets


def test_reinplace_fused_triton_gaudi_gdn_removes_both_cache_copies():
    graph_module = _make_functionalized_fused_gdn_graph(
        extra_conv_user=False)

    assert pass_reinplace_triton_gaudi_gdn_decode(_context(graph_module))

    targets = [node.target for node in graph_module.graph.nodes]
    assert torch.ops.higher_order.auto_functionalized_v2 not in targets
    assert torch.ops.aten.copy_.default not in targets
    assert targets.count(
        torch.ops.triton_gaudi.gdn_decode_conv_packed.default) == 1


def test_reinplace_fused_triton_gaudi_gdn_fails_closed_with_extra_cache_user():
    graph_module = _make_functionalized_fused_gdn_graph(extra_conv_user=True)

    assert not pass_reinplace_triton_gaudi_gdn_decode(_context(graph_module))

    targets = [node.target for node in graph_module.graph.nodes]
    assert torch.ops.higher_order.auto_functionalized_v2 in targets
    assert targets.count(torch.ops.aten.copy_.default) == 2
