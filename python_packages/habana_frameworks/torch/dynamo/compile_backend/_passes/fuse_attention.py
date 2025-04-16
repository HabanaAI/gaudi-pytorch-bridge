###############################################################################
#
#  Copyright (c) 2025 Intel Corporation
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
# mypy: allow-untyped-defs
import functools
import logging
from collections.abc import Callable, Sequence
from typing import (
    Any,
)

import torch
from torch._dynamo.utils import counters
from torch._functorch.aot_autograd import aot_function, make_boxed_func
from torch._functorch.partitioners import default_partition
from torch._inductor.compile_fx import _get_subgraph_names
from torch._inductor.fx_passes.joint_graph import constant_fold_uniform_value
from torch._inductor.fx_passes.post_grad import remove_noop_ops
from torch._inductor.pattern_matcher import (
    PatternMatcherPass,
    gen_register_replacement,
    stable_topological_sort,
)
from torch.fx import Transformer

from ..decomposition import get_hpu_decompositions

log = logging.getLogger(__name__)
aten = torch.ops.aten

patterns = PatternMatcherPass()
pass_patterns = [
    patterns,
    PatternMatcherPass(),
]


def clone_graph(input_graph: torch.fx.GraphModule) -> torch.fx.GraphModule:
    class CopyGraph(Transformer):
        def run_node(self, old_node: torch.fx.Node) -> torch.fx.Node:
            new_node = super().run_node(old_node)
            if isinstance(new_node, torch.fx.Proxy):
                new_node.node.meta.update(old_node.meta)
                new_node.node.name = self.new_graph._graph_namespace.create_name(old_node.name, None)
            return new_node

    return CopyGraph(input_graph).transform()


@torch.no_grad()
def fwd_only_hpu(
    fn: Callable[..., Any],
    args: Sequence[Any],
    *,
    run_functional_passes: bool = True,
    get_decomp_fn: Callable[..., Any] | None = None,
) -> torch.fx.GraphModule:
    """Build a normalized inference graph, for use with fx_to_pattern"""
    from torch._dispatch.python import enable_python_dispatcher
    from torch.fx.experimental.proxy_tensor import make_fx

    with enable_python_dispatcher():
        decompositions = get_hpu_decompositions()
        gm = make_fx(fn, decompositions, tracing_mode="real")(*args)

    if run_functional_passes:
        remove_noop_ops(gm.graph)
        gm.graph.eliminate_dead_code()

    gm.recompile()
    return gm


@torch.enable_grad()
def joint_fwd_bwd_hpu(fn: Callable[..., Any], args: Sequence[Any], **kwargs) -> torch.fx.GraphModule:
    """Build a normalized training graph, for use with fx_to_pattern"""
    gm: torch.fx.GraphModule | None = None
    from habana_frameworks.torch.dynamo.compile_backend import (
        config as hpu_backend_config,
    )
    from habana_frameworks.torch.dynamo.compile_backend.decomposition import (
        get_hpu_decompositions,
    )
    from habana_frameworks.torch.dynamo.compile_backend.partition_fn import (
        remove_unnecessary_clone,
    )

    def record_joint_graph(
        joint_graph: torch.fx.GraphModule, inputs: Sequence[Any], **kwargs: Any
    ) -> tuple[torch.fx.GraphModule, torch.fx.GraphModule]:
        nonlocal gm
        assert not gm
        gm = clone_graph(joint_graph)
        if hpu_backend_config.remove_unnecessary_clones:
            joint_graph = remove_unnecessary_clone(joint_graph)
        return default_partition(joint_graph, inputs, **kwargs)

    with torch._guards.tracing(None):
        aot_function(
            fn,
            lambda g, i: make_boxed_func(g),
            partition_fn=record_joint_graph,
            decompositions=get_hpu_decompositions(),
            keep_inference_input_mutations=hpu_backend_config.keep_input_mutations,
            enable_log=False,
        )(*args)
    assert gm

    gm.graph._codegen = torch.fx.graph.CodeGen()
    gm.graph.eliminate_dead_code()
    gm.recompile()

    remove_noop_ops(gm.graph)
    gm.recompile()
    return gm


def add_permute_transpose_clone(gm: torch.fx.GraphModule):
    # we have observed that if there is any view op which come from
    # either transpose or permute op , it is throwing the error becasue
    # permute/transpose create non contiguous tensor and view worked
    # on contiguous tensor only.
    # before:
    # permute/transpose->view->....
    # after :
    # permute/transpose->clone->view->....
    to_remove: list[torch.fx.Node] = []
    for node in gm.graph.nodes:
        if node.op == "call_function" and (
            node.target == torch.ops.aten.permute.default or node.target == torch.ops.aten.transpose.int
        ):
            for user in list(node.users.keys()):
                if user.target == torch.ops.aten.view.default:
                    new_op1 = torch.ops.aten.clone.default
                    new_op2 = torch.ops.aten.view.default
                    inp1, shape = list(user.args)
                    with gm.graph.inserting_before(user):
                        clone_node = gm.graph.call_function(
                            new_op1,
                            (inp1,),
                            kwargs={"memory_format": torch.contiguous_format},
                        )
                        view_new_node = gm.graph.call_function(
                            new_op2,
                            (
                                clone_node,
                                shape,
                            ),
                            {},
                        )
                        user.replace_all_uses_with(view_new_node, propagate_meta=True)
                        to_remove.append(user)
    for u in to_remove:
        gm.graph.erase_node(u)


def _sfdp_pattern_bert_large(query, key, value, attn_mask, inv_scale, dropout_p):
    # for BertLarge with dropout
    query = query.permute([0, 2, 1, 3]).contiguous()
    key = key.permute([0, 2, 1, 3]).contiguous()
    value = value.permute([0, 2, 1, 3]).contiguous()
    attention_scores = torch.matmul(query, key.transpose(-1, -2))
    attention_scores = attention_scores.to(torch.float32)
    attention_scores = attention_scores / (inv_scale)
    attention_scores = attention_scores + attn_mask
    attention_scores = attention_scores.to(torch.bfloat16)
    attention_probs = torch.nn.functional.softmax(attention_scores, dim=-1)
    attention_probs = torch.nn.functional.dropout(attention_probs, p=dropout_p, training=True).to(query.dtype)
    out = torch.matmul(attention_probs, value)
    return out


def _sfdp_replacement_bert_large(query, key, value, attn_mask, inv_scale, dropout_p):
    counters["inductor"]["fuse_attention"] += 1
    return torch.ops.aten.scaled_dot_product_attention(
        query.transpose(1, 2),
        key.transpose(1, 2),
        value.transpose(1, 2),
        attn_mask=attn_mask.to(dtype=query.dtype),
        dropout_p=dropout_p,
        is_causal=False,
        scale=1.0 / inv_scale,
    )


# this function is influenced by torch/_inductor/fx_passes/fuse_attention.py
def _get_sfdp_patterns():
    from torch._inductor.fx_passes.fuse_attention import (
        _sfdp_extra_check,
        _sfdp_pattern_1,
        _sfdp_pattern_2,
        _sfdp_pattern_3,
        _sfdp_pattern_4,
        _sfdp_replacement_1,
        _sfdp_replacement_2,
        _sfdp_replacement_3,
        _sfdp_replacement_4,
        partialize_and_update_signature,
    )

    device = "hpu"

    # sizes/values don't actually matter for initial trace
    # once we get a possible match we re-trace with the actual values and verify the match still holds
    g_inp = functools.partial(torch.empty, (2, 4, 8, 16), device=device, requires_grad=True)
    # attn_mask
    m_inp = functools.partial(torch.empty, (2, 1, 1, 4), device=device)
    # inv_scale
    c_inp = functools.partial(torch.tensor, 2.0, device=device)
    # workaround https://github.com/pytorch/pytorch/issues/97894
    # 0.113377 is a "magic" value that lets us recover the lost input arg relationship
    d = {"dropout_p": 0.113377}

    for dtype in [torch.float, torch.bfloat16]:
        g = functools.partial(g_inp, dtype=dtype)
        m = functools.partial(m_inp, dtype=dtype)
        m_float = functools.partial(m_inp, dtype=torch.float)
        c = functools.partial(c_inp, dtype=dtype)
        c_float = functools.partial(c_inp, dtype=torch.float32)
        if dtype == torch.float:
            candidates = [
                (
                    _sfdp_pattern_1,
                    _sfdp_replacement_1,
                    [g(), g(), g(), c()],
                    {},
                    _sfdp_extra_check(aten.div.Tensor),
                ),
                (
                    _sfdp_pattern_2,
                    _sfdp_replacement_2,
                    [g(), g(), g(), c()],
                    {},
                    _sfdp_extra_check(aten.mul.Tensor),
                ),
                (
                    _sfdp_pattern_3,
                    _sfdp_replacement_3,
                    [g(), g(), g(), c()],
                    d,
                    _sfdp_extra_check(aten.div.Tensor),
                ),
                (
                    _sfdp_pattern_4,
                    _sfdp_replacement_4,
                    [g(), g(), g(), c()],
                    d,
                    _sfdp_extra_check(aten.mul.Tensor),
                ),
                (
                    _sfdp_pattern_bert_large,
                    _sfdp_replacement_bert_large,
                    [g(), g(), g(), m(), c()],
                    d,
                    _sfdp_extra_check(aten.div.Tensor, disable_cuda=True),
                ),
            ]
        mask_fp32_patterns = ["bert_large"]
        if dtype == torch.bfloat16:
            # Add inputs of bf16 q/k/v and fp32 mask, for models like bert.
            candidates = [
                (
                    _sfdp_pattern_bert_large,
                    _sfdp_replacement_bert_large,
                    [g(), g(), g(), m_float(), c_float()],
                    d,
                    _sfdp_extra_check(aten.div.Tensor, disable_cuda=True),
                ),
            ]

        for pattern, replacement, args, workaround, extra_check in candidates:
            # when adding a new pattern, re-run the test with flag PYTORCH_GEN_PATTERNS=1
            # so the pattern gets serialized to a python file and does not require tracing at runtime
            # for the second time onward.
            assert isinstance(workaround, dict)
            name = pattern.__name__

            if dtype != torch.float:
                name += "_bfloat16"
                if any(p in name for p in mask_fp32_patterns) and args[3].dtype == torch.float32:
                    name += "_mask_fp32"

            training_name = name + "_training"
            yield training_name, {
                "search_fn": pattern,
                "replace_fn": replacement,
                "example_inputs": args,
                "trace_fn": joint_fwd_bwd_hpu,
                "pass_dicts": patterns,
                "extra_check": extra_check,
                "scalar_workaround": workaround,
            }

            if workaround:
                assert len(workaround) == 1 and "dropout_p" in workaround
                # functools.partial insufficient because we look at signature downstream
                pattern = partialize_and_update_signature(pattern, dropout_p=0.0)
                replacement = partialize_and_update_signature(replacement, dropout_p=0.0)
                workaround = {}

            inference_name = name + "_inference"
            yield inference_name, {
                "search_fn": pattern,
                "replace_fn": replacement,
                "example_inputs": args,
                "trace_fn": fwd_only_hpu,
                "pass_dicts": patterns,
                "extra_check": extra_check,
                "scalar_workaround": workaround,
                # with dropout turned into clone, we end up with a number of
                # semantically identical graphs
                "skip_duplicates": True,
            }


@functools.lru_cache(None)
def _sfdp_init():
    for key, register_replacement_kwargs in _get_sfdp_patterns():
        gen_register_replacement(key, **register_replacement_kwargs)


# this function is influenced by torch/_inductor/fx_passes/joint_graph.py
def hpu_joint_graph_passes(graph: torch.fx.GraphModule):
    """
    Run FX transformations on the joint forwards+backwards graph.
    """
    _sfdp_init()
    count = 0

    remove_noop_ops(graph.graph)
    constant_fold_uniform_value(graph)

    for patterns in pass_patterns:
        count += patterns.apply(graph.graph)  # type: ignore[arg-type]
    add_permute_transpose_clone(graph)

    if count:
        stable_topological_sort(graph.graph)
        graph.graph.lint()
        graph.recompile()
    return graph


def hpu_recursive_joint_graph_passes(gm):
    for subgraph_name in _get_subgraph_names(gm):
        subgraph = getattr(gm, subgraph_name)
        hpu_recursive_joint_graph_passes(subgraph)
    hpu_joint_graph_passes(gm)
