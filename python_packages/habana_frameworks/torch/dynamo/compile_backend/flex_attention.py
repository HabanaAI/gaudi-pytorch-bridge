###############################################################################
# Copyright (c) 2021-2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
###############################################################################

import functools
from collections.abc import Callable, Sequence
from typing import (
    Any,
)

from habana_frameworks.torch.hpex.kernels.PySDPA import (
    flex_attention_bwd,
    flex_attention_fwd,
)

import torch
from torch._inductor.fx_passes.post_grad import remove_noop_ops
from torch.fx import replace_pattern

from .decomposition import get_hpu_decompositions


@torch.no_grad()
def trace_on_hpu(
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


def hpu_flex_attention_bwd_pass(graph_module: torch.fx.GraphModule):
    hop_flex_attention_bwds = graph_module.graph.find_nodes(
        op="call_function",
        target=torch.ops.higher_order.flex_attention_backward,
        sort=True,
    )

    # retunr if no flex HOP bwds found
    if len(hop_flex_attention_bwds) == 0:
        return

    for hop_node in hop_flex_attention_bwds:
        qshape = hop_node.args[0].meta["val"].shape
        qstride = hop_node.args[0].meta["tensor_meta"].stride
        kshape = hop_node.args[1].meta["val"].shape
        kstride = hop_node.args[1].meta["tensor_meta"].stride
        vshape = hop_node.args[2].meta["val"].shape
        vstride = hop_node.args[2].meta["tensor_meta"].stride
        outshape = hop_node.args[3].meta["val"].shape
        ostride = hop_node.args[3].meta["tensor_meta"].stride
        logsumexpshape = hop_node.args[4].meta["val"].shape
        grad_outshape = hop_node.args[5].meta["val"].shape
        gradstride = hop_node.args[5].meta["tensor_meta"].stride
        grad_logsumexpshape = hop_node.args[6].meta["val"].shape
        is_noop_mask = False
        # dtype of q, k, v is expected to be same
        dtype = hop_node.args[2].meta["val"].dtype
        dtype_fp32 = torch.float32

        fw_graph0 = hop_node.args[7]  # score_mod fwd for recompute
        joint_graph0 = hop_node.args[8]  # score_mod for bwd
        sdpa_mask = hop_node.args[9][-1]  # mask_mod for bwd
        mask_graph_score_gm = sdpa_mask.graph.owning_module.get_submodule(sdpa_mask.name)

        for node in mask_graph_score_gm.graph.nodes:
            if node.op == "output":
                for n_arg in node.all_input_nodes:
                    if n_arg.target == torch.ops.aten.full.default and len(n_arg.args[0]) == 0:
                        is_noop_mask = True

        # extract block sizes
        block_size = hop_node.args[9][-3:-1][0]
        if block_size == 1 << 30:
            block_size = 256
        from torch._subclasses.fake_tensor import FakeTensorMode

        with FakeTensorMode():
            q_inp = functools.partial(
                torch.empty_strided, qshape, qstride, dtype=dtype, device="hpu", requires_grad=False
            )
            k_inp = functools.partial(
                torch.empty_strided, kshape, kstride, dtype=dtype, device="hpu", requires_grad=False
            )
            v_inp = functools.partial(
                torch.empty_strided, vshape, vstride, dtype=dtype, device="hpu", requires_grad=False
            )
            o_inp = functools.partial(
                torch.empty_strided, outshape, ostride, dtype=dtype, device="hpu", requires_grad=False
            )
            lse_inp = functools.partial(
                torch.empty, logsumexpshape, device="hpu", requires_grad=False, dtype=dtype_fp32
            )
            gout_inp = functools.partial(
                torch.empty_strided, grad_outshape, gradstride, dtype=dtype, device="hpu", requires_grad=False
            )
            glse_inp = functools.partial(
                torch.empty, grad_logsumexpshape, device="hpu", requires_grad=False, dtype=dtype_fp32
            )
            search_gm = trace_on_hpu(
                flex_attention_bwd,
                [
                    q_inp(),
                    k_inp(),
                    v_inp(),
                    o_inp(),
                    lse_inp(),
                    gout_inp(),
                    glse_inp(),
                    block_size,
                    is_noop_mask,
                ],
            )

        # patch fw_graph score_mod fucntion
        fw_graph_score_gm = fw_graph0.graph.owning_module.get_submodule(fw_graph0.name)

        def score_mod_pattern(attn_score, b, h, q_idx, kv_idx):
            return torch.ops.hpu.flex_attention_score_mod.default(attn_score, b, h, q_idx, kv_idx)

        replace_pattern(search_gm, score_mod_pattern, fw_graph_score_gm)

        # patch joint_graph bwd score_mod fucntion
        joint_graph_score_gm = joint_graph0.graph.owning_module.get_submodule(joint_graph0.name)
        out_node = list(joint_graph_score_gm.graph.nodes)[-1]
        out_node_args = out_node.args
        with joint_graph_score_gm.graph.inserting_before(out_node):
            new_node = joint_graph_score_gm.graph.create_node(
                "call_function",
                torch.ops.hpu.flex_attention_pack_tensors.default,
                args=out_node.args,
                kwargs={},
            )
        out_node.args = (new_node,)
        joint_graph_score_gm.recompile()

        def score_mod_pattern(attn_score, b, h, q_idx, kv_idx, grad_out):
            return torch.ops.hpu.flex_attention_bwd_score_mod.default(attn_score, b, h, q_idx, kv_idx, grad_out)

        replace_pattern(search_gm, score_mod_pattern, joint_graph_score_gm)

        flex_pack_tensors_nodes = search_gm.graph.find_nodes(
            op="call_function",
            target=torch.ops.hpu.flex_attention_pack_tensors.default,
            sort=True,
        )
        for pack_nod in flex_pack_tensors_nodes:
            args = pack_nod.args
            if any(isinstance(arg, list) for arg in args):
                list_nodes = pack_nod.args[0]
                pack_nod.replace_all_uses_with(list_nodes[0])
        search_gm.graph.eliminate_dead_code()
        search_gm.recompile()

        # revert arg list
        joint_graph_score_gm = joint_graph0.graph.owning_module.get_submodule(joint_graph0.name)
        out_node = list(joint_graph_score_gm.graph.nodes)[-1]
        out_node.args = out_node_args
        joint_graph_score_gm.graph.eliminate_dead_code()
        joint_graph_score_gm.recompile()

        # patch mask_mod fucntion
        if not is_noop_mask:

            def mask_mod_pattern(b, h, q_idx, kv_idx):
                return torch.ops.hpu.flex_attention_mask_mod.default(b, h, q_idx, kv_idx)

            replace_pattern(search_gm, mask_mod_pattern, mask_graph_score_gm)

        # decompose flex_attention on HPU
        with graph_module.graph.inserting_before(hop_node):
            new_node = graph_module.graph.create_node(
                "call_function",
                torch.ops.hpu.flex_attention_bwd,
                args=(
                    hop_node.args[0],
                    hop_node.args[1],
                    hop_node.args[2],
                    hop_node.args[3],
                    hop_node.args[4],
                    hop_node.args[5],
                    hop_node.args[6],
                    block_size,
                    is_noop_mask,
                ),
                kwargs={},
            )
            hop_node.replace_all_uses_with(new_node, propagate_meta=True)

        graph_module.graph.eliminate_dead_code()
        graph_module.recompile()

        def hpu_flex_attention_bwd(q, k, v, o, lse, dout, glse, block_size, is_apply_mask):
            ret = torch.ops.hpu.flex_attention_bwd(q, k, v, o, lse, dout, glse, block_size, is_apply_mask)
            return ret

        replace_pattern(graph_module, hpu_flex_attention_bwd, search_gm.graph)

        graph_module.graph.eliminate_dead_code()
        graph_module.recompile()

        flex_pack_tensors_node = graph_module.graph.find_nodes(
            op="call_function",
            target=torch.ops.hpu.flex_attention_pack_tensors.default,
            sort=True,
        )

        replace_nodes = {}
        for get_item in flex_pack_tensors_node[0].users:
            if len(get_item.args) >= 2:
                input_node_idx = get_item.args[1]
                replace_nodes[get_item] = flex_pack_tensors_node[0].args[input_node_idx]
                arg_node = flex_pack_tensors_node[0].args[input_node_idx]
                with graph_module.graph.inserting_before(flex_pack_tensors_node[0]):
                    get_item.replace_all_uses_with(arg_node)

        graph_module.graph.eliminate_dead_code()
        graph_module.recompile()


def hpu_flex_attention_passes(
    graph_module: torch.fx.GraphModule,
    example_inputs: list[torch.Tensor],
    is_training: bool,
    is_backward: bool,
):
    if is_backward:
        hpu_flex_attention_bwd_pass(graph_module)
        return

    hop_flex_attention_fwds = graph_module.graph.find_nodes(
        op="call_function", target=torch.ops.higher_order.flex_attention, sort=True
    )

    # retunr if no flex HOP fwds found
    if len(hop_flex_attention_fwds) == 0:
        return

    for hop_node in hop_flex_attention_fwds:
        qshape = hop_node.args[0].meta["val"].shape
        qstride = hop_node.args[0].meta["tensor_meta"].stride
        kshape = hop_node.args[1].meta["val"].shape
        kstride = hop_node.args[1].meta["tensor_meta"].stride
        vshape = hop_node.args[2].meta["val"].shape
        vstride = hop_node.args[2].meta["tensor_meta"].stride
        # dtype of q, k, v is expected to be same
        dtype = hop_node.args[2].meta["val"].dtype

        sdpa_score = hop_node.args[3]
        sdpa_mask = hop_node.args[4][-1]
        is_noop_mask = False
        is_ret_lse = False

        if len(hop_node.users) == 2:
            is_ret_lse = True

        sdpa_mask_gm = sdpa_mask.graph.owning_module.get_submodule(sdpa_mask.name)
        for node in sdpa_mask_gm.graph.nodes:
            if node.op == "output":
                for n_arg in node.all_input_nodes:
                    if n_arg.target == torch.ops.aten.full.default and len(n_arg.args[0]) == 0:
                        is_noop_mask = True

        # extract block sizes
        block_size = hop_node.args[4][-3:-1][0]
        if block_size == 1 << 30:
            block_size = 256

        q_inp = functools.partial(torch.empty_strided, qshape, qstride, dtype=dtype, device="hpu", requires_grad=False)
        k_inp = functools.partial(torch.empty_strided, kshape, kstride, dtype=dtype, device="hpu", requires_grad=False)
        v_inp = functools.partial(torch.empty_strided, vshape, vstride, dtype=dtype, device="hpu", requires_grad=False)
        search_gm = trace_on_hpu(
            flex_attention_fwd,
            [q_inp(), k_inp(), v_inp(), block_size, is_noop_mask, is_ret_lse],
        )

        # patch score_mod fucntion
        sdpa_score_gm = sdpa_score.graph.owning_module.get_submodule(sdpa_score.name)

        def score_mod_pattern(attn_score, b, h, q_idx, kv_idx):
            return torch.ops.hpu.flex_attention_score_mod.default(attn_score, b, h, q_idx, kv_idx)

        replace_pattern(search_gm, score_mod_pattern, sdpa_score_gm)

        # patch mask_mod fucntion
        if not is_noop_mask:
            sdpa_mask_gm = sdpa_mask.graph.owning_module.get_submodule(sdpa_mask.name)

            def mask_mod_pattern(b, h, q_idx, kv_idx):
                return torch.ops.hpu.flex_attention_mask_mod.default(b, h, q_idx, kv_idx)

            replace_pattern(search_gm, mask_mod_pattern, sdpa_mask_gm)

        # decompose flex_attention on HPU
        hpu_flex_attention_op = torch.ops.hpu.flex_attention_fwd

        with graph_module.graph.inserting_before(hop_node):
            new_node = graph_module.graph.create_node(
                "call_function",
                hpu_flex_attention_op,
                args=(
                    hop_node.args[0],
                    hop_node.args[1],
                    hop_node.args[2],
                    block_size,
                    is_noop_mask,
                    is_ret_lse,
                ),
                kwargs={},
            )
            hop_node.replace_all_uses_with(new_node, propagate_meta=True)

        graph_module.graph.eliminate_dead_code()
        graph_module.recompile()

        def hpu_flex_attention_fwd(q, k, v, block_size, is_apply_mask, is_ret_lse):
            ret = torch.ops.hpu.flex_attention_fwd(q, k, v, block_size, is_apply_mask, is_ret_lse)
            return ret

        replace_pattern(graph_module, hpu_flex_attention_fwd, search_gm.graph)

        graph_module.graph.eliminate_dead_code()
        graph_module.recompile()

        flex_pack_tensors_node = graph_module.graph.find_nodes(
            op="call_function",
            target=torch.ops.hpu.flex_attention_pack_tensors.default,
            sort=True,
        )

        replace_nodes = {}
        for get_item in flex_pack_tensors_node[0].users:
            if len(get_item.args) >= 2:
                input_node_idx = get_item.args[1]
                replace_nodes[get_item] = flex_pack_tensors_node[0].args[input_node_idx]
                arg_node = flex_pack_tensors_node[0].args[input_node_idx]
                with graph_module.graph.inserting_before(flex_pack_tensors_node[0]):
                    get_item.replace_all_uses_with(arg_node)

        graph_module.graph.eliminate_dead_code()
        graph_module.recompile()
