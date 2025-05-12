###############################################################################
#
#  Copyright (c) 2021-2025 Intel Corporation
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

import copy
import operator
import os
import queue
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import habana_frameworks.torch.internal.bridge_config as bc
import habana_frameworks.torch.utils.experimental as htexp
from habana_frameworks.torch.dynamo.compile_backend import config as hpu_backend_config
from habana_frameworks.torch.dynamo.debug_utils.logger import get_compile_backend_logger
from habana_frameworks.torch.dynamo.debug_utils.visualization.graph_dumping import (
    dump_fx_graph,
)
from habana_frameworks.torch.utils.debug.dynamo_utils import FxGraphAnalyzer
from habana_frameworks.torch.utils.debug.logger import LogLevel
from habana_frameworks.torch.utils.internal import Timer

import torch
import torch.fx
from torch._inductor.pattern_matcher import stable_topological_sort
from torch.fx.experimental.proxy_tensor import py_sym_types
from torch.fx.node import map_arg
from torch.fx.passes.operator_support import OperatorSupport

from ._helpers import (
    TensorInfoPropagation,
    fill_propagated_tensor_metadata_jitfork,
    get_node_args,
    is_module_dynamic,
    is_node_supported,
    is_view_node,
    jit_node_annotation_propagation,
    jit_node_shape_propagation,
    post_pass_finalize,
    propagate_meta,
    remove_duplicated_outputs,
    remove_no_effect_inplace_add,
    wrap_random_ops,
)
from ._passes.batch_as_strided import batch_as_strided, group_batch_as_strided
from ._passes.fuse_allreduce_calls import pass_fuse_collectives
from ._passes.fuse_view_chains import pass_fuse_view_chains
from ._passes.pattern_rewriter import pass_pattern_rewriter
from ._passes.propose_collective_blocks import pass_propose_collective_blocks
from ._passes.reorder_custom_ops import (
    pass_post_reorder_custom_ops,
    pass_reorder_custom_ops,
)
from ._passes.utils import (
    ColorGraph,
    OptimizationPassPlacement,
    OptimizerContext,
    SchedulePolicy,
)
from .cluster_compiler import pass_compile_clusters_jit_fork_version
from .partitioner import HabanaPartitioner
from .random_utils import is_backward_checkpoint_op
from .recipe_compiler import get_callable_recipe
from .shared_layer import is_eager_fallback_required
from .symbolic_execution import SymExprNodeManager

logger = get_compile_backend_logger()

host_call_functions = {"torch.ops.hpu.weight_permutation"}


def get_passes(stage: OptimizationPassPlacement):
    """
    This function returns optimizations passes for specific stage.
    Registering passes is done by just adding them to corresponding case here.
    Be aware that ORDER MATTERS.

    TODO: Maybe add smarter way of registering passes so we could also specify which to run
          for some debug levels? Or to add dependencies between passes instead of order?
          Could be overkill tho.
    """
    if stage == OptimizationPassPlacement.PRE_PLACEMENT:
        return [
            # this pass will flatten nested submodules by inlining
            pass_annotate_nodes_and_inline_submodule,
            # These passes will be ran once, they always get and produce a flat graph without submodules.
            pass_graph_print,
            pass_reorder_custom_ops,
            pass_propose_collective_blocks,
            pass_fuse_collectives,
            pass_allreduce_parents,
            pass_pattern_rewriter,
            pass_fake_propagation,
            pass_reinplace_inplaceable_ops_v2,
            pass_weight_permutation,
            pass_remove_unnecessary_full_copy,
            pass_remove_unnecessary_expand,
            pass_remove_unnecessary_bmm_view,
            pass_wa_mixed_devices,  # This is W/A for Adam having CPU scalar tensors parameters.
            pass_fix_arange_device,
            pass_reinplace_inplaceable_ops,
            pass_mark_collective_input,
            pass_mark_placement,
            pass_graph_print,
        ]
    elif stage == OptimizationPassPlacement.PRE_PARTITIONER:
        passes = [
            # These passes will prepare proper placement for some corner-cases.
            pass_graph_print,
            pass_eagerize_leaf_views,
            pass_reinplace_index_copy_ops,  # we need the placement information in this pass
            pass_reinplace_add_ops,
            pass_handle_negative_dims,
            pass_replace_sym_size,
            pass_inference_fuse_linear,
        ]
        return passes
    elif stage == OptimizationPassPlacement.PARTITIONER:
        return [
            pass_graph_print,
            pass_mark_frozen_params,
            pass_mark_waittensor_downstream_ops,
            # Doing stable topological sort before partitioning
            pass_stable_topological_sort,
            # These passes will prepare proper placement for some corner-cases.
            pass_propose_partitions,
            pass_post_process_partitions,
            pass_merge_paths,
            # This is final pass that creates final submoduled graph.
            pass_fuse_partitions,
            pass_add_fused_op_metadata,
            pass_reorder_collectives,
            pass_post_reorder_custom_ops,
            pass_make_symints_available,
            pass_fuse_view_chains,
            pass_batch_as_strided_groups,
            pass_graph_print,
            pass_wa_fix_output,
        ]
    elif stage == OptimizationPassPlacement.POST_PARTITIONER:
        return [
            # These passes will be ran once, they have to work on graph with submodules.
            pass_graph_print,
            pass_summarize_graph,
            pass_check_eager_fallbacks,
            pass_detect_partition_in_to_out_duplicates,
            pass_detect_reusable_inputs_for_partition,
            pass_compile_clusters,
            pass_make_boxed_graph,
        ]
    else:
        logger.error("unknown optimization stage %s", stage)
        raise


class FusedCollectiveOperatorSupport(OperatorSupport):
    def is_node_supported(self, submodules: Mapping[str, torch.nn.Module], node: torch.fx.Node) -> bool:
        if (
            self.keyword in node.meta
            and node.meta[self.keyword] == self.target_value
            and "partition_assigned" not in node.meta
            and node.meta["placement"] == "hpu_cluster"
        ):
            node.meta["partition_assigned"] = "true"
            return True
        return False


def pass_batch_as_strided_groups(ctx: OptimizerContext):
    graph_module = ctx.graph_module
    current_groups = group_batch_as_strided(graph_module)
    merged_groups = merge_paths(graph_module, current_groups)

    inserted_batch = batch_as_strided(graph_module, merged_groups)

    post_changed = post_process_partitions(
        # not having not mergable when grouping as_strided's
        ctx.graph_module,
        merged_groups,
        [],
    )

    return inserted_batch or post_changed


def pass_allreduce_parents(ctx: OptimizerContext) -> bool:
    # TODO: try to reuse torch.fx.passes.infra.partitioner._DependencyViewer
    if hpu_backend_config.enable_allreduce_graph_split:
        gm = ctx.graph_module
        allreduces = [n for n in gm.graph.nodes if n.name.startswith("all_reduce")]
        for allreduce in allreduces:
            downstream_allreduce_name = allreduce.name
            previous_nodes = allreduce.all_input_nodes
            new_previous_nodes = []
            while len(previous_nodes) > 0:
                for previous_node in previous_nodes:
                    if "downstream_allreduce_name" not in previous_node.meta:
                        previous_node.meta["downstream_allreduce_name"] = downstream_allreduce_name
                        previous_node.parent = downstream_allreduce_name
                        new_previous_nodes.extend(previous_node.all_input_nodes)
                previous_nodes = new_previous_nodes
                new_previous_nodes = []

        return len(allreduces) > 0
    return False


def pass_mark_waittensor_downstream_ops(ctx: OptimizerContext) -> bool:
    if not hpu_backend_config.enable_waittensor_graph_split:
        return False

    gm = ctx.graph_module
    waittensors = [n for n in gm.graph.nodes if n.name.startswith("wait_tensor")]
    for waittensor in waittensors:
        upstream_waittensor_name = waittensor.name
        user_nodes = list(waittensor.users.keys())
        new_user_nodes = []
        while len(user_nodes) > 0:
            for user_node in user_nodes:
                if "upstream_waittensor_name" not in user_node.meta:
                    user_node.meta["upstream_waittensor_name"] = upstream_waittensor_name
                    new_user_nodes.extend(list(user_node.users.keys()))
            user_nodes = new_user_nodes
            new_user_nodes = []

    return len(waittensors) > 0


def pass_reorder_collectives(ctx: OptimizerContext) -> bool:
    """
    This pass aims to move all_reduce nodes as early in graph as possible,
    and wait_tensor nodes as late as possible.

    For all_reduces we traverse through the graph upstream and collect all nodes
    until we reach an input then we ensure all these nodes are just after last input.
    Then we save last processed all_reduce node as a new target to append nodes to
    and repeat the process. For wait_tensor the process is analogous but we traverse downstream.
    """
    if not hpu_backend_config.enable_allreduce_graph_split:
        return False

    # Sometimes (example MLM LLama 3.1 with CAG), incoming graph doesn't have output node at the end.
    # This function requires output node to be present at the end of the graph to work correctly.
    pass_wa_fix_output(ctx)

    graph = ctx.graph_module.graph
    collective_nodes = []
    wait_tensor_nodes = []
    traversed_nodes = []
    col_move_target = None
    wt_move_target = None
    graph_changed = False
    for n in graph.nodes:
        if n.op == "placeholder":
            traversed_nodes.append(n)
            col_move_target = n  # Finding last placeholder node
        if n.op == "output":
            wt_move_target = n
        if n.name.startswith("wait_tensor"):
            wait_tensor_nodes.append(n)
            for arg in n.all_input_nodes:
                collective_nodes.append(arg)

    for col_node in collective_nodes:
        upstream_nodes = col_node.all_input_nodes
        nodes_to_move = [col_node]
        while len(upstream_nodes) > 0:
            new_upstream_nodes = []
            for upstream_node in upstream_nodes:
                if upstream_node in traversed_nodes:
                    continue
                new_upstream_nodes.extend(upstream_node.all_input_nodes)
                nodes_to_move.append(upstream_node)
            upstream_nodes = new_upstream_nodes

        traversed_nodes.extend(nodes_to_move)

        if col_move_target is not None:
            for node in nodes_to_move:
                col_move_target.append(node)

            if len(nodes_to_move) > 0:
                graph_changed = True

        col_move_target = col_node

    traversed_nodes = []
    for wt_node in reversed(wait_tensor_nodes):
        downstream_nodes = list(wt_node.users.keys())

        # if the wait_tensor has no users. move it to after the
        # corresponding collective node
        if len(downstream_nodes) == 0:
            producer = wt_node.all_input_nodes[0]
            producer.append(wt_node)
            graph_changed = True
            continue

        nodes_to_move = [wt_node]
        while len(downstream_nodes) > 0:
            new_downstream_nodes = []
            for downstream_node in downstream_nodes:
                if downstream_node in traversed_nodes:
                    continue
                new_downstream_nodes.extend(list(downstream_node.users.keys()))
                nodes_to_move.append(downstream_node)
            downstream_nodes = new_downstream_nodes

        traversed_nodes.extend(nodes_to_move)

        if wt_move_target is not None:
            for node in nodes_to_move:
                wt_move_target.prepend(node)

            if len(nodes_to_move) > 0:
                graph_changed = True

        wt_move_target = wt_node

    if graph_changed:
        ctx.graph_module.recompile()

    return graph_changed


@dataclass(frozen=True)
class InplaceableOp:
    inplace_op: Callable[..., Any]
    mutated_arg: int
    extra_check: Callable[[torch.fx.Node], bool] = lambda node: True


def _is_cpu_scale_allowed(node: torch.fx.Node, node_arg: torch.fx.Node, h2d_scales_enabled: bool) -> bool:
    # 0d float CPU scales of fp8 ops are left on the CPU device for H2D optimization.
    ops_to_scales_idx = {
        "cast_to_fp8_v2.default": (1, 2),
        "fp8_gemm_v2.default": (6, 8),
        "fp8_sdpa_fwd_dropout.default": (8, 14),
        "fp8_sdpa_fwd_non_dropout.default": (8, 14),
        "fp8_sdpa_recomp_fwd_dropout.default": (9, 15),
        "fp8_sdpa_recomp_fwd_non_dropout.default": (9, 15),
    }

    if (
        h2d_scales_enabled
        and node_arg.meta["output_dtypes"][0] in [torch.float, torch.bfloat16]
        and node_arg.meta["output_shapes"][0] == torch.Size([])
    ):
        idx_range = ops_to_scales_idx.get(node.target.__name__)
        if idx_range is not None:
            return node_arg in node.args[slice(*idx_range)]
    return False


def _check_unsupported_h2d_ops(node: torch.fx.Node):
    ops_not_yet_supported = [
        "conv2d_fp8.default",
        "mixture_of_experts.fp8",
        "mixture_of_experts.fp8_fused_weights",
        "mixture_of_experts.fp8_dynamic",
        "mixture_of_experts.fp8_fused_weights_dynamic",
    ]

    node_name = node.target.__name__
    assert (
        node_name not in ops_not_yet_supported
    ), f"{node_name} doesn't support H2D scales feature yet, but received CPU scales."


def _is_cpu_scalar_copy_required(
    node: torch.fx.Node, arg_idx: int, node_arg: torch.fx.Node, h2d_scales_enabled: bool
) -> bool:
    # This is list of scalar OPs
    scalar_ops = [
        "topk",
        "arange",
        "randperm",
        "select_scatter",
        "slice_scatter",
        "scalar_tensor",
        "logspace",
        "slice_scatter",
        "as_strided",
        "as_strided_scatter",
        "slice",
        "_roi_align_backward",
        "clamp",
        "roi_align",
        "rsub",
    ]
    copy_required = True
    if node.op == "call_function":
        node_target = node.target.__name__.split(".")[0]
        if node.target.__name__.split(".")[-1] == "Scalar":
            copy_required = False
        elif (
            node_arg.type in [int, float]
            and isinstance(node.target._schema, torch.FunctionSchema)
            and node.target._schema.arguments[arg_idx].type.annotation_str in ["number", "int"]
        ) or (node_arg.type in [int, float] and node_target in scalar_ops):
            assert node_arg.meta["output_device"] == torch.device("cpu")
            copy_required = False
        elif _is_cpu_scale_allowed(node, node_arg, h2d_scales_enabled):
            copy_required = False
        else:
            _check_unsupported_h2d_ops(node)
    return copy_required


def _is_cpu_scalar_or_symbolic_scalar(node: torch.fx.Node) -> bool:
    if node.type in [int, float]:
        assert node.meta["output_device"] == torch.device("cpu")
        return True
    else:
        return False


def is_call_function_dynamic(node: torch.fx.Node, dynamic_graph: bool) -> bool:
    """
    This function dynamicity per call_function.
    """

    def check_dynamic_meta(node: torch.fx.Node):
        meta_val = node.meta.get("val", node.meta.get("tensor_meta", None))
        return (isinstance(meta_val, FakeTensor) and meta_val._has_symbolic_sizes_strides) or isinstance(
            meta_val, py_sym_types
        )

    # early exit when the graph module is static, or when static compilation is forced
    if (not dynamic_graph) or (hpu_backend_config.force_static_compile):
        return False

    from torch._subclasses.fake_tensor import FakeTensor
    from torch.fx.experimental.proxy_tensor import py_sym_types

    is_dynamic = False
    if node.op == "call_function":
        is_dynamic = check_dynamic_meta(node)
        if not is_dynamic:
            args = get_node_args(node)
            for input in args:
                is_dynamic = check_dynamic_meta(input)
                if is_dynamic:
                    break

        logger.debug("Node %s dynamicity %s", node.name, is_dynamic)
    return is_dynamic


def get_dynamic_config_value():
    """
    This function return the is_dynamic=True if user configured
    the same while calling torch.compile. Otherwise return is_dynamic=False
    """

    is_dynamic = False
    from torch._dynamo import config

    # TODO: It is a W/A for discovering dynamic models. In final implementation
    # is should read this info from tensors.
    is_dynamic = not config.assume_static_by_default

    return is_dynamic


def is_higher_order_node(node: torch.fx.Node) -> bool:
    """
    nodes that need to be executed eagerly, while subgraph can be compiled
    """
    assert node.op == "call_function"
    supported_higher_order_ops = ["cond", "while_loop"]

    return (
        isinstance(node.target, torch._ops.HigherOrderOperator) and node.target.__name__ in supported_higher_order_ops
    )


def is_constant_for_lift_fresh_copy(node: torch.fx.Node, arg: torch.fx.Node) -> bool:
    return (
        node.op == "call_function"
        and str(node.target) == "aten.lift_fresh_copy.default"
        and arg.op == "get_attr"
        and arg.target.startswith("_tensor_constant")
    )


def is_copy_op(node: torch.fx.Node) -> bool:
    return node.op == "call_function" and node.target.__name__.split(".")[0] in [
        "_to_copy",
        "_foreach_copy",
        "_foreach_copy_",
    ]


def optimize_graph(
    stage: OptimizationPassPlacement,
    graph_module: torch.fx.GraphModule,
    graph_name: str,
    example_inputs: list[torch.Tensor],
    is_training: bool,
    is_backward: bool,
) -> bool:
    """
    This function rans optimizations of specified stage, if anything in the
    graph has changed, it will return True.

    Specific pass can be disabled by providing env in the form of:
    PT_HPU_DISABLE_<pass_name>=True

    For example:
    PT_HPU_DISABLE_pass_eagerize_leaf_views=True
    """
    # In all the three stages of partitioner, dynamicity has to be detected
    # from graph_module.
    is_dynamic = is_module_dynamic(graph_module)

    ctx = OptimizerContext(
        graph_module, graph_name, example_inputs, is_training, is_backward, is_dynamic, stage, None, None, None, None
    )

    from habana_frameworks.torch.utils.debug import _towl_emit_metrics

    def run_passes(ctx: OptimizerContext):
        graph_changed = False
        pass_counter = 0
        ctx.use_jit_fork = bc.get_pt_hpu_use_jit_fork()
        dump_fx_graph(ctx.graph_module, graph_name, stage=stage, pass_counter=pass_counter)
        for optimization_pass in get_passes(stage):
            pass_name = optimization_pass.__name__
            env_name = "PT_HPU_DISABLE_" + pass_name
            if os.getenv(env_name, "").upper() in ["ON", "1", "YES", "TRUE", "Y"]:
                logger.debug("pass %s was disabled by env at stage %s", pass_name, stage)
                continue

            logger.debug("running %s pass at stage %s", pass_name, stage)

            with Timer() as t:
                current_graph_changed = optimization_pass(ctx)

            graph_changed = current_graph_changed or graph_changed
            if current_graph_changed:
                pass_counter = pass_counter + 1
                dump_fx_graph(ctx.graph_module, graph_name, stage, pass_counter, pass_name)

            _towl_emit_metrics(f"[COMPILE] FX pass {pass_name} took (ms): ", t.elapsed * 1000)
            logger.debug(
                "pass %s at stage %s took: %.3f [s]",
                pass_name,
                stage,
                t.elapsed,
            )
        return graph_changed

    def _get_subgraph_names(gm):
        for node in gm.graph.nodes:
            if node.target == torch.ops.higher_order.cond:
                true_subgraph_name = node.args[1].name
                false_subgraph_name = node.args[2].name
                yield true_subgraph_name
                yield false_subgraph_name

    def recursive_run_passes(ctx, graph_changed, module_prefix=""):
        for submodule_name in _get_subgraph_names(ctx.graph_module):
            submodule = getattr(ctx.graph_module, submodule_name)

            submodule_inputs = [node.meta.get("val") for node in submodule.graph.nodes if node.op == "placeholder"]

            assert None not in submodule_inputs, "Metadata for one of subgraph inputs is not set"

            # create new ctx for submodule
            # outer-most graph module is dynamic while sub module is static?
            sub_ctx = OptimizerContext(
                submodule,
                submodule_name,
                submodule_inputs,
                ctx.is_training,
                ctx.is_backward,
                ctx.is_dynamic,
                ctx.stage,
                None,
                None,
                None,
                True,
            )

            submodule_qualified_name = submodule_name if module_prefix == "" else (module_prefix + "." + submodule_name)
            graph_changed = recursive_run_passes(sub_ctx, graph_changed, submodule_qualified_name)

        logger.debug(
            "Running passes of {} stage on module {}".format(
                ctx.stage, "outer_most" if module_prefix == "" else module_prefix
            )
        )
        graph_changed = run_passes(ctx) or graph_changed
        return graph_changed

    graph_changed = False
    graph_changed = recursive_run_passes(ctx, graph_changed)

    return graph_changed


def pass_annotate_nodes_and_inline_submodule(ctx: OptimizerContext) -> bool:
    """
    This pass aims to annotate node based on hints wrapped by hints_wrapper HOO.
    There are two steps:
        1. recursively annotate nodes inside nested submodules
        2. inline those nested submodules
    """

    def is_hints_wrapper_node(node: torch.fx.Node) -> bool:
        return node.op == "call_function" and node.target.__name__ == "hints_wrapper"

    def get_schedule_policy(hints: dict) -> SchedulePolicy:
        if "schedule_policy" not in hints:
            logger.warn("No schedule policy is provided, default to use strict policy.")
            return SchedulePolicy.strict

        expected_policy = hints["schedule_policy"]
        if expected_policy.lower() == "strict":
            return SchedulePolicy.strict

        logger.warn(f"Currently policy {expected_policy} is not supported, fall back to strict policy.")
        return SchedulePolicy.strict

    def get_supported_hints() -> list:
        supported_list = [
            "schedule_policy",
            "group_id",
        ]
        return supported_list

    def sanity_check_on_hints(hints: dict, n: torch.fx.Node):
        if not hints:
            # hints is empty, there is no more actions for node annotation
            logger.debug("no hints provided for node ", n)
        else:
            for h in hints.keys():
                if h not in get_supported_hints():
                    logger.warn(
                        f"hint key '{h}' is not support yet hence expect to not take effect. Supported hint keys are {get_supported_hints()}"
                    )

    def inline_hints_wrapper(
        parent_module: torch.fx.GraphModule, node_to_replace: torch.fx.Node, inline_mod: torch.fx.GraphModule
    ):
        """ "
        This is adapted from torch.fx.experimental.constant_fold._inline_module function.
        It aims to inline submodule wrapped by hints_wrapper node into parent
        module.
        """
        assert is_hints_wrapper_node(node_to_replace)

        getitem_nodes_to_be_removed = []
        for u in node_to_replace.users:
            if u.op == "call_function" and u.target.__name__ == "getitem":
                getitem_nodes_to_be_removed.append(u)

        node_args = node_to_replace.args
        # unpack input tensors
        new_node_args = []
        for arg in node_args:
            if isinstance(arg, tuple):
                for a in arg:
                    new_node_args.append(a)
                continue
            new_node_args.append(arg)

        replacement_mapping: dict[torch.fx.Node, torch.fx.Node] = {}
        # args starts from idx 1
        ph_count = 1

        def replacement_fn(node):
            new_node = replacement_mapping[node]
            return new_node

        for inline_node in inline_mod.graph.nodes:
            if inline_node.op == "placeholder":
                replacement_mapping[inline_node] = new_node_args[ph_count]
                ph_count += 1
                continue

            if inline_node.op == "output":
                outputs = inline_node.args[0]
                output_replacements = map_arg(outputs, replacement_fn)
                node_to_replace.replace_all_uses_with(output_replacements)
                continue

            with parent_module.graph.inserting_before(node_to_replace):
                new_node = parent_module.graph.node_copy(inline_node, replacement_fn)
            replacement_mapping[inline_node] = new_node

        # delete unecessary getitem nodes
        for n in getitem_nodes_to_be_removed:
            assert isinstance(n.args[0], tuple)
            arg_idx = n.args[1]
            arg_node = n.args[0][arg_idx]
            n.replace_all_uses_with(arg_node)

        parent_module.graph.eliminate_dead_code()

    def process_nested_submodule(
        parent_module: torch.fx.GraphModule,
        wrapper_node: torch.fx.Node,
        parent_hints: dict,
        module_prefix: str = "",
    ):
        sanity_check_on_hints(parent_hints, wrapper_node)

        submodule_name = wrapper_node.args[0].name
        submodule = parent_module.get_submodule(submodule_name)
        submodule_qualified_name = module_prefix + ("." if module_prefix else "") + submodule_name

        for n in submodule.graph.nodes:
            if n.op in ["placeholder", "get_attr", "output"]:
                continue
            elif is_hints_wrapper_node(n):
                cur_hints = n.kwargs.get("hints", None)
                merged_hints = {**parent_hints, **cur_hints}
                process_nested_submodule(submodule, n, merged_hints, submodule_qualified_name)
                continue

            # annotate node from here
            n.meta["context_hints"] = parent_hints
            logger.debug(f"annotated node {n} with hints {parent_hints} inside submodule {submodule_qualified_name}")

        inline_hints_wrapper(parent_module, wrapper_node, submodule)
        parent_module.delete_submodule(submodule_name)

    class StrictRunNode(torch.fx.Interpreter):
        def __init__(self, module: torch.fx.GraphModule):
            super().__init__(module)
            self.counter = 0

        def run_node(self, n: torch.fx.Node):
            if "context_hints" in n.meta:
                new_context_hints = {**n.meta["context_hints"]}
                new_context_hints["exec_order"] = self.counter
                n.meta["context_hints"] = new_context_hints
                self.counter += 1
            return super().run_node(n)

    graph = ctx.graph_module.graph
    hints_wrapper_nodes = [n for n in graph.nodes if is_hints_wrapper_node(n)]
    if not hints_wrapper_nodes:
        return False

    top_level_hints = None
    for n in hints_wrapper_nodes:
        hints_dict = n.kwargs.get("hints", None)
        if top_level_hints is None:
            top_level_hints = hints_dict
        process_nested_submodule(ctx.graph_module, n, hints_dict)

    # currently assume there is single hints_wrapper node or multiple
    # hints_wrapper nodes but with same schedule policy
    if get_schedule_policy(top_level_hints) == SchedulePolicy.strict:
        StrictRunNode(ctx.graph_module).run(*ctx.example_inputs)

    return True


def pass_mark_frozen_params(ctx: OptimizerContext) -> bool:
    for n in ctx.graph_module.graph.nodes:
        if n.op == "get_attr" or n.op == "placeholder":
            if "_frozen_param" in n.target:
                n.meta["frozen_param"] = True
    ctx.graph_module.graph.lint()
    ctx.graph_module.recompile()


def pass_replace_sym_size(ctx: OptimizerContext) -> bool:
    if not ctx.is_dynamic:
        return True

    graph_changed = False
    py_node_manager = SymExprNodeManager(ctx.graph_module)

    def _is_sym_size_node(node):
        return node.target in [torch.ops.aten.sym_size, torch.ops.aten.sym_size.int]

    def process_symsize(node):
        in_node = node.args[0]
        sym_size_dim = node.args[1]
        sym_size_expr = in_node.meta["output_shapes"][0][sym_size_dim]

        py_node = py_node_manager.get_or_create(sym_size_expr, node.type)
        py_node.meta = copy.copy(node.meta)
        list(node.users.keys())[0].replace_input_with(node, py_node)
        node.replace_all_uses_with(py_node)

    for node in ctx.graph_module.graph.nodes:
        if node.op == "placeholder":
            tmeta_val = node.meta.get("val", node.meta.get("tensor_meta", None))
            if isinstance(tmeta_val, py_sym_types):
                py_node_manager.add_sym_placeholder(tmeta_val, node)
            py_node_manager.set_insert_point(node)

        if _is_sym_size_node(node):
            process_symsize(node)
            graph_changed = True

    if graph_changed:
        # Clean up the graph and log the situation.
        ctx.graph_module.graph.eliminate_dead_code()
        ctx.graph_module.recompile()

    return True


def pass_graph_print(ctx: OptimizerContext) -> bool:
    """
    This pass just prints the graph in debug mode.
    """
    if not logger.is_enabled_for(LogLevel.DEBUG):
        return False

    assert ctx.graph_module is not None

    logger.debug("Readable:\n%s", ctx.graph_module.print_readable(False))
    logger.debug("IR:\n%s", ctx.graph_module.graph)
    logger.debug("Nodes:")
    for node in ctx.graph_module.graph.nodes:
        logger.debug("Node name: %s op: %s", node.name, node.op)
        if node.op == "call_function":
            logger.debug("    target: %s", node.target.__name__)
        if "output_device" in node.meta:
            logger.debug("    meta.output_device: %s", node.meta["output_device"])
        if "context_hints" in node.meta:
            logger.debug("    meta.context_hints: %s", node.meta["context_hints"])
    return False


def pass_make_symints_available(ctx: OptimizerContext) -> bool:
    if not ctx.is_dynamic:
        return True

    def get_all_symbolic_int_nodes():
        symint_list = ()
        for node in ctx.graph_module.graph.nodes:
            if node.op == "placeholder":
                tmeta_val = node.meta.get("val", node.meta.get("tensor_meta", None))
                if isinstance(tmeta_val, torch.SymInt):
                    symint_list += (node,)
        return symint_list

    def get_missing_symbolic_int_input_nodes(symint_list, node):
        is_arguments_present = False
        missing_symints = ()
        for symint in symint_list:
            symint_count = 0
            for node_in in node.args:
                is_arguments_present = True
                if node_in.target == symint.target:
                    symint_count += 1
                    break
            if symint_count == 0:
                missing_symints = missing_symints + (symint,)

        if is_arguments_present:
            return missing_symints

        return ()

    symint_list = get_all_symbolic_int_nodes()

    for node in ctx.graph_module.graph.nodes:
        if node.op == "call_module":
            submodule = node.graph.owning_module.get_submodule(node.target)
            # for submodules that are not dynamic, we don't need to add symints
            if not is_module_dynamic(submodule):
                continue
            missing_symint_list = get_missing_symbolic_int_input_nodes(symint_list, node)
            if missing_symint_list == ():
                continue

            node.args = missing_symint_list + node.args

            # Get the First node in the graph to insert all the SymInts at the
            # beginning of the node_list
            first_subgraph_node = node
            for sub_node in submodule.graph.nodes:
                first_subgraph_node = sub_node
                break

            for misinput in reversed(missing_symint_list):
                with submodule.graph.inserting_before(first_subgraph_node):
                    new_node = submodule.graph.create_node(
                        misinput.op,
                        misinput.target,
                        misinput.args,
                        misinput.kwargs,
                        misinput.name,
                        misinput.type,
                    )
                    new_node.meta = copy.copy(misinput.meta)
                    first_subgraph_node = new_node

    ctx.graph_module.recompile()

    return True


def pass_fake_propagation(ctx: OptimizerContext) -> bool:
    """
    This function contains FakeMode propagation implementation for PT2.1+
    """

    propagate_meta(ctx.graph_module, ctx.example_inputs, TensorInfoPropagation)

    return True


def pass_wa_fix_output(ctx: OptimizerContext) -> bool:
    """
    This pass is supposed to workaround an issue with global output not being the last
    node in the graph. Details below.
    """
    assert ctx.graph_module is not None
    graph_changed = False

    # WORKAROUND BEGIN
    # This is workaround for graphs that are not functionalized at this point.
    # Issue is that some graphs has no outputs and it will cause wrong topological
    # sort and execution when they are not functionalized. This code fixes that by
    # moving global output node to the end of graph.
    output_node = None
    is_output_last = True
    for n in reversed(ctx.graph_module.graph.nodes):
        if n.op == "output":
            output_node = n
            break

        is_output_last = False

    if not is_output_last and output_node is not None:
        logger.warn("It seems graph wasn't functionalized, fixing empty output node.")
        ctx.graph_module.graph.node_copy(output_node)
        ctx.graph_module.graph.erase_node(output_node)
        ctx.graph_module.recompile()
        graph_changed = True

    # WORKAROUND END

    return graph_changed


def pass_weight_permutation(ctx: OptimizerContext):
    """
    This pass inserts weight permutation node before convolution, handles both
    directly weight of convolution and casted weight.
    """
    graph_changed = False
    for node in ctx.graph_module.graph.nodes:
        if node.op == "call_function" and node.target == torch.ops.aten.convolution.default:
            dim = len(node.meta["tensor_meta"].shape)
            if dim in (4, 5):
                weight_node = node.args[1]
                while (
                    weight_node.op == "call_function" and "to_copy" in weight_node.target.__name__ and weight_node.args
                ):
                    node = weight_node
                    weight_node = weight_node.args[0]
                if weight_node.meta["output_device"].type == "hpu":
                    with ctx.graph_module.graph.inserting_before(node):
                        weight_permutation_node = ctx.graph_module.graph.call_function(
                            torch.ops.hpu.weight_permutation, (weight_node,), {}
                        )
                        weight_permutation_node.meta = copy.copy(weight_node.meta)
                        weight_permutation_node.val_args = weight_permutation_node.args
                        weight_permutation_node.val_kwargs = weight_permutation_node.kwargs
                    node.replace_input_with(weight_node, weight_permutation_node)
                    graph_changed = True
                    logger.info(
                        "Permute node: {} op: {} target: {} dim: {}",
                        node.name,
                        node.op,
                        node.target,
                        dim,
                    )
            else:
                logger.info("No permutation, permute weight support only 4/5D tensors")

    if graph_changed:
        ctx.graph_module.graph.lint()
        ctx.graph_module.recompile()

    return graph_changed


def pass_propose_partitions(ctx: OptimizerContext) -> bool:
    """
    This pass is supposed to run partitioner that will create proposition of partitioning.
    """
    assert ctx.stage == OptimizationPassPlacement.PARTITIONER
    assert ctx.graph_module is not None
    assert ctx.current_partitions is None
    assert ctx.current_partitions_non_mergeable is None
    assert ctx.habana_partitioner is None

    ctx.current_partitions = []
    ctx.current_partitions_non_mergeable = []
    if hpu_backend_config.enable_allreduce_graph_split:
        allreduces = [n for n in ctx.graph_module.graph.nodes if n.name.startswith("all_reduce")]
        cls = FusedCollectiveOperatorSupport
        cls.keyword = "downstream_allreduce_name"
        ctx.habana_partitioner = HabanaPartitioner(ctx.graph_module, cls)
        for allreduce in allreduces:
            cls.target_value = allreduce.name
            ctx.current_partitions_non_mergeable.extend(ctx.habana_partitioner.propose_partitions())

    if hpu_backend_config.enable_waittensor_graph_split:
        wait_tensors = [n for n in ctx.graph_module.graph.nodes if n.name.startswith("wait_tensor")]
        cls = FusedCollectiveOperatorSupport
        cls.keyword = "upstream_waittensor_name"
        ctx.habana_partitioner = HabanaPartitioner(ctx.graph_module, cls)
        for wait_tensor in wait_tensors:
            cls.target_value = wait_tensor.name
            ctx.current_partitions_non_mergeable.extend(ctx.habana_partitioner.propose_partitions())

    ctx.habana_partitioner = HabanaPartitioner(ctx.graph_module)
    ctx.current_partitions.extend(ctx.habana_partitioner.propose_partitions())

    # Nothing was really changed.
    return False


def match_full_copy_pattern(node: torch.fx.Node) -> tuple[bool, torch.fx.Node, torch.fx.Node]:
    is_full_copy_pattern = (
        node.name.startswith("full") and len(node.users) == 1 and list(node.users.keys())[0].name.startswith("copy")
    )
    if not is_full_copy_pattern:
        return (False, None, None)

    full_node, copy_node = node, list(node.users.keys())[0]
    copy_args = list(copy_node.args)
    if full_node != copy_args[0]:
        return (False, None, None)
    return True, full_node, copy_node


def post_process_partitions(
    graph_module: torch.fx.GraphModule,
    current_partitions: list[torch.fx.passes.infra.partitioner.Partition],
    current_partitions_non_mergeable: list[torch.fx.passes.infra.partitioner.Partition],
):
    """
    This pass will do some post process for those proposed partitions from hpu
    partitioner, like move some specific ops from one partition to another
    partition, to reduce some unnecessary tensor passing between partitions.
    Currently, the post process is mainly for device memory optimization.
    """
    assert None not in [graph_module, current_partitions, current_partitions_non_mergeable]
    from torch.fx.passes.infra.partitioner import Partition

    partition_changed = False

    def reassign_full_copy_to_upstream_partition(
        graph_module, assignments: dict[torch.fx.Node, int], partitions_by_id: dict[int, Partition]
    ):
        changed = False
        for node in graph_module.graph.nodes:
            matched, full_node, copy_node = match_full_copy_pattern(node)
            if not matched:
                continue

            # now, we detected a full+copy pattern
            copy_args = list(copy_node.args)
            copy_src_node = copy_args[1]
            if (
                full_node not in assignments
                or copy_node not in assignments
                or copy_src_node not in assignments
                or assignments[full_node] != assignments[copy_node]
                or assignments[full_node] == assignments[copy_src_node]
            ):
                continue

            # now we have full+copy in one partition and the copy src node in
            # another partition. we will merge the full+copy to its upstream
            # partittion
            full_copy_partition = partitions_by_id[assignments[full_node]]
            upstream_partition = partitions_by_id[assignments[copy_src_node]]
            full_copy_partition.remove_node(full_node)
            full_copy_partition.remove_node(copy_node)
            upstream_partition.add_node(full_node)
            upstream_partition.add_node(copy_node)
            changed = True
        return changed

    def reassign_copy__to_upstream_partition(
        graph_module, assignments: dict[torch.fx.Node, int], partitions_by_id: dict[int, Partition]
    ):
        changed = False
        for node in graph_module.graph.nodes:
            if not (node.op == "call_function" and node.target == torch.ops.aten.copy_.default):
                continue

            copy_node = node
            copy_args = list(copy_node.args)
            copy_src_node, copy_dst_node = copy_args[1], copy_args[0]

            if copy_dst_node.op != "placeholder":
                continue

            # now, we detected a reassignable copy_ node
            if (
                copy_node not in assignments
                or copy_src_node not in assignments
                or assignments[copy_node] == assignments[copy_src_node]
            ):
                continue

            # now we have copy_ in one partition and the copy_ src node in
            # another partition. we will merge the copy_ to its upstream
            # partittion
            copy_partition = partitions_by_id[assignments[copy_node]]
            upstream_partition = partitions_by_id[assignments[copy_src_node]]
            copy_partition.remove_node(copy_node)
            upstream_partition.add_node(copy_node)
            changed = True
        return changed

    def eagerize_partitions_with_only_view_ops(
        graph_module, assignments: dict[torch.fx.Node, int], partitions_by_id: dict[int, Partition]
    ):
        def is_partition_with_only_view_ops(part: Partition):
            return all(is_view_node(node) for node in part.nodes)

        partition_changed = False
        for _id, partition in partitions_by_id.items():
            if not is_partition_with_only_view_ops(partition):
                continue

            # clear the part to eagerize all view ops
            partition.nodes = {}
            partition_changed = True

        return partition_changed

    assignments: dict[torch.fx.Node, int] = {}  # mapping from node to partition_id
    partitions_by_id: dict[int, Partition] = {}  # mapping from partition_id to partition
    for partition in current_partitions + current_partitions_non_mergeable:
        id = partition.id
        partitions_by_id[id] = partition
        for node in list(partition.nodes):
            assignments[node] = id

    if hpu_backend_config.reassign_full_copy:
        partition_changed = reassign_full_copy_to_upstream_partition(graph_module, assignments, partitions_by_id)
    if hpu_backend_config.reassign_copy_:
        partition_changed = (
            reassign_copy__to_upstream_partition(graph_module, assignments, partitions_by_id) or partition_changed
        )

    partition_changed = (
        eagerize_partitions_with_only_view_ops(graph_module, assignments, partitions_by_id) or partition_changed
    )

    return partition_changed


def pass_post_process_partitions(ctx: OptimizerContext) -> bool:
    return post_process_partitions(ctx.graph_module, ctx.current_partitions, ctx.current_partitions_non_mergeable)


def pass_fuse_partitions(ctx: OptimizerContext) -> bool:
    """
    This pass is supposed to run partitioner that will, based on current partitioning, create
    final FX module with submodules for each HPU operations cluster.
    """
    assert ctx.stage == OptimizationPassPlacement.PARTITIONER
    assert ctx.graph_module is not None
    assert ctx.current_partitions is not None
    assert ctx.current_partitions_non_mergeable is not None
    assert ctx.habana_partitioner is not None

    ctx.habana_partitioner.fuse_partitions(ctx.current_partitions + ctx.current_partitions_non_mergeable)

    return True


def pass_add_fused_op_metadata(ctx: OptimizerContext):
    """
    This pass goes through every fused node (`call_module`) created by GraphPartitioner
    and adds output metadata according to subgraph outputs
    """
    graph_changed = False
    if not ctx.use_jit_fork:
        return graph_changed

    logger.debug("JitLowering pass_add_fused_op_metadata")

    for graph_node in ctx.graph_module.graph.nodes:
        if graph_node.op != "call_module":
            continue

        target = graph_node.target
        submod = ctx.graph_module.get_submodule(target)

        for subgraph_node in submod.graph.nodes:
            if subgraph_node.op != "output":
                continue

            args = subgraph_node.args
            if len(args) == 1 and isinstance(args[0], tuple):
                args = args[0]
            assert all(
                isinstance(x, torch.fx.Node) for x in args
            ), "Currently we are assuming that all args of output should be Nodes"
            meta_val = tuple([a.meta.get("val", None) for a in args])

        # it is possible to have zero graph outputs with KEEP_INPUT_MUTATIONS enabled
        if meta_val:
            graph_node.meta["val"] = meta_val if len(meta_val) > 1 else meta_val[0]
            graph_changed = True

    return graph_changed


def pass_wa_mixed_devices(ctx: OptimizerContext) -> bool:
    """
    This pass is supposed to find cases where HPU ops have mixed devices inputs. If for such
    OP there is non-HPU input, it will add copy to HPU on it.

    Disclaimer: this fixes an issue, but we don't know if such scenario should even occur. It
    is visible in optimizers where there are constant_tensors (like beta params) that are not
    FX graph inputs and according to device propagation they land on CPU, eventually mixing
    with HPU parameters of the model.
    """
    assert ctx.graph_module is not None

    graph_changed = False
    nodes_to_fix_list = []
    h2d_scales_enabled = htexp._get_scale_attribute_hash_id() > 0 and bc.get_pt_hpu_enable_h2d_scales()

    for node in ctx.graph_module.graph.nodes:
        if (
            node.op != "placeholder"
            and node.op != "output"
            and node.op != "get_attr"
            and not (node.op == "call_function" and "to_copy" in node.target.__name__)
            and node.meta["output_device"].type == "hpu"
            and not is_backward_checkpoint_op(node)
        ):
            for arg_idx, arg in enumerate(node.args):
                if (
                    isinstance(arg, torch.fx.Node)
                    and ("output_device" in arg.meta and arg.meta["output_device"].type != "hpu")
                    and _is_cpu_scalar_copy_required(node, arg_idx, arg, h2d_scales_enabled)
                ):
                    nodes_to_fix_list.append(node)
                    break

    for node in nodes_to_fix_list:
        for arg in node.args:
            if isinstance(arg, torch.fx.Node) and arg.meta["output_device"].type != "hpu":
                with ctx.graph_module.graph.inserting_before(node):
                    input_copy_node = ctx.graph_module.graph.call_function(
                        torch.ops.aten._to_copy.default,
                        (arg,),
                        {"device": torch.device("hpu")},
                    )
                    input_copy_node.meta["output_device"] = torch.device("hpu")
                    input_copy_node.meta["output_dtypes"] = [arg.meta["output_dtypes"][0]]
                    input_copy_node.meta["output_layouts"] = [arg.meta["output_layouts"][0]]
                    input_copy_node.meta["output_shapes"] = [arg.meta["output_shapes"][0]]
                    input_copy_node.meta["output_strides"] = [arg.meta["output_strides"][0]]
                    input_copy_node.meta["output_contiguous"] = [arg.meta["output_contiguous"][0]]
                    input_copy_node.meta["output_offset"] = [arg.meta["output_offset"][0]]
                    fill_propagated_tensor_metadata_jitfork(input_copy_node)
                    node.replace_input_with(arg, input_copy_node)
                graph_changed = True

    if graph_changed:
        # Clean up the graph and log the situation.
        ctx.graph_module.graph.eliminate_dead_code()
        ctx.graph_module.recompile()
        logger.debug("Detected mixed devices. Workaround applied.")

    return graph_changed


def pass_mark_placement(ctx: OptimizerContext) -> bool:
    """
    This pass is supposed to annotate nodes with their placement.
    There are two placement options:

    "eager"       - such OPs will not be placed inside HPU clusters
    "hpu_cluster" - such OPs will be later placed inside HPU clusters
    """
    assert ctx.graph_module is not None
    h2d_scales_enabled = htexp._get_scale_attribute_hash_id() > 0 and bc.get_pt_hpu_enable_h2d_scales()

    for node in ctx.graph_module.graph.nodes:
        placement = None
        dynamic_call_function = is_call_function_dynamic(node, ctx.is_dynamic) if node.op == "call_function" else False
        if node.op in ["placeholder", "output", "get_attr"]:
            placement = "eager"
            logger.debug(f"Node {node}: eager placement as node is one of [placeholder, output, get_attr]")
        elif node.op == "call_function" and is_higher_order_node(node):
            placement = "eager"
            logger.debug(f"Node {node}: eager placement as node is a higher order node")
        elif node.op == "call_function" and is_copy_op(node):
            args_list_with_node = [node]

            if node.target.__name__.split(".")[0] == "_to_copy":
                input_node = None
                for arg in node.args:
                    if isinstance(arg, torch.fx.Node):
                        input_node = arg
                        break

                assert input_node is not None
                args_list_with_node.append(input_node)
            else:  # foreach_copy
                for arg in node.args:
                    if isinstance(arg, list):
                        args_list_with_node.extend(arg)

            # Internal HPU copies should be placed in the clusters.
            if all(n.meta["output_device"].type == "hpu" for n in args_list_with_node):
                placement = "hpu_cluster"
            else:
                placement = "eager"
                logger.debug(f"Node {node}: eager placement as node is a non D2D copy")
        elif node.op == "call_function" and node._pretty_print_target(node.target) in host_call_functions:
            placement = "eager"
            logger.debug(f"Node {node}: eager placement as node is a host call function")
        elif node.op == "call_function" and is_eager_fallback_required(node, is_dynamic=dynamic_call_function):
            placement = "eager"
            logger.debug(f"Node {node}: eager placement as node require fallback to eager")
        elif node.meta["output_device"].type == "hpu":
            # Current assumption is that if OP outputs HPU tensor, then all its inputs are also on HPU.
            # Let's create an assert that will fire in case this assumption proves wrong.
            for arg in node.args:
                if isinstance(arg, torch.fx.Node):
                    # If you got into this assert, we might need to rewrite this part so we cluster only
                    # these OPs that also have all inputs on HPU. Or debug why this OP have mixed device
                    # tensors, that could be the original issue here.
                    if _is_cpu_scalar_or_symbolic_scalar(arg):
                        logger.debug("Argument {} to node {} is a scalar or a symbolic scalar", arg, node)
                        continue
                    elif is_backward_checkpoint_op(node) and arg.meta["output_device"] == torch.device("cpu"):
                        logger.debug("Argument {} to node {} is an rng_state - a cpu tensor by definition", arg, node)
                        continue
                    elif is_constant_for_lift_fresh_copy(node, arg):
                        logger.debug("Argument {} to node {} is a _tensor_constant get_attr", arg, node)
                        continue
                    elif _is_cpu_scale_allowed(node, arg, h2d_scales_enabled):
                        logger.debug("Argument {} to node {} is a cpu tensor for H2D optimization", arg, node)
                        continue
                    assert arg.meta["output_device"].type == "hpu"

            placement = "hpu_cluster"
        elif node.meta["output_device"].type == "cpu":
            placement = "eager"
            logger.debug(f"Node {node}: eager placement as node output_device is cpu")

        if node.op == "call_function":
            # This log line is used by the logging analysis tool. Please be cautious
            # when changing.
            logger.info(
                "Node placement. Node: {} op: {} placement: {} target: {} dynamic: {}",
                node.name,
                node.op,
                placement,
                node.target,
                dynamic_call_function,
            )
        else:
            logger.info("Node placement. Node: {} op: {} placement: {}", node.name, node.op, placement)

        assert placement is not None

        # Meta for the node should not be created yet. BUT...
        # ...it happens that placeholder nodes might be reused between FWD and BWD.
        # They are always placed in eager though, so it should not be an issue.
        if "placement" in node.meta:
            logger.debug("Node {} of type {} has had it's placement already set", node, node.op)
            assert node.meta["placement"] == placement

        node.meta["placement"] = placement

    return True


collective_ops = {torch.ops._c10d_functional.all_reduce_.default, torch.ops._c10d_functional.all_reduce.default}

view_ops_set = {torch.ops.aten.view.default, torch.ops.aten._unsafe_view.default}


def pass_mark_collective_input(ctx: OptimizerContext) -> bool:

    assert ctx.graph_module is not None

    if not hpu_backend_config.enable_sfg:
        return False

    graph = ctx.graph_module.graph

    for node in graph.nodes:
        if node.target in collective_ops:
            sfg_node_queue = queue.Queue()
            for inp in node.all_input_nodes:
                sfg_node_queue.put(inp)
            while not sfg_node_queue.empty():
                sfg_node = sfg_node_queue.get()
                if sfg_node.target in view_ops_set:
                    for inp in sfg_node.all_input_nodes:
                        sfg_node_queue.put(inp)
                else:
                    sfg_node.meta["sfg"] = True

    return False


def merge_paths(
    graph_module: torch.fx.Graph, current_partitions: list[torch.fx.passes.infra.partitioner.Partition]
) -> list[torch.fx.passes.infra.partitioner.Partition]:
    """
    This pass that will merge parallel partitions.
    """

    logger.debug(f"Merging parallel graph path. Partition cnt: {len(current_partitions)}")

    if len(current_partitions) == 1:
        logger.debug("Merging skipped for single partition graph")
        # In case of single partition there is no merging to be done
        return current_partitions

    color_graph = ColorGraph()

    # Color all nodes in every partition on the same color
    partitions_by_color = {}

    for part in current_partitions:
        partition_color = color_graph.assign_new_color(is_partition_color=True)
        for node in part.nodes:
            node.meta["merge_path_color"] = partition_color
        partitions_by_color[partition_color] = part

    # Color remaining nodes (new color for every node)
    for node in graph_module.graph.nodes:
        if "merge_path_color" not in node.meta:
            node_color = color_graph.assign_new_color()
            node.meta["merge_path_color"] = node_color

    # Build color graph
    for node in graph_module.graph.nodes:
        for user in node.users.keys():
            user_color = user.meta.get("merge_path_color")
            node_color = node.meta.get("merge_path_color")
            color_graph.add_node(user_color, node_color)
        if not node.users:
            color_graph.add_output_node(node.meta.get("merge_path_color"))

    new_partitions_desc_list = color_graph.get_parallel_blocks()

    # Update only if new partitioning is better than old one
    if len(new_partitions_desc_list) < len(current_partitions):
        logger.debug("New partition list (by colors): %s", new_partitions_desc_list)
        from torch.fx.passes.infra.partitioner import Partition

        new_partitions = []
        for desc in new_partitions_desc_list:
            new_part = Partition()
            for color in desc:
                for node in partitions_by_color[color].nodes:
                    new_part.add_node(node)
            new_partitions.append(new_part)

        current_partitions = new_partitions

        logger.debug("Merge paths done. Partition cnt: %s", len(new_partitions))
    else:
        logger.debug("No partitions suitable for merging found")

    # Cleanup coloring information from meta
    for node in graph_module.graph.nodes:
        del node.meta["merge_path_color"]

    return current_partitions


def pass_merge_paths(ctx: OptimizerContext) -> bool:
    assert ctx.stage == OptimizationPassPlacement.PARTITIONER
    assert ctx.graph_module is not None
    assert ctx.current_partitions is not None

    new_partitions = merge_paths(ctx.graph_module, ctx.current_partitions)
    graph_changed = new_partitions != ctx.current_partitions
    ctx.current_partitions = new_partitions

    return graph_changed


class resolve_negative_dim:
    node_name = ""
    view_dim_index = 0
    py_node_manager = None

    @staticmethod
    def required(node):
        node_name = node.target.__name__.split(".")[0]
        resolve_negative_dim.node_name = node_name
        # This is list of OPs with negative Dims.
        negative_dim_ops = [
            "view",
            "slice",
            "constant_pad_nd",  # it is not a neg-dim op, but requires to create a custom-schema for DS handling
        ]

        from torch.fx.experimental.proxy_tensor import py_sym_types

        if node_name in negative_dim_ops:
            if node_name == "slice" or node_name == "constant_pad_nd":
                for node_in in node.args:
                    if isinstance(node_in, torch.fx.Node):
                        meta_val = node_in.meta.get("val", node_in.meta.get("tensor_meta", None))
                        return True
                return False
            elif node_name == "view":
                node_arg0 = node.args[0]
                meta_val = node_arg0.meta.get("val", node.meta.get("tensor_meta", None))
                in_args_1 = node.args[1]
                # skip for non-iterable arg for example view(dtype)
                if not hasattr(in_args_1, "__iter__"):
                    return False

                for index, value in enumerate(in_args_1):
                    if not isinstance(value, py_sym_types):
                        if value == -1:
                            resolve_negative_dim.view_dim_index = index
                            return True
        return False

    @classmethod
    def __resolve_view_shapes(cls, ctx, node):
        if node.args[0].meta["output_device"].type == "hpu":
            new_args1 = []
            if not ctx.is_dynamic:
                meta_val = node.meta.get("val", node.meta.get("tensor_meta", None))
                new_args1 = list(meta_val.size())
            else:
                sym_size_expr = node.meta["output_shapes"][0][cls.view_dim_index]
                meta_val = node.meta.get("val", node.meta.get("tensor_meta", None))
                value = copy.copy(meta_val.shape[cls.view_dim_index])
                new_node = cls.py_node_manager.get_or_create(sym_size_expr, int)
                new_node.meta["val"] = value
                new_node.meta["placement"] = "eager"
                new_node.meta["output_device"] = torch.device("cpu")
                for arg in node.args[1]:
                    new_args1.append(arg)
                neg_node = new_args1[cls.view_dim_index]
                new_args1[cls.view_dim_index] = new_node
            # replace call_function and recompile the graph
            with ctx.graph_module.graph.inserting_before(node):
                view_new_node = ctx.graph_module.graph.call_function(
                    torch.ops.aten.view.default,
                    (
                        node.args[0],
                        new_args1,
                    ),
                    {},
                )
                node.replace_all_uses_with(view_new_node, propagate_meta=True)
        return True

    @classmethod
    def __resolve_slice_shapes(cls, ctx, node):
        if node.args[0].meta["output_device"].type == "hpu" and node.meta["placement"] != "eager":
            new_args1 = []
            if not ctx.is_dynamic:
                return False
            else:
                meta_val = node.args[0].meta.get("val", node.meta.get("tensor_meta", None))
                new_args1 = list(meta_val.size())
                for idx, arg in enumerate(list(meta_val.size())):
                    new_args1[idx] = arg
                    if isinstance(arg, py_sym_types):
                        new_node = cls.py_node_manager.get_or_create(arg, int)
                        new_node.meta["val"] = arg
                        new_node.meta["placement"] = "eager"
                        new_node.meta["output_device"] = torch.device("cpu")
                        new_args1[idx] = new_node
            # handle negative end values
            end = sys.maxsize if len(node.args) == 3 else node.args[3]
            end = new_args1[node.args[1]] if end == sys.maxsize else node.args[3]
            step = node.args[4] if len(node.args) == 5 else 1
            # replace call_function and recompile the graph
            with ctx.graph_module.graph.inserting_before(node):
                view_new_node = ctx.graph_module.graph.call_function(
                    torch.ops.hpu.slice_ds.default,
                    (
                        node.args[0],
                        node.args[1],
                        node.args[2],
                        end,
                        step,
                        new_args1,
                    ),
                    {},
                )
                node.replace_all_uses_with(view_new_node, propagate_meta=True)

        return True

    @classmethod
    def __resolve_constant_pad_nd_shapes(cls, ctx, node):
        if node.args[0].meta["output_device"].type == "hpu" and node.meta["placement"] != "eager":
            new_args1 = []
            if not ctx.is_dynamic:
                return
            else:
                meta_val = node.args[0].meta.get("val", node.meta.get("tensor_meta", None))
                new_args1 = list(meta_val.size())
                for idx, arg in enumerate(list(meta_val.size())):
                    new_args1[idx] = arg
                    if isinstance(arg, py_sym_types):
                        new_node = cls.py_node_manager.get_or_create(arg, int)
                        new_node.meta["val"] = arg
                        new_node.meta["placement"] = "eager"
                        new_node.meta["output_device"] = torch.device("cpu")
                        new_args1[idx] = new_node
            # replace call_function and recompile the graph
            val = 0 if len(node.args) == 2 else node.args[2]
            with ctx.graph_module.graph.inserting_before(node):
                view_new_node = ctx.graph_module.graph.call_function(
                    torch.ops.hpu.constant_pad_nd_ds.default,
                    (
                        node.args[0],
                        node.args[1],
                        val,
                        new_args1,
                    ),
                    {},
                )
                node.replace_all_uses_with(view_new_node, propagate_meta=True)

            ctx.graph_module.recompile()
            ctx.graph_module.graph.eliminate_dead_code()
        return True

    def __new__(cls, ctx, node):
        if cls.node_name == "view":
            return cls.__resolve_view_shapes(ctx, node)
        if cls.node_name == "slice":
            return cls.__resolve_slice_shapes(ctx, node)
        if cls.node_name == "constant_pad_nd":
            return cls.__resolve_constant_pad_nd_shapes(ctx, node)
        return False


def pass_handle_negative_dims(ctx: OptimizerContext) -> bool:
    """
    This pass goes through each node in the main module and replace
    negative dims of node with static values in non-dynamic mode and
    unrolled sympy expression with cpu operations in dynamic case
    """
    if hpu_backend_config.force_static_compile:
        return False

    graph_changed = False
    py_node_manager = SymExprNodeManager(ctx.graph_module)
    resolve_negative_dim.py_node_manager = py_node_manager
    for node in ctx.graph_module.graph.nodes:
        if node.op == "placeholder":
            tmeta_val = node.meta.get("val", node.meta.get("tensor_meta", None))
            if isinstance(tmeta_val, py_sym_types):
                py_node_manager.add_sym_placeholder(tmeta_val, node)
        if node.op == "call_function":
            if resolve_negative_dim.required(node):
                py_node_manager.set_insert_point(node.prev)
                graph_changed = resolve_negative_dim(ctx, node) or graph_changed

    if graph_changed:
        ctx.graph_module.recompile()
        ctx.graph_module.graph.eliminate_dead_code()
    return graph_changed


def pass_eagerize_leaf_views(ctx: OptimizerContext) -> bool:
    """
    This pass is supposed to find HPU nodes which are in chains of view operations that
    ultimately lead to non-HPU operations. As non-HPU operations will be placed outside
    of the module, they will become a submodule output node and we don't want to feed
    these output nodes with view tensors. In such case, we will move these HPU view OPs
    into eager mode instead, while duplicating them in some cases to avoid too much
    fragmentation.
    """

    assert ctx.stage == OptimizationPassPlacement.PRE_PARTITIONER
    assert ctx.graph_module is not None

    graph_changed = False

    # First, make sure nodes in the graph are in topological order.
    ctx.graph_module.graph.lint()

    reverse_nodes_list = list(ctx.graph_module.graph.nodes)
    reverse_nodes_list.reverse()

    # Initialize colors.
    for node in reverse_nodes_list:
        assert "pass_meta_color" not in node.meta
        node.meta["pass_meta_color"] = "none"
    # Find HPU view chains used by eager OPs ('red' color - to be eagerized).
    for node in reverse_nodes_list:
        if node.meta["placement"] == "eager" or node.meta["pass_meta_color"] == "red":
            args = get_node_args(node)
            for arg in args:
                if arg.meta["placement"] == "hpu_cluster":
                    node_target = arg.target.__name__.split(".")[0]
                    if is_view_node(arg):  # noqa SIM114
                        arg.meta["pass_meta_color"] = "red"
                    # getitem is special-cased here since it may have view args and break the view ops chain
                    elif node_target == "getitem" and is_view_node(arg.args[0]):
                        arg.meta["pass_meta_color"] = "red"
    # Find HPU view chains used by eager OPs that are also used by non-eager HPU ops ('blue' color - to be cloned).
    for node in reverse_nodes_list:
        if node.meta["pass_meta_color"] == "red":
            found_hpu_dst = False
            for dst in node.users:
                if (dst.meta["placement"] == "hpu_cluster" and dst.meta["pass_meta_color"] != "red") or (
                    dst.meta["pass_meta_color"] == "blue"
                ):
                    found_hpu_dst = True
                    break

            if found_hpu_dst:
                node.meta["pass_meta_color"] = "blue"

    # Clone each 'blue' into uncolored part that is used by non-eager HPU only and into 'red' part that is only
    # used by eager chain.
    for node in reverse_nodes_list:
        if node.meta["pass_meta_color"] == "blue":
            # Clone the node along with all inputs edges.
            with ctx.graph_module.graph.inserting_before(node):
                new_node = ctx.graph_module.graph.create_node(
                    node.op, node.target, node.args, node.kwargs, node.name, node.type
                )
                new_node.meta = copy.copy(node.meta)

            # Move non-red (HPU path) edges to the new node.
            nodes_to_change = []
            for dst in node.users:
                if dst.meta["pass_meta_color"] != "red" and dst.meta["placement"] == "hpu_cluster":
                    nodes_to_change.append(dst)
            for dst in nodes_to_change:
                dst.replace_input_with(node, new_node)

            # Change original node color back into 'red'.
            node.meta["pass_meta_color"] = "red"

            # Remove color from new node.
            new_node.meta["pass_meta_color"] = "none"

    # Mark remaining 'red' nodes as eager. Also cleanup colors altogether.
    for node in reverse_nodes_list:
        assert node.meta["pass_meta_color"] != "blue"

        if node.meta["pass_meta_color"] == "red":
            graph_changed = True
            node.meta["placement"] = "eager"
            logger.debug(
                f"{node._pretty_print_target(node.target)} fellback to eager due to being identified as leaf view node"
            )
        del node.meta["pass_meta_color"]

    ctx.graph_module.graph.lint()
    ctx.graph_module.recompile()

    return graph_changed


def pass_detect_reusable_inputs_for_partition(ctx: OptimizerContext):
    """
    This pass goes through each node in the main module. For each HPU partition,
    we will check if it's input can be reused. If the input can be reused, then
    we will record this information in the partition node's meta field.
    """

    if not hpu_backend_config.enable_synapse_input_reuse:
        return False

    from torch.fx.node import Node, map_arg

    graph_inputs: list[Node] = []
    arg_to_last_user: dict[Node, Node] = {}
    user_to_last_used_args: dict[Node, list[Node]] = {}

    def register_last_uses(arg: Node, user: Node):
        if arg not in arg_to_last_user:
            arg_to_last_user[arg] = user
            user_to_last_used_args[user].append(arg)

    def is_graph_input(node: Node):
        return node.op == "placeholder" or (
            node.op == "call_function" and node.target == operator.getitem and node.args[0].op == "placeholder"
        )

    def is_inplaced_node(node: Node):
        return node.op == "call_function" and (
            node.target.__name__.split(".")[0].endswith("_") or node.target == torch.ops.hpu.weight_permutation
        )

    def is_shared_with_partition_input(node: Node):
        if node.op == "call_module":
            out_idx = 0
            submod_target = node.target
        elif node.op == "call_function" and node.target == operator.getitem and node.args[0].op == "call_module":
            out_idx = node.args[1]
            submod_target = node.args[0].target
        else:
            return False

        submod = ctx.graph_module.get_submodule(submod_target)
        if "in_to_out_dups" not in submod.meta:
            return False
        in_to_out_dups = submod.meta["in_to_out_dups"]
        out_to_in_dups = {v: k for k, v in in_to_out_dups.items()}

        if out_idx in out_to_in_dups:
            return True

        return False

    def not_share_mem_with_others(node: Node):
        # make sure the tensor doesn't have any alias to easy the algo and ensure safety
        # TODO: consider more complex situations, and refer to the alias check in reinplacer
        no_other_alias = not any((is_view_node(user) or is_inplaced_node(user)) for user in node.users.keys())
        not_an_alias = not (is_view_node(node) or is_inplaced_node(node) or is_shared_with_partition_input(node))
        return no_other_alias and not_an_alias

    def is_frozen(node: Node):
        return node.meta.get("frozen_param", False)

    for node in reversed(ctx.graph_module.graph.nodes):
        logger.debug("Node: %s Op: %s Target: %s", node, node.op, node.target)

        if is_graph_input(node):
            graph_inputs.append(node)

        user_to_last_used_args[node] = []
        map_arg(node.args, lambda arg: register_last_uses(arg, node))

    for user, last_used_args in user_to_last_used_args.items():
        if user.op != "call_module":
            continue

        is_reusables: list[bool] = []
        for arg in user.args:
            is_last_use = arg in last_used_args
            not_graph_input = arg not in graph_inputs
            no_share_mem = not_share_mem_with_others(arg)
            not_frozen = not is_frozen(arg)

            reusable = is_last_use and not_graph_input and no_share_mem and not_frozen
            is_reusables.append(reusable)
        submod = ctx.graph_module.get_submodule(user.target)
        submod.meta["is_reusables"] = is_reusables
        logger.debug(f"Partition {user.target} has reusable input information: {is_reusables}")

    return True


def pass_stable_topological_sort(ctx: OptimizerContext):
    """
    This pass is supposed to run stable topological sort on the graph.
    """
    graph = ctx.graph_module.graph
    stable_topological_sort(graph)

    ctx.graph_module.graph.lint()
    ctx.graph_module.recompile()

    return True


def pass_compile_clusters(ctx: OptimizerContext):
    """
    This pass goes through each node in the main module. For each generated HPU cluster
    there will be "call_module" OP. For each such module create JIT IR and pass
    it to the HPU backend for recipe compilation and substitute the target with
    newly compiled one.
    """
    if ctx.use_jit_fork:
        return pass_compile_clusters_jit_fork_version(ctx)

    def generate_jit_ir_from_module(input_module: torch.fx.GraphModule):
        """
        This function generate JIT IR for specified graph module.
        """

        import copy

        from torch._functorch.compile_utils import strip_overloads
        from torch._functorch.compilers import _disable_jit_autocast

        module = copy.deepcopy(input_module)
        has_random_ops = wrap_random_ops(module)
        remove_duplicated_outputs(module)
        remove_no_effect_inplace_add(module)

        with _disable_jit_autocast():
            strip_overloads(module)

            for node in module.graph.nodes:
                new_kwargs = {}
                for k, v in node.kwargs.items():
                    if isinstance(v, torch.device):
                        v = v.type
                    new_kwargs[k] = v
                node.kwargs = new_kwargs

            module.graph.lint()
            module.recompile()

            # Strip hooks because they break jit.script functionality (habana
            # integration wraps every module with some hooks).
            from collections import OrderedDict

            saved_forward_hooks = module._forward_hooks
            saved_pre_forward_hooks = module._forward_pre_hooks
            module._forward_hooks = OrderedDict()
            module._forward_pre_hooks = OrderedDict()

            f = torch.jit.script(module)

            module._forward_hooks = saved_forward_hooks
            module._forward_pre_hooks = saved_pre_forward_hooks

            torch._C._jit_pass_remove_mutation(f.graph)

        logger.debug(
            "####PyTorch-generated JIT IR graph for this HPU graph:####\n%s",
            f.graph,
        )

        return f, module, has_random_ops

    num_subgraphs = 0
    refine_dynamic = bc.get_pt_hpu_enable_refine_dynamic_shapes()
    optim_output_sif_ds = bc.get_pt_hpu_optim_dynamic_output_sif()
    for n in ctx.graph_module.graph.nodes:
        logger.debug("Node: %s Op: %s Target: %s", n, n.op, n.target)

        if n.op == "call_module":
            assert not n.kwargs

            submod = ctx.graph_module.get_submodule(n.target)

            is_reusables: list[bool] = []
            if "is_reusables" in submod.meta:
                is_reusables = submod.meta["is_reusables"]

            jit_ir_function, submod_updated, has_random_ops = generate_jit_ir_from_module(submod)
            jit_node_annotation_propagation(jit_ir_function, submod_updated)

            is_submod_dynamic = is_module_dynamic(submod)
            if not hpu_backend_config.force_static_compile:
                if refine_dynamic:
                    is_submod_dynamic = is_submod_dynamic or get_dynamic_config_value()

                if is_submod_dynamic and optim_output_sif_ds:
                    jit_node_shape_propagation(jit_ir_function, submod_updated)

            callable_recipe = get_callable_recipe(
                jit_ir_function,
                submod,
                ctx.graph_name,
                is_training=ctx.is_training,
                is_dynamic=is_submod_dynamic,
                is_reusables=is_reusables,
                has_random_ops=has_random_ops,
            )

            ctx.graph_module.delete_submodule(n.target)
            ctx.graph_module.add_submodule(n.target, callable_recipe)

            num_subgraphs += 1

    logger.info("INFO: Number of subgraphs created:\n%s", num_subgraphs)

    return num_subgraphs != 0


def pass_summarize_graph(ctx: OptimizerContext):
    """
    This pass is just for debug.
    In case any FxGraphAnalyzer contexts are registered it counts ops occurring in FX Graph.
    """
    assert ctx.stage == OptimizationPassPlacement.POST_PARTITIONER
    assert ctx.graph_module is not None
    if not FxGraphAnalyzer.registered_contexts:
        return False

    for debug_context in FxGraphAnalyzer.registered_contexts.values():
        debug_context.count_ops(ctx.graph_module.graph.nodes, ctx)

    return False


inplaceable_ops = {}

try:
    c10d_functional = torch.ops._c10d_functional
    inplaceable_collective_ops = {
        c10d_functional.all_reduce.default: InplaceableOp(c10d_functional.all_reduce_.default, 0),
        c10d_functional.all_reduce_coalesced.default: InplaceableOp(c10d_functional.all_reduce_coalesced_.default, 0),
    }
    inplaceable_ops.update(inplaceable_collective_ops)
except AttributeError:
    # _c10d_functional ops are only available when torch
    # is built with USE_DISTRIBUTED=1.
    pass


def pass_reinplace_inplaceable_ops(ctx: OptimizerContext) -> bool:
    """
    This pass tries to replace the usage of out of place variant with the
    inplace variant of the collective op. This matches a particular variant
    of the collective where all_reduce->wait_tensor->copy is present and
    the output of copy is the same view as allreduce then the combination
    is replace with all_reduce_ which is an inplace variant of collective
    """
    graph_changed = False
    if not hpu_backend_config.use_inplace_allreduce or hpu_backend_config.use_generic_reinplacer:
        return graph_changed

    def reinplace_collective_ops(gm: torch.fx.GraphModule):
        replace_dict: dict[torch.fx.Node, torch.fx.Node] = {}

        for node in gm.graph.nodes:
            if (inplaceable_op := inplaceable_ops.get(node.target, None)) is not None:
                mutated_arg = node.args[inplaceable_op.mutated_arg]
                node_users = list(node.users)
                if len(node_users) == 1 and node_users[0].target == torch.ops._c10d_functional.wait_tensor.default:
                    wait_tensor_node = node_users[0]
                    wait_tensor_node_users = list(wait_tensor_node.users)
                    if (
                        len(wait_tensor_node_users) == 1
                        and wait_tensor_node_users[0].target == torch.ops.aten.copy.default
                    ):

                        copy_node = wait_tensor_node_users[0]
                        if copy_node.args[0] == mutated_arg:
                            replace_dict[copy_node] = copy_node.args[1]
                            node.target = inplaceable_op.inplace_op
                            graph_changed = True

        for node, replacement in replace_dict.items():
            while replacement in replace_dict:
                replacement = replace_dict[replacement]
            replace_dict[node] = replacement
            node.replace_all_uses_with(replacement)
            gm.graph.erase_node(node)

        gm.recompile()

    reinplace_collective_ops(ctx.graph_module)

    return graph_changed


def pass_reinplace_inplaceable_ops_v2(ctx: OptimizerContext) -> bool:
    """
    Note: this pass must be called after pass_fake_propagation, since it relies
    on the node.meta["val"] information.
    """
    if not hpu_backend_config.use_generic_reinplacer:
        return False

    from ._passes.reinplace import reinplace_inplaceable_ops

    graph_changed = reinplace_inplaceable_ops(ctx.graph_module.graph)
    if graph_changed:
        ctx.graph_module.recompile()

    return graph_changed


def pass_reinplace_index_copy_ops(ctx: OptimizerContext) -> bool:
    """
    This pass tries to replace the usage of out of place variant with the
    inplace variant of the index_copy op. This matches a particular variant
    of the index_copy where index_copy->copy_ is present, then the combination
    is replace with index_copy_ which is an inplace variant of index_copy
    """
    graph_changed = False
    if not hpu_backend_config.use_inplace_index_copy or hpu_backend_config.use_generic_reinplacer:
        return graph_changed

    graph = ctx.graph_module.graph

    def has_any_eager_users(node: torch.fx.Node):
        user_nodes = list(node.users.keys())
        for user_node in user_nodes:
            if user_node.meta.get("placement", "") == "eager":
                return True
        return False

    def reinplace_index_copy_ops(gm: torch.fx.GraphModule):
        inplaceable_index_copy_ops = {
            torch.ops.aten.index_copy.default: InplaceableOp(torch.ops.aten.index_copy_.default, 0),
        }

        replace_dict: dict[torch.fx.Node, torch.fx.Node] = {}

        for node in gm.graph.nodes:
            if (inplaceable_op := inplaceable_index_copy_ops.get(node.target, None)) is not None:
                mutated_arg = node.args[inplaceable_op.mutated_arg]
                mutated_arg_users = list(mutated_arg.users)
                if (
                    len(mutated_arg_users) == 2
                    and (
                        mutated_arg_users[0].target == torch.ops.aten.copy_.default
                        or mutated_arg_users[1].target == torch.ops.aten.copy_.default
                    )
                    and not (mutated_arg.op == "call_function" and is_view_node(mutated_arg))
                    and not has_any_eager_users(node)  # index_copy_ output can't be the partition output
                ):
                    # the mutated arg is only used by one index_copy op and one
                    # copy_ op, and it's not a view tensor
                    inplace_copy_node = (
                        mutated_arg_users[0]
                        if mutated_arg_users[0].target == torch.ops.aten.copy_.default
                        else mutated_arg_users[1]
                    )
                    # modify index_copy to index_copy_ directly
                    node.target = inplaceable_op.inplace_op
                    # connect copy_'s uses to index_copy_
                    replace_dict[inplace_copy_node] = node

        for node, replacement in replace_dict.items():
            node.replace_all_uses_with(replacement)
            gm.graph.erase_node(node)

        gm.recompile()

    reinplace_index_copy_ops(ctx.graph_module)

    return graph_changed


def pass_reinplace_add_ops(ctx: OptimizerContext):
    """
    In this pass, we will reinplace all possible out-of-place
    torch.ops.aten.add.Tensor ops, to optimize the memory consumption.
    """
    if not hpu_backend_config.reinplace_add or hpu_backend_config.use_generic_reinplacer:
        return False

    graph_changed = False

    def is_eligible_add_node(node: torch.fx.Node):
        def is_view_op(_node: torch.fx.Node):
            return _node.op == "call_function" and is_view_node(_node)

        is_add = node.op == "call_function" and node.target == torch.ops.aten.add.Tensor
        if not is_add:
            return False

        src0, src1 = node.args[0], node.args[1]
        is_add_two_tensors = type(src0) is torch.fx.Node and type(src1) is torch.fx.Node
        is_float_dtype = (
            is_add_two_tensors
            and src0.meta["output_dtypes"][0] == src1.meta["output_dtypes"][0]
            and src0.meta["output_dtypes"][0] in (torch.float32, torch.bfloat16, torch.float16)
        )
        is_eligible = is_float_dtype and src0.op != "placeholder" and not is_view_op(src0)
        if not is_eligible:
            return False

        # add must be the last user of its src0
        src0_users = list(src0.users.keys())
        is_eligible = (
            is_eligible
            and not any(user > node for user in src0_users)
            and not any(is_view_op(user) for user in src0_users)
        )

        return is_eligible

    for node in ctx.graph_module.graph.nodes:
        if not is_eligible_add_node(node):
            continue

        node.target = torch.ops.aten.add_.Tensor
        graph_changed = True

    return graph_changed


def pass_detect_partition_in_to_out_duplicates(ctx: OptimizerContext):
    """
    This pass will detect the duplicated inputs outputs caused by inplace ops
    inside partition.

    Note: Some ops whose name ends with "__" (double underscore), like
    __rshift__, __lshift__,  are not inplaced, we filter them out
    """

    # currently, we only consider the duplications caused by inplace op.
    def detect_in_to_out_duplicates(graph_module: torch.fx.GraphModule):
        in_nodes = []
        for node in graph_module.graph.nodes:
            if node.op == "placeholder":
                in_nodes.append(node)

        in_to_out_dups = {}
        visited = set()
        for in_idx, in_node in enumerate(in_nodes):
            queue = [in_node]
            while queue:
                current = queue.pop(0)
                visited.add(current)
                for user_node in current.users:
                    if user_node in visited:
                        continue
                    elif (
                        user_node.op == "call_function"
                        and user_node.target.__name__.split(".")[0].endswith("_")
                        and not user_node.target.__name__.split(".")[0].endswith("__")
                        and user_node.args[0] == current
                    ):
                        # inplace op, and current op is the mutable arg (we
                        # assume mutable arg is always the arg0)
                        queue.append(user_node)
                    elif user_node.op == "output":
                        outs = list(user_node.args[0]) if type(user_node.args[0]) is tuple else [user_node.args[0]]
                        for out_idx, out in enumerate(outs):
                            if out == current:
                                in_to_out_dups[in_idx] = out_idx
        return in_to_out_dups

    changed = False
    for n in ctx.graph_module.graph.nodes:
        logger.debug("Node: %s Op: %s Target: %s", n, n.op, n.target)

        if n.op == "call_module":
            assert not n.kwargs
            submod = ctx.graph_module.get_submodule(n.target)
            in_to_out_dups = detect_in_to_out_duplicates(submod)
            if len(in_to_out_dups) > 0:
                submod.meta["in_to_out_dups"] = in_to_out_dups
                changed = True
    return changed


def pass_remove_unnecessary_full_copy(ctx: OptimizerContext):
    """
    The following pattern is quite redudent:
    def fowrard():
        a = op0(xxx)
        full = torch.ops.full.default(yyy)
        b = full.copy(a)
        return b
    We can match such pattern and transform them to:
    def fowrard():
        a = op0(xxx)
        return a
    """
    to_remove = []
    for node in ctx.graph_module.graph.nodes:
        matched, full_node, copy_node = match_full_copy_pattern(node)
        if not matched:
            continue

        def match(lhs, rhs) -> bool:
            return lhs is not None and rhs is not None and lhs == rhs

        copy_args = list(copy_node.args)
        dst, src = copy_args[0], copy_args[1]
        if not (
            match(dst.meta["output_device"], src.meta["output_device"])
            and match(dst.meta["output_shapes"][0], src.meta["output_shapes"][0])
            and match(dst.meta["output_dtypes"][0], src.meta["output_dtypes"][0])
            and match(dst.meta["output_layouts"][0], src.meta["output_layouts"][0])
            and match(dst.meta["output_strides"][0], src.meta["output_strides"][0])
            and match(dst.meta["output_contiguous"][0], src.meta["output_contiguous"][0])
        ):
            continue

        copy_src_node = copy_args[1]
        copy_node.replace_all_uses_with(copy_src_node)
        to_remove.append(copy_node)
        to_remove.append(full_node)

    for node in to_remove:
        ctx.graph_module.graph.erase_node(node)

    graph_changed = len(to_remove) > 0
    return graph_changed


def pass_check_eager_fallbacks(ctx: OptimizerContext):
    """
    This pass is for testing purposes with use of PT_HPU_USE_EAGER_FALLBACK=0.
    It goes through nodes in graph and in case any ops fall to eager
    while PT_HPU_USE_EAGER_FALLBACK env variable is set to 0
    it throws an assertion error
    """
    assert ctx.stage == OptimizationPassPlacement.POST_PARTITIONER
    assert ctx.graph_module is not None
    if not hpu_backend_config.use_eager_fallback:
        eager_nodes = []
        for node in ctx.graph_module.graph.nodes:
            if (
                node.op in {"call_function", "call_method"}
                and node._pretty_print_target(node.target)
                not in {
                    "operator.getitem",
                    "habana_frameworks.torch.dynamo.compile_backend.symbolic_execution.symexpr_python",
                }
                and node._pretty_print_target(node.target) not in host_call_functions
            ):
                if node.meta["placement"] == "eager":
                    eager_nodes.append(str(node) + ":" + node._pretty_print_target(node.target))
        assert len(eager_nodes) == 0, f"Eager fallback in nodes: {eager_nodes}"
    return False


def pass_inference_fuse_linear(ctx: OptimizerContext) -> bool:
    """
    Runs iff inference mode is set for the input GraphModule
    This pass goes through the input GraphModule and fuses all instances of
    t + mm or t + addmm back to linear. It also removes redundant reshapes added
    for the t + mm or t + addmm pattern. It returns a status indicating if the
    module changed
    """
    graph_changed = False

    if ctx.is_training or ctx.is_backward:
        return graph_changed

    for node in ctx.graph_module.graph.nodes:
        if (
            node.op != "call_function"
            or
            # aten.t is decomposed into aten.transpose.int
            (node.target != torch.ops.aten.transpose.int or str(node.meta.get("original_aten", "")) != "aten.t.default")
            or not is_node_supported(node=node)
        ):
            continue
        to_remove = []
        for u in node.users:
            if u.op != "call_function" or not is_node_supported(node=u):
                break
            if u.target == torch.ops.aten.addmm.default:
                # transpose should be addmm's third input to be fused to linear
                if len(u.args) < 3 or node is not u.args[2]:
                    break
                bias, inp, _ = list(u.args)
                weight = list(node.args)[0]
                new_args = (inp, weight, bias)
            elif u.target == torch.ops.aten.mm.default:
                # transpose should be mm's second input to be fused to linear
                if len(u.args) < 2 or node is not u.args[1]:
                    break
                inp, _ = list(u.args)
                weight = list(node.args)[0]
                new_args = (inp, weight)
            else:
                continue

            graph_changed = True
            new_op = torch.ops.aten.linear
            with ctx.graph_module.graph.inserting_after(u):
                new_node = ctx.graph_module.graph.create_node(
                    "call_function",
                    new_op,
                    args=new_args,
                    kwargs=u.kwargs,
                )
                u.replace_all_uses_with(new_node, propagate_meta=True)
                to_remove.append(u)
        for u in to_remove:
            ctx.graph_module.graph.erase_node(u)

    if not graph_changed:
        return graph_changed

    ctx.graph_module = post_pass_finalize(input_module=ctx.graph_module)

    """
    The following sub-graph rewriter removes the redundant reshapes that are added
    by aot autograd as part of lowering linear to t + mm/addmm as the above rewriter
    has replaced the pattern with linear
    """
    for node in ctx.graph_module.graph.nodes:
        if node.op != "call_function" or node.target != torch.ops.aten.linear or not is_node_supported(node=node):
            continue
        before = node.args[0]
        after = next(iter(node.users))
        cond_after = False
        if len(node.users) == 1 and after.target == torch.ops.aten.view.default and is_node_supported(after):
            cond_after = True
        cond_before = False
        if len(before.users) == 1 and before.target == torch.ops.aten.view.default and is_node_supported(before):
            cond_before = True

        """
        After replaced with Linear op if the subgraph looks like

        view_1
          |
        linear
          |
        view_2

        we will check view_1 input tensor and view_2 output tensor rank is same or not,
        if it is same we will also check except last dim all the dims are same or not.
        if it is not same we are discarding this pattern matching using the below checks.
        """
        if cond_after and cond_before:
            if (len(before.args[0].meta["output_shapes"][0]) == len(after.meta["output_shapes"][0])) and (
                before.args[0].meta["output_shapes"][0][:-1] == after.meta["output_shapes"][0][:-1]
            ):
                real_input = before.args[0]
                new_args = list(node.args)
                new_args[0] = real_input
                node.args = tuple(new_args)
                after.replace_all_uses_with(node)
                node.meta.update(after.meta)

    ctx.graph_module = post_pass_finalize(input_module=ctx.graph_module)

    return graph_changed


# Note: This pass must run after any passes that rely on pass_fake_propagation,
# because it will make bmm op consume non-3D input tensors and cause "batch1
# must be a 3D tensor" error.
def pass_remove_unnecessary_bmm_view(ctx: OptimizerContext):
    """
    This pass remove the view ops which are the neightbors of bmm op
    In fx graph, bmm args will change to 4d tensor but output tensor shape will show as
    3d, internally in synapse we are handling and creating 4d tensor[ref jira: SW-191200]
    we can easily understand to see the fx graph below.
    before :
    def forward(self, arg0_1: "bf16[2, 3, 5, 4]", arg1_1: "bf16[2, 3, 4, 5]"):
         # File: /home/pradas/qnpu/pt/src/pytorch-integration/tests/pytest_working/any_mode/test_hpu_mytest.py:1873 in matmul_4d, code: return torch.matmul(A, B)* 0.2
        expand: "bf16[2, 3, 4, 5]" = torch.ops.aten.expand.default(arg1_1, [2, 3, 4, 5]);  arg1_1 = None
        view: "bf16[6, 4, 5]" = torch.ops.aten.view.default(expand, [6, 4, 5]);  expand = None
        expand_1: "bf16[2, 3, 5, 4]" = torch.ops.aten.expand.default(arg0_1, [2, 3, 5, 4]);  arg0_1 = None
        view_1: "bf16[6, 5, 4]" = torch.ops.aten.view.default(expand_1, [6, 5, 4]);  expand_1 = None
        bmm: "bf16[6, 4, 4]" = torch.ops.aten.bmm.default(view, view_1);  view = view_1 = None
        view_2: "bf16[2, 3, 4, 4]" = torch.ops.aten.view.default(bmm, [2, 3, 4, 4]);  bmm = None
        mul: "bf16[2, 3, 4, 4]" = torch.ops.aten.mul.Tensor(view_2, 0.2);  view_2 = None
        return (mul,)

    after :
    def forward(self, arg0_1: "bf16[2, 3, 5, 4]", arg1_1: "bf16[2, 3, 4, 5]"):
         # File: /home/pradas/qnpu/pt/src/pytorch-integration/tests/pytest_working/any_mode/test_hpu_mytest.py:1873 in matmul_4d, code: return torch.matmul(A, B)* 0.2
        bmm: "bf16[6, 4, 4]" = torch.ops.aten.bmm.default(arg1_1, arg0_1);  arg1_1 = arg0_1 = None
        mul: "bf16[2, 3, 4, 4]" = torch.ops.aten.mul.Tensor(bmm, 0.2);  bmm = None
        return (mul,)
    So here, though the bmm output is 3d but mul is working on 4d tensor and give the result as 4d.
    Note: It is really dangerous, we need to be very carefull about this pass.
    """

    def is_view_node(node):
        view_ops = {torch.ops.aten.view.default, torch.ops.aten._unsafe_view.default}
        return node.target in view_ops

    def is_permute_node(node):
        permute_ops = {torch.ops.aten.permute.default}
        return node.target in permute_ops

    def get_node_dim(node):
        tensor_meta = node.meta.get("tensor_meta", None)
        if tensor_meta:
            return len(tensor_meta.shape)
        else:
            return None

    graph_changed = False

    for node in ctx.graph_module.graph.nodes:
        if node.op == "call_function" and node.target == torch.ops.aten.bmm.default:
            bmm_input_left, bmm_input_right = node.args
            bmm_output = list(node.users.keys())[0]

            if all(map(is_view_node, [bmm_input_left, bmm_input_right, bmm_output])):
                neighbor_nodes_for_view = (
                    bmm_input_left.all_input_nodes + bmm_input_right.all_input_nodes + list(bmm_output.users.keys())
                )
                if all(map(is_permute_node, neighbor_nodes_for_view)):
                    """
                    here we are checking the view nodes `args` which appears before the bmm and view nodes `outputs` which appears after
                    the bmm are `permute op` or not, if yes we simply skip this bmm node as this is unsafe.
                    for example below fx graph should not be impacted for this pass:
                    def forward(self, arg0_1: "bf16[1, 128, 108, 1, 108]", arg1_1: "bf16[1, 1, 1, 512, 108]"):
                        permute_2: "bf16[128, 108, 108, 1, 1]" = torch.ops.aten.permute.default(arg0_1, [1, 2, 4, 0, 3]);  permute = None
                        view: "bf16[1, 13824, 108]" = torch.ops.aten.view.default(permute_2, [1, 13824, 108]);  permute_2 = None
                        permute_3: "bf16[108, 1, 512, 1, 1]" = torch.ops.aten.permute.default(arg1_1, [4, 0, 3, 1, 2]);  permute_1 = None
                        view_1: "bf16[1, 108, 512]" = torch.ops.aten.view.default(permute_3, [1, 108, 512]);  permute_3 = None
                        bmm: "bf16[1, 13824, 512]" = torch.ops.aten.bmm.default(view, view_1);  view = view_1 = None
                        view_2: "bf16[128, 108, 1, 1, 512]" = torch.ops.aten.view.default(bmm, [128, 108, 1, 1, 512]);  bmm = None
                        permute_4: "bf16[1, 128, 108, 512, 1]" = torch.ops.aten.permute.default(view_2, [3, 0, 1, 4, 2]);  view_2 = None
                        return (permute_4,)
                    """
                    continue

                left_dim = get_node_dim(bmm_input_left.args[0])
                right_dim = get_node_dim(bmm_input_right.args[0])

                if left_dim in {4, 5} and left_dim == right_dim:
                    node.replace_input_with(bmm_input_left, bmm_input_left.args[0])
                    node.replace_input_with(bmm_input_right, bmm_input_right.args[0])

                    for bmm_output_user in list(bmm_output.users.keys()):
                        bmm_output_user.replace_input_with(bmm_output, node)

                    if "output_shapes" in node.meta:
                        node.meta["output_shapes"] = bmm_output.meta.get("output_shapes", None)
                        node.meta["output_strides"] = bmm_output.meta.get("output_strides", None)
                        fill_propagated_tensor_metadata_jitfork(node)
                    else:
                        logger.warn("There is no 'output_shapes' for bmm nodes")

                    graph_changed = True

    if graph_changed:
        logger.debug("####### Removed unnecessary bmm view nodes")
        ctx.graph_module.graph.eliminate_dead_code()
        ctx.graph_module.graph.lint()
        ctx.graph_module.recompile()

    return graph_changed


def pass_remove_unnecessary_expand(ctx: OptimizerContext):
    def get_node_shape(node):
        shape = None
        if isinstance(node, torch.fx.Node):
            tensor_meta = node.meta.get("val", node.meta.get("tensor_meta"))
            if tensor_meta is not None:
                if isinstance(tensor_meta, torch.Tensor):
                    shape = tensor_meta.shape
                elif isinstance(tensor_meta, py_sym_types):
                    shape = tensor_meta
        elif isinstance(node, int):
            shape = node
        return shape

    graph_changed = False

    for node in ctx.graph_module.graph.nodes:
        if node.op == "call_function" and node.target == torch.ops.aten.expand.default:
            input_node, target_shape_params = node.args
            input_shape = get_node_shape(input_node)
            target_shape = []
            for dim in target_shape_params:
                target_shape.append(get_node_shape(dim))

            if input_shape and list(input_shape) == target_shape:
                for user in list(node.users.keys()):
                    user.replace_input_with(node, input_node)
                graph_changed = True

    if graph_changed:
        logger.debug("####### Removed unnecessary expand nodes")
        ctx.graph_module.graph.eliminate_dead_code()
        ctx.graph_module.graph.lint()
        ctx.graph_module.recompile()


def pass_make_boxed_graph(ctx: OptimizerContext) -> bool:
    """
    This pass converts the graph inputs to a single list. So that we are able to
    clear the elements in the list to free inputs memory sooner.
    """

    # submodules are not called in boxed convention, so we don't make boxed graph for them.
    if ctx.is_submod or not hpu_backend_config.use_boxed_input:
        return False

    list_placeholder = None
    list_getitems = []

    # Step 1: make the graph input boxed, which is converting the non-list
    # inputs to a single list
    orig_inputs = []
    for node in ctx.graph_module.graph.nodes:
        if node.op == "placeholder":
            orig_inputs.append(node)

    with ctx.graph_module.graph.inserting_before():
        list_placeholder = ctx.graph_module.graph.placeholder("input_list", type_expr=list)
        list_placeholder.meta["output_device"] = torch.device("hpu")

    for i, orig_input in enumerate(orig_inputs):
        with ctx.graph_module.graph.inserting_after(orig_input):
            list_getitem = ctx.graph_module.graph.call_function(operator.getitem, args=(list_placeholder, i), kwargs={})
            list_getitem.meta = copy.copy(orig_input.meta)
            orig_input.replace_all_uses_with(list_getitem)
            list_getitems.append(list_getitem)
        ctx.graph_module.graph.erase_node(orig_input)

    # Step 2: insert the list clear op after the last getitem. If the graph
    # doesn't have any placeholder, the list_getitems length will be 0 and we
    # don't need to clear the input list.
    if len(list_getitems) > 0:
        last_getitem = list_getitems[-1]
        with ctx.graph_module.graph.inserting_after(last_getitem):
            list_clear = ctx.graph_module.graph.call_function(lambda x: x.clear(), args=(list_placeholder,), kwargs={})
            list_clear.meta["output_device"] = torch.device("cpu")

    ctx.graph_module.graph.lint()
    ctx.graph_module.recompile()
    return True


def pass_fix_arange_device(ctx: OptimizerContext):
    """
    Eager supports:

        aten.index(hpu_tensor, torch.arange(..., device="cpu"))

    But this results in an implicit host-device-copy and breaks graphs. Rewrite the arange to use hpu.
    Refer the fx graph without and with this pass for the test case at following location:

    Test File: ~/qnpu/pt/src/pytorch-integration/tests/pytest_working/compile/test_passes_fix_arange_device.py
    Before (without this pass):
    def forward(self, arg0_1: "f32[64, 64]"):
        # File: ~/qnpu/pt/src/pytorch-integration/tests/pytest_working/compile/test_passes_fix_arange_device.py:34 in func, code: return x[torch.arange(32)]
        arange: "i64[32]" = torch.ops.aten.arange.start_step(0, 32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        index: "f32[32, 64]" = torch.ops.aten.index.Tensor(arg0_1, [arange]);  arg0_1 = arange = None
        return (index,)

    After (with this pass):
    def forward(self, arg0_1: "f32[64, 64]"):
        # File: ~/qnpu/pt/src/pytorch-integration/tests/pytest_working/compile/test_passes_fix_arange_device.py:34 in func, code: return x[torch.arange(32)]
        arange_start_step: "i64[32]" = torch.ops.aten.arange.start_step(0, 32, layout = torch.strided, device = device(type='hpu', index=0), pin_memory = False)
        index: "f32[32, 64]" = torch.ops.aten.index.Tensor(arg0_1, [arange_start_step]);  arg0_1 = arange_start_step = None
        return index
    """
    replace_node_list = []
    for node in ctx.graph_module.graph.nodes:
        if node.op == "call_function" and node.target in (
            torch.ops.aten.arange,
            torch.ops.aten.arange.start,
            torch.ops.aten.arange.start_step,
        ):
            replace_node_list.append(node)

    graph_changed = False
    for node in replace_node_list:
        valid_node = True
        user_devices: set[torch.device] = set()
        for user in node.users:
            if (
                user.op == "call_function"
                and user.target in (torch.ops.aten.index.Tensor, torch.ops.aten.index_put.default)
                and hasattr(user.meta.get("val"), "device")
            ):
                user_devices.add(user.meta["val"].device)  # type: ignore[union-attr]
            else:
                valid_node = False
                break  # bail out

        if valid_node and len(user_devices) == 1 and "val" in node.meta:
            node_device = node.meta["val"].device
            (user_device,) = user_devices
            if node_device.type != user_device.type:
                repl_kwargs = {}
                for k, v in node.kwargs.items():
                    repl_kwargs[k] = v
                repl_kwargs["device"] = user_device

                with ctx.graph_module.graph.inserting_before(node):
                    repl = ctx.graph_module.graph.call_function(
                        node.target,
                        node.args,
                        repl_kwargs,
                    )
                    repl.meta.update(node.meta)
                    repl.meta["val"] = repl.meta["val"].to(user_device)
                    repl.meta["output_device"] = user_device
                    repl.val_args = node.args
                    repl.val_kwargs = repl_kwargs
                    node.replace_all_uses_with(repl)

                ctx.graph_module.graph.erase_node(node)
                graph_changed = True

    if graph_changed:
        logger.debug("####### Pass to fix arange op device")
        ctx.graph_module.graph.lint()
        ctx.graph_module.recompile()
