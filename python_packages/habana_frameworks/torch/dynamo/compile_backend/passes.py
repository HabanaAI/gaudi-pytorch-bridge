###############################################################################
#
#  Copyright (c) 2021-2024 Intel Corporation
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

import contextlib
import copy
import operator
import os
import queue
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

import habana_frameworks.torch.internal.bridge_config as bc
import torch
import torch.fx
from habana_frameworks.torch.dynamo.compile_backend import config as hpu_backend_config
from habana_frameworks.torch.dynamo.debug_utils.logger import get_compile_backend_logger
from habana_frameworks.torch.dynamo.debug_utils.visualization.graph_dumping import dump_fx_graph
from habana_frameworks.torch.utils.debug.dynamo_utils import FxGraphAnalyzer
from habana_frameworks.torch.utils.internal import Timer
from packaging.version import Version
from torch.fx.experimental.proxy_tensor import py_sym_types
from torch.fx.node import map_arg
from torch.fx.passes.operator_support import OperatorSupport

from ._helpers import *
from ._passes.fuse_allreduce_calls import pass_fuse_collectives
from ._passes.fuse_view_chains import pass_fuse_view_chains
from ._passes.pattern_rewriter import pass_pattern_rewriter
from ._passes.propose_collective_blocks import pass_propose_collective_blocks
from ._passes.utils import ColorGraph, OptimizationPassPlacement, OptimizerContext, SchedulePolicy
from .partitioner import HabanaPartitioner
from .random_utils import (
    backward_random_op_inputs,
    is_backward_checkpoint_op,
    is_multi_output_op,
    is_random_op,
    random_op_inputs,
)
from .recipe_compiler import get_callable_recipe
from .shared_layer import is_eager_fallback_required
from .symbolic_execution import HPUExprPrinter, SymExprNodeManager, substitute_sympyfn, sympify_expression

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
            pass_propose_collective_blocks,
            pass_fuse_collectives,
            pass_allreduce_parents,
            pass_pattern_rewriter,
            pass_fake_propagation,
            pass_weight_permutation,
            pass_remove_unnecessary_full_copy,
            pass_remove_unnecessary_expand,
            pass_remove_unnecessary_bmm_view,
            pass_wa_mixed_devices,  # This is W/A for Adam having CPU scalar tensors parameters.
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
        if Version(Version(torch.__version__).base_version) < Version("2.4.0"):
            passes.insert(0, pass_handle_view_before_inplace_compute_ops)

        return passes
    elif stage == OptimizationPassPlacement.PARTITIONER:
        return [
            pass_graph_print,
            # These passes will prepare proper placement for some corner-cases.
            pass_propose_partitions,
            pass_post_process_partitions,
            pass_merge_paths,
            # This is final pass that creates final submoduled graph.
            pass_fuse_partitions,
            pass_reorder_allreduce,
            pass_make_symints_available,
            pass_fuse_view_chains,
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
            pass_compile_clusters,
            pass_make_boxed_graph,
        ]
    else:
        logger.error("unknown optimization stage %s", stage)
        raise


class FusedCollectiveOperatorSupport(OperatorSupport):
    def is_node_supported(self, submodules: Mapping[str, torch.nn.Module], node: torch.fx.Node) -> bool:
        if (
            "downstream_allreduce_name" in node.meta
            and node.meta["downstream_allreduce_name"] == self.allreduce_name
            and "partition_assigned" not in node.meta
            and node.meta["placement"] == "hpu_cluster"
        ):
            node.meta["partition_assigned"] = "true"
            return True
        return False


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
                        setattr(previous_node, "parent", downstream_allreduce_name)
                        new_previous_nodes.extend(previous_node.all_input_nodes)
                previous_nodes = new_previous_nodes
                new_previous_nodes = []

        return len(allreduces) > 0
    return False


def pass_reorder_allreduce(ctx: OptimizerContext) -> bool:
    if hpu_backend_config.enable_allreduce_graph_split:
        graph = ctx.graph_module.graph
        allreduces = [n for n in graph.nodes if n.name.startswith("all_reduce")]
        graph_changed = False
        for allreduce in allreduces:
            upstream_nodes = allreduce.all_input_nodes
            nodes_to_move = [allreduce]
            while len(upstream_nodes) > 0:
                new_upstream_nodes = []
                for upstream_node in upstream_nodes:
                    if not upstream_node.name.startswith("fused"):
                        new_upstream_nodes.extend(upstream_node.all_input_nodes)
                        nodes_to_move.append(upstream_node)
                    else:
                        fused = upstream_node
                upstream_nodes = new_upstream_nodes

            for node in nodes_to_move:
                fused.append(node)

            if len(nodes_to_move) > 0:
                graph_changed = True

        waittensors = [n for n in graph.nodes if n.name.startswith("wait_tensor")]
        for waittensor in waittensors:
            downstream_nodes = list(waittensor.users.keys())

            # if the wait_tensor has no users. move it to after the
            # corresponding collective node
            if len(downstream_nodes) == 0:
                producer = waittensor.all_input_nodes[0]
                producer.append(waittensor)
                graph_changed = True
                continue

            fused = None

            nodes_to_move = [waittensor]
            while len(downstream_nodes) > 0:
                new_downstream_nodes = []
                for downstream_node in downstream_nodes:
                    if not downstream_node.name.startswith("fused"):
                        new_downstream_nodes.extend(list(downstream_node.users.keys()))
                        nodes_to_move.append(downstream_node)
                    else:
                        fused = downstream_node
                downstream_nodes = new_downstream_nodes

            if fused is None:
                continue

            for node in nodes_to_move:
                fused.prepend(node)

            if len(nodes_to_move) > 0:
                graph_changed = True

        if graph_changed:
            ctx.graph_module.recompile()

        return graph_changed
    return False


@dataclass(frozen=True)
class InplaceableOp:
    inplace_op: Callable[..., Any]
    mutated_arg: int
    extra_check: Callable[[torch.fx.Node], bool] = lambda node: True


def _is_cpu_scalar_copy_required(node: torch.fx.Node, node_arg: torch.fx.Node) -> bool:
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
        if node_arg.type in [int, float] and node_target in scalar_ops:
            assert node_arg.meta["output_device"] == torch.device("cpu")
            copy_required = False
    return copy_required


def _is_cpu_scalar_or_symbolic_scalar(node: torch.fx.Node) -> bool:
    if node.type in [int, float]:
        assert node.meta["output_device"] == torch.device("cpu")
        return True
    else:
        return False


def _is_legacy_pt():
    if Version(Version(torch.__version__).base_version) < Version("2.1"):
        return True
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


def is_module_dynamic(input_module: torch.fx.GraphModule) -> bool:
    """
    This function dynamicity per graph module.
    """

    from torch._subclasses.fake_tensor import FakeTensor
    from torch.fx.experimental.proxy_tensor import py_sym_types
    from torch.fx.passes.shape_prop import TensorMetadata

    is_dynamic = False
    for node in input_module.graph.nodes:
        if node.op == "placeholder":
            meta_val = node.meta.get("val", node.meta.get("tensor_meta", None))
            if (isinstance(meta_val, FakeTensor) and meta_val._has_symbolic_sizes_strides) or isinstance(
                meta_val, py_sym_types
            ):
                is_dynamic = True
                break

    logger.debug("Module dynamicity %s", is_dynamic)
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
    if _is_legacy_pt():
        is_dynamic = config.dynamic_shapes
    else:
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


def optimize_graph(
    stage: OptimizationPassPlacement,
    graph_module: torch.fx.GraphModule,
    graph_name: str,
    example_inputs: List[torch.Tensor],
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
        graph_module,
        graph_name,
        example_inputs,
        is_training,
        is_backward,
        is_dynamic,
        stage,
        None,
        None,
    )

    def run_passes(ctx: OptimizerContext):
        graph_changed = False

        pass_counter = 0

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

            # create new ctx for submodule
            # outer-most graph module is dynamic while sub module is static?
            sub_ctx = OptimizerContext(
                submodule,
                submodule_name,
                ctx.example_inputs,
                ctx.is_training,
                ctx.is_backward,
                ctx.is_dynamic,
                ctx.stage,
                None,
                None,
                None,
                True,
            )

            submodule_qualified_name = submodule_name if module_prefix is "" else (module_prefix + "." + submodule_name)
            graph_changed = recursive_run_passes(sub_ctx, graph_changed, submodule_qualified_name)

        logger.debug(
            "Running passes of {} stage on module {}".format(
                ctx.stage, "outer_most" if module_prefix is "" else module_prefix
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
        return node.op == "call_function" and "hints_wrapper" == node.target.__name__

    def get_schedule_policy(hints: dict) -> SchedulePolicy:
        if "schedule_policy" not in hints:
            logger.warn("No schedule policy is provided, default to use strict policy.")
            return SchedulePolicy.strict

        expected_policy = hints["schedule_policy"]
        if expected_policy.lower() == "strict":
            return SchedulePolicy.strict

        logger.warn("Currently policy {} is not supported, fall back to strict policy.".format(expected_policy))
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
                        "hint key '{}' is not support yet hence expect to not take effect. Supported hint keys are {}".format(
                            h, get_supported_hints()
                        )
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

        replacement_mapping: Dict[torch.fx.Node, torch.fx.Node] = {}
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
        return

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
            logger.debug(
                "annotated node {} with hints {} inside submodule {}".format(n, parent_hints, submodule_qualified_name)
            )

        inline_hints_wrapper(parent_module, wrapper_node, submodule)
        parent_module.delete_submodule(submodule_name)
        return

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
    if (not ctx.is_dynamic) or (hpu_backend_config.force_static_compile):
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
            missing_symint_list = get_missing_symbolic_int_input_nodes(symint_list, node)
            if missing_symint_list == ():
                continue

            node.args = missing_symint_list + node.args
            submodule = node.graph.owning_module.get_submodule(node.target)

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


def pass_fake_propagation_current(ctx: OptimizerContext) -> bool:
    """
    This function contains FakeMode propagation implementation for PT2.1+
    """

    from torch._dynamo.utils import detect_fake_mode
    from torch._subclasses.fake_tensor import FakeTensorMode

    class TensorInfoPropagation(torch.fx.Interpreter):
        """
        This class is responsible for tracing through the graph module, and
        propagating all the necessary tensor information. All is done using
        fake_tensors so it does not make any real computations.
        """

        def __init__(
            self,
            graph_module: torch.fx.GraphModule,
            fake_mode: Optional[FakeTensorMode] = None,
        ):
            super().__init__(graph_module)
            if fake_mode is None:
                fake_mode = FakeTensorMode()
            self._mode = fake_mode

        def run_node(self, node: torch.fx.Node):
            args = kwargs = result = None
            if SymExprNodeManager.node_name in node.name:
                result = node.meta["val"]
                args, kwargs = self.fetch_args_kwargs_from_env(node)
            else:
                result = super().run_node(node)
                args, kwargs = self.fetch_args_kwargs_from_env(node)
            node.val_args = args
            node.val_kwargs = kwargs
            fill_propagated_tensor_metadata_to_node(result, node)

            return result

        def propagate(self, *args):
            fake_args = [self._mode.from_tensor(a) if isinstance(a, torch.Tensor) else a for a in args]
            return self.propagate_dont_convert_inputs(*fake_args)

        def propagate_dont_convert_inputs(self, *args):
            with self._mode:
                return super().run(*args)

    fake_mode = detect_fake_mode(ctx.example_inputs)
    with torch.autocast(enabled=False, device_type="hpu"), torch.autocast(enabled=False, device_type="cpu"):
        # Disabling autocast in fake tensor propagation as autocasting has been
        # already done and all dtypes has been already deduced.
        if not fake_mode:
            fake_mode = torch._subclasses.FakeTensorMode(allow_non_fake_inputs=True)
            TensorInfoPropagation(ctx.graph_module, fake_mode).propagate(*ctx.example_inputs)
        else:
            TensorInfoPropagation(ctx.graph_module, fake_mode).propagate_dont_convert_inputs(*ctx.example_inputs)

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
    last_node_after_output = None
    for n in ctx.graph_module.graph.nodes:
        if output_node:
            last_node_after_output = n

        if n.op == "output":
            output_node = n
    if last_node_after_output is not None:
        logger.warn("It seems graph wasn't functionalized, fixing empty output node.")
        ctx.graph_module.graph.node_copy(output_node)
        ctx.graph_module.graph.erase_node(output_node)
        ctx.graph_module.recompile()
        graph_changed = True

    # WORKAROUND END

    return graph_changed


def pass_fake_propagation_legacy(ctx: OptimizerContext) -> bool:
    """
    This function contains FakeMode propagation implementation for PT2.0
    """

    from torch._dynamo.utils import deepcopy_to_fake_tensor, fake_mode_from_tensors
    from torch.utils._python_dispatch import _get_current_dispatch_mode_stack

    class LegacyTensorInfoPropagation(torch.fx.Interpreter):
        """
        This class is responsible for tracing through the graph module, and
        propagating all the necessary tensor information. All is done using
        fake_tensors so it does not make any real computations.
        """

        def __init__(
            self,
            graph_module: torch.fx.GraphModule,
            fakemode_already_enabled: bool,
            fake_mode: torch._subclasses.FakeTensorMode,
        ):
            super().__init__(graph_module)
            if fakemode_already_enabled:
                self.fake_mode = contextlib.nullcontext()
            else:
                self.fake_mode = fake_mode

        def run_node(self, node: torch.fx.Node):
            with self.fake_mode:
                result = super().run_node(node)

            args, kwargs = self.fetch_args_kwargs_from_env(node)
            node.val_args = args
            node.val_kwargs = kwargs

            fill_propagated_tensor_metadata_to_node(result, node)

            return result

        def propagate(self, *args):
            return super().run(*args)

    # We need to make sure we run in fake_mode.
    fakemode_already_enabled = False
    for mode in _get_current_dispatch_mode_stack():
        if isinstance(mode, torch._subclasses.FakeTensorMode):
            fakemode_already_enabled = True
            break

    fake_mode = None
    fake_inputs = ctx.example_inputs
    if not fakemode_already_enabled:
        fake_mode = fake_mode_from_tensors(ctx.example_inputs)
        if fake_mode is None:
            fake_mode = torch._subclasses.FakeTensorMode()
            fake_inputs = deepcopy_to_fake_tensor(ctx.example_inputs, fake_mode)

    with torch.autocast(enabled=False, device_type="hpu"), torch.autocast(enabled=False, device_type="cpu"):
        # Disabling autocast in fake tensor propagation as autocasting has been
        # already done and all dtypes has been already deduced.
        LegacyTensorInfoPropagation(ctx.graph_module, fakemode_already_enabled, fake_mode).propagate(*fake_inputs)

    return True


def pass_fake_propagation(ctx: OptimizerContext) -> bool:
    """
    This pass makes sure that input tensors are in fake mode so we don't
    make any actual computation. Then it propagates tensor metadata into nodes.
    """
    if _is_legacy_pt():
        return pass_fake_propagation_legacy(ctx)
    else:
        return pass_fake_propagation_current(ctx)


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
        ctx.habana_partitioner = HabanaPartitioner(ctx.graph_module, cls)
        for allreduce in allreduces:
            setattr(cls, "allreduce_name", allreduce.name)
            ctx.current_partitions_non_mergeable.extend(ctx.habana_partitioner.propose_partitions())
    ctx.habana_partitioner = HabanaPartitioner(ctx.graph_module)
    ctx.current_partitions.extend(ctx.habana_partitioner.propose_partitions())

    # Nothing was really changed.
    return False


def match_full_copy_pattern(node: torch.fx.Node) -> Tuple[bool, torch.fx.Node, torch.fx.Node]:
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


def pass_post_process_partitions(ctx: OptimizerContext):
    """
    This pass will do some post process for those proposed partitions from hpu
    partitioner, like move some specific ops from one partition to another
    partition, to reduce some unnecessary tensor passing between partitions.
    Currently, the post process is mainly for device memory optimization.
    """
    from torch.fx.passes.infra.partitioner import Partition

    partition_changed = False

    def reassign_full_copy_to_upstream_partition(
        graph_module, assignments: Dict[torch.fx.Node, int], partitions_by_id: Dict[int, Partition]
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
        graph_module, assignments: Dict[torch.fx.Node, int], partitions_by_id: Dict[int, Partition]
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

    assignments: Dict[torch.fx.Node, int] = {}  # mapping from node to partition_id
    partitions_by_id: Dict[int, Partition] = {}  # mapping from partition_id to partition
    for partition in ctx.current_partitions + ctx.current_partitions_non_mergeable:
        id = partition.id
        partitions_by_id[id] = partition
        for node in list(partition.nodes):
            assignments[node] = id

    if hpu_backend_config.reassign_full_copy:
        partition_changed = partition_changed or reassign_full_copy_to_upstream_partition(
            ctx.graph_module, assignments, partitions_by_id
        )
    if hpu_backend_config.reassign_copy_:
        partition_changed = partition_changed or reassign_copy__to_upstream_partition(
            ctx.graph_module, assignments, partitions_by_id
        )

    return partition_changed


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
    for node in ctx.graph_module.graph.nodes:
        if (
            node.op != "placeholder"
            and node.op != "output"
            and node.op != "get_attr"
            and not (node.op == "call_function" and "to_copy" in node.target.__name__)
            and node.meta["output_device"].type == "hpu"
            and not is_backward_checkpoint_op(node)
        ):
            for arg in node.args:
                if (
                    isinstance(arg, torch.fx.Node)
                    and ("output_device" in arg.meta and arg.meta["output_device"].type != "hpu")
                    and _is_cpu_scalar_copy_required(node, arg)
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

    for node in ctx.graph_module.graph.nodes:
        placement = None
        dynamic_call_function = is_call_function_dynamic(node, ctx.is_dynamic) if node.op == "call_function" else False
        if node.op in ["placeholder", "output", "get_attr"]:
            placement = "eager"
        elif node.op == "call_function" and is_higher_order_node(node):
            placement = "eager"
        elif node.op == "call_function" and "to_copy" in node.target.__name__:
            input_node = None
            for arg in node.args:
                if isinstance(arg, torch.fx.Node):
                    input_node = arg
                    break

            assert input_node is not None

            # Internal HPU copies should be placed in the clusters.
            if all([n.meta["output_device"].type == "hpu" for n in [input_node, node]]):
                placement = "hpu_cluster"
            else:
                placement = "eager"
                logger.debug(
                    f"{node._pretty_print_target(node.target)} fellback to eager becouse it was identified as non D2D copy"
                )
        elif node.op == "call_function" and node._pretty_print_target(node.target) in host_call_functions:
            placement = "eager"
        elif node.op == "call_function" and is_eager_fallback_required(node, is_dynamic=dynamic_call_function):
            placement = "eager"
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
                    assert arg.meta["output_device"].type == "hpu"

            placement = "hpu_cluster"
        elif node.meta["output_device"].type == "cpu":
            placement = "eager"

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


collective_ops = set(
    [
        torch.ops._c10d_functional.all_reduce_.default,
        torch.ops._c10d_functional.all_reduce.default,
    ]
)

view_ops_set = set(
    [
        torch.ops.aten.view.default,
        torch.ops.aten._unsafe_view.default,
    ]
)


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


def pass_merge_paths(ctx: OptimizerContext) -> bool:
    """
    This pass that will merge parallel partitions.
    """
    assert ctx.stage == OptimizationPassPlacement.PARTITIONER
    assert ctx.graph_module is not None
    assert ctx.current_partitions is not None

    logger.debug(f"Merging parallel graph path. Partition cnt: {len(ctx.current_partitions)}")

    graph_changed = False

    if len(ctx.current_partitions) == 1:
        logger.debug(f"Merging skipped for single partition graph")
        # In case of single partition there is no merging to be done
        return graph_changed

    color_graph = ColorGraph()

    # Color all nodes in every partition on the same color
    partitions_by_color = dict()

    for part in ctx.current_partitions:
        partition_color = color_graph.assign_new_color(is_partition_color=True)
        for node in part.nodes:
            node.meta["merge_path_color"] = partition_color
        partitions_by_color[partition_color] = part

    # Color remaining nodes (new color for every node)
    for node in ctx.graph_module.graph.nodes:
        if "merge_path_color" not in node.meta:
            node_color = color_graph.assign_new_color()
            node.meta["merge_path_color"] = node_color

    # Build color graph
    for node in ctx.graph_module.graph.nodes:
        for user in node.users.keys():
            user_color = user.meta.get("merge_path_color")
            node_color = node.meta.get("merge_path_color")
            color_graph.add_node(user_color, node_color)
        if not node.users:
            color_graph.add_output_node(node.meta.get("merge_path_color"))

    new_partitions_desc_list = color_graph.get_parallel_blocks()

    # Update only if new partitioning is better than old one
    if len(new_partitions_desc_list) < len(ctx.current_partitions):
        logger.debug("New partition list (by colors): %s", new_partitions_desc_list)
        from torch.fx.passes.infra.partitioner import Partition

        new_partitions = list()
        for desc in new_partitions_desc_list:
            new_part = Partition()
            for color in desc:
                for node in partitions_by_color[color].nodes:
                    new_part.add_node(node)
            new_partitions.append(new_part)

        ctx.current_partitions = new_partitions
        graph_changed = True
        logger.debug("Merge paths done. Partition cnt: %s", len(ctx.current_partitions))
    else:
        logger.debug("No partitions suitable for merging found")

    # Cleanup coloring information from meta
    for node in ctx.graph_module.graph.nodes:
        del node.meta["merge_path_color"]

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

        from torch._subclasses.fake_tensor import FakeTensor
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
                idx = 0
                new_args1 = list(meta_val.size())
                for arg in list(meta_val.size()):
                    new_args1[idx] = arg
                    if isinstance(arg, py_sym_types):
                        new_node = cls.py_node_manager.get_or_create(arg, int)
                        new_node.meta["val"] = arg
                        new_node.meta["placement"] = "eager"
                        new_node.meta["output_device"] = torch.device("cpu")
                        new_args1[idx] = new_node
                    idx += 1
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
                idx = 0
                new_args1 = list(meta_val.size())
                for arg in list(meta_val.size()):
                    new_args1[idx] = arg
                    if isinstance(arg, py_sym_types):
                        new_node = cls.py_node_manager.get_or_create(arg, int)
                        new_node.meta["val"] = arg
                        new_node.meta["placement"] = "eager"
                        new_node.meta["output_device"] = torch.device("cpu")
                        new_args1[idx] = new_node
                    idx += 1
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
                graph_changed = resolve_negative_dim(ctx, node)

    if graph_changed:
        ctx.graph_module.recompile()
        ctx.graph_module.graph.eliminate_dead_code()
    return graph_changed


def pass_handle_view_before_inplace_compute_ops(ctx: OptimizerContext) -> bool:
    """
    This pass is actually a fix for https://github.com/pytorch/pytorch/pull/104689.
    This PR force a HPU op to generate contiguous outputs, however AOTAutograd
    functionalization is not aware of this modification, thus cannot help handle
    this. This pass helps restore the correct strides for the output of inplace op.

    Consider below case:

    def fn(a):
        b = a.t()
        b.mul_(2)
        return b

    The generated FX graph may be like:

    def forward(self, arg0_1: f32[2, 3], arg1_1: i64[3, 2]):
        t: f32[3, 2] = torch.ops.aten.t.default(arg0_1);  arg0_1 = None
        mul: f32[3, 2] = torch.ops.aten.mul.Tensor(t, arg1_1);  t = arg1_1 = None
        t_1: f32[2, 3] = torch.ops.aten.t.default(mul);  mul = None
        t_2: f32[3, 2] = torch.ops.aten.t.default(t_1)
        return (t_1, t_2)

    Normally, the output of mul node has stride [1, 3], and then t_2 (b) will
    have stride [1, 3]. In this way, we can get correct result. But after applying
    https://github.com/pytorch/pytorch/pull/104689, the output of mul node will
    be contiguous, which means stride is [2, 1]. Then finally, it leads to t_2 (b)
    having stride [2, 1]. The output stride is mismatched with expected stride.

    With this pass, `as_strided` node will be inserted before t_2 (b). And the
    output strides of `as_strided` node is filled with strides of original
    strides. The strides propagation flow of above graph is like below:

    [3, 1]
       |
       t (prefix view node)
       |
    [1, 3]    Scalar
        \     /
          mul (anchor node)
           |
         [2, 1] (forced to be contiguous)
           |
          t_1 (leaf view node)
           |
         [1, 2]
           |
           |    <----------- inserting point: as_strided
           |                                       |
          t_2 (leaf view node)                   [3, 1]
           |                                       |
        [2, 1] -> b                               t_2 (leaf view node)
                                                   |
                                                 [1, 3] -> b
    """

    def is_strides_special_case(node):
        # expand_as operator make some of strides zero at dimensions being expanded.
        # The expanded tensor return is_contiguos as False, this function detects
        # such a case, where all stride elements except with value 0 are same.

        if "output_shapes" not in node.meta or "output_strides" not in node.meta:
            return False
        contiguous_strides = calculate_default_strides(node.meta["output_shapes"][0])
        actual_strides = node.meta["output_strides"][0]
        if len(contiguous_strides) != len(actual_strides):
            return False

        special_case = False
        for i in range(len(actual_strides)):
            if not (actual_strides[i] == contiguous_strides[i] or actual_strides[i] == 0):
                return False
            else:
                special_case = special_case or (actual_strides[i] == 0)

        return special_case

    def is_output_contiguous_strides(node):
        if "output_shapes" not in node.meta or "output_contiguous" not in node.meta:
            return False

        contiguous_strides = calculate_default_strides(node.meta["output_shapes"][0])
        if not contiguous_strides:
            return False
        actual_strides = node.meta["output_strides"][0]
        return node.meta["output_contiguous"][0] or (contiguous_strides == list(actual_strides))

    def get_as_strided_src_sizes_and_strides(gm, meta_val, symbolic_sizes, symbolic_strides):
        py_node_manager = SymExprNodeManager(gm)

        for node in gm.graph.nodes:
            if node.op == "placeholder":
                tmeta_val = node.meta.get("val", node.meta.get("tensor_meta", None))
                if isinstance(tmeta_val, py_sym_types):
                    py_node_manager.add_sym_placeholder(tmeta_val, node)
                py_node_manager.set_insert_point(node)

        def convert_symexpr_to_py_node(symbolic_shape):
            var_shape = ()
            logger.debug("convert_symexpr_to_py_node symbolic_shape:", symbolic_shape)
            for dim_size in symbolic_shape:
                if isinstance(dim_size, int):
                    var_shape = var_shape + (dim_size,)
                elif isinstance(dim_size, torch.SymInt):
                    var_node = py_node_manager.get_match_sym_placeholder(dim_size)
                    if var_node is None:
                        var_node = py_node_manager.get_or_create(dim_size, int)
                        var_node.meta["val"] = dim_size
                        var_node.meta["placement"] = "eager"
                        var_node.meta["output_device"] = torch.device("cpu")
                        var_node.meta["output_dtypes"] = [None]
                        var_node.meta["output_layouts"] = [None]
                        var_node.meta["output_shapes"] = [None]
                    var_shape = var_shape + (var_node,)
            return var_shape

        # Process sizes
        var_sizes = convert_symexpr_to_py_node(symbolic_sizes)
        # Process strides
        var_strides = convert_symexpr_to_py_node(symbolic_strides)
        return var_sizes, var_strides

    def insert_as_strided_after(ctx, node_insert_point, node_src_meta):
        with ctx.graph_module.graph.inserting_after(node_insert_point):
            # input node
            new_args = [
                node_insert_point,
            ]

            src_sizes = node_src_meta.meta["output_shapes"][0]
            src_strides = node_src_meta.meta["output_strides"][0]
            if ctx.is_dynamic:
                meta_val = node_src_meta.meta.get("val", node_src_meta.meta.get("tensor_meta", None))
                src_sizes, src_strides = get_as_strided_src_sizes_and_strides(
                    ctx.graph_module,
                    meta_val,
                    node_src_meta.meta["output_shapes"][0],
                    node_src_meta.meta["output_strides"][0],
                )

            # sizes of inserted as_strided node
            new_args.append(src_sizes)
            # strides of inserted as_strided node
            new_args.append(src_strides)
            new_kwargs = None
            as_strided_custom = ctx.graph_module.graph.create_node(
                node_insert_point.op,
                torch.ops.aten.as_strided.default,
                tuple(new_args),
                new_kwargs,
                "as_strided_custom_0",
                node_insert_point.type,
            )
            as_strided_custom.meta = copy.copy(node_src_meta.meta)
            # reset output_strides in case it can be propagated later
            as_strided_custom.meta["output_strides"] = None
        return as_strided_custom

    def check_same_target_and_parameters(ref, comp) -> bool:
        if ref.target != comp.target or len(comp.args) != len(ref.args):
            return False
        comp_args = comp.args
        idx = 0
        for ref_arg in ref.args:
            # only check those int or list[int] parameters
            # for example, 0 and 1 in aten.transpose(a, 0, 1)
            # or [1, 2, 0, 3] in aten.permute(a, [1, 2, 0, 3])
            if isinstance(ref_arg, int) or isinstance(ref_arg, list):
                if ref_arg != comp_args[idx]:
                    return False

            idx += 1
        return True

    def is_call_function_node(node):
        return isinstance(node, torch.fx.Node) and node.op == "call_function"

    def is_input_mutation_node(node):
        if not (node.op == "call_function" and node.target == torch.ops.aten.copy_.default):
            return False
        args = get_node_args(node)
        # check if first arg is actually a graph input
        return args[0].op == "placeholder"

    assert ctx.graph_module is not None
    graph_changed = False
    # This is general checking for input mutation and output alias which will
    # filter out those cases early this pass doesn't target for.
    is_input_mutation_in_graph = False
    is_only_output_alias_in_graph = False
    input_mutations_nodes = []
    # see usage of torch._guards.TracingContext at _functorch/aot_autograd.py.
    if torch._guards.TracingContext.get():
        fw_metadata = torch._guards.TracingContext.get().fw_metadata
        # there exist inplace or alias
        if Version(torch.__version__) > Version("2.1.2"):
            is_input_mutation_in_graph = fw_metadata.num_mutated_inp_runtime_indices > 0
        else:
            is_input_mutation_in_graph = fw_metadata.num_mutated_inputs > 0

        # extra check for config.keep_input_mutations = 1
        if hpu_backend_config.keep_input_mutations:
            input_mutations_nodes = [node for node in ctx.graph_module.graph.nodes if is_input_mutation_node(node)]
            if input_mutations_nodes:
                is_input_mutation_in_graph = True

        is_only_output_alias_in_graph = not is_input_mutation_in_graph and fw_metadata.num_outputs_aliased > 0

    if not is_input_mutation_in_graph and not is_only_output_alias_in_graph:
        return graph_changed

    # make sure the graph nodes are topologically sorted
    ctx.graph_module.graph.lint()

    # get output nodes in graph
    fw_output_node = [node for node in ctx.graph_module.graph.nodes if node.op == "output"][0]
    fw_outputs = fw_output_node.args[0]

    # Step 0: fast path for the case where only detach node exists in graph.
    # Unlike nodes like aten.tranpose, aten.detach is a special node which never
    # changes shape/strides. aten.transpose may not change shape/stride if
    # providing parametes like aten.transpose(self, 0, 0).
    #
    # An example FX graph:
    # class <lambda>(torch.nn.Module):
    # def forward(self, arg0_1: f32[10, 5]):
    #     t: f32[5, 10] = torch.ops.aten.t.default(arg0_1);  arg0_1 = None
    #     alias: f32[5, 10] = torch.ops.aten.alias.default(t);  t = None
    #     return (alias,)
    def is_alias_node(node):
        return is_call_function_node(node) and "alias" in node.target.__name__

    if is_only_output_alias_in_graph:
        for out_node in fw_outputs:
            if not is_alias_node(out_node):
                continue
            prefix_node = get_node_args(out_node)[0]
            if is_call_function_node(prefix_node) and not is_output_contiguous_strides(prefix_node):
                # The special case here is because of expand_as which makes tensor strides like (0, 1)
                # If alias on this node is output, then we have as_strided on (0, 1), which we can't do
                # So option is to not add as_strided and call empty_strided if we detect such a case.
                # test_hpu_views_detach.py has testcase with this scenario.
                if is_strides_special_case(out_node):
                    out_node.meta["output_strides_has_zero"] = [True]
                    continue
                as_strided_node = insert_as_strided_after(ctx, out_node, prefix_node)
                # connect as_stride node as input of following users
                list(out_node.users.keys())[0].replace_input_with(out_node, as_strided_node)
                graph_changed = True

        if graph_changed:
            ctx.graph_module.recompile()
            pass_fake_propagation(ctx)
        return graph_changed

    # record pair of (prefix view node, inserting point)
    prefix_view_node_insert_point_pair = []
    for node in ctx.graph_module.graph.nodes:
        # skip for args nodes
        if not is_call_function_node(node):
            continue

        # find node which is derived from, or decomposed from an inplace node.
        # we don't need to check placement ('hpu_cluster') as it doesn't matter
        # what strides compute node generates, the later inserted as_strided
        # node should use the correct strides
        if not (
            is_compute_node(node)
            or
            # for aten.addr node, it will be decomposed into view ops +
            # other ops, here need to filter out those decomposed nodes
            is_decomposed_from_inplace_node(node)
        ):
            continue

        # Step 1: find the leaf view node pair (t_1 and t_2 in below graph) and
        # inserting point
        #
        # Example FX graph:
        # def forward(self, arg0_1: f32[2, 3]):
        #     t: f32[3, 2] = torch.ops.aten.t.default(arg0_1);  arg0_1 = None
        #     mul: f32[3, 2] = torch.ops.aten.mul.Tensor(t, 1.0);  t = None
        #     t_1: f32[2, 3] = torch.ops.aten.t.default(mul);  mul = None
        #     t_2: f32[3, 2] = torch.ops.aten.t.default(t_1)
        #     return (t_1, t_2)
        #
        # records nodes already searched in current path
        nodes_in_current_path = set()
        nodes_in_current_path.add(node)

        # The first while loop aims to find the anchor node following with pair
        # of leaf view nodes in a top-down manner
        anchor_node = node
        next_node = get_node_users(anchor_node)[0]
        while next_node is not None:
            # exit the search when reaches to the graph output or finds the
            # leaf view node pair
            if next_node.op == "output" or (
                is_call_function_node(next_node)
                and is_view_node(next_node)
                # leaf view node lies in graph outputs or has second consumer
                # node (copy_) which mutate input tensor when
                # keep_input_mutations is turned on.
                and (next_node in fw_outputs or [u for u in get_node_users(next_node) if u in input_mutations_nodes])
            ):
                break

            anchor_node = next_node
            nodes_in_current_path.add(next_node)
            # look at next user node
            next_node = get_node_users(next_node)[0]

        if next_node is None or next_node.op == "output":
            logger.debug("Not found valid leaf view node pair.")
            break

        # record leaf view nodes
        leaf_view_nodes_first = next_node
        leaf_view_nodes_second = None
        for u in get_node_users(leaf_view_nodes_first):
            if u.op == "call_function" and is_view_node(u):
                # found
                leaf_view_nodes_second = u
                break
        if leaf_view_nodes_second is None:
            logger.debug("Not found valid leaf view node pair.")
            break

        # Step 2: bottom up to find prefix view node
        #
        # The second while loop aims to locate the prefix view node in a
        # bottom-up manner. Here the loop robustly traverses all potential paths
        # to find prefix view nodes. For example for a aten.mul node which has
        # two arguments. The prefix view node might be inserted before the first
        # argument, or the second one.
        prefix_view_node = anchor_node
        nodes_queue_to_search = []
        while prefix_view_node.op != "placeholder":
            prefix_view_node_args = get_node_args(prefix_view_node)
            if prefix_view_node_args:
                # workaround for aten.where whose data input is third argument
                if prefix_view_node.target == torch.ops.aten.where.self:
                    nodes_queue_to_search.insert(0, prefix_view_node_args[2])
                else:
                    # insert input args at the front of queue
                    node_args = [
                        n for n in prefix_view_node_args if isinstance(n, torch.fx.Node) and n.op != "placeholder"
                    ]
                    nodes_queue_to_search[0:0] = node_args

                if not nodes_queue_to_search:
                    prefix_view_node = None
                    break

                # inverse depth-first search
                prefix_view_node = nodes_queue_to_search.pop(0)
            else:
                if not nodes_queue_to_search:
                    prefix_view_node = None
                    break
                # pop last node in queue to research
                prefix_view_node = nodes_queue_to_search.pop()

            if (
                is_call_function_node(prefix_view_node)
                and (is_view_node(prefix_view_node) and not is_decomposed_from_inplace_node(prefix_view_node))
                and prefix_view_node.target == leaf_view_nodes_first.target
                and prefix_view_node not in nodes_queue_to_search
            ):
                # additional check if leaf view node and prefix view node has
                # matched parameters
                if leaf_view_nodes_second in fw_outputs and not check_same_target_and_parameters(
                    leaf_view_nodes_second, prefix_view_node
                ):
                    prefix_view_node = None
                # found
                break

        if prefix_view_node is None or prefix_view_node.op == "placeholder":
            logger.debug("Not found prefix view node")
            break

        # Step 3: decide whther to insert as_strided node by checking contiguity
        # of nodes
        if not is_output_contiguous_strides(prefix_view_node) and is_output_contiguous_strides(anchor_node):
            prefix_view_node_insert_point_pair = [prefix_view_node, leaf_view_nodes_first]
            break
        else:
            # debug purpose
            prefix_view_node_output_contiguity = "True" if is_output_contiguous_strides(prefix_view_node) else "False"
            anchor_node_output_contiguity = "True" if is_output_contiguous_strides(anchor_node) else "False"
            logger.debug("prefix_view_node contiguity: %s", prefix_view_node_output_contiguity)
            logger.debug("anchor_node contiguity: %s", anchor_node_output_contiguity)

    # Step 4: insert as_strided node and then recompile graph
    if prefix_view_node_insert_point_pair:
        prefix_view_node = prefix_view_node_insert_point_pair[0]
        inserting_point = prefix_view_node_insert_point_pair[1]

        # node which has original strides information
        arg_prefix_view_node = get_node_args(prefix_view_node)[0]
        as_strided_node = insert_as_strided_after(ctx, inserting_point, arg_prefix_view_node)
        list(inserting_point.users.keys())[0].replace_input_with(inserting_point, as_strided_node)

        ctx.graph_module.recompile()
        # another metadata (only strides) propagation needed due to newly
        # inserted node
        pass_fake_propagation(ctx)
        graph_changed = True

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
                    if is_view_node(arg):
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


def wrap_random_ops(input_module: torch.fx.GraphModule):
    """
    This pass goes through habana cluster and:
    - replaces run_and_save_rng_state ops with habana wrappers,
    - replaces run_with_rng_state ops with habana checkpoint wrappers,
    - replaces random ops with habana wrappers,
    - creates seed and counter tensor for habana_seed_generator,
    - feeds habana wrappers with generated seed tensors.
    """

    random_ops = [node for node in input_module.graph.nodes if is_random_op(node)]
    backward_random_ops = [node for node in input_module.graph.nodes if is_backward_checkpoint_op(node)]

    # run_with_rng_state op is replaced with the actual random op with seed acquired from
    # the run_with_rng_state's first input.
    if len(backward_random_ops) > 0:
        for node in backward_random_ops:
            with input_module.graph.inserting_before(node):
                random_node = input_module.graph.call_function(*backward_random_op_inputs(node))
                node.replace_all_uses_with(random_node, propagate_meta=True)
                random_node.meta.update(node.meta)
                input_module.graph.erase_node(node)

        input_module.recompile()

    if len(random_ops) == 0:
        return

    with input_module.graph.inserting_before():
        counter_pl = input_module.graph.placeholder("counter_pl")
        seed_pl = input_module.graph.placeholder("seed_pl")

    with input_module.graph.inserting_after(counter_pl):
        seeds = input_module.graph.call_function(
            torch.ops.hpu.habana_seed_generator, (counter_pl, seed_pl, len(random_ops)), {}
        )
        add_inplace = input_module.graph.call_function(torch.ops.aten.add_, (counter_pl, len(random_ops)), {})

    multi_output_ops = []

    for i, node in enumerate(random_ops):
        with input_module.graph.inserting_before(node):
            seed = input_module.graph.call_function(torch.select, (seeds, 0, i), {})
            random_node = input_module.graph.call_function(*random_op_inputs(node, seed))
            node.replace_all_uses_with(random_node, propagate_meta=True)
            random_node.meta.update(node.meta)
            if is_multi_output_op(node):
                multi_output_ops.append(random_node)
            input_module.graph.erase_node(node)

    for node in multi_output_ops:
        for getitem in list(node.users):
            if getitem.args[1] == 1:
                for selector in list(getitem.users):
                    idx = selector.args[1]
                    selector.args = (node, idx + 1)

    input_module.recompile()


def remove_duplicated_outputs(input_module: torch.fx.GraphModule):
    """
    This function will remove those outputs which are duplicated with inputs in
    the fx graph. So that the generated JIT graph won't have duplicated output.
    This function run before we convert fx graph to jit graph.

    For example, the add_1 output in following graph will be removed. def
    forward(self, mm: "bf16[4,4]", relu: "bf16[4,4]", _to_copy_1: "bf16[4,4]"):
        add: "bf16[4, 4]" = torch.ops.aten.add_.Tensor(mm, relu) relu_1:
        "bf16[4, 4]" = torch.ops.aten.relu.default(_to_copy_1) add_1: "bf16[4,
        4]" = torch.ops.aten.add_.Tensor(add, relu_1) relu_2: "bf16[4, 4]" =
        torch.ops.aten.relu.default(add_1) return (add_1, relu_2)
    """
    in_to_out_dups = input_module.meta.get("in_to_out_dups", None)
    if in_to_out_dups is None:
        return

    duplicated_out_indexes = list(in_to_out_dups.values())
    for node in input_module.graph.nodes:
        if node.op == "output":
            output_node = node
            break  # expect only one output node per fx graph

    # remove the duplicated outputs
    outs = list(output_node.args[0]) if type(output_node.args[0]) == tuple else [output_node.args[0]]
    for idx in duplicated_out_indexes:
        outs.remove(outs[idx])

    # create a new output node
    input_module.graph.output(outs[0] if len(outs) == 1 else tuple(outs))
    input_module.graph.erase_node(output_node)
    input_module.graph.lint()
    return


def remove_no_effect_inplace_add(graph_module: torch.fx.GraphModule):
    """
    This function will convert some reinpalced add_ ops back to out-of-place
    version if they don't cause partition input/output duplications. This is a
    WA since those add_ ops will be converted back to out-of-place version
    during generating jit graph by _jit_pass_remove_mutation, and that jit pass
    will change the ops order inside the graph, and make the
    jit_node_shape_propagation failed.
    """
    for node in graph_module.graph.nodes:
        if not (node.op == "call_function" and node.target == torch.ops.aten.add_.Tensor):
            continue

        src0 = node.args[0]
        if not (src0.op == "placeholder" or src0.target.__name__.split(".")[0].endswith("_")):
            # this inplace add_ op doesn't have possbility to change the arg, so
            # convert it to out-of-place version.
            node.target = torch.ops.aten.add.Tensor
    return


def pass_compile_clusters(ctx: OptimizerContext):
    """
    This pass goes through each node in the main module. For each generated HPU cluster
    there will be "call_module" OP. For each such module create JIT IR and pass
    it to the HPU backend for recipe compilation and substitute the target with
    newly compiled one.
    """

    # It seems that this pass assumes that the fx graph and the jit graph must
    # have same ops order. Otherwise, the shape propagation may fail. However,
    # the _jit_pass_remove_mutation pass has possiblity to change the jit graph
    # ops order, and may break the assumption.
    def jit_node_shape_propagation(jit_ir, fx_module):
        Jit_graph = jit_ir.graph
        logger.debug("JIT processing shape propagation JIT graph:", Jit_graph)
        logger.debug("JIT processing shape propagation FX graph:", fx_module.print_readable(False))
        fx_nodes = list(fx_module.graph.nodes)
        jit_node_skip_list = ["prim::Constant", "prim::ListConstruct"]

        fx_count = 0
        for node in fx_module.graph.nodes:
            if node.op == "placeholder":
                fx_count += 1
            else:
                break

        def get_fx_subname(jit_node_name):
            changed_name = jit_node_name.replace("::", ".")
            return changed_name.split(".")[1]

        def get_matched_fx_node(fx_nodes, fx_idx, jit_node_name):
            size = len(fx_nodes)
            next_fx_idx = None
            curr_fx_node = None
            logger.debug("Matching Jit node:", jit_node_name, "from FX node index:", fx_idx)
            while fx_idx < size:
                fx_node = fx_nodes[fx_idx]
                if fx_node.op == "placeholder" or fx_node.op == "output":
                    fx_idx += 1
                    continue
                if fx_node.target.__name__.count(jit_node_name) > 0:
                    fx_idx += 1
                    next_fx_idx = fx_idx
                    curr_fx_node = fx_node
                    break
                else:
                    fx_idx += 1
            return next_fx_idx, curr_fx_node

        def create_output_size(tensor_size):
            from .symbolic_execution import PythonPrinter

            pexpr = PythonPrinter().doprint
            pexpr_output_shape = HPUExprPrinter().doprint

            def convert_tsize_to_str(tsize):
                shape = tsize
                dims = len(shape)
                tsize_str = "["
                for dim, sz in enumerate(shape):
                    sz_str = pexpr(sz)
                    sz_str_sympy = sympify_expression(sz_str)
                    sz_str_sympy = substitute_sympyfn(sz_str_sympy)
                    logger.debug("pexpr_output_shape input sz_str_sympy:", sz_str_sympy)
                    sz_str = pexpr_output_shape(sz_str_sympy)
                    tsize_str = tsize_str + str(sz_str)
                    if dim < dims - 1:
                        tsize_str += ","
                tsize_str += "]"
                return tsize_str

            output_len = len(tensor_size)
            output_size_str = "["
            for idx, tsize in enumerate(tensor_size):
                tsize_str = convert_tsize_to_str(tsize)
                logger.debug("create_output_size tsize_str:", tsize_str)
                output_size_str = output_size_str + tsize_str
                if idx < output_len - 1:
                    output_size_str += ";"

            output_size_str += "]"
            logger.debug("create_output_size output_size_str:", output_size_str)
            return output_size_str

        for node in Jit_graph.nodes():
            if node.kind() in jit_node_skip_list:
                continue

            fx_subname = get_fx_subname(node.kind())
            backup_fx_count = fx_count
            next_fx_idx, fx_node = get_matched_fx_node(fx_nodes, fx_count, fx_subname)
            fx_count = next_fx_idx
            # If a Jit node didnot find in the FX, then the move to next
            # Jit node and start from next FX node index.
            if fx_count is None:
                fx_count = backup_fx_count + 1

            if fx_node is None:
                logger.debug("Not found a matching FX node for node name: %s !!!", fx_subname)
                continue

            output_size_str = "[[]]"
            if "output_shapes" in fx_node.meta:
                logger.debug(
                    "Matched nodes, Jit node name formated: %s FX node: %s fx_count: %d, output_shapes:%s",
                    fx_subname,
                    fx_node,
                    fx_count,
                    fx_node.meta["output_shapes"],
                )
                output_size_str = create_output_size(fx_node.meta["output_shapes"])
            node.s_("output_shapes", output_size_str)

    def jit_node_annotation_propagation(jit_ir, fx_module):
        """
        This pass aims to directly manipulate JIT IR to set hints to node's
        attribute.
        """

        # Filter inputs/output and getitem nodes from fx graph, as they are not
        # present in jit
        fx_nodes = list(
            filter(
                lambda x: ((x.op == "call_function") and ("getitem" not in x.target.__name__)),
                fx_module.graph.nodes,
            )
        )

        jit_graph = jit_ir.graph
        # Filter prim nodes, as they are not present in fx
        jit_graph_nodes = list(
            filter(
                lambda x: ("prim::" not in x.kind()),
                jit_graph.nodes(),
            )
        )

        if len(fx_nodes) != len(jit_graph_nodes):
            logger.debug("Jit graph and FX graph should have same number of nodes: ")
            logger.debug("FX nodes: ", fx_nodes)
            logger.debug("JIT graph nodes: ", jit_graph_nodes)
            return

        is_annotated_graph = False
        for jit_node, fx_node in zip(jit_graph_nodes, fx_nodes):
            fx_node_name = fx_node.target.__name__.split(".")[0]
            if fx_node_name not in jit_node.kind():
                logger.debug("FX node {} doesn't match with Jit node {}".format(fx_node_name, jit_node.kind()))
                break

            # extract hints from FX node metadata
            context_hints = fx_node.meta.get("context_hints", None)
            if context_hints:
                logger.debug("node {} has context hints {}".format(fx_node_name, context_hints))
                # combine hints into a single string in format "name1:value1;[name2:value2;]"
                hints_str = ""
                for k, v in context_hints.items():
                    hints_str += "".join([k, ":", str(v), ";"])
                jit_node.s_("hints", hints_str)
                logger.debug("set hints for jit node", jit_node)
                is_annotated_graph = True

            if "sfg" in fx_node.meta:
                jit_node.s_("sfg", "true")
                logger.debug("sfg marked for jit node", jit_node)
                is_annotated_graph = True

        if is_annotated_graph:
            logger.debug(
                "####Annotated JIT IR graph for this HPU graph:####\n%s",
                jit_graph,
            )

        return

    def generate_jit_ir_from_module(input_module: torch.fx.GraphModule):
        """
        This function generate JIT IR for specified graph module.
        """

        import copy

        from torch._functorch.compile_utils import strip_overloads
        from torch._functorch.compilers import _disable_jit_autocast

        module = copy.deepcopy(input_module)
        wrap_random_ops(module)
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

        return f, module

    num_subgraphs = 0
    refine_dynamic = bc.get_pt_hpu_enable_refine_dynamic_shapes()
    optim_output_sif_ds = bc.get_pt_hpu_optim_dynamic_output_sif()

    for n in ctx.graph_module.graph.nodes:
        logger.debug("Node: %s Op: %s Target: %s", n, n.op, n.target)

        if n.op == "call_module":
            assert not n.kwargs
            submod = ctx.graph_module.get_submodule(n.target)

            jit_ir_function, submod_updated = generate_jit_ir_from_module(submod)
            jit_node_annotation_propagation(jit_ir_function, submod_updated)

            is_submod_dynamic = False

            if not hpu_backend_config.force_static_compile:
                # Submodule dynamicity has to recheck and set to the collable.
                is_submod_dynamic = is_module_dynamic(submod)

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
    if not hpu_backend_config.use_inplace_allreduce:
        return graph_changed

    graph = ctx.graph_module.graph

    def reinplace_collective_ops(gm: torch.fx.GraphModule):
        replace_dict: Dict[torch.fx.Node, torch.fx.Node] = {}

        for idx, node in enumerate(gm.graph.nodes):
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


def pass_reinplace_index_copy_ops(ctx: OptimizerContext) -> bool:
    """
    This pass tries to replace the usage of out of place variant with the
    inplace variant of the index_copy op. This matches a particular variant
    of the index_copy where index_copy->copy_ is present, then the combination
    is replace with index_copy_ which is an inplace variant of index_copy
    """
    graph_changed = False
    if not hpu_backend_config.use_inplace_index_copy:
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

        replace_dict: Dict[torch.fx.Node, torch.fx.Node] = {}

        for idx, node in enumerate(gm.graph.nodes):
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
    if not hpu_backend_config.reinplace_add:
        return False

    graph_changed = False

    def is_eligible_add_node(node: torch.fx.Node):
        def is_view_op(_node: torch.fx.Node):
            return _node.op == "call_function" and is_view_node(_node)

        is_add = node.op == "call_function" and node.target == torch.ops.aten.add.Tensor
        if not is_add:
            return False

        src0, src1 = node.args[0], node.args[1]
        is_add_two_tensors = type(src0) == torch.fx.Node and type(src1) == torch.fx.Node
        is_float_dtype = (
            is_add_two_tensors
            and src0.meta["output_dtypes"][0] == src1.meta["output_dtypes"][0]
            and src0.meta["output_dtypes"][0] in (torch.float32, torch.bfloat16, torch.float16)
        )
        is_eligible = is_float_dtype and not src0.op == "placeholder" and not is_view_op(src0)
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

    Note: Currently, we only detect the duplicates caused by inplace add_ ops
    (the WA 1), because some ops whose name ends with "_" are actually not
    inplaced, such as __rshift__.

    Take the following graph as an example, if the output of __rshift__ is not
    returned from the fx graph, then the converted jit graph will not contain
    rshift op:

    def forward(self, arg1_1, arg0_1):
        rshift = torch.ops.aten.__rshift__.Tensor(arg1_1, arg0_1)
        return

    The generated JIT graph:

    graph(%self : __torch__.torch.fx.graph_module.GraphModule,
            %arg1_1.1 : Tensor,
            %arg0_1.1 : Tensor):
        %20 : () = prim::Constant[value=()]()
        return (%20)

    """

    # currently, we only consider the duplications caused by inplace op.
    def detect_in_to_out_duplicates(graph_module: torch.fx.GraphModule):
        in_nodes = []
        for node in graph_module.graph.nodes:
            if node.op == "placeholder":
                in_nodes.append(node)

        in_to_out_dups = dict()
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
                        and user_node.target == torch.ops.aten.add_.Tensor  # WA 1
                        and user_node.args[0] == current
                    ):
                        # inplace op, and current op is the mutable arg (we
                        # assume mutable arg is always the arg0)
                        queue.append(user_node)
                    elif user_node.op == "output":
                        outs = list(user_node.args[0]) if type(user_node.args[0]) == tuple else [user_node.args[0]]
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


def pass_remove_unnecessary_bmm_view(ctx: OptimizerContext):
    def is_view_node(node):
        view_ops = {torch.ops.aten.view.default, torch.ops.aten._unsafe_view.default}
        return node.target in view_ops

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
                left_dim = get_node_dim(bmm_input_left.args[0])
                right_dim = get_node_dim(bmm_input_right.args[0])

                if left_dim in {4, 5} and left_dim == right_dim:
                    node.replace_input_with(bmm_input_left, bmm_input_left.args[0])
                    node.replace_input_with(bmm_input_right, bmm_input_right.args[0])

                    for bmm_output_user in list(bmm_output.users.keys()):
                        bmm_output_user.replace_input_with(bmm_output, node)

                    if "output_shapes" in node.meta:
                        node.meta["output_shapes"] = bmm_output.meta.get("output_shapes", None)

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
