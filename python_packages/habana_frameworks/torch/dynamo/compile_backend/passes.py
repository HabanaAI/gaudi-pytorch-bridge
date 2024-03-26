###############################################################################
# Copyright (C) 2023-2024 Habana Labs, Ltd. an Intel Company
# All Rights Reserved.
#
# Unauthorized copying of this file or any element(s) within it, via any medium
# is strictly prohibited.
# This file contains Habana Labs, Ltd. proprietary and confidential information
# and is subject to the confidentiality and license agreements under which it
# was provided.
#
###############################################################################

import contextlib
import copy
import os
import sys
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Callable, List, Mapping, Optional

import habana_frameworks.torch.internal.bridge_config as bc
import torch
from habana_frameworks.torch.dynamo.compile_backend import config as hpu_backend_config
from habana_frameworks.torch.utils.debug.dynamo_utils import FxGraphAnalyzer
from habana_frameworks.torch.utils.internal import Timer
from habana_frameworks.torch.utils.visualization import graph_visualizer
from packaging.version import Version
from torch.distributed._spmd.graph_utils import find_node
from torch.fx.experimental.proxy_tensor import py_sym_types
from torch.fx.passes.operator_support import OperatorSupport

from ._passes.fuse_allreduce_calls import fuse_allreduce_calls
from ._passes.utils import OptimizationPassPlacement, OptimizerContext
from .logger import get_compile_backend_logger
from .partitioner import HabanaPartitioner
from .random_utils import is_random_op, random_op_inputs
from .recipe_compiler import get_callable_recipe
from .shared_layer import is_eager_fallback_required
from .symbolic_execution import SymExprNodeManager, sympify_expression

logger = get_compile_backend_logger()


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
    if bc.get_pt_hpu_enable_allreduce_graph_split():
        gm = ctx.graph_module
        allreduces = find_node(gm.graph, lambda n: n.name.startswith("all_reduce"))
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
    if bc.get_pt_hpu_enable_allreduce_graph_split():
        graph = ctx.graph_module.graph
        allreduces = find_node(graph, lambda n: n.name.startswith("all_reduce"))
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

        waittensors = find_node(graph, lambda n: n.name.startswith("wait_tensor"))
        for waittensor in waittensors:
            downstream_nodes = list(waittensor.users.keys())
            nodes_to_move = [waittensor]
            while len(downstream_nodes) > 0:
                new_downstream_nodes = []
                for downstream_node in downstream_nodes:
                    if not downstream_node.name.startswith("fused"):
                        new_downstream_nodes.extend(list(waittensor.users.keys()))
                        nodes_to_move.append(downstream_node)
                    else:
                        fused = downstream_node
                downstream_nodes = new_downstream_nodes

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
        "as_strided_scatter",
        "slice",
        "_roi_align_backward",
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
        if (isinstance(meta_val, FakeTensor) and meta_val._has_symbolic_sizes_strides) or isinstance(
            meta_val, py_sym_types
        ):
            return True

    # early exit when the graph module is static
    if not dynamic_graph:
        return False

    from torch._subclasses.fake_tensor import FakeTensor
    from torch.fx.experimental.proxy_tensor import py_sym_types

    is_dynamic = False
    if node.op == "call_function":
        is_dynamic = check_dynamic_meta(node)
        if not is_dynamic:
            args = helper_get_node_args(node)
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


def optimize_graph(
    stage: OptimizationPassPlacement,
    graph_module: torch.fx.GraphModule,
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
        example_inputs,
        is_training,
        is_backward,
        is_dynamic,
        stage,
        None,
    )

    graph_changed = False
    visualization_mode = bc.get_pt_hpu_graph_dump_mode()
    visualisation_enabled = visualization_mode in ["all", "compile", "compile_fx"]
    with graph_visualizer(
        graph_module=graph_module,
        active_stage=stage,
        final_stage=OptimizationPassPlacement.POST_PARTITIONER,
        disable=not visualisation_enabled,
    ) as gv:
        for optimization_pass in get_passes(stage):
            pass_name = optimization_pass.__name__
            env_name = "PT_HPU_DISABLE_" + pass_name
            if os.getenv(env_name, "").upper() in ["ON", "1", "YES", "TRUE", "Y"]:
                logger.debug("pass %s was disabled by env at stage %s", pass_name, stage)
            else:
                logger.debug("running %s pass at stage %s", pass_name, stage)

                with Timer() as t:
                    current_graph_changed = optimization_pass(ctx)

                graph_changed = current_graph_changed or graph_changed
                if current_graph_changed:
                    gv.visualize_graph(graph_module, optimization_pass.__name__)

                logger.debug(
                    "pass %s at stage %s took: %.3f [s]",
                    pass_name,
                    stage,
                    t.elapsed,
                )

    return graph_changed


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
            # These passes will be ran once, they always get and produce a flat graph without submodules.
            pass_graph_print,
            fuse_allreduce_calls,
            pass_allreduce_parents,
            pass_pattern_rewriter,
            pass_fake_propagation,
            pass_wa_mixed_devices,  # This is W/A for Adam having CPU scalar tensors parameters.
            pass_reinplace_inplaceable_ops,
            pass_mark_placement,
            pass_graph_print,
        ]
    elif stage == OptimizationPassPlacement.PRE_PARTITIONER:
        return [
            # These passes will prepare proper placement for some corner-cases.
            pass_handle_view_before_inplace_compute_ops,
            pass_graph_print,
            pass_eagerize_leaf_views,
            pass_handle_negative_dims,
            pass_replace_sym_size,
            pass_inference_fuse_linear,
        ]
    elif stage == OptimizationPassPlacement.PARTITIONER:
        return [
            # These passes will prepare proper placement for some corner-cases.
            pass_propose_partitions,
            pass_merge_paths,
            # This is final pass that creates final submoduled graph.
            pass_fuse_partitions,
            pass_reorder_allreduce,
            pass_make_symints_available,
            pass_graph_print,
            pass_wa_fix_output,
        ]
    elif stage == OptimizationPassPlacement.POST_PARTITIONER:
        return [
            # These passes will be ran once, they have to work on graph with submodules.
            pass_graph_print,
            pass_summarize_graph,
            pass_compile_clusters,
        ]
    else:
        logger.error("unknown optimization stage %s", stage)
        raise


def helper_is_view_node(node):
    node_target = node.target.__name__.split(".")[0]

    # This is list of view OPs.
    view_ops = [
        "view",
        "_unsafe_view",
        "as_strided",
        "as_strided_scatter",
        "slice",
        "select",
        "squeeze",
        "unsqueeze",
        "expand",
        "transpose",
        "t",
        "permute",
        "split",
        "split_with_sizes",
        "alias",
    ]

    return node_target in view_ops


def helper_get_node_args(node: torch.fx.Node):
    """
    This helper function get inputs to specific node. It should support
    various corner cases.
    """
    args = node.args
    if "output" in node.op and isinstance(node.args, tuple):
        # Output args could be a single-element tuple containing all outputs as well,
        # so let's support that.
        assert len(node.args) == 1

        # There are two cases, resulting unwrapped args could be again a tuple or directly a node.
        # Code assumes something iterable so if it's just a a single node, then do not unwrap it.
        if (
            isinstance(node.args[0], tuple)
            or isinstance(node.args[0], list)
            or isinstance(node.args[0], torch.fx.immutable_collections.immutable_list)
        ):
            args = node.args[0]

    if (
        isinstance(args, tuple)
        or isinstance(args, list)
        or isinstance(args, torch.fx.immutable_collections.immutable_list)
    ):
        cleaned_args = []
        for arg in args:
            if isinstance(arg, torch.fx.Node):
                cleaned_args.append(arg)
    else:
        cleaned_args = args

    return cleaned_args


def helper_handle_noncontiguous_output(node: torch.fx.Node, result: torch.Tensor):
    """
    This function aims to handle non-contiguous output, see details at:
    https://github.com/pytorch/pytorch/issues/103650 and
    https://github.com/pytorch/pytorch/pull/104689. The public fix is not
    complete since besides `torch/_refs/__init__.py`, there are still some ops
    whose meta function is defined at `pytorch/torch/_meta_registrations.py`.
    """
    if node.op != "call_function":
        return result

    node_target_list = [
        "round.default",
        "round.decimals",
    ]
    if node.target.__name__ in node_target_list:
        result = result.contiguous()
    return result


def helper_post_pass_finalize(input_module: torch.fx.GraphModule):
    """
    Run this pass iff the input graph changed for each submodule
    for each pass
    """
    # Clean up the graph and log the situation.
    input_module.graph.eliminate_dead_code()
    input_module.graph.lint()
    input_module.recompile()

    return input_module


def helper_is_node_supported(node: torch.fx.Node) -> bool:
    """
    Returns true if the node is on HPU and is part of
    the proposed fused partition
    """
    return node.meta["output_device"].type == "hpu" and node.meta["placement"] == "hpu_cluster"


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


def fill_propagated_tensor_metadata_to_node(result: torch.Tensor, node: torch.fx.Node):
    """
    This function takes out basic information from propagated fake tensor, like
    dtype, layout and device and puts it to the node that created it.
    """

    result = helper_handle_noncontiguous_output(node, result)

    device = None
    dtypes = []
    layouts = []
    output_shapes = []
    output_strides = []
    output_contiguous = []

    result_type_to_node_type: dict[type, type] = {
        torch.SymInt: int,
        torch.SymBool: bool,
        torch.SymFloat: float,
        int: int,
        float: float,
        bool: bool,
        type(None): None,
    }

    if (
        type(result) is torch._subclasses.FakeTensor
        or type(result) is torch._subclasses.fake_tensor.FakeTensor
        or type(result) is torch.Tensor
        or type(result) is torch.nn.parameter.Parameter
    ):
        device = result.device
        dtypes = [result.dtype]
        layouts = [result.layout]
        output_shapes = [result.size()]
        output_strides = [result.stride()]
        output_contiguous = [result.is_contiguous()]

        logger.debug("    result shape: %s", result.shape)
        logger.debug("    result stride: %s", result.stride())
    elif type(result) in result_type_to_node_type:
        device = torch.device("cpu")
        dtypes = [None]
        layouts = [None]
        output_shapes = [None]
        output_strides = [None]
        output_contiguous = [None]
        node.type = result_type_to_node_type[type(result)]
    elif str(node.target) == "inductor.accumulate_grad_.default":
        device = torch.device("hpu")
        dtypes = [None]
        layouts = [None]
        output_shapes = [None]
        output_strides = [None]
        output_contiguous = [None]
    else:
        devices = []
        assert isinstance(result, Iterable), "expecting iterable at this point"
        for res in result:
            if res is None:
                continue

            if hasattr(res, "device"):
                devices.append(res.device)
            if hasattr(res, "dtype"):
                dtypes.append(res.dtype)
            if hasattr(res, "layout"):
                layouts.append(res.layout)

            if hasattr(res, "shape"):
                output_shapes.append(res.shape)
                output_contiguous.append(res.is_contiguous())
                output_strides.append(res.storage_offset())
                output_strides.append(res.stride())
                logger.debug("    result shape: %s", res.shape)

        if len(devices) > 0:
            if devices.count(devices[0]) != len(devices) and "output" not in node.op:
                logger.error(
                    "multiple devices in single node\n%s\n at node: %s",
                    devices,
                    node,
                )
                raise
            else:
                device = devices[0]

    if "output" not in node.op:
        assert device is not None
        assert len(dtypes) != 0
        assert len(layouts) != 0
    else:
        device = None

    # Meta for the node should not be created yet. BUT...
    # ...it happens that placeholder nodes might be reused between FWD and BWD.
    # This is fine, I guess, as long as nothing has changed between those.
    # There is an exception for propagating strides information for newly inserted nodes
    if (
        "output_device" in node.meta
        or "output_dtypes" in node.meta
        or "output_layouts" in node.meta
        or "output_shapes" in node.meta
    ):
        if node.meta["output_device"] is not None and device is not None:
            assert node.meta["output_device"].type == device.type
        else:
            assert node.meta["output_device"] == device
        assert node.meta["output_dtypes"] == dtypes
        assert node.meta["output_layouts"] == layouts
        assert node.meta["output_shapes"] == output_shapes

    node.meta["output_device"] = device
    node.meta["output_dtypes"] = dtypes
    node.meta["output_layouts"] = layouts
    node.meta["output_shapes"] = output_shapes
    node.meta["output_strides"] = output_strides
    node.meta["output_contiguous"] = output_contiguous


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


def pass_propose_partitions(ctx: OptimizerContext) -> bool:
    """
    This pass is supposed to run partitioner that will create proposition of partitioning.
    """
    assert ctx.stage == OptimizationPassPlacement.PARTITIONER
    assert ctx.graph_module is not None
    assert ctx.current_partitions is None

    ctx.current_partitions = []
    if bc.get_pt_hpu_enable_allreduce_graph_split():
        allreduces = find_node(ctx.graph_module.graph, lambda n: n.name.startswith("all_reduce"))
        for allreduce in allreduces:
            cls = FusedCollectiveOperatorSupport
            setattr(cls, "allreduce_name", allreduce.name)
            ctx.current_partitions.extend(HabanaPartitioner(ctx.graph_module, cls).propose_partitions())
    ctx.current_partitions.extend(HabanaPartitioner(ctx.graph_module).propose_partitions())

    # Nothing was really changed.
    return False


def pass_fuse_partitions(ctx: OptimizerContext) -> bool:
    """
    This pass is supposed to run partitioner that will, based on current partitioning, create
    final FX module with submodules for each HPU operations cluster.
    """
    assert ctx.stage == OptimizationPassPlacement.PARTITIONER
    assert ctx.graph_module is not None
    assert ctx.current_partitions is not None

    HabanaPartitioner(ctx.graph_module).fuse_partitions(ctx.current_partitions)

    return True


def pass_pattern_rewriter(ctx: OptimizerContext):
    """
    Rewrite problematic:
        div(Scalar, Tensor, rounding_mode)
        floor_divide(Scalar, Tensor)
    that are unable to find proper variant.
    """
    fx_graph = ctx.graph_module

    def replace_rewrite_div(fx_graph):
        def pattern(scalar_input, tensor_input):
            x = torch.ops.aten.div.Tensor_mode(scalar_input, tensor_input, rounding_mode=None)
            return x

        def replace(scalar_input, tensor_input):
            x = torch.ops.aten.scalar_tensor(scalar_input)
            x = torch.ops.aten.div.Tensor_mode(x, tensor_input, rounding_mode=None)
            return x

        def filter(match, *args, **kwargs):
            return not isinstance(match.placeholder_nodes[0], torch.fx.node.Node)

        torch.fx.subgraph_rewriter.replace_pattern_with_filters(fx_graph, pattern, replace, [filter])

    def replace_rewrite_div_floor(fx_graph):
        def pattern(scalar_input, tensor_input):
            x = torch.ops.aten.div.Tensor_mode(scalar_input, tensor_input, rounding_mode="floor")
            return x

        def replace(scalar_input, tensor_input):
            x = torch.ops.aten.scalar_tensor(scalar_input)
            x = torch.ops.aten.div.Tensor_mode(x, tensor_input, rounding_mode="floor")
            return x

        def filter(match, *args, **kwargs):
            return not isinstance(match.placeholder_nodes[0], torch.fx.node.Node)

        torch.fx.subgraph_rewriter.replace_pattern_with_filters(fx_graph, pattern, replace, [filter])

    def replace_rewrite_div_trunc(fx_graph):
        def pattern(scalar_input, tensor_input):
            x = torch.ops.aten.div.Tensor_mode(scalar_input, tensor_input, rounding_mode="trunc")
            return x

        def replace(scalar_input, tensor_input):
            x = torch.ops.aten.scalar_tensor(scalar_input)
            x = torch.ops.aten.div.Tensor_mode(x, tensor_input, rounding_mode="trunc")
            return x

        def filter(match, *args, **kwargs):
            return not isinstance(match.placeholder_nodes[0], torch.fx.node.Node)

        torch.fx.subgraph_rewriter.replace_pattern_with_filters(fx_graph, pattern, replace, [filter])

    def replace_rewrite_floor_divide(fx_graph):
        def pattern(scalar_input, tensor_input):
            x = torch.ops.aten.floor_divide.default(scalar_input, tensor_input)
            return x

        def replace(scalar_input, tensor_input):
            x = torch.ops.aten.scalar_tensor(scalar_input)
            x = torch.ops.aten.floor_divide.default(x, tensor_input)
            return x

        def filter(match, *args, **kwargs):
            return not isinstance(match.placeholder_nodes[0], torch.fx.node.Node)

        torch.fx.subgraph_rewriter.replace_pattern_with_filters(fx_graph, pattern, replace, [filter])

    replace_rewrite_div(fx_graph)
    replace_rewrite_div_floor(fx_graph)
    replace_rewrite_div_trunc(fx_graph)
    replace_rewrite_floor_divide(fx_graph)


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
            and not (node.op == "call_function" and "to_copy" in node.target.__name__)
            and node.meta["output_device"].type == "hpu"
        ):
            for arg in node.args:
                if (
                    isinstance(arg, torch.fx.Node)
                    and arg.meta["output_device"].type != "hpu"
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


def pass_accumulate_grads(ctx: OptimizerContext) -> bool:
    """
    This pass collects inputs (variable, new_grad) from inductor.accumulate_grad_ nodes
    in the graph, and passes them as TensorLists to the custom op hpu.accumulate_grads_.
    hpu.accumulate_grads_ op is executed eagerly.
    All accumulate_grad_ nodes are then removed.
    """
    assert ctx.graph_module is not None

    graph = ctx.graph_module.graph

    variables = []
    grads = []
    accumulate_grad_nodes = []

    for node in graph.nodes:
        if str(node.target) == "inductor.accumulate_grad_.default":
            variables.append(node.args[0])
            grads.append(node.args[1])
            accumulate_grad_nodes.append(node)

    if variables:
        last_accumulate_grad = accumulate_grad_nodes[-1]
        with graph.inserting_before(last_accumulate_grad):
            accumulate_grads_ = graph.call_function(torch.ops.hpu.accumulate_grads_, (variables, grads), {})
            accumulate_grads_.meta["placement"] = "eager"
            accumulate_grads_.meta["output_device"] = last_accumulate_grad.meta["output_device"]
            last_accumulate_grad.replace_all_uses_with(accumulate_grads_, propagate_meta=False)
        for accumulate_grad in accumulate_grad_nodes:
            graph.erase_node(accumulate_grad)

        graph.lint()
        ctx.graph_module.recompile()

        logger.debug(f"inductor.accumulate_grad_ nodes were wrapped into hpu.accumulate_grads op.")

        return True

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

    class ColorGraph:
        OUTPUT_COLOR = 0

        def __init__(self):
            self.all_colors = set()
            self.partition_colors = set()
            self.output_colors = set()
            self.colors_to_remove = set()
            self._last_color = 0
            self._graph = dict()

        def new_color(self):
            self._last_color += 1
            self.all_colors.add(self._last_color)
            return self._last_color

        def new_partition_color(self):
            color = self.new_color()
            self.partition_colors.add(color)
            return color

        def add_node(self, user_color, color):
            if user_color != color:
                if color in self._graph:
                    self._graph[color].add(user_color)
                else:
                    self._graph[color] = set()
                    self._graph[color].add(user_color)

        def _update_internal_sets(self):
            for color in self.all_colors:
                if color not in self._graph.keys():
                    continue
                if color not in self.partition_colors:
                    self.colors_to_remove.add(color)
                    continue

        def _merge_non_partition_colors(self):
            for color in self.colors_to_remove:
                replacement_set = self._graph[color]
                for v in self._graph.values():
                    if color in v:
                        v.remove(color)
                        v.update(replacement_set)
            for color in self.colors_to_remove:
                del self._graph[color]
                self.all_colors.remove(color)
            self.colors_to_remove = set()

        def extract_new_partitions(self):
            logger.debug("Color graph (initial): \n%s", self)
            self._update_internal_sets()
            logger.debug("Color graph (replaced output): \n%s", self)
            self._merge_non_partition_colors()
            logger.debug("Color graph (partitions only): \n%s", self)
            partitions = dict()
            for color, user_set in self._graph.items():
                user_frozen_set = frozenset(user_set)
                if user_frozen_set in partitions:
                    partitions[user_frozen_set].add(color)
                else:
                    partitions[user_frozen_set] = set()
                    partitions[user_frozen_set].add(color)
            return list(partitions.values())

        def __str__(self):
            lines = []
            lines.append(f"Partition colors: {self.partition_colors}")
            lines.extend([f"{k} --> {v}" for k, v in self._graph.items()])
            return "\n".join(lines)

    color_graph = ColorGraph()

    # Color all nodes in every partition on the same color
    partitions_by_color = dict()

    for part in ctx.current_partitions:
        partition_color = color_graph.new_partition_color()
        for node in part.nodes:
            node.meta["merge_path_color"] = partition_color
        partitions_by_color[partition_color] = part

    # Color remaining nodes (new color for every node)
    for node in ctx.graph_module.graph.nodes:
        if "merge_path_color" not in node.meta:
            node_color = color_graph.new_color()
            node.meta["merge_path_color"] = node_color

    # Build color graph
    for node in ctx.graph_module.graph.nodes:
        for user in node.users.keys():
            user_color = user.meta.get("merge_path_color")
            node_color = node.meta.get("merge_path_color")
            color_graph.add_node(user_color, node_color)
        if not node.users:
            color_graph.add_node(ColorGraph.OUTPUT_COLOR, node.meta.get("merge_path_color"))

    new_partitions_desc_list = color_graph.extract_new_partitions()

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
    is_dynamic = False
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
                        if (isinstance(meta_val, FakeTensor) and meta_val._has_symbolic_sizes_strides) or isinstance(
                            meta_val, py_sym_types
                        ):
                            resolve_negative_dim.is_dynamic = True
                            return True
                return False
            elif node_name == "view":
                node_arg0 = node.args[0]
                meta_val = node_arg0.meta.get("val", node.meta.get("tensor_meta", None))
                if (isinstance(meta_val, FakeTensor) and meta_val._has_symbolic_sizes_strides) or isinstance(
                    meta_val, py_sym_types
                ):
                    resolve_negative_dim.is_dynamic = True
                in_args_1 = node.args[1]
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
            if not cls.is_dynamic:
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
            if not cls.is_dynamic:
                return False
            else:
                meta_val = node.args[0].meta.get("val", node.meta.get("tensor_meta", None))
                idx = 0
                new_args1 = list(meta_val.size())
                for arg in list(meta_val.size()):
                    new_args1[idx] = arg
                    if isinstance(arg, py_sym_types):
                        new_node = cls.py_node_manager.get_or_create(arg, int)
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
            if not cls.is_dynamic:
                return
            else:
                meta_val = node.args[0].meta.get("val", node.meta.get("tensor_meta", None))
                idx = 0
                new_args1 = list(meta_val.size())
                for arg in list(meta_val.size()):
                    new_args1[idx] = arg
                    if isinstance(arg, py_sym_types):
                        new_node = cls.py_node_manager.get_or_create(arg, int)
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

    def helper_is_compute_node(node):
        # return false if node is a view node, input node or output node
        return (not helper_is_view_node(node)) and (node.op != "placeholder") and (node.op != "output")

    def helper_is_decomposed_from_inplace_node(node):
        if node.op != "call_function":
            return False
        node_target = node.target.__name__
        if ("original_aten" not in node.meta) or ("from_node" not in node.meta):
            return False

        return node_target != node.meta["original_aten"].__name__ and (node.meta["from_node"][0][0].endswith("_"))

    def helper_calculate_default_strides(sizes):
        # Calculate default strides for given size
        if sizes is None or len(sizes) == 0:
            return []

        reversed_strides = [1]
        for size in reversed(sizes[1:]):
            reversed_strides.append(size * reversed_strides[-1])
        return list(reversed(reversed_strides))

    def is_strides_special_case(node):
        # expand_as operator make some of strides zero at dimensions being expanded.
        # The expanded tensor return is_contiguos as False, this function detects
        # such a case, where all stride elements except with value 0 are same.

        if "output_shapes" not in node.meta or "output_strides" not in node.meta:
            return False
        contiguous_strides = helper_calculate_default_strides(node.meta["output_shapes"][0])
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

        contiguous_strides = helper_calculate_default_strides(node.meta["output_shapes"][0])
        if not contiguous_strides:
            return False
        actual_strides = node.meta["output_strides"][0]
        return node.meta["output_contiguous"][0] or (contiguous_strides == list(actual_strides))

    def helper_get_node_users(node):
        if not isinstance(node, torch.fx.Node):
            return [None]
        node_list = list(node.users.keys())
        if len(node_list) == 0:
            return [None]
        return node_list

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
        args = helper_get_node_args(node)
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
            prefix_node = helper_get_node_args(out_node)[0]
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
            helper_is_compute_node(node)
            or
            # for aten.addr node, it will be decomposed into view ops +
            # other ops, here need to filter out those decomposed nodes
            helper_is_decomposed_from_inplace_node(node)
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
        next_node = helper_get_node_users(anchor_node)[0]
        while next_node is not None:
            # exit the search when reaches to the graph output or finds the
            # leaf view node pair
            if next_node.op == "output" or (
                is_call_function_node(next_node)
                and helper_is_view_node(next_node)
                # leaf view node lies in graph outputs or has second consumer
                # node (copy_) which mutate input tensor when
                # keep_input_mutations is turned on.
                and (
                    next_node in fw_outputs
                    or [u for u in helper_get_node_users(next_node) if u in input_mutations_nodes]
                )
            ):
                break

            anchor_node = next_node
            nodes_in_current_path.add(next_node)
            # look at next user node
            next_node = helper_get_node_users(next_node)[0]

        if next_node is None or next_node.op == "output":
            logger.debug("Not found valid leaf view node pair.")
            break

        # record leaf view nodes
        leaf_view_nodes_first = next_node
        leaf_view_nodes_second = None
        for u in helper_get_node_users(leaf_view_nodes_first):
            if u.op == "call_function" and helper_is_view_node(u):
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
            prefix_view_node_args = helper_get_node_args(prefix_view_node)
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
                and (
                    helper_is_view_node(prefix_view_node)
                    and not helper_is_decomposed_from_inplace_node(prefix_view_node)
                )
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
        arg_prefix_view_node = helper_get_node_args(prefix_view_node)[0]
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
            args = helper_get_node_args(node)
            for arg in args:
                if arg.meta["placement"] == "hpu_cluster":
                    node_target = arg.target.__name__.split(".")[0]
                    if helper_is_view_node(arg):
                        arg.meta["pass_meta_color"] = "red"
                    # getitem is special-cased here since it may have view args and break the view ops chain
                    elif node_target == "getitem" and helper_is_view_node(arg.args[0]):
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

        del node.meta["pass_meta_color"]

    ctx.graph_module.graph.lint()
    ctx.graph_module.recompile()

    return graph_changed


def wrap_random_ops(input_module: torch.fx.GraphModule):
    """
    This pass goes through habana cluster and:
    - replaces random ops with habana wrappers,
    - creates seed and counter tensor for habana_seed_generator,
    - feeds habana wrappers with generated seed tensors.
    """

    random_ops = [node for node in input_module.graph.nodes if is_random_op(node)]

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

    for i, node in enumerate(random_ops):
        with input_module.graph.inserting_before(node):
            seed = input_module.graph.call_function(torch.select, (seeds, 0, i), {})
            random_node = input_module.graph.call_function(*random_op_inputs(node, seed))
            node.replace_all_uses_with(random_node, propagate_meta=True)
            random_node.meta.update(node.meta)
            input_module.graph.erase_node(node)

    input_module.recompile()


def pass_compile_clusters(ctx: OptimizerContext):
    """
    This pass goes through each node in the main module. For each generated HPU cluster
    there will be "call_module" OP. For each such module create JIT IR and pass
    it to the HPU backend for recipe compilation and substitute the target with
    newly compiled one.
    """

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

            shape = tensor_size[0]
            dims = len(shape)
            output_size_str = "["
            for dim, sz in enumerate(shape):
                sz_str = pexpr(sz)
                sz_str = sympify_expression(sz_str)
                output_size_str = output_size_str + str(sz_str)
                if dim < dims - 1:
                    output_size_str += ","
            output_size_str += "]"
            return output_size_str

        for node in Jit_graph.nodes():
            if node.kind() in jit_node_skip_list:
                continue

            fx_subname = get_fx_subname(node.kind())
            next_fx_idx, fx_node = get_matched_fx_node(fx_nodes, fx_count, fx_subname)
            logger.debug("Matched nodes, FX node: %s JIT node: %s fx_count: %d", fx_subname, fx_node, fx_count)
            fx_count = next_fx_idx

            if fx_node is None:
                logger.debug("Not found a matching FX node for node name: %s !!!", fx_subname)
                continue

            output_size_str = "[]"
            if "output_shapes" in fx_node.meta:
                output_size_str = create_output_size(fx_node.meta["output_shapes"])
            node.s_("output_shapes", output_size_str)

    def jit_node_annotation_propagation(jit_ir, fx_module):
        """
        This pass aims to directly manipulate JIT IR to set hints to node's
        attribute.
        """

        def extract_dict_from_str(hints_str):
            hint_values = None
            if hints_str is None:
                return None
            else:
                assert isinstance(hints_str, str)
                hints_str = hints_str.strip()
                if hints_str:
                    hint_values = eval(hints_str)

            if hint_values and isinstance(hint_values, dict):
                return hint_values

            return None

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
            context_hints = extract_dict_from_str(fx_node.meta.get("context_hints", None))
            if context_hints is None:
                continue
            else:
                logger.debug("node {} has context hints {}".format(fx_node_name, context_hints))
                # combine hints into a single string in format "name1:value1;[name2:value2;]"
                hints_str = ""
                for k, v in context_hints.items():
                    hints_str += "".join([k, ":", str(v), ";"])
                jit_node.s_("hints", hints_str)
                logger.debug("set hints for jit node", jit_node)
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

            # Submodule dynamicity has to recheck and set to the collable.
            is_submod_dynamic = is_module_dynamic(submod)

            if refine_dynamic:
                is_submod_dynamic = is_submod_dynamic or get_dynamic_config_value()

            if is_submod_dynamic and optim_output_sif_ds:
                jit_node_shape_propagation(jit_ir_function, submod_updated)

            callable_recipe = get_callable_recipe(
                jit_ir_function,
                submod,
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


from torch.fx.passes.reinplace import _FunctionalizationMetadataProp

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
                for node in mutated_arg.users:
                    node_users = list(node.users)
                    if len(node_users) == 1 and node_users[0].target == torch.ops._c10d_functional.wait_tensor.default:

                        wait_tensor_node = node_users[0]
                        wait_tensor_node_users = list(wait_tensor_node.users)
                        if (
                            len(wait_tensor_node_users) == 1
                            and wait_tensor_node_users[0].target == torch.ops.aten.copy.default
                        ):

                            copy_node = wait_tensor_node_users[0]
                            dst = node.args[0]
                            src = node.args[1]
                            dst_base = (
                                dst.meta["view_of"].meta["fake_result"]
                                if "view_of" in dst.meta
                                else dst.meta["fake_result"]
                            )
                            arg_base = (
                                mutated_arg.meta["view_of"].meta["fake_result"]
                                if "view_of" in mutated_arg.meta
                                else mutated_arg.meta["fake_result"]
                            )
                            if dst_base.untyped_storage()._cdata == arg_base.untyped_storage()._cdata:
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

    _FunctionalizationMetadataProp(ctx.graph_module).propagate(*(ctx.example_inputs))
    reinplace_collective_ops(ctx.graph_module)

    return graph_changed


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
            or not helper_is_node_supported(node=node)
        ):
            continue
        to_remove = []
        for u in node.users:
            if u.op != "call_function" or not helper_is_node_supported(node=u):
                break
            if u.target == torch.ops.aten.addmm.default:
                bias, inp, _ = list(u.args)
                weight = list(node.args)[0]
                new_args = (inp, weight, bias)
            elif u.target == torch.ops.aten.mm.default:
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

    ctx.graph_module = helper_post_pass_finalize(input_module=ctx.graph_module)

    """
    The following sub-graph rewriter removes the redundant reshapes that are added
    by aot autograd as part of lowering linear to t + mm/addmm as the above rewriter
    has replaced the pattern with linear
    """
    for node in ctx.graph_module.graph.nodes:
        if (
            node.op != "call_function"
            or node.target != torch.ops.aten.linear
            or not helper_is_node_supported(node=node)
        ):
            continue
        before = node.args[0]
        after = next(iter(node.users))
        cond_after = False
        if len(node.users) == 1 and after.target == torch.ops.aten.view.default and helper_is_node_supported(after):
            cond_after = True
        cond_before = False
        if len(before.users) == 1 and before.target == torch.ops.aten.view.default and helper_is_node_supported(before):
            cond_before = True
        if cond_after and cond_before:
            real_input = before.args[0]
            new_args = list(node.args)
            new_args[0] = real_input
            node.args = tuple(new_args)
            after.replace_all_uses_with(node)
            node.meta.update(after.meta)
            ctx.graph_module = helper_post_pass_finalize(input_module=ctx.graph_module)

    return graph_changed
