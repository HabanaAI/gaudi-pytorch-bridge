###############################################################################
# Copyright (C) 2023 Habana Labs, Ltd. an Intel Company
# All Rights Reserved.
#
# Unauthorized copying of this file or any element(s) within it, via any medium
# is strictly prohibited.
# This file contains Habana Labs, Ltd. proprietary and confidential information
# and is subject to the confidentiality and license agreements under which it
# was provided.
#
###############################################################################

import os
import copy
import torch
import contextlib
import habana_frameworks.torch.internal.bridge_config as bc

from enum import Enum
from typing import List, Optional
from dataclasses import dataclass
from packaging.version import Version
from torch.fx.experimental.proxy_tensor import py_sym_types

from .shared_layer import is_eager_fallback_required
from .recipe_compiler import get_callable_recipe
from .logger import get_compile_backend_logger
from .random_utils import is_random_op, random_op_inputs
from .symbolic_execution import SymExprNodeManager
from habana_frameworks.torch.utils.debug.dynamo_utils import FxGraphAnalyzer

logger = get_compile_backend_logger()

# Copy of partitoner module from native pytroch-fork along with the
# mentioend PR changes are kept in .partitioner.py file. Below code will
# be rolled back to .partitioner.py file once the below PR is merged.
# PR: https://github.com/pytorch/pytorch/pull/115621
from typing import Mapping
from torch.fx.passes.operator_support import OperatorSupport
from .partitioner import CapabilityBasedPartitioner

class HabanaClusterOperatorSupport(OperatorSupport):
    def is_node_supported(self, submodules: Mapping[str, torch.nn.Module], node: torch.fx.Node) -> bool:
        return node.meta["placement"] == "hpu_cluster"


class HabanaPartitioner(CapabilityBasedPartitioner):
    def __init__(self, graph_module: torch.fx.GraphModule):
        super().__init__(
            graph_module,
            HabanaClusterOperatorSupport(),
            allows_single_node_partition=True,
        )

def _is_cpu_scalar_or_symbolic_scalar(node: torch.fx.Node) -> bool:
    if node.type in [int, float]:
        assert node.meta["output_device"] == torch.device('cpu')
        return True
    else:
        return False

def _is_legacy_pt():
    if Version(Version(torch.__version__).base_version) < Version("2.1"):
        return True
    return False


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
            if (
                isinstance(meta_val, FakeTensor)
                and meta_val._has_symbolic_sizes_strides
            ) or isinstance(meta_val, py_sym_types):
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

class OptimizationPassPlacement(Enum):
    PRE_PARTITIONER = 1
    PARTITIONER = 2
    POST_PARTITIONER = 3


@dataclass
class OptimizerContext:
    graph_module: torch.fx.GraphModule
    example_inputs: List[torch.Tensor]
    is_training: bool
    is_backward: bool
    is_dynamic: bool
    uses_aot: bool
    stage: OptimizationPassPlacement
    current_partitions: List


def optimize_graph(
    stage: OptimizationPassPlacement,
    graph_module: torch.fx.GraphModule,
    example_inputs: List[torch.Tensor],
    is_training: bool,
    is_backward: bool,
    uses_aot: bool,
) -> bool:
    """
    This function rans optimizations of specified stage, if anything in the
    graph has changed, it will return True.

    Specific pass can be disabled by providing env in the form of:
    PT_HPU_DISABLE_<pass_name>=True

    For example:
    PT_HPU_DISABLE_pass_eagerize_leaf_views=True
    """
    if not uses_aot:
        # If backend used by the user does not use AOT then we cannot be sure whether
        # it is properly functionalized, meaning we should not call eliminate_dead_code
        # over it as it is not sound usage of Dead Code Elimination:
        # https://github.com/pytorch/pytorch/issues/68301
        # To satisfy above, we will monkey patch this function for optimizer scope so no
        # pass can do this silently in non-aot mode.
        original_dce_func = torch.fx.Graph.eliminate_dead_code

        def dummy_dce_raise(*args, **kwargs):
            raise Exception(
                "Tried to call DCE in possibly non-functionalized graph."
                "Make sure you add proper check in your code"
            )

        torch.fx.Graph.eliminate_dead_code = dummy_dce_raise

    # In all the three stages of partitioner, dynamicity has to be detected
    # from graph_module.
    is_dynamic = is_module_dynamic(graph_module)

    ctx = OptimizerContext(
        graph_module,
        example_inputs,
        is_training,
        is_backward,
        is_dynamic,
        uses_aot,
        stage,
        None,
    )

    graph_changed = False
    for optimization_pass in get_passes(stage):
        pass_name = optimization_pass.__name__
        env_name = "PT_HPU_DISABLE_" + pass_name
        if os.getenv(env_name, "").upper() in ["ON", "1", "YES", "TRUE", "Y"]:
            logger.debug("pass %s was disabled by env at stage %s", pass_name, stage)
        else:
            logger.debug("running %s pass at stage %s", pass_name, stage)
            graph_changed = optimization_pass(ctx) or graph_changed

    if not uses_aot:
        # Bring back original state.
        torch.fx.Graph.eliminate_dead_code = original_dce_func

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
    if stage == OptimizationPassPlacement.PRE_PARTITIONER:
        return [
            # These passes will be ran once, they always get and produce a flat graph without submodules.
            pass_graph_print,
            pass_fake_propagation,
            pass_wa_mixed_devices,  # This is W/A for Adam having CPU scalar tensors parameters.
            pass_mark_placement,
            pass_graph_print,
        ]
    elif stage == OptimizationPassPlacement.PARTITIONER:
        return [
            # These passes will prepare proper placement for some corner-cases.
            pass_handle_negative_dims,
            pass_handle_view_before_inplace_compute_ops,
            pass_graph_print,
            pass_eagerize_leaf_views,
            pass_replace_sym_size,
            pass_propose_partitions,
            pass_merge_paths,
            # This is final pass that creates final submoduled graph.
            pass_fuse_partitions,
            pass_make_symints_available,
            pass_graph_print,
            # Workarounds after partitioner phase.
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
        "slice",
        "select",
        "squeeze",
        "unsqueeze",
        "expand",
        "transpose",
        "t",
        "permute",
        "split",
    ]

    return node_target in view_ops


def helper_get_node_args(node: torch.fx.Node):
    """
    This helper function get inputs to specific node. It should supports
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


def pass_replace_sym_size(ctx: OptimizerContext) -> bool:
    if not ctx.is_dynamic:
        return True

    graph_changed = False
    py_node_manager = SymExprNodeManager(ctx.graph_module)

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
            tmeta_val = node.meta.get('val', node.meta.get('tensor_meta', None))
            if isinstance(tmeta_val, py_sym_types):
                py_node_manager.add_sym_placeholder(tmeta_val, node)
            py_node_manager.set_insert_point(node)

        if node.target == torch.ops.aten.sym_size:
            process_symsize(node)
            graph_changed = True

    if graph_changed:
        # Clean up the graph and log the situation.
        if ctx.uses_aot:
            ctx.graph_module.graph.eliminate_dead_code()
        else:
            # Running DCE on graph that might not be functionalized in unsafe:
            # https://github.com/pytorch/pytorch/issues/68301
            logger.warning("Disallowed to run DCE in non-aot mode.")
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
            missing_symint_list = get_missing_symbolic_int_input_nodes(
                symint_list, node
            )
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

    device = None
    dtypes = []
    layouts = []
    output_shapes = []
    output_strides = []
    output_contiguous = []

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
    elif type(result) is torch.SymInt:
        device = torch.device("cpu")
        dtypes = [None]
        layouts = [None]
        output_shapes = [None]
        output_strides = [None]
        output_contiguous = [None]

        node.type = int
    elif type(result) is torch.SymFloat:
        device = torch.device("cpu")
        dtypes = [None]
        layouts = [None]
        output_shapes = [None]
        output_strides = [None]
        output_contiguous = [None]

        node.type = float
    elif isinstance(result, int):
        device = torch.device("cpu")
        dtypes = [None]
        layouts = [None]
        output_shapes = [None]
        output_strides = [None]
        output_contiguous = [None]

        node.type = int

    elif isinstance(result, float):
        device = torch.device("cpu")
        dtypes = [None]
        layouts = [None]
        output_shapes = [None]
        output_strides = [None]
        output_contiguous = [None]

        node.type = float
    else:
        devices = []
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

    from torch._subclasses.fake_tensor import FakeTensorMode
    from torch._dynamo.utils import detect_fake_mode

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
            result = super().run_node(node)
            args, kwargs = self.fetch_args_kwargs_from_env(node)
            node.val_args = args
            node.val_kwargs = kwargs
            fill_propagated_tensor_metadata_to_node(result, node)

            return result

        def propagate(self, *args):
            fake_args = [
                self._mode.from_tensor(a) if isinstance(a, torch.Tensor) else a
                for a in args
            ]
            return self.propagate_dont_convert_inputs(*fake_args)

        def propagate_dont_convert_inputs(self, *args):
            with self._mode:
                return super().run(*args)

    fake_mode = detect_fake_mode(ctx.example_inputs)
    with torch.autocast(enabled=False, device_type="hpu"), torch.autocast(
        enabled=False, device_type="cpu"
    ):
        # Disabling autocast in fake tensor propagation as autocasting has been
        # already done and all dtypes has been already deduced.
        if not fake_mode or not ctx.uses_aot:
            fake_mode = torch._subclasses.FakeTensorMode(allow_non_fake_inputs=True)
            TensorInfoPropagation(ctx.graph_module, fake_mode).propagate(
                *ctx.example_inputs
            )
        else:
            TensorInfoPropagation(
                ctx.graph_module, fake_mode
            ).propagate_dont_convert_inputs(*ctx.example_inputs)

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
        logger.warn(
            "It seems graph wasn't functionalized, fixing empty output node."
        )
        ctx.graph_module.graph.erase_node(output_node)
        ctx.graph_module.graph.node_copy(output_node)
        ctx.graph_module.recompile()
        graph_changed = True

    # WORKAROUND END

    return graph_changed


def pass_fake_propagation_legacy(ctx: OptimizerContext) -> bool:
    """
    This function contains FakeMode propagation implementation for PT2.0
    """

    from torch._dynamo.utils import fake_mode_from_tensors, deepcopy_to_fake_tensor
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

    with torch.autocast(enabled=False, device_type="hpu"), torch.autocast(
        enabled=False, device_type="cpu"
    ):
        # Disabling autocast in fake tensor propagation as autocasting has been
        # already done and all dtypes has been already deduced.
        LegacyTensorInfoPropagation(
            ctx.graph_module, fakemode_already_enabled, fake_mode
        ).propagate(*fake_inputs)

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

    ctx.current_partitions = HabanaPartitioner(ctx.graph_module).propose_partitions()

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
                    and not _is_cpu_scalar_or_symbolic_scalar(arg)
                ):
                    nodes_to_fix_list.append(node)
                    break

    for node in nodes_to_fix_list:
        for arg in node.args:
            if (
                isinstance(arg, torch.fx.Node)
                and arg.meta["output_device"].type != "hpu"
            ):
                with ctx.graph_module.graph.inserting_before(node):
                    input_copy_node = ctx.graph_module.graph.call_function(
                        torch.ops.aten._to_copy.default,
                        (arg,),
                        {"device": torch.device("hpu")},
                    )
                    input_copy_node.meta["output_device"] = torch.device("hpu")
                    input_copy_node.meta["output_dtypes"] = [
                        arg.meta["output_dtypes"][0]
                    ]
                    input_copy_node.meta["output_layouts"] = [
                        arg.meta["output_layouts"][0]
                    ]
                    input_copy_node.meta["output_shapes"] = [
                        arg.meta["output_shapes"][0]
                    ]
                    input_copy_node.meta["output_strides"] = [
                        arg.meta["output_strides"][0]
                    ]
                    input_copy_node.meta["output_contiguous"] = [
                        arg.meta["output_contiguous"][0]
                    ]
                    node.replace_input_with(arg, input_copy_node)
                graph_changed = True

    if graph_changed:
        # Clean up the graph and log the situation.
        if ctx.uses_aot:
            ctx.graph_module.graph.eliminate_dead_code()
        else:
            # Running DCE on graph that might not be functionalized in unsafe:
            # https://github.com/pytorch/pytorch/issues/68301
            logger.warn("Disallowed to run DCE in non-aot mode.")

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
        if node.op == "placeholder" or node.op == "output" or node.op == "get_attr":
            placement = "eager"
        elif node.op == "call_function" and "to_copy" in node.target.__name__:
            input_node = None
            for arg in node.args:
                if isinstance(arg, torch.fx.Node):
                    input_node = arg
                    break

            assert input_node is not None

            # Internal HPU copies should be placed in the clusters.
            if (
                input_node.meta["output_device"].type == "hpu"
                and node.meta["output_device"].type == "hpu"
            ):
                placement = "hpu_cluster"
            else:
                placement = "eager"
        elif node.op == "call_function" and is_eager_fallback_required(
            node, is_dynamic=ctx.is_dynamic
        ):
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

        assert placement is not None

        # Meta for the node should not be created yet. BUT...
        # ...it happens that placeholder nodes might be reused between FWD and BWD.
        # They are always placed in eager though, so it should not be an issue.
        if "placement" in node.meta:
            logger.debug("Node {} of type {} has had it's placement already set" ,node, node.op)
            assert node.meta["placement"] == placement

        node.meta["placement"] = placement

    return True


def pass_merge_paths(ctx: OptimizerContext) -> bool:
    """
    This pass that will merge parallel partitions.
    """
    assert ctx.stage == OptimizationPassPlacement.PARTITIONER
    assert ctx.graph_module is not None
    assert ctx.current_partitions is not None

    logger.debug(
        f"Merging parallel graph path. Partition cnt: {len(ctx.current_partitions)}"
    )

    graph_changed = False

    if len(ctx.current_partitions) == 1:
        logger.debug(f"Merging skipped for single partition graph")
        # In case of single partition there is no merging to be done
        return graph_changed

    class ColorGraph:
        def __init__(self):
            self.OUTPUT_COLOR = 0
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
                    self.output_colors.add(color)
                    continue
                if color not in self.partition_colors:
                    self.colors_to_remove.add(color)
                    continue

        def _merge_outputs(self):
            for output_color in self.output_colors:
                for v in self._graph.values():
                    if output_color in v:
                        v.remove(output_color)
                        v.add(self.OUTPUT_COLOR)
                self.all_colors.remove(output_color)
            self.output_colors = set()
            self.output_colors.add(self.OUTPUT_COLOR)
            self.all_colors.add(self.OUTPUT_COLOR)

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
            self._merge_outputs()
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
    node_name = ''
    view_dim_index = 0
    py_node_manager = None

    @staticmethod
    def required(node):
        node_name = node.target.__name__.split(".")[0]
        resolve_negative_dim.node_name = node_name
        # This is list of OPs with negative Dims.
        negative_dim_ops = [
            "view",
        ]

        from torch.fx.experimental.proxy_tensor import py_sym_types
        from torch._subclasses.fake_tensor import FakeTensor
        if node_name in negative_dim_ops:
            if node_name == 'view':
                node_arg0 = node.args[0]
                meta_val = node_arg0.meta.get('val', node.meta.get('tensor_meta', None))
                if (
                    (isinstance(meta_val, FakeTensor) and meta_val._has_symbolic_sizes_strides)
                    or isinstance(meta_val, py_sym_types)
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
                meta_val = node.meta.get('val', node.meta.get('tensor_meta', None))
                new_args1 = list(meta_val.size())
            else:
                sym_size_expr = node.meta["output_shapes"][0][cls.view_dim_index]
                meta_val = node.meta.get('val', node.meta.get('tensor_meta', None))
                value = copy.copy(meta_val.shape[cls.view_dim_index])
                new_node = cls.py_node_manager.get_or_create(sym_size_expr, int)
                new_node.meta['val'] = value
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
                    (node.args[0], new_args1,),
                    {},
                )
                node.replace_all_uses_with(view_new_node, propagate_meta=True)

            ctx.graph_module.recompile()
            ctx.graph_module.graph.eliminate_dead_code()
        return True

    def __new__(cls, ctx, node):
        if cls.node_name == 'view':
            return cls.__resolve_view_shapes(ctx, node)
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
            tmeta_val = node.meta.get('val', node.meta.get('tensor_meta', None))
            if isinstance(tmeta_val, py_sym_types):
                py_node_manager.add_sym_placeholder(tmeta_val, node)
        if node.op == "call_function":
            if resolve_negative_dim.required(node):
                py_node_manager.set_insert_point(node.prev)
                graph_changed = resolve_negative_dim(ctx, node)
    return graph_changed

def helper_is_compute_node(node):
    # return false if node is a view node, input node or output node
    return (
        (not helper_is_view_node(node))
        and (node.op != "placeholder")
        and (node.op != "output")
    )


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
    output strides of `as_strided` node is filled with strides of original strides.
    The strides propagation flow of above graph is like below:

    [3, 1]
       |
       t
       |
    [1, 3]    Scalar
        \     /
          mul
           |
         [2, 1] (contiguous)
           |
          t_1
           |
        [1, 2] (copy to original input due to input mutation)
           |
       as_strided (newly inserted, restore un-viewed strides)
           |
        [3, 1]
           |
          t_2
           |
        [1, 3] -> b
    """

    def helper_calculate_default_strides(sizes):
        # Calculate default strides for given size
        if len(sizes) == 0:
            return []

        reversed_strides = [1]
        for size in reversed(sizes[1:]):
            reversed_strides.append(size * reversed_strides[-1])
        return list(reversed(reversed_strides))

    def is_output_contiguous_strides(node):
        contiguous_strides = helper_calculate_default_strides(
            node.meta["output_shapes"][0]
        )
        actual_strides = node.meta["output_strides"][0]
        return contiguous_strides == list(actual_strides)

    def helper_get_node_users(node):
        assert isinstance(node, torch.fx.Node)
        node_list = list(node.users.keys())
        if len(node_list) == 0:
            return [None]
        return node_list

    assert ctx.graph_module is not None
    graph_changed = False

    fw_output_node = [
        node for node in ctx.graph_module.graph.nodes if node.op == "output"
    ][0]
    fw_outputs = fw_output_node.args[0]
    # no inplace op
    if len(fw_outputs) < 2:
        return graph_changed

    # make sure the graph nodes is topologically sorted
    ctx.graph_module.graph.lint()

    nodes_is_visted = set()
    for node in ctx.graph_module.graph.nodes:
        if node.op != "call_function" or node in nodes_is_visted:
            continue

        # mark current node as visited
        nodes_is_visted.add(node)

        view_in_node = None
        # find HPU node which is actually an inplace node
        # we may not need to check placement ('hpu_cluster') due to no matter
        # what strides compute node generates, the later inserted as_strided node
        # should use the correct strides
        if not helper_is_compute_node(node):
            continue

        is_duplicate_chain = False
        current_chain = [node]
        # take node as the start point of potential inplace op chain
        # try to find the end point of this chain
        end_node_in_chain = node
        next_node = helper_get_node_users(end_node_in_chain)[0]
        while next_node is not None:
            if next_node in nodes_is_visted:
                # next_node is in previous chain, exit current chain search
                is_duplicate_chain = True
                break

            # found first node which has view user, assumes it's the end
            # point of chain. Or it reaches the graph output
            if next_node.op == "output" or (
                next_node.op == "call_function"
                and helper_is_view_node(next_node)
                and next_node in fw_outputs
            ):
                break

            end_node_in_chain = next_node
            nodes_is_visted.add(next_node)
            current_chain.append(next_node)
            # look at next user node
            next_node = helper_get_node_users(next_node)[0]

        # detect duplicate chain or no valid consequent view nodes
        if is_duplicate_chain or next_node is None or next_node.op == "output":
            break

        # collect candidate view nodes before this chain
        # bottom up from the end_node_in_chain
        view_in_node = end_node_in_chain
        while view_in_node.op != "placeholder":
            view_in_node_args = helper_get_node_args(view_in_node)
            if len(view_in_node_args) > 0:
                # workaround for where whose data input is third argument
                if view_in_node.target == torch.ops.aten.where.self:
                    view_in_node = view_in_node_args[2]
                else:
                    view_in_node = view_in_node_args[0]
            else:
                view_in_node = None
                break

            if (
                view_in_node.op == "call_function"
                and helper_is_view_node(view_in_node)
                and not view_in_node in current_chain
            ):
                # found
                break

        if view_in_node is None or view_in_node.op == "placeholder":
            continue

        nodes_to_change = []
        # check if current node's output node is the same view node as input node
        for user in end_node_in_chain.users:
            out_node = helper_get_node_users(user)[0]
            # there are same view before / after the current node
            # actually we may also need check if the same arguments, for
            # example dim0 and dim1 for torch.transpose
            if (
                user.target == view_in_node.target
                # view-pair for inplace op
                and (out_node is not None and out_node.target == view_in_node.target)
                # this based on the fact that HPU node only produces contiguous output
                and (not is_output_contiguous_strides(view_in_node))
                and (is_output_contiguous_strides(end_node_in_chain))
            ):
                nodes_to_change.append(user)

        # node which has original strides information
        arg_view_in_node = helper_get_node_args(view_in_node)[0]

        # found ops which has wrong strides
        if nodes_to_change:
            # node which has original strides information
            arg_view_in_node = helper_get_node_args(view_in_node)[0]
            node_to_change = nodes_to_change[0]

            with ctx.graph_module.graph.inserting_after(node_to_change):
                # input node
                new_args = [
                    node_to_change,
                ]
                # sizes of inserted as_strided_0 node
                new_args.append(arg_view_in_node.meta["output_shapes"][0])
                # strides of inserted as_strided_0 node
                new_args.append(arg_view_in_node.meta["output_strides"][0])
                new_kwargs = None
                as_strided_0 = ctx.graph_module.graph.create_node(
                    node_to_change.op,
                    torch.ops.aten.as_strided.default,
                    tuple(new_args),
                    new_kwargs,
                    "as_strided_0",
                    node_to_change.type,
                )
                as_strided_0.meta = copy.copy(arg_view_in_node.meta)
                # reset output_strides in case it can be propagated later
                as_strided_0.meta["output_strides"] = None

            # connect as_stride_0 node to original node_to_change's user
            list(node_to_change.users.keys())[0].replace_input_with(
                node_to_change, as_strided_0
            )

            graph_changed = True

    ctx.graph_module.recompile()

    # another metadata (only strides) propagation needed due to newly inserted node
    if graph_changed:
        pass_fake_propagation(ctx)
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

    assert ctx.stage == OptimizationPassPlacement.PARTITIONER
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
                if (
                    dst.meta["placement"] == "hpu_cluster"
                    and dst.meta["pass_meta_color"] != "red"
                ) or (dst.meta["pass_meta_color"] == "blue"):
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
                if (
                    dst.meta["pass_meta_color"] != "red"
                    and dst.meta["placement"] == "hpu_cluster"
                ):
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

    random_count = 0
    random_ops = []

    for node in input_module.graph.nodes:
        if is_random_op(node):
            random_count += 1
            random_ops.append(node)

    if random_count == 0:
        return

    with input_module.graph.inserting_before():
        counter_pl = input_module.graph.placeholder("counter_pl")
        seed_pl = input_module.graph.placeholder("seed_pl")

    with input_module.graph.inserting_after(counter_pl):
        seeds = input_module.graph.call_function(
            torch.ops.hpu.habana_seed_generator, (counter_pl, seed_pl, random_count), {}
        )
        add_inplace = input_module.graph.call_function(
            torch.ops.aten.add_, (counter_pl, random_count), {}
        )

    for i, node in enumerate(random_ops):
        with input_module.graph.inserting_before(node):
            seed = input_module.graph.call_function(torch.select, (seeds, 0, i), {})
            random_node = input_module.graph.call_function(
                *random_op_inputs(node, seed)
            )
            node.replace_all_uses_with(random_node, propagate_meta=True)
            input_module.graph.erase_node(node)

    input_module.recompile()


def pass_compile_clusters(ctx: OptimizerContext):
    """
    This pass goes through each node in the main module. For each generated HPU cluster
    there will be "call_module" OP. For each such module create JIT IR and pass
    it to the HPU backend for recipe compilation and substitute the target with
    newly compiled one.
    """

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

        return f

    num_subgraphs = 0
    refine_dynamic = bc.get_pt_hpu_enable_refine_dynamic_shapes()
    for n in ctx.graph_module.graph.nodes:
        logger.debug("Node: %s Op: %s Target: %s", n, n.op, n.target)

        if n.op == "call_module":
            assert not n.kwargs
            submod = ctx.graph_module.get_submodule(n.target)

            jit_ir_function = generate_jit_ir_from_module(submod)

            # Submodule dynamicity has to recheck and set to the collable.
            is_submod_dynamic = is_module_dynamic(submod)
            if refine_dynamic:
                is_submod_dynamic = is_submod_dynamic or  get_dynamic_config_value()

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
