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

from enum import Enum
from typing import List, Optional
from dataclasses import dataclass

from .shared_layer import is_cpu_fallback_required
from .partitioner import HabanaPartitioner
from .recipe_compiler import get_callable_recipe
from .config import configuration_flags
from .logger import get_compile_backend_logger

logger = get_compile_backend_logger()


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
    stage: OptimizationPassPlacement
    current_partitions: List


def optimize_graph(
    stage: OptimizationPassPlacement,
    graph_module: torch.fx.GraphModule,
    example_inputs: List[torch.Tensor],
    is_training: bool,
    is_backward: bool
) -> bool:
    """
    This function rans optimizations of specified stage, if anything in the
    graph has changed, it will return True.

    Specific pass can be disabled by providing env in the form of:
    PT_HPU_DISABLE_<pass_name>=True

    For example:
    PT_HPU_DISABLE_pass_eagerize_leaf_views=True
    """
    from torch._dynamo import config
    is_dynamic = config.dynamic_shapes
    ctx = OptimizerContext(graph_module, example_inputs, is_training, is_backward, is_dynamic, stage, None)

    graph_changed = False
    for optimization_pass in get_passes(stage):
        pass_name = optimization_pass.__name__
        env_name = "PT_HPU_DISABLE_" + pass_name
        if os.getenv(env_name, "").upper() in ["ON", "1", "YES", "TRUE", "Y"]:
            logger.debug("pass %s was disabled by env at stage %s", pass_name, stage)
        else:
            logger.debug("running %s pass at stage %s", pass_name, stage)
            graph_changed = optimization_pass(ctx) or graph_changed

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
            pass_mark_fallbacks,
            pass_transform_fallbacks,
            pass_skip_copies,
            pass_graph_print,
        ]
    elif stage == OptimizationPassPlacement.PARTITIONER:
        return [
            # These passes will prepare proper placement for some corner-cases.
            pass_eagerize_leaf_views,
            pass_propose_partitions,
            pass_merge_paths,

            # This is final pass that creates final submoduled graph.
            pass_fuse_partitions,
        ]
    elif stage == OptimizationPassPlacement.POST_PARTITIONER:
        return [
            # These passes will be ran once, they have to work on graph with submodules.
            pass_graph_print,
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
        if isinstance(node.args[0], tuple) or isinstance(node.args[0], list) or isinstance(node.args[0], torch.fx.immutable_collections.immutable_list):
            args = node.args[0]

    if isinstance(args, tuple) or isinstance(args, list) or isinstance(args, torch.fx.immutable_collections.immutable_list):
        cleaned_args = []
        for arg in args:
            if isinstance(arg, torch.fx.Node):
                cleaned_args.append(arg)
    else:
        cleaned_args = args

    return cleaned_args

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

def fill_propagated_tensor_metadata_to_node(result: torch.Tensor, node: torch.fx.Node):
    """
    This function takes out basic information from propagated fake tensor, like
    dtype, layout and device and puts it to the node that created it.
    """

    device = None
    dtypes = []
    layouts = []

    if (
        type(result) is torch._subclasses.FakeTensor
        or type(result) is torch._subclasses.fake_tensor.FakeTensor
        or type(result) is torch.Tensor
        or type(result) is torch.nn.parameter.Parameter
    ):
        device = result.device
        dtypes = [result.dtype]
        layouts = [result.layout]

        logger.debug("    result shape: %s", result.shape)
    elif type(result) is torch.SymInt:
        device = torch.device("cpu")
        dtypes = [None]
        layouts = [None]

        node.type = int
    elif type(result) is torch.SymFloat:
        device = torch.device("cpu")
        dtypes = [None]
        layouts = [None]

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
    if "output_device" in node.meta or "output_dtypes" in node.meta or "output_layouts" in node.meta:
        assert node.op == "placeholder"

        assert node.meta["output_device"] == device
        assert node.meta["output_dtypes"] == dtypes
        assert node.meta["output_layouts"] == layouts

    node.meta["output_device"] = device
    node.meta["output_dtypes"] = dtypes
    node.meta["output_layouts"] = layouts

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

        def __init__(self, graph_module: torch.fx.GraphModule, fake_mode: Optional[FakeTensorMode] = None):
            super().__init__(graph_module)
            if fake_mode is None:
                fake_mode = FakeTensorMode()
            self._mode = fake_mode

        def run_node(self, node: torch.fx.Node):
            result = super().run_node(node)

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
    if not fake_mode:
        fake_mode = torch._subclasses.FakeTensorMode(allow_non_fake_inputs=True)
        TensorInfoPropagation(ctx.graph_module, fake_mode).propagate(*ctx.example_inputs)
    else:
        TensorInfoPropagation(ctx.graph_module, fake_mode).propagate_dont_convert_inputs(
            *ctx.example_inputs
        )

    return True


def pass_fake_propagation_legacy(ctx: OptimizerContext) -> bool:
    """
    This function contains FakeMode propagation implementation for PT2.0
    """

    from torch._dynamo.utils import (
        fake_mode_from_tensors,
        deepcopy_to_fake_tensor
    )
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

    LegacyTensorInfoPropagation(ctx.graph_module, fakemode_already_enabled, fake_mode).propagate(*fake_inputs)

    return True


def pass_fake_propagation(ctx: OptimizerContext) -> bool:
    """
    This pass makes sure that input tensors are in fake mode so we don't
    make any actual computation. Then it propagates tensor metadata into nodes.
    """
    from packaging.version import Version

    if Version(torch.__version__) < Version("2.1"):
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
                if isinstance(arg, torch.fx.Node) and arg.meta["output_device"].type != "hpu":
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

    def is_op_unsupported_in_graph(node):
        node_target = node.target.__name__.split(".")[0]

        # This is list of OPs that need to be ran eagerly at this point of time.
        unsupported_ops = [
            # Tensor creation OPs.
            "zeros",
            "ones",
            # Random OPs.
            "seed",
            "manual_seed",
            "initial_seed",
            "get_rng_state",
            "set_rng_state",
            "rand",
            "randn",
            "randint",
            "rand_like",
            "randn_like",
            "randint_like",
            "randperm",
            "poisson",
            "multinomial",
            "normal",
            # Other
            "slice_backward", # SW-146680
            "addcmul",
            "index",  # SW-146773
            "split",   # SW-149515
            "new_empty_strided"   # SW-149882
        ]

        if node_target in unsupported_ops:
            return True

        # This is a list of OPs that are not supported in the graph mode only for specific dtypes.
        unsupported_types = {
            "permute": torch.int64
        }

        if node_target in unsupported_types:
            for output_dtype in node.meta["output_dtypes"]:
                if output_dtype == unsupported_types[node_target]:
                    return True

        return False

    for node in ctx.graph_module.graph.nodes:
        placement = None
        if node.op == "placeholder" or node.op == "output":
            placement = "eager"
        elif node.op == "call_function" and is_op_unsupported_in_graph(node):
            placement = "eager"
        elif node.op == "call_function" and "to_copy" in node.target.__name__:
            input_node = None
            for arg in node.args:
                if isinstance(arg, torch.fx.Node):
                    input_node = arg
                    break

            assert input_node is not None

            # Internal HPU copies should be placed in the clusters.
            if input_node.meta["output_device"].type == "hpu" and node.meta["output_device"].type == "hpu":
                placement = "hpu_cluster"
            else:
                placement = "eager"
        elif node.meta["output_device"].type == "hpu":
            # Current assumption is that if OP outputs HPU tensor, then all its inputs are also on HPU.
            # Let's create an assert that will fire in case this assumption proves wrong.
            for arg in node.args:
                if isinstance(arg, torch.fx.Node):
                    # If you got into this assert, we might need to rewrite this part so we cluster only
                    # these OPs that also have all inputs on HPU. Or debug why this OP have mixed device
                    # tensors, that could be the original issue here.
                    assert arg.meta["output_device"].type == "hpu"

            placement = "hpu_cluster"
        elif node.meta["output_device"].type == "cpu":
            placement = "eager"

        assert placement is not None

        # Meta for the node should not be created yet. BUT...
        # ...it happens that placeholder nodes might be reused between FWD and BWD.
        # They are always placed in eager though, so it should not be an issue.
        if "placement" in node.meta:
            assert node.op == "placeholder"

        node.meta["placement"] = placement

    return True


def pass_mark_fallbacks(ctx: OptimizerContext) -> bool:
    """
    This pass is supposed to find nodes requiring CPU fallback.
    If such node is found, mark it as requriing fallbacking.
    """

    assert ctx.graph_module is not None
    graph_changed = False
    for node in ctx.graph_module.graph.nodes:
        if is_cpu_fallback_required(node):
            if node.meta["placement"] != "cpufallback":
                node.meta["placement"] = "cpufallback"
                graph_changed = True

    return graph_changed


def pass_transform_fallbacks(ctx: OptimizerContext) -> bool:
    """
    This pass is supposed to find nodes annotated as requiring fallbacks,
    for each such node it will materialize the copies and move the actual
    OP to CPU.
    """

    assert ctx.graph_module is not None
    graph_changed = False
    fallbacked_ops_counter = {}

    for node in ctx.graph_module.graph.nodes:
        if node.meta["placement"] == "cpufallback":
            if node.target.__name__ in fallbacked_ops_counter:
                fallbacked_ops_counter[node.target.__name__] += 1
            else:
                fallbacked_ops_counter[node.target.__name__] = 1

            graph_changed = True
            with ctx.graph_module.graph.inserting_before(node):
                for arg in node.args:
                    if isinstance(arg, torch.fx.Node):
                        input_copy_node = ctx.graph_module.graph.call_function(
                            torch.ops.aten._to_copy.default,
                            (arg,),
                            {"device": torch.device("cpu")},
                        )
                        input_copy_node.meta["placement"] = "eager"
                        input_copy_node.meta["output_device"] = torch.device("cpu")
                        input_copy_node.meta["output_dtypes"] = [arg.meta["output_dtypes"][0]]
                        input_copy_node.meta["output_layouts"] = [arg.meta["output_layouts"][0]]
                        node.replace_input_with(arg, input_copy_node)

            # Check if this is tuple based output.
            is_tuple_output = False
            for user in node.users:
                if user.op == "call_function" and "getitem" in user.target.__name__:
                    is_tuple_output = True
                    break

            if not is_tuple_output:
                with ctx.graph_module.graph.inserting_after(node):
                    output_copy_node = ctx.graph_module.graph.call_function(
                        torch.ops.aten._to_copy.default,
                        (node,),
                        {"device": node.meta["output_device"]},
                    )
                    output_copy_node.meta["placement"] = "eager"
                    output_copy_node.meta["output_device"] = node.meta["output_device"]
                    output_copy_node.meta["output_dtypes"] = [node.meta["output_dtypes"][0]]
                    output_copy_node.meta["output_layouts"] = [node.meta["output_layouts"][0]]
                    node.replace_all_uses_with(output_copy_node)

                    # Above line will also replace the input of output
                    # conversion to itself.... fix it back.
                    output_copy_node.replace_input_with(output_copy_node, node)
            else:
                # We need to fall back getitems following the output tuple
                # instead of just the output itself.
                for user in node.users:
                    assert isinstance(user, torch.fx.Node)
                    assert "getitem" in user.target.__name__
                    with ctx.graph_module.graph.inserting_after(user):
                        output_copy_node = ctx.graph_module.graph.call_function(
                            torch.ops.aten._to_copy.default,
                            (user,),
                            {"device": user.meta["output_device"]},
                        )
                        output_copy_node.meta["placement"] = "eager"
                        output_copy_node.meta["output_device"] = user.meta["output_device"]
                        output_copy_node.meta["output_dtypes"] = [user.meta["output_dtypes"][0]]
                        output_copy_node.meta["output_layouts"] = [user.meta["output_layouts"][0]]
                        user.replace_all_uses_with(output_copy_node)

                        # Above line will also replace the input of output
                        # conversion to itself.... fix it back.
                        output_copy_node.replace_input_with(output_copy_node, user)

                    user.meta["placement"] = "eager"
                    user.meta["output_device"] = torch.device("cpu")

            node.meta["placement"] = "eager"
            node.meta["output_device"] = torch.device("cpu")

    return graph_changed


def pass_skip_copies(ctx: OptimizerContext) -> bool:
    """
    This pass is supposed to find cases where input to the node is copy between devices, which
    was created by the opposite copy. In that scenario we can just skip these copies. After we
    do that, we should remove dead code and recompile the FX graph.
    """

    assert ctx.graph_module is not None
    graph_changed = False

    for node in ctx.graph_module.graph.nodes:
        # For this specific node, check every input for copies. If there is a copy, follow the chain
        # to skip as many copies as possible.
        args = helper_get_node_args(node)

        for arg in args:
            if isinstance(arg, torch.fx.Node) and arg.op == "call_function" and "to_copy" in arg.target.__name__:
                # Candidate_node is a node that produces output we would
                # like out original input to skip to.
                candidate_node = None

                # Found copy, save original input metadata.
                original_device = arg.meta["output_device"]
                original_dtype = arg.meta["output_dtypes"][0]
                original_layout = arg.meta["output_layouts"][0]

                current_node = arg

                valid_chain = True
                while valid_chain:
                    chain_arg = helper_get_node_args(current_node)[0]

                    assert isinstance(chain_arg, torch.fx.Node)

                    # Follow the chain till dtype and layouts are matching.
                    chain_arg_device = chain_arg.meta["output_device"]
                    chain_arg_dtype = chain_arg.meta["output_dtypes"][0]
                    chain_arg_layout = chain_arg.meta["output_layouts"][0]

                    if chain_arg_dtype == original_dtype and chain_arg_layout == original_layout:
                        # Dtypes and layouts still match. If device is the same as original, save as
                        # candidate for final skip.
                        if chain_arg_device == original_device:
                            candidate_node = chain_arg
                    else:
                        # This chain is not longer valid, bail out.
                        valid_chain = False

                    if chain_arg.op == "call_function" and "to_copy" in chain_arg.target.__name__:
                        # This node is also a copy, let's go deeper.
                        current_node = chain_arg
                    else:
                        # This chain finished, bail out.
                        valid_chain = False

                if candidate_node is not None:
                    # We have candidate for skip. Skip it then.
                    node.replace_input_with(arg, candidate_node)
                    graph_changed = True

    # Clean up the graph.
    # The circular copy back to the input is being considered as dead code.
    # The whole graph is incorrectly eliminated
    # skip this optimization when keep_input_mutation is enabled
    if (configuration_flags["keep_input_mutations"] is False):
        ctx.graph_module.graph.eliminate_dead_code()
    ctx.graph_module.recompile()

    return graph_changed


def pass_merge_paths(ctx: OptimizerContext) -> bool:
    """
    Placeholder for pass that will merge parallel partitions.
    """

    assert ctx.stage == OptimizationPassPlacement.PARTITIONER
    assert ctx.graph_module is not None
    assert ctx.current_partitions is not None

    graph_changed = False

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
                if arg.meta["placement"] == "hpu_cluster" and helper_is_view_node(arg):
                    arg.meta["pass_meta_color"] = "red"

    # Find HPU view chains used by eager OPs that are also used by non-eager HPU ops ('blue' color - to be cloned).
    for node in reverse_nodes_list:
        if node.meta["pass_meta_color"] == "red":
            found_hpu_dst = False
            for dst in node.users:
                if (dst.meta["placement"] == "hpu_cluster" and dst.meta["pass_meta_color"] != "red") or (dst.meta["pass_meta_color"] == "blue"):
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
                new_node = ctx.graph_module.graph.create_node(node.op, node.target, node.args, node.kwargs, node.name, node.type)
                new_node.meta = copy.copy(node.meta)

            # Move non-red (HPU path) edges to the new node.
            nodes_to_change = []
            for dst in node.users:
                if dst.meta["pass_meta_color"] != "red":
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
    for n in ctx.graph_module.graph.nodes:
        logger.debug("Node: %s Op: %s Target: %s", n, n.op, n.target)

        if n.op == "call_module":
            assert not n.kwargs
            submod = ctx.graph_module.get_submodule(n.target)

            jit_ir_function = generate_jit_ir_from_module(submod)
            callable_recipe = get_callable_recipe(
                jit_ir_function, submod, is_training=ctx.is_training, is_dynamic=ctx.is_dynamic)

            ctx.graph_module.delete_submodule(n.target)
            ctx.graph_module.add_submodule(n.target, callable_recipe)

            num_subgraphs += 1

    logger.info("INFO: Number of subgraphs created:\n%s", num_subgraphs)

    return num_subgraphs != 0
