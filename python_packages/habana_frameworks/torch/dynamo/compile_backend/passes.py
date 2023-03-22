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
import logging
import torch
import contextlib

from enum import Enum
from typing import List
from dataclasses import dataclass

from .shared_layer import is_cpu_fallback_required
from .partitioner import HabanaPartitioner
from .recipe_compiler import get_callable_recipe

logger = logging.getLogger("aot_hpu_backend")


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
    ids_to_nodes: dict
    stage: OptimizationPassPlacement


def optimize_graph(
    stage: OptimizationPassPlacement,
    graph_module: torch.fx.GraphModule,
    example_inputs: List[torch.Tensor],
    is_training: bool,
    is_backward: bool,
    ids_to_nodes: dict = None,
) -> bool:
    """
    This function rans optimizations of specified stage, if anything in the
    graph has changed, it will return True.

    `ids_to_nodes` parameter is used for partitioner passes where we might
    work on both currently proposed partitioning and original `writeback` graph.

    Specific pass can be disabled by providing env in the form of:
    PT_HPU_DISABLE_<pass_name>=True

    For example:
    PT_HPU_DISABLE_pass_eagerize_leaf_views=True
    """
    from torch._dynamo import config
    is_dynamic = config.dynamic_shapes
    ctx = OptimizerContext(graph_module, example_inputs, is_training, is_backward, is_dynamic, ids_to_nodes, stage)

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
            pass_wa_mixed_devices, # This is W/A for Adam having CPU scalar tensors parameters.
            pass_mark_placement,
            pass_mark_fallbacks,
            pass_transform_fallbacks,
            pass_skip_copies,
            pass_graph_print,
        ]
    elif stage == OptimizationPassPlacement.PARTITIONER:
        return [
            # These passes are going to be ran in loop till we get satisfying partitioning to submodules.
            pass_partition_and_fuse,
            pass_graph_print,
            pass_eagerize_leaf_views,
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


def helper_get_node_args(node: torch.fx.Node):
    """
    This helper function get inputs to specific node. It should supports
    various corner cases (currently - for output node only).
    """
    # Output args could be a single-element tuple containing all outputs as well,
    # so let's support that.
    if "output" in node.op and isinstance(node.args, tuple):
        assert len(node.args) == 1

        # There are two cases, resulting unwrapped args could be again a tuple or directly a node.
        # Code assumes something iterable so if it's just a a single node, then do not unwrap it.
        if not isinstance(node.args[0], tuple):
            args = node.args
        else:
            args = node.args[0]
    else:
        args = node.args

    return args


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


def pass_fake_propagation(ctx: OptimizerContext) -> bool:
    """
    This pass makes sure that input tensors are in fake mode so we don't
    make any actual computation. Then it propagates tensor metadata into nodes.
    """

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

    class TensorInfoPropagation(torch.fx.Interpreter):
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

            fill_propagated_tensor_metadata_to_node(result, node)

            return result

        def propagate(self, *args):
            return super().run(*args)

    from torch._dynamo.utils import (
        fake_mode_from_tensors,
        deepcopy_to_fake_tensor
    )
    from torch.utils._python_dispatch import _get_current_dispatch_mode_stack

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

    TensorInfoPropagation(ctx.graph_module, fakemode_already_enabled, fake_mode).propagate(*fake_inputs)

    return True


def pass_partition_and_fuse(ctx: OptimizerContext) -> bool:
    """
    This pass is supposed to run partitioner that will create propisition of partitioning.
    This is special kind of pass, it will be ran always as first pass in the loop, and it
    always needs to return False (graph_changed=False) because we always have to run it
    and it will always change the graph, but we might still want to exit the loop if other
    passes didn't find optimization/fixes opportunities.
    """
    assert ctx.stage == OptimizationPassPlacement.PARTITIONER
    assert ctx.graph_module is not None
    assert ctx.ids_to_nodes is not None

    HabanaPartitioner(ctx.graph_module).partition_and_fuse()

    # This is special case, this is main pass that always needs to be ran and it should
    # always return that nothing was changed.
    return False


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
            "empty",
            "zeros",
            "ones",
            "clone",  # SW-136398
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
            "bernoulli",
            "multinomial",
            "normal",
            "max_pool2d_with_indices",
            # Other
            "convolution",  # SW-137174
            "_native_batch_norm_legit_functional",  # SW-137176
        ]

        return node_target in unsupported_ops

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
    ctx.graph_module.graph.eliminate_dead_code()
    ctx.graph_module.recompile()

    return graph_changed

def pass_eagerize_leaf_views(ctx: OptimizerContext) -> bool:
    """
    This pass is supposed to work on subgraphs and find nodes which are views and that emit these
    views to the output node. It also supports finding chains of such views.

    When such view node is found, mark original graph node as placed into `eager` so it does not
    end within clustered HPU submodules during repartition phase.
    """

    assert ctx.stage == OptimizationPassPlacement.PARTITIONER
    assert ctx.graph_module is not None
    assert ctx.ids_to_nodes is not None

    def is_view_node(node_name):
        is_view = False
        view_ops = [
            "view",
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
        for view_name in view_ops:
            # check if the op name begins with a view name and ensure that it not a scatter op
            # TODO are these conditions sufficient to avoid false positives?
            if (node_name.find(view_name) == 0) and ("scatter" not in node_name):
                if len(view_name) == len(node_name):
                    is_view = True
                    break
                else:
                    # can have some numbering ex: t_1
                    assert len(node_name) > len(view_name)
                    if node_name[len(view_name)] == "_":
                        is_view = True
                        break

        return is_view

    def fix_node_input_views(node):
        graph_changed = False

        args = helper_get_node_args(node)
        for arg in args:
            if isinstance(arg, torch.fx.Node) and is_view_node(arg.name):
                # Recursively find all views chains.
                graph_changed = fix_node_input_views(arg) or graph_changed

                # Transform current node.
                key = arg.meta["unique_id"]
                if key in ctx.ids_to_nodes:
                    original_node = ctx.ids_to_nodes[key]
                    if original_node.meta["placement"] != "eager":
                        original_node.meta["placement"] = "eager"
                        graph_changed = True

        return graph_changed

    # Get all submodules that are used in original graph.
    # On each of them, find `output` node.
    # Then, eagerize all views being used by this node.
    graph_changed = False
    for n in ctx.graph_module.graph.nodes:
        if n.op == "call_module":
            assert not n.kwargs
            subgraph = ctx.graph_module.get_submodule(n.target)
            for node in subgraph.graph.nodes:
                # Search for output node.
                if "output" in node.op:
                    graph_changed = fix_node_input_views(node) or graph_changed

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
                if (
                    node.target == torch.ops.aten._to_copy
                    and len(node.args) == 1
                    and len(node.kwargs) == 1
                    and "dtype" in node.kwargs
                ):
                    node.target = torch.ops.aten.to

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
