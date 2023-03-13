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

from enum import Enum

from .shared_layer import is_cpu_fallback_required
from .partitioner import HabanaPartitioner

logger = logging.getLogger("aot_hpu_backend")


class OptimizationPassPlacement(Enum):
    PRE_PARTITIONER = 1
    PARTITIONER = 2
    POST_PARTITIONER = 3


def optimize_graph(
    stage: OptimizationPassPlacement, graph_module: torch.fx.GraphModule, ids_to_nodes: dict = None
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
    graph_changed = False
    for optimization_pass in get_passes(stage):
        pass_name = optimization_pass.__name__
        env_name = "PT_HPU_DISABLE_" + pass_name
        if env_name in os.environ and os.environ[env_name] == "True":
            logger.debug("pass %s was disabled by env at stage %s", pass_name, stage)
        else:
            logger.debug("running %s pass at stage %s", pass_name, stage)
            graph_changed = optimization_pass(stage, graph_module, ids_to_nodes) or graph_changed

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
            pass_wa_fix_output,
            pass_graph_print,
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


def pass_graph_print(stage: OptimizationPassPlacement, graph_module: torch.fx.GraphModule, *args) -> bool:
    """
    This pass just prints the graph in debug mode.
    """
    assert graph_module is not None

    logger.debug("pass_graph_print at stage%s:\n%s", stage, graph_module.print_readable(False))
    return False


def pass_partition_and_fuse(
    stage: OptimizationPassPlacement, graph_module: torch.fx.GraphModule, ids_to_nodes: dict = None
) -> bool:
    """
    This pass is supposed to run partitioner that will create propisition of partitioning.
    This is special kind of pass, it will be ran always as first pass in the loop, and it
    always needs to return False (graph_changed=False) because we always have to run it
    and it will always change the graph, but we might still want to exit the loop if other
    passes didn't find optimization/fixes opportunities.
    """
    assert stage == OptimizationPassPlacement.PARTITIONER
    assert graph_module is not None
    assert ids_to_nodes is not None

    HabanaPartitioner(graph_module).partition_and_fuse()

    # This is special case, this is main pass that always needs to be ran and it should
    # always return that nothing was changed.
    return False


def pass_mark_placement(stage: OptimizationPassPlacement, graph_module: torch.fx.GraphModule, *args) -> bool:
    """
    This pass is supposed to annotate nodes with their placement.
    There are two placement options:

    "eager"       - such OPs will not be placed inside HPU clusters
    "hpu_cluster" - such OPs will be later placed inside HPU clusters
    """
    assert graph_module is not None

    def is_op_unsupported_in_graph(node):
        node_target = node.target.__name__.split(".")[0]

        unsupported_ops = [
            # Tensor creation OPs.
            "empty",
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
            "bernoulli",
            "multinomial",
            "normal",
        ]

        return node_target in unsupported_ops

    for node in graph_module.graph.nodes:
        assert "placement" not in node.meta

        placement = None
        if node.op == "placeholder" or node.op == "output":
            placement = "eager"
        elif node.op == "call_function" and is_op_unsupported_in_graph(node):
            placement = "eager"
        elif "to_copy" in node.name:
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

        node.meta["placement"] = placement

    return True


def pass_mark_fallbacks(stage: OptimizationPassPlacement, graph_module: torch.fx.GraphModule, *args) -> bool:
    """
    This pass is supposed to find nodes requiring CPU fallback.
    If such node is found, mark it as requriing fallbacking.
    """

    assert graph_module is not None
    graph_changed = False
    for node in graph_module.graph.nodes:
        if is_cpu_fallback_required(node):
            if node.meta["placement"] != "cpufallback":
                node.meta["placement"] = "cpufallback"
                graph_changed = True

    return graph_changed


def pass_transform_fallbacks(stage: OptimizationPassPlacement, graph_module: torch.fx.GraphModule, *args) -> bool:
    """
    This pass is supposed to find nodes annotated as requiring fallbacks,
    for each such node it will materialize the copies and move the actual
    OP to CPU.
    """

    assert graph_module is not None
    graph_changed = False
    fallbacked_ops_counter = {}

    for node in graph_module.graph.nodes:
        if node.meta["placement"] == "cpufallback":
            if node.target.__name__ in fallbacked_ops_counter:
                fallbacked_ops_counter[node.target.__name__] += 1
            else:
                fallbacked_ops_counter[node.target.__name__] = 1

            graph_changed = True
            with graph_module.graph.inserting_before(node):
                for arg in node.args:
                    if isinstance(arg, torch.fx.Node):
                        input_copy_node = graph_module.graph.call_function(
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
                with graph_module.graph.inserting_after(node):
                    output_copy_node = graph_module.graph.call_function(
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
                    with graph_module.graph.inserting_after(user):
                        output_copy_node = graph_module.graph.call_function(
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


def pass_skip_copies(stage: OptimizationPassPlacement, graph_module: torch.fx.GraphModule, *args) -> bool:
    """
    This pass is supposed to find cases where input to the node is copy between devices, which
    was created by the opposite copy. In that scenario we can just skip these copies. After we
    do that, we should remove dead code and recompile the FX graph.
    """

    assert graph_module is not None
    graph_changed = False

    for node in graph_module.graph.nodes:
        # For this specific node, check every input for copies. If there is a copy, follow the chain
        # to skip as many copies as possible.
        args = helper_get_node_args(node)

        for arg in args:
            if isinstance(arg, torch.fx.Node) and "to_copy" in arg.name:
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

                    if "to_copy" in chain_arg.name:
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
    graph_module.graph.eliminate_dead_code()
    graph_module.recompile()

    return graph_changed


def pass_wa_fix_output(stage: OptimizationPassPlacement, graph_module: torch.fx.GraphModule, *args) -> bool:
    """
    This pass is supposed to workaround an issue with global output not being the last
    node in the graph. Details below.
    """
    assert graph_module is not None
    graph_changed = False

    # AOT Autograd BUG WORKAROUND
    # (https://github.com/pytorch/pytorch/issues/92245):
    # This is workaround for optimizer graphs that are not functionalized at
    # this point. Issue is that optimizer has no outputs and it will cause
    # wrong topological sort and execution. After optimizer is correctly
    # functionalized by AOT Autograd, this code should be removed.
    output_node = None
    last_node_after_output = None
    for n in graph_module.graph.nodes:
        if output_node:
            last_node_after_output = n

        if n.op == "output":
            output_node = n
    if last_node_after_output is not None:
        logger.warning("It seems graph wasn't functionalized, fixing output node.")
        graph_module.graph.erase_node(output_node)
        graph_module.graph.node_copy(output_node)
        graph_module.recompile()
        graph_changed = True

    # WORKAROUND END

    return graph_changed


def pass_eagerize_leaf_views(
    stage: OptimizationPassPlacement, graph_module: torch.fx.GraphModule, ids_to_nodes: dict = None
) -> bool:
    """
    This pass is supposed to work on subgraphs and find nodes which are views and that emit these
    views to the output node. It also supports finding chains of such views.

    When such view node is found, mark original graph node as placed into `eager` so it does not
    end within clustered HPU submodules during repartition phase.
    """

    assert stage == OptimizationPassPlacement.PARTITIONER
    assert graph_module is not None
    assert ids_to_nodes is not None

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
                if key in ids_to_nodes:
                    original_node = ids_to_nodes[key]
                    if original_node.meta["placement"] != "eager":
                        original_node.meta["placement"] = "eager"
                        graph_changed = True

        return graph_changed

    # Get all submodules that are used in original graph.
    # On each of them, find `output` node.
    # Then, eagerize all views being used by this node.
    graph_changed = False
    for n in graph_module.graph.nodes:
        if n.op == "call_module":
            assert not n.kwargs
            subgraph = graph_module.get_submodule(n.target)
            for node in subgraph.graph.nodes:
                # Search for output node.
                if "output" in node.op:
                    graph_changed = fix_node_input_views(node) or graph_changed

    return graph_changed
