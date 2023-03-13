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

import logging
import torch
import copy

from typing import List
from .recipe_compiler import get_callable_recipe
from .passes import OptimizationPassPlacement, optimize_graph

logger = logging.getLogger("aot_hpu_backend")


def optimize_pre_partitioner(graph_module: torch.fx.GraphModule):
    """
    This function is supposed to run optimizations passes on a graph that
    wasn't yet partitioned.
    """
    optimize_graph(OptimizationPassPlacement.PRE_PARTITIONER, graph_module)


def optimize_post_partitioner(graph_module: torch.fx.GraphModule):
    """
    This function is supposed to run optimizations passes on a graph that
    was already partitioned.
    """
    optimize_graph(OptimizationPassPlacement.POST_PARTITIONER, graph_module)


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
        fake_mode: torch._subclasses.FakeTensorMode = None,
    ):
        super().__init__(graph_module)
        self.fake_mode = fake_mode

    def run_node(self, node: torch.fx.Node):
        logger.debug("Node: %s Op: %s Target: %s", node, node.op, node.target)

        with self.fake_mode:
            result = super().run_node(node)

        fill_propagated_tensor_metadata_to_node(result, node)

        return result

    def propagate(self, *args):
        return super().run(*args)


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


def preprocess_module(graph_module: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
    """
    This function makes sure that input tensors are in fake mode so we don't
    make any actual computation. Then it propagates tensor metadata into nodes.
    """

    from torch._dynamo.utils import (
        fake_mode_from_tensors,
        deepcopy_to_fake_tensor,
    )

    fake_mode = fake_mode_from_tensors(example_inputs)
    logger.debug("example_inputs fake mode: %s", fake_mode)
    logger.debug("####input graph_module:####\n%s", graph_module.print_readable(False))
    if fake_mode is None:
        fake_mode = torch._subclasses.FakeTensorMode()
        fake_inputs = deepcopy_to_fake_tensor(example_inputs, fake_mode)
    else:
        fake_inputs = example_inputs
    TensorInfoPropagation(graph_module, fake_mode).propagate(*fake_inputs)


def cluster_module(graph_module: torch.fx.GraphModule) -> torch.fx.GraphModule:
    """
    We need to make it a bit convoluted because we will analyze graph that was already
    partitioned and might have different set of nodes than original that we need apply the
    placement changes into. It will also have different layout of subgraphs.
    This is what we gonna do:
    1. To each original node add metadata entry containing unique ID.
    2. Grab all original nodes into dictionary using these IDs as keys.
    3. When we decide that specific >partitioned< node placement needs to be
        changed, we do following inside the optimization passes:
        I.   Check if it contains metadata with unique ID.
            (If it does not, ignore it as it do not exist in original graph.)
        II.  Use the unique ID to find original node in dictionary.
        III. Change original node placement.
    """
    ids_to_original_nodes = {}

    for node in graph_module.graph.nodes:
        key = node.meta["unique_id"] = hash(node)

        if key in ids_to_original_nodes:
            logger.error("key collision @ %d", key)
            raise

        ids_to_original_nodes[key] = node

    graph_changed = True
    while graph_changed:
        # Deep copy the graph because currently used CapabilityBasedPartitioner will
        # modify the graph in-place and we want to re-start from original on each iteration.
        copied_graph_module = copy.deepcopy(graph_module)

        # OUTPUT META BUG WORKAROUND
        # Fun fact - node.meta is supposed to be guaranteed to be copied when graph
        # is cloned. It's usually true, but it seems it's not for `output` nodes.
        # Let's W/A it, assume that output is last node in the graph.
        original_output_node = next(iter(reversed(graph_module.graph.nodes)))
        copied_output_node = next(iter(reversed(copied_graph_module.graph.nodes)))
        assert "output" in original_output_node.op
        assert "output" in copied_output_node.op

        copied_output_node.meta = copy.deepcopy(original_output_node.meta)
        # WORKAROUND END

        graph_changed = optimize_graph(
            OptimizationPassPlacement.PARTITIONER, copied_graph_module, ids_to_original_nodes
        )

    # Return graph_module that was partitioned last.
    return copied_graph_module


def compile_clusters(graph_module: torch.fx.GraphModule):
    """
    Go through each node in the main module. For each generated HPU cluster
    there will be "call_module" OP. For each such module create JIT IR and pass
    it to the HPU backend for recipe compilation and substitute the target with
    newly compiled one.
    """

    num_subgraphs = 0
    for n in graph_module.graph.nodes:
        logger.debug("Node: %s Op: %s Target: %s", n, n.op, n.target)

        if n.op == "call_module":
            assert not n.kwargs
            submod = graph_module.get_submodule(n.target)

            jit_ir_function = generate_jit_ir_from_module(submod)
            callable_recipe = get_callable_recipe(jit_ir_function, submod)

            graph_module.delete_submodule(n.target)
            graph_module.add_submodule(n.target, callable_recipe)

            num_subgraphs += 1

    logger.info("INFO: Number of subgraphs created:\n%s", num_subgraphs)
