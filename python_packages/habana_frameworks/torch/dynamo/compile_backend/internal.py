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

from typing import List
from .shared_layer import is_cpu_fallback_required
from .recipe_compiler import get_callable_recipe
from .partitioner import HabanaPartitioner

logger = logging.getLogger("aot_hpu_backend")


def annotate_cpu_fallback(node: torch.fx.Node):
    """
    This function will forward the query to shared_layer so we know whether
    fallback is needed for this node.

    If yes, then annotate the node meta so following code can recognize such
    fallbacked node.
    """

    if is_cpu_fallback_required(node):
        node.meta["placement"] = "cpufallback"


def transform_cpu_fallbacks(graph_module: torch.fx.GraphModule):
    """
    This function is supposed to find nodes annotated as requiring fallbacks,
    for each such node it will materialize the copies and move the actual
    OP to CPU.
    """

    modified = False
    fallbacked_ops_counter = {}

    for node in graph_module.graph.nodes:
        if node.meta["placement"] == "cpufallback":
            if node.target.__name__ in fallbacked_ops_counter:
                fallbacked_ops_counter[node.target.__name__] += 1
            else:
                fallbacked_ops_counter[node.target.__name__] = 1

            modified = True
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

    if modified:
        graph_module.recompile()
        logger.debug(
            "#### Graph module after CPU fallback transformation:####\n%s",
            graph_module.print_readable(False),
        )

    return fallbacked_ops_counter


def fill_propagated_tensor_metadata_to_node(result: torch.Tensor, node: torch.fx.Node):
    """
    This function takes out basic information from propagated fake tensor, like
    dtype, layout and device and puts it to the node that created it. It will
    also place annotation about proposed placement. There are two options:

    "eager"       - such OPs will not be placed inside HPU clusters
    "hpu_cluster" - such OPs will be later placed inside HPU clusters
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

    placement = None
    if "placeholder" in node.op or "output" in node.op:
        placement = "eager"
    elif "to_copy" in node.name:
        # If this is dtype/layout copy from hpu to hpu, then leave it in
        # hpu_cluster. In any other case put it to eager.
        input_node = None
        for arg in node.args:
            if isinstance(arg, torch.fx.Node):
                input_node = arg
                break

        assert input_node is not None

        if input_node.meta["output_device"].type == "hpu" and node.meta["output_device"].type == "hpu":
            placement = "hpu_cluster"
        else:
            placement = "eager"
    elif node.meta["output_device"].type == "hpu":
        placement = "hpu_cluster"
    elif node.meta["output_device"].type == "cpu":
        placement = "eager"

    assert placement is not None

    node.meta["placement"] = placement

    logger.debug("placement: %s", placement)


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
        annotate_cpu_fallback(node)

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
    make any actual computation. Then it propagates tensor metadata, like
    device placement or layout, into nodes.
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


def cluster_module(graph_module: torch.fx.GraphModule):
    """
    This function will use partitioner to cluster in all
    habana-supported operations.
    """

    partitioner = HabanaPartitioner(graph_module)
    clustered_module = partitioner.partition_and_fuse()

    logger.debug("clustered module:\n%s", clustered_module.print_readable(False))

    return clustered_module


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

    # WORKAROUND END

    logger.info("INFO: Number of subgraphs created:\n%s", num_subgraphs)
