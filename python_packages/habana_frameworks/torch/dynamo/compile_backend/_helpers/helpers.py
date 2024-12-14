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

from collections.abc import Iterable

import habana_frameworks.torch.internal.bridge_config as bc
import torch
from habana_frameworks.torch.dynamo.debug_utils.logger import get_compile_backend_logger

from ..random_utils import (
    backward_random_op_inputs,
    is_backward_checkpoint_op,
    is_multi_output_op,
    is_random_op,
    random_op_inputs,
)

logger = get_compile_backend_logger()


def is_view_node(node):
    # view nodes should only be a node with call_function op
    # when passing a node with different op target wil be of type str
    if node.op != "call_function":
        return False
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


def get_node_args(node: torch.fx.Node):
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


def handle_noncontiguous_output(node: torch.fx.Node, result: torch.Tensor):
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


def post_pass_finalize(input_module: torch.fx.GraphModule):
    """
    Run this pass iff the input graph changed for each submodule
    for each pass
    """
    # Clean up the graph and log the situation.
    input_module.graph.eliminate_dead_code()
    input_module.graph.lint()
    input_module.recompile()

    return input_module


def is_node_supported(node: torch.fx.Node) -> bool:
    """
    Returns true if the node is on HPU and is part of
    the proposed fused partition
    """
    return node.meta["output_device"].type == "hpu" and node.meta["placement"] == "hpu_cluster"


def is_compute_node(node):
    # return false if node is a view node, input node or output node
    return (not is_view_node(node)) and (node.op != "placeholder") and (node.op != "output")


def is_decomposed_from_inplace_node(node):
    if node.op != "call_function":
        return False
    node_target = node.target.__name__
    if ("original_aten" not in node.meta) or ("from_node" not in node.meta):
        return False

    return node_target != node.meta["original_aten"].__name__ and (node.meta["from_node"][0][0].endswith("_"))


def calculate_default_strides(sizes):
    # Calculate default strides for given size
    if sizes is None or len(sizes) == 0:
        return []

    reversed_strides = [1]
    for size in reversed(sizes[1:]):
        reversed_strides.append(size * reversed_strides[-1])
    return list(reversed(reversed_strides))


def get_node_users(node):
    if not isinstance(node, torch.fx.Node):
        return [None]
    node_list = list(node.users.keys())
    if len(node_list) == 0:
        return [None]
    return node_list


def is_symbolic_shape(shape):
    """
    This function checks if the shape is symbolic.
    """
    from torch.fx.experimental.symbolic_shapes import is_symbolic

    if isinstance(shape, torch.Size):
        return any(is_symbolic(dim) for dim in shape)
    return False


def fill_propagated_tensor_metadata_to_node(result: torch.Tensor, node: torch.fx.Node):
    """
    This function takes out basic information from propagated fake tensor, like
    dtype, layout and device and puts it to the node that created it.
    """
    if not bc.get_pt_hpu_use_jit_fork():
        # todo - cleanup [SW-199903]
        # just skip for get_attr node since it's not necessary
        if node.op == "get_attr":
            return

    if node.meta.get("val") is None:
        node.meta["val"] = result

    result = handle_noncontiguous_output(node, result)

    device = None
    dtypes = []
    layouts = []
    output_shapes = []
    output_strides = []
    output_contiguous = []
    output_offset = []

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
        output_offset = [result.storage_offset()]

        logger.debug("    result shape: %s", result.shape)
        logger.debug("    result stride: %s", result.stride())
        logger.debug("    result offset: %s", result.storage_offset())
    elif type(result) in result_type_to_node_type:
        device = torch.device("cpu")
        dtypes = [None]
        layouts = [None]
        output_shapes = [()]
        output_strides = [()]
        output_contiguous = [None]
        output_offset = [()]
        node.type = result_type_to_node_type[type(result)]
    elif str(node.target) == "inductor.accumulate_grad_.default":
        device = torch.device("hpu")
        dtypes = [None]
        layouts = [None]
        output_shapes = [None]
        output_strides = [None]
        output_contiguous = [None]
        output_offset = [None]
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
                # todo https://jira.habana-labs.com/browse/SW-199903:
                #  this must be a bug!
                # output_strides.append(res.storage_offset())
                output_offset.append(res.storage_offset())
                output_strides.append(res.stride())
                logger.debug("    result shape: %s", res.shape)

        if len(devices) > 0:
            # run_and_save_rng_state op has first output always on cpu, so the device
            # is set based on the second output.
            if str(node.target) == "run_and_save_rng_state":
                device = devices[1] if len(devices) > 1 else result[1][0].device
            elif devices.count(devices[0]) != len(devices) and "output" not in node.op:
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
        or "output_offset" in node.meta
    ):
        if node.meta["output_device"] is not None and device is not None:
            assert node.meta["output_device"].type == device.type
        else:
            assert node.meta["output_device"] == device
        assert node.meta["output_dtypes"] == dtypes
        assert node.meta["output_layouts"] == layouts
        if not any(is_symbolic_shape(shape) for shape in output_shapes):
            assert node.meta["output_shapes"] == output_shapes
        assert node.meta["output_offset"] == output_offset

    node.meta["output_device"] = device
    node.meta["output_dtypes"] = dtypes  # list expected
    node.meta["output_layouts"] = layouts  # list expected
    node.meta["output_shapes"] = output_shapes  # list expected
    node.meta["output_strides"] = output_strides  # list expected
    node.meta["output_contiguous"] = output_contiguous  # list expected
    node.meta["output_offset"] = output_offset  # list expected

    if bc.get_pt_hpu_use_jit_fork():
        logger.debug('Filling metadata "valX" for Lowering pass')
        with torch._subclasses.fake_tensor.FakeTensorMode():
            meta_output_vals = []
            for i in range(len(dtypes)):
                meta_output_vals.append(  # output_strides consists of storage_offset, strides, acccess only strides
                    torch.empty_strided(
                        output_shapes[i],
                        output_strides[i],
                        dtype=dtypes[i],
                        device=device,
                    )
                )
        node.meta["valX"] = meta_output_vals[0] if len(meta_output_vals) == 1 else tuple(meta_output_vals)


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
        _ = input_module.graph.call_function(torch.ops.aten.add_, (counter_pl, len(random_ops)), {})

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
