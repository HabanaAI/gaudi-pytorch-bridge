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


def fill_propagated_tensor_metadata_to_node(result: torch.Tensor, node: torch.fx.Node):
    """
    This function takes out basic information from propagated fake tensor, like
    dtype, layout and device and puts it to the node that created it.
    """
    # just skip for get_attr node since it's not necessary
    if node.op == "get_attr":
        return

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
        output_shapes = [None]
        output_strides = [None]
        output_contiguous = [None]
        output_offset = [None]
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
                output_strides.append(res.storage_offset())
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
        assert node.meta["output_shapes"] == output_shapes
        assert node.meta["output_offset"] == output_offset

    node.meta["output_device"] = device
    node.meta["output_dtypes"] = dtypes
    node.meta["output_layouts"] = layouts
    node.meta["output_shapes"] = output_shapes
    node.meta["output_strides"] = output_strides
    node.meta["output_contiguous"] = output_contiguous
    node.meta["output_offset"] = output_offset
