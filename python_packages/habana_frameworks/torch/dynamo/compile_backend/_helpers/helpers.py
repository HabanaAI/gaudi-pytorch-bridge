###############################################################################
# Copyright (c) 2021-2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
###############################################################################

from collections.abc import Iterable

import habana_frameworks.torch.internal.bridge_config as bc
from habana_frameworks.torch.dynamo.debug_utils.logger import get_compile_backend_logger

import torch
from torch._dispatch.python import enable_python_dispatcher
from torch._dynamo.utils import detect_fake_mode
from torch._subclasses.fake_tensor import FakeTensorMode

from ..random_utils import (
    backward_random_op_inputs,
    is_backward_checkpoint_op,
    is_random_op,
    is_run_and_save_rng_state,
    random_op_inputs,
)
from ..symbolic_execution import (
    HPUExprPrinter,
    SymExprNodeManager,
    substitute_sympyfn,
    sympify_expression,
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
        if not len(node.args) == 1:
            raise AssertionError("Incorrect args number")

        # There are two cases, resulting unwrapped args could be again a tuple or directly a node.
        # Code assumes something iterable so if it's just a a single node, then do not unwrap it.
        if isinstance(node.args[0], tuple | list | torch.fx.immutable_collections.immutable_list):
            args = node.args[0]

    if isinstance(args, tuple | list | torch.fx.immutable_collections.immutable_list):
        cleaned_args = [arg for arg in args if isinstance(arg, torch.fx.Node)]
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
    return (not is_view_node(node)) and (node.op not in ["placeholder", "output"])


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
    if node.meta.get("val") is None:
        node.meta["val"] = result

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
        torch.fx.experimental._backward_state.BackwardState: None,
    }

    logger.debug("node name: %s", node.name)
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
        if not isinstance(result, Iterable):
            raise AssertionError("Expecting iterable at this point")

        def collect_result(
            result,
            devices=devices,
            dtypes=dtypes,
            layouts=layouts,
            output_shapes=output_shapes,
            output_contiguous=output_contiguous,
            output_offset=output_offset,
            output_strides=output_strides,
        ):
            if hasattr(result, "device"):
                devices.append(result.device)
            if hasattr(result, "dtype"):
                dtypes.append(result.dtype)
            if hasattr(result, "layout"):
                layouts.append(result.layout)

            if hasattr(result, "shape"):
                output_shapes.append(result.shape)
                output_contiguous.append(result.is_contiguous())
                output_offset.append(result.storage_offset())
                output_strides.append(result.stride())
                logger.debug("    result shape: %s", result.shape)

        for res in result:
            if res is None:
                continue
            if isinstance(res, tuple | list):
                devices_list = []
                dtypes_list = []
                layouts_list = []
                output_shapes_list = []
                output_contiguous_list = []
                output_offset_list = []
                output_strides_list = []

                for r in res:
                    collect_result(
                        r,
                        devices=devices_list,
                        dtypes=dtypes_list,
                        layouts=layouts_list,
                        output_shapes=output_shapes_list,
                        output_contiguous=output_contiguous_list,
                        output_offset=output_offset_list,
                        output_strides=output_strides_list,
                    )

                devices.append(tuple(devices_list))
                dtypes.append(tuple(dtypes_list))
                layouts.append(tuple(layouts_list))
                output_shapes.append(tuple(output_shapes_list))
                output_contiguous.append(tuple(output_contiguous_list))
                output_offset.append(tuple(output_offset_list))
                output_strides.append(tuple(output_strides_list))
            else:
                collect_result(res)

        if len(devices) > 0:
            # run_and_save_rng_state op has first output always on cpu, so the device
            # is set based on the second output.
            if str(node.target) == "run_and_save_rng_state":
                device = devices[1] if len(devices) > 1 else result[1][0].device
                if isinstance(device, tuple):
                    device = device[0]
            elif devices.count(devices[0]) != len(devices) and "output" not in node.op:
                logger.error(
                    "multiple devices in single node\n%s\n at node: %s",
                    devices,
                    node,
                )
                raise AssertionError("Multiple devices in single node")
            else:
                device = devices[0]

    if "output" not in node.op:
        if device is None:
            raise AssertionError("Missing device")
        if len(dtypes) == 0:
            raise AssertionError("Missing dtypes")
        if len(layouts) == 0:
            raise AssertionError("Missing layouts")
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
            if not node.meta["output_device"].type == device.type:
                raise AssertionError("Device mismatch")
        elif node.meta["output_device"] != device:
            raise AssertionError("Device mismatch")
        if not node.meta["output_dtypes"] == dtypes:
            raise AssertionError("Dtype mismatch")
        if not node.meta["output_layouts"] == layouts:
            raise AssertionError("Layout mismatch")
        if (
            not any(is_symbolic_shape(shape) for shape in output_shapes)
            and not node.meta["output_shapes"] == output_shapes
        ):
            raise AssertionError("Output shape mismatch")
        if not node.meta["output_offset"] == output_offset:
            raise AssertionError("Output offset mismatch")

    node.meta["output_device"] = device
    node.meta["output_dtypes"] = dtypes  # list expected
    node.meta["output_layouts"] = layouts  # list expected
    node.meta["output_shapes"] = output_shapes  # list expected
    node.meta["output_strides"] = output_strides  # list expected
    node.meta["output_contiguous"] = output_contiguous  # list expected
    node.meta["output_offset"] = output_offset  # list expected


def fill_propagated_tensor_metadata_jitfork(node: torch.fx.Node):
    if bc.get_pt_hpu_use_jit_fork():
        logger.debug('Filling metadata "val" for node: %s', node.name)
        meta_output_vals = []
        for i in range(len(node.meta["output_dtypes"])):
            meta_output_vals.append(  # noqa PERF401
                # output_strides consists of storage_offset, strides, acccess only strides
                torch.empty_strided(
                    node.meta["output_shapes"][i],
                    node.meta["output_strides"][i],
                    dtype=node.meta["output_dtypes"][i],
                    device=node.meta["output_device"],
                )
            )

        node.meta["val"] = meta_output_vals[0] if len(meta_output_vals) == 1 else tuple(meta_output_vals)


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
    outs = list(output_node.args[0]) if type(output_node.args[0]) is tuple else [output_node.args[0]]
    duplicated_out_indexes.sort()
    for idx in reversed(duplicated_out_indexes):
        del outs[idx]

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


def is_module_dynamic(input_module: torch.fx.GraphModule) -> bool:
    """
    This function dynamicity per graph module.
    """

    from torch._subclasses.fake_tensor import FakeTensor
    from torch.fx.experimental.proxy_tensor import py_sym_types

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
                random_node.meta["deterministic"] = True
                input_module.graph.erase_node(node)

        input_module.recompile()

    if len(random_ops) == 0:
        return False

    with input_module.graph.inserting_before():
        counter_pl = input_module.graph.placeholder("counter_pl")
        seed_pl = input_module.graph.placeholder("seed_pl")

    with input_module.graph.inserting_after(counter_pl):
        seeds = input_module.graph.call_function(
            torch.ops.hpu.habana_seed_generator, (counter_pl, seed_pl, len(random_ops)), {}
        )
        _ = input_module.graph.call_function(
            torch.ops.aten.add_.Tensor, (counter_pl, len(random_ops)), {}, name="add__from_random"
        )

    for i, node in enumerate(random_ops):
        with input_module.graph.inserting_before(node):
            seed = input_module.graph.call_function(torch.select, (seeds, 0, i), {})
            random_node = input_module.graph.call_function(*random_op_inputs(node, seed))

            if is_run_and_save_rng_state(node):
                random_node.meta["deterministic"] = True
                for getitem_node in list(node.users):
                    index = getitem_node.args[1]
                    if index not in [0, 1]:
                        raise AssertionError(f"Expecting {node} to produce only two outputs")

                    new_out = seed if index == 0 else random_node
                    getitem_node.replace_all_uses_with(new_out, propagate_meta=False)
                    input_module.graph.erase_node(getitem_node)
            else:
                node.replace_all_uses_with(random_node, propagate_meta=True)
                random_node.meta.update(node.meta)

            input_module.graph.erase_node(node)

    input_module.recompile()
    return True


class TensorInfoPropagation(torch.fx.Interpreter):
    """
    This class is responsible for tracing through the graph module, and
    propagating all the necessary tensor information. All is done using
    fake_tensors so it does not make any real computations.
    """

    def __init__(self, graph_module: torch.fx.GraphModule, fake_mode: FakeTensorMode | None):
        super().__init__(graph_module)
        self._mode = fake_mode if fake_mode else FakeTensorMode()

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
        with self._mode, enable_python_dispatcher():
            return super().run(*args)


def propagate_meta(graph_module: torch.fx.GraphModule, example_inputs: list[torch.Tensor], propagator_class):
    fake_mode = detect_fake_mode(example_inputs)
    with torch.autocast(enabled=False, device_type="hpu"), torch.autocast(enabled=False, device_type="cpu"):
        # Disabling autocast in fake tensor propagation as autocasting has been
        # already done and all dtypes has been already deduced.
        if not fake_mode:
            fake_mode = torch._subclasses.FakeTensorMode(allow_non_fake_inputs=True)
            propagator_class(graph_module, fake_mode).propagate(*example_inputs)
        else:
            propagator_class(graph_module, fake_mode).propagate_dont_convert_inputs(*example_inputs)


def jit_node_annotation_propagation(jit_graph, fx_module):
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
    for jit_node, fx_node in zip(jit_graph_nodes, fx_nodes, strict=False):
        fx_node_name = fx_node.target.__name__.split(".")[0]
        if fx_node_name not in jit_node.kind():
            logger.debug(f"FX node {fx_node_name} doesn't match with Jit node {jit_node.kind()}")
            break

        # extract hints from FX node metadata
        context_hints = fx_node.meta.get("context_hints", None)
        if context_hints:
            logger.debug(f"node {fx_node_name} has context hints {context_hints}")
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

        if "deterministic" in fx_node.meta:
            jit_node.i_("deterministic", True)
            logger.debug("deterministic marked for jit node", jit_node)
            is_annotated_graph = True

    if is_annotated_graph:
        logger.debug(
            "####Annotated JIT IR graph for this HPU graph:####\n%s",
            jit_graph,
        )

    return


def get_dynamic_config_value():
    """
    This function return the is_dynamic=True if user configured
    the same while calling torch.compile. Otherwise return is_dynamic=False
    """

    is_dynamic = False
    from torch._dynamo import config

    # TODO: It is a W/A for discovering dynamic models. In final implementation
    # is should read this info from tensors.
    is_dynamic = not config.assume_static_by_default

    return is_dynamic


# It seems that this pass assumes that the fx graph and the jit graph must
# have same ops order. Otherwise, the shape propagation may fail. However,
# the _jit_pass_remove_mutation pass has possiblity to change the jit graph
# ops order, and may break the assumption.
def jit_node_shape_propagation(jit_graph, fx_module):
    logger.debug("JIT processing shape propagation JIT graph:", jit_graph)
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
            if fx_node.op in ("placeholder", "output"):
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
        from ..symbolic_execution import PythonPrinter

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

    for node in jit_graph.nodes():
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
