###############################################################################
# Copyright (C) 2024 Habana Labs, Ltd. an Intel Company
# All Rights Reserved.
#
# Unauthorized copying of this file or any element(s) within it, via any medium
# is strictly prohibited.
# This file contains Habana Labs, Ltd. proprietary and confidential information
# and is subject to the confidentiality and license agreements under which it
# was provided.
#
###############################################################################

import importlib
import os
from functools import partial
from typing import Any, Dict, List

import functorch
import torch
from habana_frameworks.torch.dynamo.compile_backend.logger import get_compile_backend_logger
from torch._dynamo.backends.common import aot_autograd
from torch.fx import Node
from torch.fx.passes.utils.source_matcher_utils import SourcePartition, get_source_partitions

logger = get_compile_backend_logger()

habana_quantization_map_queue = []
export_module_record = dict()
param_id = 0


# ======================================================================================
# Habana Quantization Manager defined for torch.compile backend
# This is the module we use for actual support of quantization
# ======================================================================================
class HabanaQuantWrapperModule(torch.nn.Module):
    def __init__(self, graph_module, module_key):
        super().__init__()
        self._module_key = module_key
        self._preprocessed = False
        self._prepared = False
        self._converted = False
        self._fx_module = graph_module
        self._prepared_module = None
        self._observed_module = None
        self._converted_module = None

    def preprocess(self, *args):
        discover_and_materialize_params(self._fx_module, *args)
        self._preprocessed = True

    def __call__(self, *args, **kwargs):
        logger.debug(
            f"HabanaQuantWrapperModule::__call__ [{self._module_key}] ID:",
            id(self),
            f"\tpreprocessed={self._preprocessed}" f"\tprepared={self._prepared}" f"\tconverted={self._converted}",
        )

        if not self._preprocessed:
            self.preprocess(*args)

        assert len(habana_quantization_map_queue[self._module_key]) == 1
        queue_element = habana_quantization_map_queue[self._module_key][0]
        if queue_element["task"] == "prepare_pt2e":
            if not self._prepared:
                # Apply pytorch prepare_pt2e on each fx graph
                from torch.ao.quantization.quantize_pt2e import prepare_pt2e

                self._prepared_module = prepare_pt2e(self._fx_module, queue_element["quantizer"])

                # Now we use torch.compilation with hpu_backend.
                # hpu_backend internally uses aot_autograd which extracts the forward definition of
                # observer class and replaces the observer specific call_module nodes with corresponding
                # inlined forward definitions.
                # However, as the same storage is still used for holding the observer state, the
                # result of calibration (i.e. all stat updates) remains available from the original
                # _prepared_module that we use later at conversion stage.
                with torch.no_grad():
                    self._observed_module = torch.compile(self._prepared_module, backend="hpu_backend")

                self._prepared = True

            return self._observed_module(*args, **kwargs)

        elif queue_element["task"] == "convert_pt2e":
            if not self._converted:
                if not self._prepared:
                    logger.error(
                        "Attempt to convert an unprepared module!. Please use PT2E quant flow, i.e."
                        "Export -> prepare_pt2e -> calibrate -> convert_pt2e -> Ref_Quantized_Model, as recommended in"
                        "https://pytorch.org/tutorials/prototype/quantization_in_pytorch_2_0_export_tutorial.html"
                    )
                    raise

                # Apply pytorch convert_pt2e on each fx graph
                from torch.ao.quantization.quantize_pt2e import convert_pt2e

                self._converted_module = convert_pt2e(
                    self._prepared_module, use_reference_representation=False, fold_quantize=False
                )

                # Now we call hpu_inference_compiler to convert it into synapse graph.
                with torch.no_grad():
                    self._converted_module = torch.compile(self._converted_module, backend="hpu_backend")

                self._converted = True

            return self._converted_module(*args, **kwargs)


def habana_quant_compiler_fw(
    module: torch.fx.GraphModule, example_inputs: List[torch.Tensor], module_key: torch.fx.GraphModule
):
    # This backend only sets up runtime wrapper to run real compilation once we have real tensors.
    return functorch.compile.make_boxed_func(HabanaQuantWrapperModule(module, module_key))


def habana_quant_compiler_bw_raise(graph_module: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
    raise Exception("tried to call backward pass compiler in inference backend")


def habana_quant_backend(
    graph_module: torch.fx.GraphModule, example_inputs: List[torch.Tensor], module_key: torch.fx.GraphModule
):
    """
    This function implements interface for Habana's PT2E quantization backend.
    """
    from habana_frameworks.torch.dynamo.compile_backend.decomposition import get_hpu_decompositions

    return aot_autograd(
        fw_compiler=partial(habana_quant_compiler_fw, module_key=module_key),
        bw_compiler=habana_quant_compiler_bw_raise,
        decompositions=get_hpu_decompositions(),
    )(graph_module, example_inputs)


# ======================================================================================
# Habana export() to register torch.compile backend
# ======================================================================================
def export(module):
    logger.debug("Habana's implementation of PT2E based quantization flow: [export]")
    global export_module_record
    global habana_quantization_map_queue
    id_module = id(module)
    if id_module in export_module_record.keys():
        return export_module_record[id_module], True
    else:
        module_key = len(habana_quantization_map_queue)
        module = torch.compile(module, backend=partial(habana_quant_backend, module_key=module_key), dynamic=False)
        habana_quantization_map_queue.append([])
        export_module_record[id_module] = module
        setattr(module, "meta_hb_quant_id", module_key)
        return module, False


# ======================================================================================
# Habana prepare_pt2e() to set "prepare_pt2e" cmd for HabanaQuantWrapperModule
# ======================================================================================
def prepare_pt2e(module, quantizer):
    logger.debug("Habana's implementation of PT2E based quantization flow: [prepare_pt2e]")
    global habana_quantization_map_queue
    module_key = getattr(module, "meta_hb_quant_id")
    habana_quantization_map_queue[module_key] = []
    habana_quantization_map_queue[module_key].append({"task": "prepare_pt2e", "quantizer": quantizer})
    return module


# ======================================================================================
# Habana convert_pt2e() to set "convert_pt2e" cmd for HabanaQuantWrapperModule
# ======================================================================================
def convert_pt2e(module, use_reference_representation=False):
    logger.debug("Habana's implementation of PT2E based quantization flow: [convert_pt2e]")
    global habana_quantization_map_queue
    module_key = getattr(module, "meta_hb_quant_id")
    habana_quantization_map_queue[module_key] = []
    habana_quantization_map_queue[module_key].append({"task": "convert_pt2e"})
    return module


# ======================================================================================
# Freeze parameters for linear op, as is done in case of torch.export()
# ======================================================================================
def preprocess_linears(placeholder_map, module: torch.fx.GraphModule, tupled_args, *args):
    linear_module_partitions = get_source_partitions(module.graph, [torch.nn.Linear, torch.nn.functional.linear])

    if len(linear_module_partitions) == 0:
        return

    global param_id
    module_changed = False
    for module_or_fn_type, partitions in linear_module_partitions.items():
        if module_or_fn_type == torch.nn.Linear or module_or_fn_type == torch.nn.functional.linear:
            for p in partitions:
                weight_node = None
                bias_node = None
                compute_node = None
                for node in p.nodes:
                    if node.op == "call_function":
                        if node.target.__name__ == "linear.default":
                            weight_node = node.args[1]
                            if len(node.args) > 2:
                                bias_node = node.args[2]
                            compute_node = node
                            break
                        elif node.target.__name__ == "addmm.default":
                            weight_node = node.args[0]
                            bias_node = node.args[2]
                            compute_node = node
                            break
                        elif node.target.__name__ == "mm.default":
                            weight_node = node.args[1]
                            compute_node = node
                            break

                if compute_node is None:
                    logger.warn("Ignoring cases, where linear is decomposed into (t + bmm).")
                    continue

                assert weight_node is not None

                # Now let's follow addmm node inputs till we find nodes on partition list to get
                # original primals. We do that to go before any view/t ops we could have here.
                # We assume that all ops in such chain take single input.
                if weight_node in p.input_nodes:
                    # Already a primal.
                    weight_node_first_user = compute_node
                else:
                    weight_node_first_user = weight_node
                    while True:
                        if weight_node in p.input_nodes:
                            break
                        assert len(weight_node.args) >= 1
                        weight_node_first_user = weight_node
                        weight_node = weight_node.args[0]

                if bias_node is not None:
                    if bias_node in p.input_nodes:
                        # Already a primal.
                        bias_node_first_user = compute_node
                    else:
                        bias_node_first_user = bias_node
                        while True:
                            if bias_node in p.input_nodes:
                                break
                            assert len(bias_node.args) >= 1
                            bias_node_first_user = bias_node
                            bias_node = bias_node.args[0]

                # Now, clone original parameters primals into actual params within self and add
                # FX graph nodes to use them instead of inputs.
                with module.graph.inserting_before(weight_node_first_user):
                    module_changed = module_changed or True
                    attr_name = "_param_constant_l" + str(param_id)
                    param_tensor = tupled_args[placeholder_map[weight_node.name]]
                    setattr(module, attr_name, torch.nn.parameter.Parameter(param_tensor.detach()))
                    new_attr_node = module.graph.create_node("get_attr", attr_name)
                    weight_node_first_user.replace_input_with(weight_node, new_attr_node)
                    param_id = param_id + 1

                    # Fix source code meta for annotations detection.
                    new_attr_node.meta["source_fn_stack"] = compute_node.meta.get("source_fn_stack", None)
                    new_attr_node.meta["stack_trace"] = compute_node.meta.get("stack_trace", None)
                    new_attr_node.meta["tensor_meta"] = compute_node.meta.get("tensor_meta", None)
                    new_attr_node.meta["val"] = compute_node.meta.get("val", None)

                if bias_node is not None:
                    with module.graph.inserting_before(bias_node_first_user):
                        module_changed = module_changed or True
                        attr_name = "_param_constant_l" + str(param_id)
                        param_tensor = tupled_args[placeholder_map[bias_node.name]]
                        setattr(module, attr_name, torch.nn.parameter.Parameter(param_tensor.detach()))
                        new_attr_node = module.graph.create_node("get_attr", attr_name)
                        bias_node_first_user.replace_input_with(bias_node, new_attr_node)
                        param_id = param_id + 1

                        # Fix source code meta for annotations detection.
                        new_attr_node.meta["source_fn_stack"] = compute_node.meta.get("source_fn_stack", None)
                        new_attr_node.meta["stack_trace"] = compute_node.meta.get("stack_trace", None)
                        new_attr_node.meta["tensor_meta"] = compute_node.meta.get("tensor_meta", None)
                        new_attr_node.meta["val"] = compute_node.meta.get("val", None)

    if module_changed:
        module.graph.lint()
        module.recompile()


# ======================================================================================
# Freeze parameters for conv op, as is done in case of torch.export()
# ======================================================================================
def preprocess_convs(placeholder_map, module: torch.fx.GraphModule, tupled_args):
    conv_module_partitions = get_source_partitions(module.graph, [torch.nn.Conv2d, torch.nn.functional.conv2d])

    if len(conv_module_partitions) == 0:
        return

    # TODO add support for convs without bias.

    global param_id
    for module_or_fn_type, partitions in conv_module_partitions.items():
        if module_or_fn_type == torch.nn.Conv2d or module_or_fn_type == torch.nn.functional.conv2d:
            for p in partitions:
                weight_node = None
                bias_node = None
                compute_node = None
                for node in p.nodes:
                    # Find addmm node and get first input. We cannot use partitions input list
                    # to get params as it is changing inputs order.
                    if node.op == "call_function" and node.target.__name__ == "convolution.default":
                        weight_node = node.args[1]
                        bias_node = node.args[2]
                        compute_node = node
                        break

                assert weight_node is not None and compute_node is not None

                # Now let's follow addmm node inputs till we find nodes on partition list to get
                # original primals. We do that to go before any view/t ops we could have here.
                # We assume that all ops in such chain take single input.
                if weight_node in p.input_nodes:
                    # Already a primal.
                    weight_node_first_user = compute_node
                else:
                    weight_node_first_user = weight_node
                    while True:
                        if weight_node in p.input_nodes:
                            break
                        assert len(weight_node.args) >= 1
                        weight_node_first_user = weight_node
                        weight_node = weight_node.args[0]

                if bias_node in p.input_nodes:
                    # Already a primal.
                    bias_node_first_user = compute_node
                else:
                    bias_node_first_user = bias_node
                    while True:
                        if bias_node in p.input_nodes:
                            break
                        assert len(bias_node.args) >= 1
                        bias_node_first_user = bias_node
                        bias_node = bias_node.args[0]

                # Now, clone original parameters primals into actual params within self and add
                # FX graph nodes to use them instead of inputs.
                with module.graph.inserting_before(weight_node_first_user):
                    attr_name = "_param_constant_c" + str(param_id)
                    param_tensor = tupled_args[placeholder_map[weight_node.name]]
                    setattr(module, attr_name, torch.nn.parameter.Parameter(param_tensor.detach()))
                    new_attr_node = module.graph.create_node("get_attr", attr_name)
                    weight_node_first_user.replace_input_with(weight_node, new_attr_node)
                    param_id = param_id + 1

                    # Fix source code meta for annotations detection.
                    new_attr_node.meta["source_fn_stack"] = compute_node.meta.get("source_fn_stack", None)
                    new_attr_node.meta["stack_trace"] = compute_node.meta.get("stack_trace", None)
                    new_attr_node.meta["tensor_meta"] = compute_node.meta.get("tensor_meta", None)
                    new_attr_node.meta["val"] = compute_node.meta.get("val", None)

                with module.graph.inserting_before(bias_node_first_user):
                    attr_name = "_param_constant_c" + str(param_id)
                    param_tensor = tupled_args[placeholder_map[bias_node.name]]
                    setattr(module, attr_name, torch.nn.parameter.Parameter(param_tensor.detach()))
                    new_attr_node = module.graph.create_node("get_attr", attr_name)
                    bias_node_first_user.replace_input_with(bias_node, new_attr_node)
                    param_id = param_id + 1

                    # Fix source code meta for annotations detection.
                    new_attr_node.meta["source_fn_stack"] = compute_node.meta.get("source_fn_stack", None)
                    new_attr_node.meta["stack_trace"] = compute_node.meta.get("stack_trace", None)
                    new_attr_node.meta["tensor_meta"] = compute_node.meta.get("tensor_meta", None)
                    new_attr_node.meta["val"] = compute_node.meta.get("val", None)

    module.graph.lint()
    module.recompile()


# ======================================================================================
# Change FX graph so that it resembles one that would be generated by torch.export()
# ======================================================================================
def discover_and_materialize_params(module: torch.fx.GraphModule, *args):

    # Get placeholder map from FX graph.
    placeholder_map = {}
    placeholder_count = 0
    for node in module.graph.nodes:
        if node.op == "placeholder":
            placeholder_map[node.name] = placeholder_count
            placeholder_count = placeholder_count + 1

    tupled_args = tuple(args)

    # Handle following custom linear modules in deepspeed
    def handle_custom_linear_modules(module):
        for node in module.graph.nodes:
            source_fn_stack = node.meta.get("source_fn_stack", None)
            nn_module_stack = node.meta.get("nn_module_stack", None)
            if source_fn_stack is not None and nn_module_stack is not None:
                node.meta["source_fn_stack_original"] = source_fn_stack
                nn_module_stack_last_value = str(list(nn_module_stack.values())[-1])
                custom_linear_modules = [
                    "LinearLayer",
                    "LinearAllreduce",
                    "ScopedLinearAllReduce",
                    "LmHeadLinearAllreduce",
                ]
                if any(substring in nn_module_stack_last_value for substring in custom_linear_modules):
                    del source_fn_stack[-1]
                    source_fn_stack.append((list(nn_module_stack.keys())[-1], torch.nn.Linear))
                    node.meta["source_fn_stack"] = source_fn_stack

    # Due to custom linear modules in deepspeed, "source_fn_stack" node meta
    # of post-decomposition "mm" nodes does not include the original source
    # information. Hence, pytorch's get_source_partitions() utility fails to
    # to identify the "mm" nodes that originally belong to linear modules.
    # Till we have a proper 'parameter freezing' mechanism in place, we can
    # use "nn_module_stack" node meta to refill the missing information.
    if importlib.util.find_spec("deepspeed") and os.getenv("WORLD_SIZE", "0") != "0":
        handle_custom_linear_modules(module)

    preprocess_linears(placeholder_map, module, tupled_args, *args)
    preprocess_convs(placeholder_map, module, tupled_args)
