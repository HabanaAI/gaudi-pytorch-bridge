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
import operator
import os
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import functorch
import torch
from habana_frameworks.torch.dynamo.compile_backend.logger import get_compile_backend_logger
from torch._dynamo.backends.common import aot_autograd
from torch.ao.quantization.quantizer import Quantizer
from torch.fx import GraphModule, Node
from torch.fx.passes.utils.source_matcher_utils import SourcePartition, get_source_partitions

from .torch_overwrites import _native_pt2e_quantization_interface

logger = get_compile_backend_logger()

habana_quantization_map_queue = []
export_model_record = dict()
habana_pt2e_quant_context = None
param_id = 0


# ======================================================================================
# Habana's model level context manager for multi-graph PT2E-Quantization
# ======================================================================================
class HabanaPT2EQuantContext:
    def __init__(self, model_key, input):
        super().__init__()
        self._model_key = model_key
        self._total_number_of_graphs = 0
        self._input_for_tracing = input
        self._graph_list = list()
        self._graphs = ""
        self._model = None

    def append_graph(self, graph):
        self._graph_list.append(graph)
        self._graphs = self._graphs + "\n\n" + f"{graph}"
        self._total_number_of_graphs = self._total_number_of_graphs + 1
        setattr(self._model, "graph", self._graphs)

    def get_total_number_of_graphs(self):
        return self._total_number_of_graphs

    def clear_graphs(self):
        self._graph_list.clear()
        self._graphs = ""
        self._total_number_of_graphs = 0

    def get_input_for_tracing(self):
        return self._input_for_tracing

    def set_input_for_tracing(self, input):
        pass
        # # Tap input of the 1st fx graph
        # if self._total_number_of_graphs == 1:
        #     self._input_for_tracing = [input, ]

    def set_model(self, model):
        self._model = model
        setattr(
            self._model,
            "graph",
            "If you haven't provided sample input during export, run the model at least once with actual input to cature the graphs.",
        )


# ======================================================================================
# Habana's torch.compile based graph break detector
# ======================================================================================
def graph_breaks(
    f: torch.nn.Module,
    args: Tuple[Any],
    kwargs: Optional[Dict[str, Any]] = None,
) -> bool:
    """
    Graph break detector for Habana's PT2E-Quantization flow.
    If graph breaks, Habana's PT2E-Quant flow is used. Else, native PT2E-Quant flow is used.
    """

    # If sample input is not specified during export, Habana's PT2E-Quant flow is used
    if args == None:
        return True

    # Next, check for user instruction, if any. 3 possibilities:
    # a. graph_break_present: False [Can be set only when user is sure about no graph breaks]
    # b. graph_break_present: True  [Can be set only when user is sure about graph breaks]
    # c. graph_break_present: unspecified
    if kwargs != None and "graph_break_present" in kwargs:
        return kwargs["graph_break_present"]

    # Finally, try and figure out if there is any graph break.
    try:

        def detect_graph_break(model: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
            return model

        torch._dynamo.config.suppress_errors = True
        model = torch.compile(f, backend=detect_graph_break, fullgraph=True)
        model(*args)
        logger.info("......................................................................!")
        logger.info("NO GRAPH BREAK DETECTED............USING Native PT2E-QUANT FLOW.......!")
        logger.info("......................................................................!")
        return False
    except:
        logger.info("......................................................................!")
        logger.info("GRAPH BREAK DETECTED...............USING HABANA PT2E-QUANT FLOW.......!")
        logger.info("......................................................................!")
        return True


# ======================================================================================
# Habana Quantization Manager defined for torch.compile backend
# This is the module we use for actual support of quantization
# ======================================================================================
class HabanaQuantWrapperModule(torch.nn.Module):
    def __init__(self, graph_module, module_key, pt2e_quant_context):
        super().__init__()
        self._module_key = module_key
        self._preprocessed = False
        self._prepared = False
        self._converted = False
        self._fx_module = graph_module
        self._prepared_module = None
        self._observed_module = None
        self._converted_module = None
        self._pt2e_quant_context = pt2e_quant_context

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

        if habana_quantization_map_queue[self._module_key] == []:
            if self._pt2e_quant_context != None:
                self._pt2e_quant_context.append_graph(self._fx_module.graph)
            return self._fx_module(*args, **kwargs)

        assert len(habana_quantization_map_queue[self._module_key]) == 1
        queue_element = habana_quantization_map_queue[self._module_key][0]
        if queue_element["task"] == "prepare_pt2e":
            if not self._prepared:
                # Apply pytorch prepare_pt2e on each fx graph
                self._prepared_module = _native_pt2e_quantization_interface("prepare_pt2e")(
                    self._fx_module, queue_element["quantizer"]
                )

                if self._pt2e_quant_context != None:
                    self._pt2e_quant_context.append_graph(self._prepared_module.graph)

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

            self._pt2e_quant_context.set_input_for_tracing(args[-1])
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
                self._converted_module = _native_pt2e_quantization_interface("convert_pt2e")(
                    self._prepared_module,
                    use_reference_representation=queue_element["use_reference_representation"],
                    fold_quantize=queue_element["fold_quantize"],
                )

                if os.getenv("USE_FX_GRAPH_PATTERN_MATCHING", "0") != "0":
                    logger.debug("=================BEFORE PASS================")
                    logger.debug(self._converted_module.graph)
                    logger.debug("============================================")
                    replace_pattern_quant_dequant_mm_addmm(self._converted_module)
                    replace_pattern_quant_dequant_bmm(self._converted_module)
                    replace_quantize_with_cast(self._converted_module)
                    # replace_pattern_quant_dequant_softmax(self._converted_module)
                    logger.debug("=================AFTER PASS================")
                    logger.debug(self._converted_module.graph)
                    logger.debug("===========================================")
                    global nodes_replaced
                    logger.debug("=================TOTAL CHANGES {}================".format(nodes_replaced))

                if self._pt2e_quant_context != None:
                    self._pt2e_quant_context.append_graph(self._converted_module.graph)

                # Now we call hpu_inference_compiler to convert it into synapse graph.
                if os.getenv("USE_FX_GRAPH_FREEZING", "0") != "0":
                    with torch.no_grad():
                        self._converted_module = torch.compile(
                            self._converted_module, backend="hpu_backend", options={"use_graph_freezing": True}
                        )
                else:
                    with torch.no_grad():
                        self._converted_module = torch.compile(self._converted_module, backend="hpu_backend")

                self._converted = True

            self._pt2e_quant_context.set_input_for_tracing(args[-1])
            return self._converted_module(*args, **kwargs)


def habana_quant_compiler_fw(
    module: torch.fx.GraphModule,
    example_inputs: List[torch.Tensor],
    module_key: torch.fx.GraphModule,
    pt2e_quant_context: HabanaPT2EQuantContext,
):
    # This backend only sets up runtime wrapper to run real compilation once we have real tensors.
    return functorch.compile.make_boxed_func(HabanaQuantWrapperModule(module, module_key, pt2e_quant_context))


def habana_quant_compiler_bw_raise(graph_module: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
    raise Exception("tried to call backward pass compiler in inference backend")


def habana_quant_backend(
    graph_module: torch.fx.GraphModule,
    example_inputs: List[torch.Tensor],
    module_key: torch.fx.GraphModule,
    pt2e_quant_context: HabanaPT2EQuantContext,
    **kwargs,
):
    """
    This function implements interface for Habana's PT2E quantization backend.
    """
    from habana_frameworks.torch.dynamo.compile_backend import config as habana_quant_backend_config
    from habana_frameworks.torch.dynamo.compile_backend.decomposition import (
        get_hpu_decompositions,
        override_composite_ops,
    )

    options = kwargs["options"] if "options" in kwargs else None
    with habana_quant_backend_config.patch(options), override_composite_ops():
        return aot_autograd(
            fw_compiler=habana_quant_backend_config.patch(options)(
                partial(habana_quant_compiler_fw, module_key=module_key, pt2e_quant_context=pt2e_quant_context)
            ),
            bw_compiler=habana_quant_compiler_bw_raise,
            decompositions=get_hpu_decompositions(),
            keep_inference_input_mutations=habana_quant_backend_config.keep_input_mutations,
        )(graph_module, example_inputs)


# ======================================================================================
# Habana's implementation of PT2E like multi-graph export
# Note: It uses torch.compile based approach with custom quantization backend
# ======================================================================================
def export(
    f: torch.nn.Module,
    args: Tuple[Any] = None,
    kwargs: Optional[Dict[str, Any]] = None,
    dynamic_shapes: Optional[Union[Dict[str, Any], Tuple[Any]]] = None,
) -> torch.nn.Module:
    """
    Habana's implementation of PT2E like multi-graph export
    Note: It uses torch.compile based approach with custom quantization backend
    """
    logger.debug("Habana's implementation of PT2E based quantization flow: [export]")

    id_model = hash((id(f), type(f).__name__))
    global export_model_record
    global habana_pt2e_quant_context
    if id_model in export_model_record.keys():
        habana_pt2e_quant_context = export_model_record[id_model][1]
        return export_model_record[id_model][0]

    if graph_breaks(f, args, kwargs):
        habana_pt2e_quant_context = HabanaPT2EQuantContext(id_model, args)
        global habana_quantization_map_queue
        model_key = len(habana_quantization_map_queue)
        habana_quantization_map_queue = {model_key: []}
        torch._dynamo.reset()
        model = torch.compile(
            f,
            backend=partial(habana_quant_backend, module_key=model_key, pt2e_quant_context=habana_pt2e_quant_context),
            dynamic=False,
            options={"keep_input_mutations": True},
        )
        setattr(model, "meta_hb_quant_id", model_key)
        habana_pt2e_quant_context.set_model(model)
        if args != None:
            model(*args)
            logger.debug(f"Graph after pt2e kind of export:\n {model.graph}")

        setattr(model, "multi_graph", True)
        export_model_record[id_model] = [model, habana_pt2e_quant_context]
        return model
    else:
        habana_pt2e_quant_context = None
        if kwargs != None and "graph_break_present" in kwargs:
            kwargs.pop("graph_break_present")
        model = _native_pt2e_quantization_interface("export")(f, args, kwargs, dynamic_shapes)
        logger.debug(f"Graph after pt2 export:\n {model.graph}")
        setattr(model, "multi_graph", False)
        export_model_record[id_model] = [model, habana_pt2e_quant_context]
        return model


# ======================================================================================
# Habana's implementation of prepare_pt2e for multi-graph scenario
# ======================================================================================
def prepare_pt2e(
    model: GraphModule,
    quantizer: Quantizer,
) -> GraphModule:
    """
    Habana's implementation of prepare_pt2e for multi-graph scenario
    """
    logger.debug("Habana's implementation of PT2E based quantization flow: [prepare_pt2e]")

    multi_graph = getattr(model, "multi_graph", False)
    if multi_graph:
        # Set "prepare_pt2e" cmd for HabanaQuantWrapperModule
        global habana_quantization_map_queue
        model_key = getattr(model, "meta_hb_quant_id")
        habana_quantization_map_queue[model_key] = []
        habana_quantization_map_queue[model_key].append({"task": "prepare_pt2e", "quantizer": quantizer})

        global habana_pt2e_quant_context
        habana_pt2e_quant_context.clear_graphs()
        habana_pt2e_quant_context.set_model(model)
        if habana_pt2e_quant_context.get_input_for_tracing() != None:
            model(*habana_pt2e_quant_context.get_input_for_tracing())
            logger.debug(f"Graph after prepare_pt2e:\n {model.graph}")
        setattr(model, "multi_graph", True)
        return model
    else:
        model = _native_pt2e_quantization_interface("prepare_pt2e")(model, quantizer)
        logger.debug(f"Graph after prepare_pt2e:\n {model.graph}")
        setattr(model, "multi_graph", False)
        return model


# ======================================================================================
# Habana's implementation of convert_pt2e for multi-graph scenario
# ======================================================================================
def convert_pt2e(
    model: GraphModule,
    use_reference_representation: bool = False,
    fold_quantize: bool = True,
) -> GraphModule:
    """
    Habana's implementation of convert_pt2e for multi-graph scenario
    """
    logger.debug("Habana's implementation of PT2E based quantization flow: [convert_pt2e]")

    multi_graph = getattr(model, "multi_graph", False)
    if multi_graph:
        # Set "convert_pt2e" cmd for HabanaQuantWrapperModule
        global habana_quantization_map_queue
        model_key = getattr(model, "meta_hb_quant_id")
        habana_quantization_map_queue[model_key] = []
        habana_quantization_map_queue[model_key].append(
            {
                "task": "convert_pt2e",
                "use_reference_representation": use_reference_representation,
                "fold_quantize": False,
            }
        )

        global habana_pt2e_quant_context
        habana_pt2e_quant_context.clear_graphs()
        habana_pt2e_quant_context.set_model(model)
        if habana_pt2e_quant_context.get_input_for_tracing() != None:
            model(*habana_pt2e_quant_context.get_input_for_tracing())
            logger.debug(f"Graph after convert_pt2e:\n {model.graph}")
        setattr(model, "multi_graph", True)
        return model
    else:
        model = _native_pt2e_quantization_interface("convert_pt2e")(
            model, use_reference_representation, fold_quantize=fold_quantize
        )
        logger.debug(f"Graph after convert_pt2e:\n {model.graph}")
        setattr(model, "multi_graph", False)
        return model


def is_node(node, name):
    if node and node.op == "call_function" and node.target.__name__ == name:
        return True
    return False


nodes_replaced = 0


def replace_pattern_quant_dequant_softmax(graph_module: torch.fx.GraphModule):
    # Iterate through all nodes in the graph
    graph = graph_module.graph
    nodes_to_remove = []

    graph_changed = False
    for node in graph.nodes:
        # Check if the node is a softmax.default operation
        if is_node(node, "_softmax.default"):
            logger.debug(f"Found softmax.default node: {node.name}")
            softmax_users_node = next(iter(node.users), None)
            output_view_node = None
            output_quant_node = None
            input_quant_node = None
            input_dequant_node = None
            if is_node(softmax_users_node, "quantize_per_tensor.default"):
                output_quant_node = softmax_users_node
            if is_node(node.args[0], "dequantize_per_tensor.default"):
                input_dequant_node = node.args[0]
                if is_node(input_dequant_node.args[0], "quantize_per_tensor.default"):
                    input_quant_node = input_dequant_node.args[0]
            if output_quant_node is None or input_quant_node is None:
                logger.debug("======Failed to match the expected pattern for Softmax========")
                continue
            with graph.inserting_before(node):
                graph_changed = True
                output_scale = output_quant_node.args[1]
                input_scale = input_quant_node.args[1]
                logger.debug(
                    f"replace_softmax_quant_dequant_nodes: input_scale={input_scale} output_scale={output_scale}"
                )
                input_scale_tensor = graph.call_function(torch.ops.aten.scalar_tensor, (input_scale,))
                output_scale_tensor = graph.call_function(torch.ops.aten.scalar_tensor, (output_scale,))
                softmax_fp8_node = graph.call_function(
                    torch.ops.hpu.softmax_fp8,
                    args=(
                        input_quant_node.args[0],
                        node.args[1],
                        input_scale_tensor,
                        output_scale_tensor,
                        # torch.Tensor([1.0]),
                        # torch.Tensor([1.0 / input0_scale]),
                        # torch.nn.Parameter(torch.tensor([1.0])),
                        # torch.nn.Parameter(torch.tensor([1.0 / input0_scale])),
                        # None,
                    ),
                )

            output_quant_node.replace_all_uses_with(softmax_fp8_node)
            nodes_to_remove.extend([output_quant_node, node, input_dequant_node, input_quant_node])

    if not graph_changed:
        assert nodes_to_remove == []
        return

    # Remove marked nodes from the graph
    for node in nodes_to_remove:
        graph.erase_node(node)
    graph_module.recompile()


def get_dequant_node(node):
    view_nodes = [
        "view.default",
        "expand.default",
        "clone.default",
        "_unsafe_view.default",
        "slice.Tensor",
    ]
    while node and node.op == "call_function":
        if is_node(node, "dequantize_per_tensor.default"):
            return node
        if node.target.__name__ not in view_nodes:
            logger.debug("Traced back to a non view node {}".format(node.target.__name__))
            break
        node = node.args[0]
    return None


def replace_quantize_with_cast(module: torch.fx.GraphModule):
    # Iterate through all nodes in the graph
    graph = module.graph

    graph_changed = False
    nodes_to_remove = []
    for node in graph.nodes:
        # Check if the node is a bmm.default operation
        if is_node(node, "quantize_per_tensor.default"):

            with graph.inserting_before(node):
                invert_scale_node = 1 / node.args[1]
                cast_node = graph.call_function(
                    torch.ops.hpu.cast_to_fp8_v2,
                    args=(
                        node.args[0],
                        invert_scale_node,
                        False,
                        False,
                        torch.float8_e4m3fn,
                    ),
                )
                quant_out = module.graph.call_function(operator.getitem, args=(cast_node, 0))
            node.replace_all_uses_with(quant_out)
            nodes_to_remove.extend(
                [
                    node,
                ]
            )

    for node in nodes_to_remove:
        graph.erase_node(node)
    graph.lint()  # Ensure graph integrity
    module.recompile()


def replace_pattern_quant_dequant_bmm(module: torch.fx.GraphModule):
    # Iterate through all nodes in the graph
    graph = module.graph
    nodes_to_remove = []
    number_of_bmm_replacements_done = 0

    graph_changed = False
    for node in graph.nodes:
        # Check if the node is a bmm.default operation
        if is_node(node, "bmm.default"):
            input0_dequant_node = get_dequant_node(node.args[0])
            input1_dequant_node = get_dequant_node(node.args[1])
            if input0_dequant_node is not None:
                logger.debug("Dequant node found for input0")
            if input1_dequant_node is not None:
                logger.debug("Dequant node found for input1")
            if input0_dequant_node and input1_dequant_node:
                gemm_users_node = next(iter(node.users), None)
                output_view_node = None
                if is_node(gemm_users_node, "view.default"):
                    output_view_node = gemm_users_node

                buffer_counter = len(module.state_dict())
                input0_scale_attr = f"__param_constant_{buffer_counter}"
                buffer_counter = buffer_counter + 1
                module.register_buffer(input0_scale_attr, torch.tensor(input0_dequant_node.args[1], device="hpu"))
                input1_scale_attr = f"__param_constant_{buffer_counter}"
                buffer_counter = buffer_counter + 1
                module.register_buffer(input1_scale_attr, torch.tensor(input1_dequant_node.args[1], device="hpu"))

                with graph.inserting_before(input0_dequant_node):
                    input0_scale_node = graph.get_attr(input0_scale_attr)
                    input1_scale_node = graph.get_attr(input1_scale_attr)

                with graph.inserting_before(node):
                    graph_changed = True
                    logger.debug(
                        f"replace_bmm_quant_dequant_nodes: input0_scale={input0_dequant_node.args[1]}, input1_scale={input1_dequant_node.args[1]}"
                    )
                    gemm_fp8_node = graph.call_function(
                        torch.ops.hpu.fp8_gemm_v2,
                        args=(
                            node.args[0],
                            False,
                            node.args[1],
                            False,
                            None,
                            torch.bfloat16,
                            input0_scale_node,
                            input1_scale_node,
                            None,
                            False,
                        ),
                    )
                if output_view_node is not None:
                    output_view_node.args = (gemm_fp8_node,) + output_view_node.args[1:]
                    node.replace_all_uses_with(output_view_node)
                else:
                    node.replace_all_uses_with(gemm_fp8_node)

                input0_dequant_node.replace_all_uses_with(input0_dequant_node.args[0])
                input1_dequant_node.replace_all_uses_with(input1_dequant_node.args[0])
                nodes_to_remove.extend(
                    [
                        node,
                        input0_dequant_node,
                        input1_dequant_node,
                    ]
                )
            number_of_bmm_replacements_done = number_of_bmm_replacements_done + 1

    if not graph_changed:
        assert nodes_to_remove == []
        return

    global nodes_replaced
    nodes_replaced = nodes_replaced + number_of_bmm_replacements_done

    # Remove marked nodes from the graph
    for node in nodes_to_remove:
        graph.erase_node(node)

    graph.lint()  # Ensure graph integrity
    module.recompile()


# ======================================================================================
# Replace quant-dequant node with fp8 ops
# ======================================================================================
def replace_pattern_quant_dequant_mm_addmm(module: torch.fx.GraphModule):
    graph = module.graph
    nodes_to_remove = []
    graph_changed = False
    number_of_mm_addmm_replacements_done = 0

    for node in graph.nodes:
        if not (is_node(node, "mm.default") or is_node(node, "addmm.default")):
            continue

        gemm_node = node
        source_fn_stack = node.meta.get("source_fn_stack", None)
        weight_transpose = source_fn_stack and source_fn_stack[-1][1] in [torch.nn.Linear, torch.nn.functional.linear]

        gemm_users_node = next(iter(gemm_node.users), None)
        output_view_node = gemm_users_node if is_node(gemm_users_node, "_unsafe_view.default") else None

        is_addmm_node = is_node(gemm_node, "addmm.default")
        weight_idx = 2 if is_addmm_node else 1
        input_idx = 1 if is_addmm_node else 0
        bias = gemm_node.args[0] if is_addmm_node else None

        weight_transpose_node = (
            gemm_node.args[weight_idx] if is_node(gemm_node.args[weight_idx], "transpose.int") else None
        )
        weight_dequant_node = (
            weight_transpose_node.args[0]
            if weight_transpose_node and is_node(weight_transpose_node.args[0], "dequantize_per_tensor.default")
            else None
        )

        if not weight_dequant_node:
            logger.debug("Weight pattern match failed")
            continue

        weight_quant_node = weight_dequant_node.args[0]

        input_dequant_node = None
        input_view_node = gemm_node.args[input_idx] if is_node(gemm_node.args[input_idx], "view.default") else None
        if input_view_node and is_node(input_view_node.args[0], "dequantize_per_tensor.default"):
            input_dequant_node = input_view_node.args[0]
        elif is_node(gemm_node.args[input_idx], "dequantize_per_tensor.default"):
            input_dequant_node = gemm_node.args[input_idx]

        if not input_dequant_node:
            logger.debug("Input pattern match failed")
            continue

        input_quant_node = input_dequant_node.args[0]

        is_view = input_view_node is not None and output_view_node is not None
        if is_view:
            input_view_node.args = (input_quant_node,) + input_view_node.args[1:]

        gemm_input = input_view_node if is_view else input_quant_node
        insertion_node = output_view_node if is_view else gemm_node

        buffer_counter = len(module.state_dict())
        input_scale_attr = f"__param_constant_{buffer_counter}"
        buffer_counter = buffer_counter + 1
        # Store the scalar value as a buffer in the module
        module.register_buffer(input_scale_attr, torch.tensor(input_quant_node.args[1], device="hpu"))
        weight_scale_attr = f"__param_constant_{buffer_counter}"
        buffer_counter = buffer_counter + 1
        module.register_buffer(weight_scale_attr, torch.tensor(weight_quant_node.args[1], device="hpu"))

        with graph.inserting_before(input_quant_node):
            input_scale_node = graph.get_attr(input_scale_attr)
            weight_scale_node = graph.get_attr(weight_scale_attr)

        # input_quant_node.args = (input_quant_node.args[0], ) + (input_scale_node, ) + input_scale_node.args[2:]
        with graph.inserting_before(insertion_node):
            graph_changed = True
            gemm_fp8_node = graph.call_function(
                torch.ops.hpu.fp8_gemm_v2,
                args=(
                    gemm_input,
                    False,
                    weight_quant_node,
                    weight_transpose,
                    None,
                    torch.bfloat16,
                    input_scale_node,
                    weight_scale_node,
                    bias,
                ),
            )

        if is_view:
            output_view_node.args = (gemm_fp8_node,) + output_view_node.args[1:]
            gemm_node.replace_all_uses_with(output_view_node)
        else:
            gemm_node.replace_all_uses_with(gemm_fp8_node)

        nodes_to_remove.extend([gemm_node, input_dequant_node, weight_transpose_node, weight_dequant_node])

        number_of_mm_addmm_replacements_done += 1

    if graph_changed:
        global nodes_replaced
        nodes_replaced += number_of_mm_addmm_replacements_done

        for node in nodes_to_remove:
            graph.erase_node(node)

        graph.lint()
        module.recompile()


# ======================================================================================
# Freeze parameters for linear op, as is done in case of torch.export()
# ======================================================================================
def preprocess_linears(placeholder_map, model: torch.fx.GraphModule, tupled_args, *args):
    linear_module_partitions = get_source_partitions(model.graph, [torch.nn.Linear, torch.nn.functional.linear])

    if len(linear_module_partitions) == 0:
        return

    global param_id
    model_changed = False
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
                with model.graph.inserting_before(weight_node_first_user):
                    model_changed = model_changed or True
                    attr_name = "_param_constant_l" + str(param_id)
                    param_tensor = tupled_args[placeholder_map[weight_node.name]]
                    setattr(model, attr_name, torch.nn.parameter.Parameter(param_tensor.detach()))
                    new_attr_node = model.graph.create_node("get_attr", attr_name)
                    weight_node_first_user.replace_input_with(weight_node, new_attr_node)
                    param_id = param_id + 1

                    # Fix source code meta for annotations detection.
                    new_attr_node.meta["source_fn_stack"] = compute_node.meta.get("source_fn_stack", None)
                    new_attr_node.meta["stack_trace"] = compute_node.meta.get("stack_trace", None)
                    new_attr_node.meta["tensor_meta"] = compute_node.meta.get("tensor_meta", None)
                    new_attr_node.meta["val"] = compute_node.meta.get("val", None)

                if bias_node is not None:
                    with model.graph.inserting_before(bias_node_first_user):
                        model_changed = model_changed or True
                        attr_name = "_param_constant_l" + str(param_id)
                        param_tensor = tupled_args[placeholder_map[bias_node.name]]
                        setattr(model, attr_name, torch.nn.parameter.Parameter(param_tensor.detach()))
                        new_attr_node = model.graph.create_node("get_attr", attr_name)
                        bias_node_first_user.replace_input_with(bias_node, new_attr_node)
                        param_id = param_id + 1

                        # Fix source code meta for annotations detection.
                        new_attr_node.meta["source_fn_stack"] = compute_node.meta.get("source_fn_stack", None)
                        new_attr_node.meta["stack_trace"] = compute_node.meta.get("stack_trace", None)
                        new_attr_node.meta["tensor_meta"] = compute_node.meta.get("tensor_meta", None)
                        new_attr_node.meta["val"] = compute_node.meta.get("val", None)

    if model_changed:
        model.graph.lint()
        model.recompile()


# ======================================================================================
# Freeze parameters for conv op, as is done in case of torch.export()
# ======================================================================================
def preprocess_convs(placeholder_map, model: torch.fx.GraphModule, tupled_args):
    conv_module_partitions = get_source_partitions(model.graph, [torch.nn.Conv2d, torch.nn.functional.conv2d])

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
                with model.graph.inserting_before(weight_node_first_user):
                    attr_name = "_param_constant_c" + str(param_id)
                    param_tensor = tupled_args[placeholder_map[weight_node.name]]
                    setattr(model, attr_name, torch.nn.parameter.Parameter(param_tensor.detach()))
                    new_attr_node = model.graph.create_node("get_attr", attr_name)
                    weight_node_first_user.replace_input_with(weight_node, new_attr_node)
                    param_id = param_id + 1

                    # Fix source code meta for annotations detection.
                    new_attr_node.meta["source_fn_stack"] = compute_node.meta.get("source_fn_stack", None)
                    new_attr_node.meta["stack_trace"] = compute_node.meta.get("stack_trace", None)
                    new_attr_node.meta["tensor_meta"] = compute_node.meta.get("tensor_meta", None)
                    new_attr_node.meta["val"] = compute_node.meta.get("val", None)

                with model.graph.inserting_before(bias_node_first_user):
                    attr_name = "_param_constant_c" + str(param_id)
                    param_tensor = tupled_args[placeholder_map[bias_node.name]]
                    setattr(model, attr_name, torch.nn.parameter.Parameter(param_tensor.detach()))
                    new_attr_node = model.graph.create_node("get_attr", attr_name)
                    bias_node_first_user.replace_input_with(bias_node, new_attr_node)
                    param_id = param_id + 1

                    # Fix source code meta for annotations detection.
                    new_attr_node.meta["source_fn_stack"] = compute_node.meta.get("source_fn_stack", None)
                    new_attr_node.meta["stack_trace"] = compute_node.meta.get("stack_trace", None)
                    new_attr_node.meta["tensor_meta"] = compute_node.meta.get("tensor_meta", None)
                    new_attr_node.meta["val"] = compute_node.meta.get("val", None)

    model.graph.lint()
    model.recompile()


# ======================================================================================
# Change FX graph so that it resembles one that would be generated by torch.export()
# ======================================================================================
def discover_and_materialize_params(model: torch.fx.GraphModule, *args):

    # Get placeholder map from FX graph.
    placeholder_map = {}
    placeholder_count = 0
    for node in model.graph.nodes:
        if node.op == "placeholder":
            placeholder_map[node.name] = placeholder_count
            placeholder_count = placeholder_count + 1

    tupled_args = tuple(args)

    # Handle following custom linear modules in deepspeed
    def handle_custom_linear_modules(model):
        for node in model.graph.nodes:
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
        handle_custom_linear_modules(model)

    preprocess_linears(placeholder_map, model, tupled_args, *args)
    preprocess_convs(placeholder_map, model, tupled_args)
