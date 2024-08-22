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

                if self._pt2e_quant_context != None:
                    self._pt2e_quant_context.append_graph(self._converted_module.graph)

                # Now we call hpu_inference_compiler to convert it into synapse graph.
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
):
    """
    This function implements interface for Habana's PT2E quantization backend.
    """
    from habana_frameworks.torch.dynamo.compile_backend.decomposition import get_hpu_decompositions

    return aot_autograd(
        fw_compiler=partial(habana_quant_compiler_fw, module_key=module_key, pt2e_quant_context=pt2e_quant_context),
        bw_compiler=habana_quant_compiler_bw_raise,
        decompositions=get_hpu_decompositions(),
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

    id_model = id(f)
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
