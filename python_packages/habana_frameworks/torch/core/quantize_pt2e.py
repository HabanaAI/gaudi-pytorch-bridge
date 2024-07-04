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

# Note - A significant part of this implementation is taken from quantze_pt2e toy example
# https://gist.github.com/leslie-fang-intel/b78ed682aa9b54d2608285c5a4897cfc#file-toy_example_quantization_2_0-py
# E.g. BackendQuantizer and get_symmetric_quantization_config
# However, they have been renamed and amended as per the present need.

import copy
import importlib
import itertools
import operator
import os
from functools import partial
from typing import Any, Callable, Dict, List, Optional

import functorch
import torch
from habana_frameworks.torch import hpu
from habana_frameworks.torch.dynamo.compile_backend.logger import get_compile_backend_logger
from torch._dynamo.backends.common import aot_autograd
from torch.ao.quantization.observer import MinMaxObserver, PerChannelMinMaxObserver, PlaceholderObserver
from torch.ao.quantization.qconfig import _ObserverOrFakeQuantizeConstructor
from torch.ao.quantization.quantizer.x86_inductor_quantizer import (
    OperatorConfig,
    QuantizationAnnotation,
    QuantizationConfig,
    QuantizationSpec,
    Quantizer,
    SharedQuantizationSpec,
)
from torch.ao.quantization.quantizer.xnnpack_quantizer_utils import (
    _annotate_input_qspec_map,
    _annotate_output_qspec,
    get_bias_qspec,
    get_input_act_qspec,
    get_output_act_qspec,
    get_weight_qspec,
)
from torch.fx import Node
from torch.fx.passes.utils.source_matcher_utils import SourcePartition, get_source_partitions

logger = get_compile_backend_logger()

QUANTIZER_MIN_MAX = {torch.int8: (-128, 127), torch.float8_e4m3fn: (-240, 240), torch.float8_e5m2: (-240, 240)}
extra_args_act: Dict[str, Any] = {"for_observer": {"eps": 2**-12}, "margin": 2}
extra_args_weight: Dict[str, Any] = {"for_observer": {"eps": 2**-12}, "margin": 0}

scale_history_of_last_linear_or_conv_weight = dict()
habana_quantization_map_queue = []
export_module_record = dict()
quant_dtype_used = None
param_id = 0


# ======================================================================================
# Utility functions for internal testing only
# ======================================================================================
def set_activation_backoff_margin(margin=0):
    extra_args_act["margin"] = margin


def set_weight_backoff_margin(margin=0):
    extra_args_weight["margin"] = margin


def get_weight_scale_history():
    return scale_history_of_last_linear_or_conv_weight.copy()


# ======================================================================================
# Utility functions used by Habana Quantizer definition
# ======================================================================================
def _mark_nodes_as_annotated(nodes: List[Node]):
    for node in nodes:
        if node is not None:
            if "quantization_annotation" not in node.meta:
                node.meta["quantization_annotation"] = QuantizationAnnotation()
            node.meta["quantization_annotation"]._annotated = True


def _is_annotated(nodes: List[Node]):
    annotated = False
    for node in nodes:
        annotated = annotated or (
            "quantization_annotation" in node.meta and node.meta["quantization_annotation"]._annotated
        )
    return annotated


def _update_input_qspec_map(partition: SourcePartition, input_node: Node, qspec: QuantizationSpec) -> None:
    input_node_user = None
    for n in partition.nodes:
        if n in input_node.users:
            input_node_user = n
            break
    if input_node_user is None:
        raise ValueError("Could not find a user within source partition.")
    _annotate_input_qspec_map(
        input_node_user,
        input_node,
        qspec,
    )


def _update_output_qspec(output_node: Node, qspec: QuantizationSpec) -> None:
    if _is_annotated([output_node]) is False:
        _annotate_output_qspec(output_node, qspec)


# ======================================================================================
# Habana Quantizer definition
# ======================================================================================
class habana_quantizer(Quantizer):

    def __init__(self):
        super().__init__()
        self.global_config: QuantizationConfig = None  # type: ignore[assignment]
        self.operator_type_config: Dict[str, Optional[QuantizationConfig]] = {}

    def set_global(self, quantization_config: QuantizationConfig):
        """set global QuantizationConfig used for the backend.
        QuantizationConfig is defined in torch/ao/quantization/_pt2e/quantizer/quantizer.py.
        """
        self.global_config = quantization_config
        return self

    def annotate(self, model: torch.fx.GraphModule) -> torch.fx.GraphModule:
        """annotate nodes in the graph with observer or fake quant constructors
        to convey the desired way of quantization.
        """
        global_config = self.global_config
        self.annotate_symmetric_config(model, global_config)

        return model

    def annotate_symmetric_config(
        self, model: torch.fx.GraphModule, config: QuantizationConfig
    ) -> torch.fx.GraphModule:
        self._annotate_linear(model, config)
        self._annotate_matmul(model, config)
        self._annotate_conv2d(model, config)
        self._annotate_maxpool2d(model, config)

        return model

    def _annotate_conv2d(self, gm: torch.fx.GraphModule, quantization_config: QuantizationConfig) -> None:
        conv_partitions = get_source_partitions(gm.graph, [torch.nn.Conv2d, torch.nn.functional.conv2d])

        if len(conv_partitions) == 0:
            return

        conv_partitions = list(itertools.chain(*conv_partitions.values()))

        for conv_partition in conv_partitions:
            if len(conv_partition.output_nodes) > 1:
                raise ValueError("conv partition has more than one output node")
            conv_node = conv_partition.output_nodes[0]
            if conv_node.op != "call_function" or conv_node.target != torch.ops.aten.convolution.default:
                raise ValueError(f"{conv_node} is not an aten conv2d operator")
            # skip annotation if it is already annotated
            if _is_annotated([conv_node]):
                continue

            input_qspec_map = {}
            input_act = conv_node.args[0]
            assert isinstance(input_act, Node)
            input_qspec_map[input_act] = get_input_act_qspec(quantization_config)

            weight = conv_node.args[1]
            assert isinstance(weight, Node)
            input_qspec_map[weight] = get_weight_qspec(quantization_config)

            bias = conv_node.args[2]
            if isinstance(bias, Node):
                input_qspec_map[bias] = get_bias_qspec(quantization_config)

            conv_node.meta["quantization_annotation"] = QuantizationAnnotation(
                input_qspec_map=input_qspec_map,
                output_qspec=get_output_act_qspec(quantization_config),
                _annotated=True,
            )

    def _annotate_linear(self, gm: torch.fx.GraphModule, quantization_config: QuantizationConfig) -> None:
        module_partitions = get_source_partitions(gm.graph, [torch.nn.Linear, torch.nn.functional.linear])

        if len(module_partitions) == 0:
            return

        act_qspec = get_input_act_qspec(quantization_config)
        weight_qspec = get_weight_qspec(quantization_config)
        bias_qspec = get_bias_qspec(quantization_config)
        for module_or_fn_type, partitions in module_partitions.items():
            if module_or_fn_type == torch.nn.Linear or module_or_fn_type == torch.nn.functional.linear:
                for p in partitions:
                    act_node = p.input_nodes[0]
                    output_node = p.output_nodes[0]
                    weight_node = None
                    bias_node = None
                    for node in p.params:
                        weight_or_bias = getattr(gm, node.target)  # type: ignore[arg-type]
                        if weight_or_bias.ndim == 2:  # type: ignore[attr-defined]
                            weight_node = node
                        if weight_or_bias.ndim == 1:  # type: ignore[attr-defined]
                            bias_node = node

                    if weight_node is None:
                        logger.warn("No weight found in Linear pattern")
                        continue

                    _update_input_qspec_map(p, act_node, act_qspec)
                    _update_input_qspec_map(p, weight_node, weight_qspec)
                    if bias_node:
                        _update_input_qspec_map(p, bias_node, bias_qspec)
                    _update_output_qspec(output_node, act_qspec)

                    nodes_to_mark_annotated = list(p.nodes)
                    _mark_nodes_as_annotated(nodes_to_mark_annotated)

    def _annotate_matmul(self, gm: torch.fx.GraphModule, quantization_config: QuantizationConfig) -> None:
        matmul_partitions = get_source_partitions(gm.graph, [torch.matmul])

        if len(matmul_partitions) == 0:
            return

        act_qspec = get_input_act_qspec(quantization_config)
        for module_or_fn_type, partitions in matmul_partitions.items():
            for p in partitions:
                assert len(p.input_nodes) == 2
                act_node1 = p.input_nodes[0]
                act_node2 = p.input_nodes[1]
                assert len(p.output_nodes) == 1
                output_node = p.output_nodes[0]

                _update_input_qspec_map(p, act_node1, act_qspec)
                _update_input_qspec_map(p, act_node2, act_qspec)
                _update_output_qspec(output_node, act_qspec)

                nodes_to_mark_annotated = list(p.nodes)
                _mark_nodes_as_annotated(nodes_to_mark_annotated)

    def _annotate_maxpool2d(self, gm: torch.fx.GraphModule, quantization_config: QuantizationConfig) -> None:
        module_partitions = get_source_partitions(gm.graph, [torch.nn.MaxPool2d, torch.nn.functional.max_pool2d])

        if len(module_partitions) == 0:
            return

        maxpool_partitions = list(itertools.chain(*module_partitions.values()))

        for maxpool_partition in maxpool_partitions:
            output_node = maxpool_partition.output_nodes[0]
            maxpool_node = None
            for n in maxpool_partition.nodes:
                if n.target == torch.ops.aten.max_pool2d_with_indices.default:
                    maxpool_node = n
            if _is_annotated([output_node, maxpool_node]):  # type: ignore[list-item]
                continue

            input_act = maxpool_node.args[0]  # type: ignore[union-attr]
            assert isinstance(input_act, Node)

            act_qspec = get_input_act_qspec(quantization_config)
            maxpool_node.meta["quantization_annotation"] = QuantizationAnnotation(  # type: ignore[union-attr]
                input_qspec_map={
                    input_act: act_qspec,
                },
                _annotated=True,
            )
            output_node.meta["quantization_annotation"] = QuantizationAnnotation(
                output_qspec=SharedQuantizationSpec((input_act, maxpool_node)),
                _annotated=True,
            )

    def validate(self, model: torch.fx.GraphModule) -> None:
        """validate if the annotated graph is supported by the backend"""
        pass

    @classmethod
    def get_supported_operators(cls) -> List[OperatorConfig]:
        return []


# ======================================================================================
# Habana Quant Config definition
# ======================================================================================
def habana_quant_config_symmetric(quant_dtype):
    logger.debug(f"habana_quant_config_symmetric: quantizer dtype is {quant_dtype}")
    quant_min, quant_max = QUANTIZER_MIN_MAX[quant_dtype]

    act_observer_or_fake_quant_ctr: _ObserverOrFakeQuantizeConstructor = MinMaxObserver
    act_observer_or_fake_quant_args = extra_args_act.get("for_observer").copy()
    if quant_dtype == torch.float8_e4m3fn:
        act_observer_or_fake_quant_args["eps"] = 0
    act_quantization_spec = QuantizationSpec(
        dtype=quant_dtype,
        quant_min=quant_min,
        quant_max=quant_max,
        qscheme=torch.per_tensor_symmetric,  # Due to this, MinMaxObserver acts as AbsMaxObserver
        is_dynamic=False,
        observer_or_fake_quant_ctr=act_observer_or_fake_quant_ctr.with_args(**act_observer_or_fake_quant_args),
    )

    weight_observer_or_fake_quant_ctr: _ObserverOrFakeQuantizeConstructor = MinMaxObserver
    weight_observer_or_fake_quant_args = extra_args_weight.get("for_observer").copy()
    if quant_dtype == torch.float8_e4m3fn:
        weight_observer_or_fake_quant_args["eps"] = 0
    weight_quantization_spec = QuantizationSpec(
        dtype=quant_dtype,
        quant_min=quant_min,
        quant_max=quant_max,
        qscheme=torch.per_tensor_symmetric,  # Due to this, MinMaxObserver acts as AbsMaxObserver
        ch_axis=0,
        is_dynamic=False,
        observer_or_fake_quant_ctr=weight_observer_or_fake_quant_ctr.with_args(**weight_observer_or_fake_quant_args),
    )

    bias_observer_or_fake_quant_ctr: _ObserverOrFakeQuantizeConstructor = PlaceholderObserver
    bias_quantization_spec = QuantizationSpec(
        dtype=torch.float, observer_or_fake_quant_ctr=bias_observer_or_fake_quant_ctr
    )
    quantization_config = QuantizationConfig(
        act_quantization_spec,
        act_quantization_spec,
        weight_quantization_spec,
        bias_quantization_spec,
    )
    global quant_dtype_used
    quant_dtype_used = quant_dtype
    return quantization_config


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

                # Adjust the scale values as per H/W requirements
                adjust_scale_val(self._converted_module)

                # Default datatype for output of dequantize is torch.float32 but we run models in torch.bfloat16
                change_output_dtype_of_dequant(self._converted_module)

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
# Utility functions used for aligning scale values as per HW requirement
# Note: This code is basically taken from quantization_toolkit. Please refer:
#       quantization_toolkit/habana_quantization_toolkit/_core/fp_utils.py
# ======================================================================================
def scale_to_pow2_hw(old_scale, eps, margin, is_weight, quant_dtype):
    import habana_frameworks.torch.utils.experimental as htexp

    GAUDI2 = htexp.synDeviceType.synDeviceGaudi2
    GAUDI3 = htexp.synDeviceType.synDeviceGaudi3

    EXP_WIDTH = {torch.float8_e4m3fn: 4, torch.float8_e5m2: 5}

    def get_default_exp_bias(dtype):
        exp_width = EXP_WIDTH[dtype]
        return 2 ** (exp_width - 1) - 1

    EXP_BIAS_SETS = {
        (GAUDI2, torch.float8_e4m3fn): [3, 7, 11, 15],
        (GAUDI2, torch.float8_e5m2): [15],
        (GAUDI3, torch.float8_e4m3fn): range(0, 63),
        (GAUDI3, torch.float8_e5m2): range(0, 63),
    }

    MAX_RANGE = {
        torch.float8_e4m3fn: 2 ** ((2**4 - 2 - get_default_exp_bias(torch.float8_e4m3fn))) * (2 - 2 ** -(8 - 1 - 4)),
        torch.float8_e5m2: 2 ** ((2**5 - 2 - get_default_exp_bias(torch.float8_e5m2))) * (2 - 2 ** -(8 - 1 - 5)),
    }

    def get_fullscale(dtype, exp_bias=None):
        default_exp_bias = get_default_exp_bias(dtype)
        fullscale = MAX_RANGE[dtype]
        exp_bias = default_exp_bias if exp_bias == None else exp_bias
        fullscale = fullscale * (2 ** (default_exp_bias - exp_bias))
        return fullscale

    def get_fullscales_by_expbias_set(dtype, expbias_set):
        return [get_fullscale(dtype, exp_bias=eb) for eb in expbias_set]

    def get_fp8_hw_alligned_scales(dtype, device):
        exp_bias_set = EXP_BIAS_SETS.get((device, dtype), None)
        return (
            None
            if exp_bias_set == None
            else [x / MAX_RANGE[dtype] for x in get_fullscales_by_expbias_set(dtype, exp_bias_set)]
        )

    DEVICES_SCALE_FACTORS = {GAUDI2: 4, GAUDI3: 1}
    FP8_143_SCALES = {
        device: get_fp8_hw_alligned_scales(quant_dtype, device) for device in DEVICES_SCALE_FACTORS.keys()
    }
    FP8_143_SCALES_TRAITS = {
        device: (min(FP8_143_SCALES[device]), max(FP8_143_SCALES[device]), DEVICES_SCALE_FACTORS[device])
        for device in DEVICES_SCALE_FACTORS.keys()
    }

    def scale_to_pow2(scale):
        scale_pow2 = 2 ** torch.ceil(torch.log2(scale))
        return scale_pow2

    def scale_after_backoff_adjustment(scale, eps, margin, is_weight):
        if is_weight:
            scale_history_of_last_linear_or_conv_weight["convert_pt2e_scale"] = scale.item()
        scale = scale * (2**margin)
        scale = max(scale, eps)
        if is_weight:
            scale_history_of_last_linear_or_conv_weight["backed_off_scale"] = scale.item()
        return scale

    scale_pow2 = scale_to_pow2(scale_after_backoff_adjustment(old_scale, eps, margin, is_weight))
    min_scale, max_scale, scale_factor = FP8_143_SCALES_TRAITS[GAUDI2]
    scale_pow2_hw = torch.minimum(
        torch.maximum(
            2 ** (torch.ceil(torch.log2(scale_pow2) / scale_factor) * scale_factor),
            torch.tensor(min_scale, dtype=old_scale.dtype, device=old_scale.device),
        ),
        torch.tensor(max_scale, dtype=old_scale.dtype, device=old_scale.device),
    )

    if is_weight:
        scale_history_of_last_linear_or_conv_weight["final_hw_scale"] = scale_pow2_hw.item()

    return scale_pow2_hw


def get_eps_and_backoff_margin(module: torch.fx.GraphModule, quant_node: torch.fx.node):
    # Check if quant_node's input arg is a param_constant or not
    input_node = quant_node.args[0]
    is_param_constant = input_node.op == "get_attr" and "_param_constant_l" in str(input_node.target)
    logger.debug(f"get_eps_and_backoff_margin: is_param_constant = {is_param_constant}")

    eps = 0
    backoff_margin = 0
    is_weight_param = False
    if is_param_constant:
        # Check if the param_constant is actually weight param constant or not
        dquant_node = None
        for n in module.graph.nodes:
            if (
                dquant_node == None
                and n.op == "call_function"
                and n.target.__name__ == "dequantize_per_tensor.default"
                and n.args[0] == quant_node
            ):
                dquant_node = n
                continue

            # Assuming topologically sorted graph
            if dquant_node:
                if n.op == "call_function":
                    if (n.target.__name__ == "addmm.default" and n.args[0] == dquant_node) or (
                        n.target.__name__ in ["mm.default", "linear.default", "convolution.default"]
                        and n.args[1] == dquant_node
                    ):
                        is_weight_param = True
                        break
                    else:
                        # This must be transpose kind of node. Just follow the path.
                        if dquant_node in n.args:
                            dquant_node = n

        logger.debug(f"get_eps_and_backoff_margin: is_weight_param = {is_weight_param}")
        if is_weight_param:
            # If weight param, use eps and backoff margin from extra_args_weight
            eps = extra_args_weight.get("for_observer").get("eps")
            backoff_margin = extra_args_weight.get("margin")
    else:
        # else, use eps and backoff margin from extra_args_act
        eps = extra_args_act.get("for_observer").get("eps")
        backoff_margin = extra_args_act.get("margin")

    logger.debug(f"get_eps_and_backoff_margin: (eps, backoff_margin) = ({eps}, {backoff_margin})")
    return eps, backoff_margin, is_weight_param


# ======================================================================================
# Adjust the scale as per HW requirement for float8 quantized dtype
# Align scale value to 2**n
# ======================================================================================
def adjust_scale_val(module: torch.fx.GraphModule):
    if quant_dtype_used != torch.float8_e4m3fn:
        return

    ## PART 1 - check quantization nodes
    nodes_to_change = []
    eps_and_backoff_margin = []
    for node in module.graph.nodes:
        if node.op == "call_function" and node.target.__name__ == "quantize_per_tensor.default":
            nodes_to_change.append(node)
            eps_and_backoff_margin.append(get_eps_and_backoff_margin(module, node))

    count = 0
    hw_aligned_scale = []
    hw_aligned_scales = []
    for node in nodes_to_change:
        node_args = list(node.args)
        old_scale = torch.tensor(node_args[1], dtype=torch.float32, device=torch.device("hpu"))
        new_scale = scale_to_pow2_hw(old_scale, *eps_and_backoff_margin[count], quant_dtype=node_args[5]).item()
        node_args[1] = new_scale
        node.args = tuple(node_args)
        count = count + 1

        num_users = len(set(node.users))
        hw_aligned_scale = [new_scale] * num_users
        hw_aligned_scales = hw_aligned_scales + hw_aligned_scale

    ## PART 2 - check dequantization nodes
    nodes_to_change = []
    for node in module.graph.nodes:
        if node.op == "call_function" and node.target.__name__ == "dequantize_per_tensor.default":
            nodes_to_change.append(node)

    assert len(hw_aligned_scales) == len(nodes_to_change)
    count = 0
    for node in nodes_to_change:
        node_args = list(node.args)
        node_args[1] = hw_aligned_scales[count]
        node.args = tuple(node_args)
        count = count + 1

    module.graph.lint()
    module.recompile()


# ======================================================================================
# Add argument out_dtype in dequantize_per_tensor op
# ======================================================================================
def change_output_dtype_of_dequant(module: torch.fx.GraphModule):
    global quant_dtype_used
    for node in module.graph.nodes:
        if node.op == "call_function" and node.target.__name__ == "dequantize_per_tensor.default":
            new_kwargs = node.kwargs.copy()
            new_kwargs["out_dtype"] = (
                torch.float32 if quant_dtype_used == torch.int8 else torch.bfloat16
            )  # Add dtype argument
            node.kwargs = new_kwargs
    module.graph.lint()
    module.recompile()


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
                    setattr(module, attr_name, torch.nn.parameter.Parameter(torch.clone(param_tensor.detach())))
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
                        setattr(module, attr_name, torch.nn.parameter.Parameter(torch.clone(param_tensor.detach())))
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
                    setattr(module, attr_name, torch.nn.parameter.Parameter(torch.clone(param_tensor.detach())))
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
                    setattr(module, attr_name, torch.nn.parameter.Parameter(torch.clone(param_tensor.detach())))
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
