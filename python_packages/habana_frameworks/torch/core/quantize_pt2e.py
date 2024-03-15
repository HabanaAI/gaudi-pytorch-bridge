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
import itertools
import operator
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
habana_quantization_map_queue = []
export_module_record = dict()


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
            if module_or_fn_type == torch.nn.Linear:
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
    act_observer_or_fake_quant_ctr: _ObserverOrFakeQuantizeConstructor = MinMaxObserver
    logger.debug(f"quantizer dtype is {quant_dtype}")
    quant_min, quant_max = QUANTIZER_MIN_MAX[quant_dtype]
    act_quantization_spec = QuantizationSpec(
        dtype=quant_dtype,
        quant_min=quant_min,
        quant_max=quant_max,
        qscheme=torch.per_tensor_symmetric,
        is_dynamic=False,
        observer_or_fake_quant_ctr=act_observer_or_fake_quant_ctr.with_args(eps=2**-12),
    )

    weight_observer_or_fake_quant_ctr: _ObserverOrFakeQuantizeConstructor = MinMaxObserver
    extra_args: Dict[str, Any] = {"eps": 2**-12}
    weight_quantization_spec = QuantizationSpec(
        dtype=quant_dtype,
        quant_min=quant_min,
        quant_max=quant_max,
        qscheme=torch.per_tensor_symmetric,
        ch_axis=0,
        is_dynamic=False,
        observer_or_fake_quant_ctr=weight_observer_or_fake_quant_ctr.with_args(**extra_args),
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

        from habana_frameworks.torch.dynamo.compile_backend.compilers import hpu_inference_compiler_noaot

        assert len(habana_quantization_map_queue[self._module_key]) == 1
        queue_element = habana_quantization_map_queue[self._module_key][0]
        if queue_element["task"] == "prepare_pt2e":
            if not self._prepared:
                from torch.ao.quantization.quantize_pt2e import prepare_pt2e

                self._prepared_module = prepare_pt2e(self._fx_module, queue_element["quantizer"])
                self._prepared = True

                # Save prepared module and work on active now, we will need prepared one later
                # so we can feed it to convert after we modify its statistics basing on active values.
                self._observed_module = copy.deepcopy(self._prepared_module)

                # Change the module so it can be fed into mid layer. We cannot do that just right
                # away because we need to remove module calls and unroll them to simple primitives.
                unroll_observers(self._observed_module)

                # Now we call hpu_inference_compiler_noaot to convert it into synapse graph.
                hpu_inference_compiler_noaot(self._observed_module, args)

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

                # Reconstruct prepared module with active stats.
                reconstruct_observers(self._prepared_module, self._observed_module)

                from torch.ao.quantization.quantize_pt2e import convert_pt2e

                # Take active module that we gathered stats on and convert it to final module.
                self._converted_module = convert_pt2e(self._prepared_module, use_reference_representation=False)
                self._converted = True

                # After this funtion we will be left with quantize/dequantize ops in the graph.
                # Unfortunately we do not support them directly so we need to make some manual
                # pattern matching here.
                decompose_quant_ops(self._converted_module)

                # Now we call hpu_inference_compiler_noaot to convert it into synapse graph.
                hpu_inference_compiler_noaot(self._converted_module, args)

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
# Decompose Quant into div + round + add + clamp
# Decompose DeQuant into sub + mul
# ======================================================================================
def decompose_quant_ops(module: torch.fx.GraphModule):
    ## PART 1 - quantizations
    nodes_to_change = []
    for node in module.graph.nodes:
        if node.op == "call_function" and node.target.__name__ == "quantize_per_tensor.default":
            nodes_to_change.append(node)

    quantization_src_dtypes = []
    quantization_dst_dtypes = []
    for node in nodes_to_change:
        num_users = len(set(node.users))
        quantization_src_dtype = [node.args[0].meta["tensor_meta"].dtype] * num_users
        quantization_src_dtypes = quantization_src_dtypes + quantization_src_dtype
        arg_param = node.args[0]
        arg_scale = node.args[1]
        arg_zero_point = node.args[2]
        arg_min = node.args[3]
        arg_max = node.args[4]
        arg_type = node.args[5]
        quantization_dst_dtype = [arg_type] * num_users
        quantization_dst_dtypes = quantization_dst_dtypes + quantization_dst_dtype

        with module.graph.inserting_before(node):
            if arg_type == torch.float8_e4m3fn:
                invert_scale_node = module.graph.call_function(torch.ops.aten.div.Tensor, (1, arg_scale))
                quant_node = module.graph.call_function(
                    torch.ops.hpu.cast_to_fp8_v2, args=(arg_param, invert_scale_node, False, False, arg_type)
                )
                quant_out = module.graph.call_function(operator.getitem, args=(quant_node, 0))
                final_typed_node = module.graph.call_function(
                    torch.ops.aten._to_copy.default,
                    (quant_out,),
                    {"dtype": arg_type},
                )
            else:
                div_node = module.graph.call_function(torch.ops.aten.div.Tensor, (arg_param, arg_scale))
                round_node = module.graph.call_function(torch.ops.aten.round.default, (div_node,))
                add_node = module.graph.call_function(torch.ops.aten.add.Tensor, (round_node, arg_zero_point))
                clamp_node = module.graph.call_function(torch.ops.aten.clamp.default, (add_node, arg_min, arg_max))
                final_typed_node = module.graph.call_function(
                    torch.ops.aten._to_copy.default,
                    (clamp_node,),
                    {"dtype": arg_type},
                )

        users_to_change = []
        for dst in node.users:
            users_to_change.append(dst)

        for dst in users_to_change:
            dst.replace_input_with(node, final_typed_node)

        module.graph.erase_node(node)

    ## PART 2 - dequantizations
    nodes_to_change = []
    for node in module.graph.nodes:
        if node.op == "call_function" and node.target.__name__ == "dequantize_per_tensor.default":
            nodes_to_change.append(node)

    assert len(quantization_src_dtypes) == len(nodes_to_change)
    assert len(quantization_dst_dtypes) == len(nodes_to_change)
    count = 0
    for node in nodes_to_change:
        src_dtype = quantization_dst_dtypes[count]
        dst_dtype = quantization_src_dtypes[count]
        count = count + 1

        arg_param = node.args[0]
        arg_scale = node.args[1]
        arg_zero_point = node.args[2]

        with module.graph.inserting_before(node):
            if src_dtype == torch.float8_e4m3fn:
                mul_node = module.graph.call_function(
                    torch.ops.hpu.cast_from_fp8, args=(arg_param, arg_scale, dst_dtype)
                )
            else:
                casted_input = module.graph.call_function(
                    torch.ops.aten._to_copy.default,
                    (arg_param,),
                    {"dtype": dst_dtype},
                )
                sub_node = module.graph.call_function(torch.ops.aten.sub.Tensor, (casted_input, arg_zero_point))
                mul_node = module.graph.call_function(torch.ops.aten.mul.Tensor, (sub_node, arg_scale))

        users_to_change = []
        for dst in node.users:
            users_to_change.append(dst)

        for dst in users_to_change:
            dst.replace_input_with(node, mul_node)

        module.graph.erase_node(node)

    module.graph.lint()
    module.recompile()


# ======================================================================================
# Function to remove Observers module calls and unroll them to simple primitives.
# ======================================================================================
def unroll_observers(module: torch.fx.GraphModule):

    nodes_to_change = []
    for node in module.graph.nodes:
        if node.op == "call_module":
            assert not node.kwargs
            submod = module.get_submodule(node.target)

            if (
                isinstance(submod, torch.ao.quantization.observer.MinMaxObserver)
                or isinstance(submod, torch.ao.quantization.observer.PerChannelMinMaxObserver)
                or isinstance(submod, torch.ao.quantization.observer.PlaceholderObserver)
            ):
                nodes_to_change.append(node)
            else:
                # Please add support for new observer if you are here.
                assert False

    observer_id = 0
    for node in nodes_to_change:
        submod = module.get_submodule(node.target)
        if isinstance(submod, torch.ao.quantization.observer.MinMaxObserver):
            old_min_stat = torch.clone(submod.min_val.detach())
            old_max_stat = torch.clone(submod.max_val.detach())
            min_attr_name = "_observer" + str(observer_id) + "_min_val"
            max_attr_name = "_observer" + str(observer_id) + "_max_val"
            setattr(module, min_attr_name, old_min_stat)
            setattr(module, max_attr_name, old_max_stat)

            with module.graph.inserting_before(node):
                min_attr_node = module.graph.create_node("get_attr", min_attr_name)
                max_attr_node = module.graph.create_node("get_attr", max_attr_name)

                casted_input = module.graph.call_function(
                    torch.ops.aten._to_copy.default,
                    (node.args[0],),
                    {"dtype": old_min_stat.dtype},
                )

                # Get min and max of a tensor and update stats.
                input_max = module.graph.call_function(torch.ops.aten.amax.default, (casted_input,))
                input_min = module.graph.call_function(torch.ops.aten.amin.default, (casted_input,))

                max_value = module.graph.call_function(torch.ops.aten.maximum.default, (input_max, max_attr_node))
                min_value = module.graph.call_function(torch.ops.aten.minimum.default, (input_min, min_attr_node))

                module.graph.call_function(torch.ops.aten.copy_.default, (max_attr_node, max_value))
                module.graph.call_function(torch.ops.aten.copy_.default, (min_attr_node, min_value))

            users_to_change = []
            for dst in node.users:
                users_to_change.append(dst)

            for dst in users_to_change:
                dst.replace_input_with(node, node.args[0])

            module.graph.erase_node(node)
            module.delete_submodule(node.target)

        elif isinstance(submod, torch.ao.quantization.observer.PerChannelMinMaxObserver):
            # Not really implemented, this is just sample on how/where to add new observers.
            assert False

        elif isinstance(submod, torch.ao.quantization.observer.PlaceholderObserver):
            users_to_change = []
            for dst in node.users:
                users_to_change.append(dst)

            for dst in users_to_change:
                dst.replace_input_with(node, node.args[0])

            module.graph.erase_node(node)
            module.delete_submodule(node.target)

        else:
            # You should not be here, really..
            assert False

        observer_id = observer_id + 1

    module.graph.lint()
    module.recompile()


# ======================================================================================
# Function to reconstruct Observers modules (from prepared_pt2e) with collected stats.
# ======================================================================================
def reconstruct_observers(module_prepared: torch.fx.GraphModule, module_active: torch.fx.GraphModule):

    nodes_to_reconstruct = []
    for node in module_prepared.graph.nodes:
        if node.op == "call_module":
            assert not node.kwargs
            submod = module_prepared.get_submodule(node.target)

            if (
                isinstance(submod, torch.ao.quantization.observer.MinMaxObserver)
                or isinstance(submod, torch.ao.quantization.observer.PerChannelMinMaxObserver)
                or isinstance(submod, torch.ao.quantization.observer.PlaceholderObserver)
            ):
                nodes_to_reconstruct.append(node)
            else:
                # Please add support for new observer if you are here.
                assert False

    observer_id = 0
    for node in nodes_to_reconstruct:
        submod = module_prepared.get_submodule(node.target)
        if isinstance(submod, torch.ao.quantization.observer.MinMaxObserver):
            min_attr_name = "_observer" + str(observer_id) + "_min_val"
            max_attr_name = "_observer" + str(observer_id) + "_max_val"

            submod.min_val = getattr(module_active, min_attr_name)
            submod.max_val = getattr(module_active, max_attr_name)

        elif isinstance(submod, torch.ao.quantization.observer.PerChannelMinMaxObserver):
            # Not really implemented, this is just sample on how/where to add new observers.
            assert False

        elif isinstance(submod, torch.ao.quantization.observer.PlaceholderObserver):
            pass

        else:
            # You should not be here, really..
            assert False

        observer_id = observer_id + 1

    module_prepared.graph.lint()
    module_prepared.recompile()


# ======================================================================================
# Freeze parameters for linear op, as is done in case of torch.export()
# ======================================================================================
def preprocess_linears(placeholder_map, module: torch.fx.GraphModule, tupled_args, *args):
    linear_module_partitions = get_source_partitions(module.graph, [torch.nn.Linear, torch.nn.functional.linear])

    if len(linear_module_partitions) == 0:
        return

    param_id = 0
    module_changed = False
    for module_or_fn_type, partitions in linear_module_partitions.items():
        if module_or_fn_type == torch.nn.Linear:
            for p in partitions:
                weight_node = None
                bias_node = None
                compute_node = None
                for node in p.nodes:
                    # Find addmm node and get first input. We cannot use partitions input list
                    # to get params as it is changing inputs order.
                    if node.op == "call_function" and node.target.__name__ == "addmm.default":
                        weight_node = node.args[0]
                        bias_node = node.args[2]
                        compute_node = node
                        break
                    else:
                        if node.op == "call_function" and node.target.__name__ == "mm.default":
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

    param_id = 0
    for module_or_fn_type, partitions in conv_module_partitions.items():
        if module_or_fn_type == torch.nn.Conv2d:
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

    module.graph.lint()
    module.recompile()


# ======================================================================================
# Preprocess maxpool op, as is done in case of torch.export()
# ======================================================================================
def preprocess_maxpools(placeholder_map, module: torch.fx.GraphModule, tupled_args):
    pool_module_partitions = get_source_partitions(module.graph, [torch.nn.MaxPool2d, torch.nn.functional.max_pool2d])

    if len(pool_module_partitions) == 0:
        return

    for module_or_fn_type, partitions in pool_module_partitions.items():
        if module_or_fn_type == torch.nn.MaxPool2d:
            for p in partitions:
                for node in p.nodes:
                    if node.op == "call_function" and node.target.__name__ == "max_pool2d_with_indices.default":
                        assert len(node.users) == 2

                        getitem_0 = list(node.users.keys())[0]
                        getitem_1 = list(node.users.keys())[1]

                        assert getitem_0.op == "call_function" and getitem_0.target.__name__ == "getitem"
                        assert getitem_1.op == "call_function" and getitem_1.target.__name__ == "getitem"

                        assert len(getitem_1.users) == 1

                        getitem_1_output_user = list(getitem_1.users.keys())[0]
                        assert getitem_1_output_user.op == "output"

                        getitem_1_output_user.replace_input_with(getitem_1, getitem_0)
                        break

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

    preprocess_linears(placeholder_map, module, tupled_args, *args)
    preprocess_convs(placeholder_map, module, tupled_args)
    preprocess_maxpools(placeholder_map, module, tupled_args)
