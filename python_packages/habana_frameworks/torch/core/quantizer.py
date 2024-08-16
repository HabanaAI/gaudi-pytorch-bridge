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

"""
This module implements Habana quantizers that can be used in PT2E-Quantization.
"""

# Note - A significant part of this implementation is taken from quantze_pt2e toy example
# https://gist.github.com/leslie-fang-intel/b78ed682aa9b54d2608285c5a4897cfc#file-toy_example_quantization_2_0-py
# E.g. BackendQuantizer and get_symmetric_quantization_config
# However, they have been renamed and amended as per the present need.

import itertools
from typing import Any, Dict, List, Optional

import torch
from habana_frameworks.torch.core.observer import AbsMaxObserver, SimpleAbsMaxObserver
from habana_frameworks.torch.dynamo.compile_backend.logger import get_compile_backend_logger
from torch.ao.quantization.observer import PlaceholderObserver
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
extra_args_act: Dict[str, Any] = {"for_observer": {"eps": 2**-12, "backoff_margin": 2}}
extra_args_weight: Dict[str, Any] = {"for_observer": {"eps": 2**-12, "backoff_margin": 0}}


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
        # self._annotate_softmax(model, config)

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

        input_act_qspec = get_input_act_qspec(quantization_config)
        output_act_qspec = get_output_act_qspec(quantization_config)
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

                    _update_input_qspec_map(p, act_node, input_act_qspec)
                    _update_input_qspec_map(p, weight_node, weight_qspec)
                    if bias_node:
                        _update_input_qspec_map(p, bias_node, bias_qspec)
                    _update_output_qspec(output_node, output_act_qspec)

                    nodes_to_mark_annotated = list(p.nodes)
                    _mark_nodes_as_annotated(nodes_to_mark_annotated)

    def _annotate_matmul(self, gm: torch.fx.GraphModule, quantization_config: QuantizationConfig) -> None:
        matmul_partitions = get_source_partitions(gm.graph, [torch.matmul])

        if len(matmul_partitions) == 0:
            return

        input_act_qspec = get_input_act_qspec(quantization_config)
        output_act_qspec = get_output_act_qspec(quantization_config)
        for module_or_fn_type, partitions in matmul_partitions.items():
            for p in partitions:
                assert len(p.input_nodes) == 2
                act_node1 = p.input_nodes[0]
                act_node2 = p.input_nodes[1]
                assert len(p.output_nodes) == 1
                output_node = p.output_nodes[0]

                _update_input_qspec_map(p, act_node1, input_act_qspec)
                _update_input_qspec_map(p, act_node2, input_act_qspec)
                _update_output_qspec(output_node, output_act_qspec)

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

    def _annotate_softmax(self, gm: torch.fx.GraphModule, quantization_config: QuantizationConfig) -> None:
        softmax_partitions = get_source_partitions(gm.graph, [torch.softmax, torch.nn.functional.softmax])
        # breakpoint()

        if len(softmax_partitions) == 0:
            return

        output_act_qspec = get_input_act_qspec(quantization_config)
        input_act_qspec = get_input_act_qspec(quantization_config)
        for module_or_fn_type, partitions in softmax_partitions.items():
            for p in partitions:
                assert len(p.input_nodes) == 1
                act_node = p.input_nodes[0]
                assert len(p.output_nodes) == 1
                output_node = p.output_nodes[0]

                _update_input_qspec_map(p, act_node, input_act_qspec)
                _update_output_qspec(output_node, output_act_qspec)

                nodes_to_mark_annotated = list(p.nodes)
                _mark_nodes_as_annotated(nodes_to_mark_annotated)

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

    act_observer_or_fake_quant_ctr: _ObserverOrFakeQuantizeConstructor = AbsMaxObserver
    act_observer_or_fake_quant_args = extra_args_act.get("for_observer").copy()
    act_quantization_spec = QuantizationSpec(
        dtype=quant_dtype,
        quant_min=quant_min,
        quant_max=quant_max,
        qscheme=torch.per_tensor_symmetric,
        is_dynamic=False,
        observer_or_fake_quant_ctr=act_observer_or_fake_quant_ctr.with_args(**act_observer_or_fake_quant_args),
    )

    weight_observer_or_fake_quant_ctr: _ObserverOrFakeQuantizeConstructor = AbsMaxObserver
    weight_observer_or_fake_quant_args = extra_args_weight.get("for_observer").copy()
    weight_quantization_spec = QuantizationSpec(
        dtype=quant_dtype,
        quant_min=quant_min,
        quant_max=quant_max,
        qscheme=torch.per_tensor_symmetric,
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
        None,
        weight_quantization_spec,
        bias_quantization_spec,
    )

    return quantization_config
