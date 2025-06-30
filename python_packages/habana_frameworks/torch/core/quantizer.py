###############################################################################
#
#  Copyright (c) 2021-2025 Intel Corporation
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


"""
This module implements Habana quantizers that can be used in PT2E-Quantization.
"""

# Note - A significant part of this implementation is taken from quantze_pt2e toy example
# https://gist.github.com/leslie-fang-intel/b78ed682aa9b54d2608285c5a4897cfc#file-toy_example_quantization_2_0-py
# E.g. BackendQuantizer and get_symmetric_quantization_config
# However, they have been renamed and amended as per the present need.

import itertools
from typing import Any

import habana_frameworks.torch.internal.bridge_config as bc
from habana_frameworks.torch import hpu
from habana_frameworks.torch.core.observer import AbsMaxObserver, SimplePerChannelAbsMaxObserver
from habana_frameworks.torch.dynamo.debug_utils.logger import get_compile_backend_logger

import torch
from torch._ops import OpOverload as TorchOpOverload
from torch.ao.quantization.observer import PlaceholderObserver
from torch.ao.quantization.qconfig import _ObserverOrFakeQuantizeConstructor
from torch.ao.quantization.quantizer.x86_inductor_quantizer import (
    QuantizationAnnotation,
    QuantizationConfig,
    QuantizationSpec,
    Quantizer,
    SharedQuantizationSpec,
)
from torch.ao.quantization.quantizer.xnnpack_quantizer_utils import (
    OperatorConfig,
    _annotate_input_qspec_map,
    _annotate_output_qspec,
    get_bias_qspec,
    get_input_act_qspec,
    get_output_act_qspec,
    get_weight_qspec,
)
from torch.fx import Node
from torch.fx.passes.utils.source_matcher_utils import (
    SourcePartition,
    get_source_partitions,
)

logger = get_compile_backend_logger()


def is_fp8_e4m3_on_gaudi2():
    device_name = hpu.get_device_name()
    if device_name == "GAUDI2":
        try:
            fp8_e4m3_range_on_gaudi2 = torch.finfo(torch.float8_e4m3fnuz)
            return True
        except:
            pass
    return False


QUANTIZER_MIN_MAX = {
    torch.int8: (int(torch.iinfo(torch.int8).min), int(torch.iinfo(torch.int8).max)),
    torch.float8_e4m3fn: (int(torch.finfo(torch.float8_e4m3fnuz).min), int(torch.finfo(torch.float8_e4m3fnuz).max))
    if is_fp8_e4m3_on_gaudi2()
    else (int(torch.finfo(torch.float8_e4m3fn).min), int(torch.finfo(torch.float8_e4m3fn).max)),
    torch.float8_e5m2: (int(torch.finfo(torch.float8_e5m2).min), int(torch.finfo(torch.float8_e5m2).max)),
}
extra_args_act: dict[str, Any] = {"for_observer": {"eps": 2**-12, "backoff_margin": 2}}
extra_args_weight: dict[str, Any] = {"for_observer": {"eps": 2**-12, "backoff_margin": 1}}


# ======================================================================================
# Utility functions used by Habana Quantizer definition
# ======================================================================================
def _mark_nodes_as_annotated(nodes: list[Node]):
    for node in nodes:
        if node is not None:
            if "quantization_annotation" not in node.meta:
                node.meta["quantization_annotation"] = QuantizationAnnotation()
            node.meta["quantization_annotation"]._annotated = True


def _is_annotated(nodes: list[Node]):
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
        self.operator_type_config: dict[str, QuantizationConfig | None] = {}

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
        if bc.get_pt_hpu_pt2eq_kvcq():
            self._annotate_kvcache(model, config)

        self._annotate_conv2d(model, config)
        self._annotate_linear(model, config)
        self._annotate_matmul(model, config)
        self._annotate_maxpool2d(model, config)
        self._annotate_sdpa(model, config)

        return model

    def _annotate_kvcache(self, gm: torch.fx.GraphModule, quantization_config: QuantizationConfig) -> None:
        from .pattern_matcher import is_node

        for node in gm.graph.nodes:
            # For prefill / prompt stage
            # full --> copy
            if is_node(node, "full.default"):
                logger.debug(f"Found full.default node: {node.name}")
                assert len(node.users) == 1
                full_user_node = next(iter(node.users), None)

                if is_node(full_user_node, "copy.default"):
                    logger.debug(f"Found copy.default node: {node.name}")

                    input_qspec_map = {}
                    input_src = full_user_node.args[1]
                    assert isinstance(input_src, Node)
                    input_qspec_map[input_src] = get_input_act_qspec(quantization_config)

                    full_user_node.meta["quantization_annotation"] = QuantizationAnnotation(
                        input_qspec_map=input_qspec_map,
                        output_qspec=None,
                        _annotated=True,
                    )

            # For token generation stage
            # index_copy --> copy_
            if is_node(node, "index_copy.default"):
                logger.debug(f"Found index_copy.default node: {node.name}")
                input_qspec_map = {}
                input_3 = node.args[3]
                assert isinstance(input_3, Node)
                input_qspec_map[input_3] = get_input_act_qspec(quantization_config)

                node.meta["quantization_annotation"] = QuantizationAnnotation(
                    input_qspec_map=input_qspec_map,
                    output_qspec=get_input_act_qspec(quantization_config),
                    _annotated=True,
                )

    def _annotate_conv2d(self, gm: torch.fx.GraphModule, quantization_config: QuantizationConfig) -> None:
        conv_partitions = get_source_partitions(gm.graph, [torch.nn.Conv2d, torch.nn.functional.conv2d])

        if len(conv_partitions) == 0:
            return

        def is_conv_node(node):
            return node.op == "call_function" and (
                node.target == torch.ops.aten.convolution.default
                or node.target == torch.ops.aten.conv2d.default
                or (
                    isinstance(node.target, TorchOpOverload)
                    and (node.target._name == "aten::convolution" or node.target._name == "aten::conv2d")
                )
            )

        conv_partitions = list(itertools.chain(*conv_partitions.values()))

        for conv_partition in conv_partitions:
            if len(conv_partition.output_nodes) > 1:
                raise ValueError("conv partition has more than one output node")
            conv_node = conv_partition.output_nodes[0]
            if not is_conv_node(conv_node):
                raise ValueError(f"{conv_node} is not a convolution operator!")
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
                    assert output_node.op == "call_function"
                    weight_node = None
                    bias_node = None

                    if output_node.target in [
                        torch.ops.aten.view.default,
                        torch.ops.aten._unsafe_view.default,
                    ]:
                        output_node = output_node.args[0]

                    if output_node.target in [
                        torch.ops.aten.linear.default,
                    ]:
                        weight_node = output_node.args[1]
                        if len(output_node.args) > 2:
                            bias_node = output_node.args[2]

                    if output_node.target in [
                        torch.ops.aten.mm.default,
                        torch.ops.aten.addmm.default,
                    ]:
                        transpose_node = None
                        for node in p.nodes:
                            if node.op == "call_function" and node.target in [
                                torch.ops.aten.transpose.int,
                            ]:
                                transpose_node = node
                                break
                        assert transpose_node is not None
                        weight_node = transpose_node.args[0]
                        for node in p.params:
                            if node.op == "get_attr" and (node != weight_node):
                                bias_node = node
                                break

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
        for _, partitions in matmul_partitions.items():
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

        def is_max_pool_node(node):
            return node.op == "call_function" and (
                node.target == torch.ops.aten.max_pool2d_with_indices.default
                or node.target == torch.ops.aten.max_pool2d.default
                or (
                    isinstance(node.target, TorchOpOverload)
                    and (
                        node.target._name == "aten::max_pool2d_with_indices" or node.target._name == "aten::max_pool2d"
                    )
                )
            )

        maxpool_partitions = list(itertools.chain(*module_partitions.values()))

        for maxpool_partition in maxpool_partitions:
            output_node = maxpool_partition.output_nodes[0]
            maxpool_node = None
            for n in maxpool_partition.nodes:
                if is_max_pool_node(n):
                    maxpool_node = n
                    break
            if _is_annotated([output_node, maxpool_node]):  # type: ignore[list-item]
                continue

            input_act = maxpool_node.args[0]  # type: ignore[union-attr]
            assert isinstance(input_act, Node)

            act_qspec = get_input_act_qspec(quantization_config)
            maxpool_node.meta["quantization_annotation"] = QuantizationAnnotation(  # type: ignore[union-attr]
                input_qspec_map={
                    input_act: act_qspec,
                },
                output_qspec=SharedQuantizationSpec((input_act, maxpool_node)),
                _annotated=True,
            )

    def _annotate_softmax(self, gm: torch.fx.GraphModule, quantization_config: QuantizationConfig) -> None:
        softmax_partitions = get_source_partitions(gm.graph, [torch.softmax, torch.nn.functional.softmax])

        if len(softmax_partitions) == 0:
            return

        output_act_qspec = get_input_act_qspec(quantization_config)
        input_act_qspec = get_input_act_qspec(quantization_config)
        for _, partitions in softmax_partitions.items():
            for p in partitions:
                assert len(p.input_nodes) == 1
                act_node = p.input_nodes[0]
                assert len(p.output_nodes) == 1
                output_node = p.output_nodes[0]

                _update_input_qspec_map(p, act_node, input_act_qspec)
                _update_output_qspec(output_node, output_act_qspec)

                nodes_to_mark_annotated = list(p.nodes)
                _mark_nodes_as_annotated(nodes_to_mark_annotated)

    def _annotate_sdpa(self, gm: torch.fx.GraphModule, quantization_config: QuantizationConfig) -> None:
        module_partitions = get_source_partitions(
            gm.graph,
            [
                torch.ops.hpu.sdpa_recomp_fwd_non_dropout.default,
                torch.ops.hpu.sdpa_recomp_fwd,
            ],
        )

        if len(module_partitions) == 0:
            return

        def is_qkv(node):
            while node and node.op == "call_function":
                if (
                    node.target.__name__ == "index_copy.default"
                    or node.target.__name__ == "rotary_pos_embedding.default"
                    or node.target.__name__ == "mm.default"
                ):
                    return True
                node = node.args[0]
            return False

        input_act_qspec = get_input_act_qspec(quantization_config)
        output_act_qspec = get_output_act_qspec(quantization_config)
        for module_or_fn_type, partitions in module_partitions.items():
            if (
                module_or_fn_type == torch.ops.hpu.sdpa_recomp_fwd
                or module_or_fn_type == torch.ops.hpu.sdpa_recomp_fwd_non_dropout.default
            ):
                for p in partitions:
                    output_node = p.output_nodes[0]
                    if p.input_nodes[0] and is_qkv(p.input_nodes[0]):
                        _update_input_qspec_map(p, p.input_nodes[0], input_act_qspec)
                    if p.input_nodes[1] and is_qkv(p.input_nodes[1]):
                        _update_input_qspec_map(p, p.input_nodes[1], input_act_qspec)
                    if p.input_nodes[2] and is_qkv(p.input_nodes[2]):
                        _update_input_qspec_map(p, p.input_nodes[2], input_act_qspec)
                    if len(p.input_nodes) > 3 and p.input_nodes[3] and is_qkv(p.input_nodes[3]):
                        _update_input_qspec_map(p, p.input_nodes[3], input_act_qspec)
                    _update_output_qspec(output_node, output_act_qspec)

                    nodes_to_mark_annotated = list(p.nodes)
                    _mark_nodes_as_annotated(nodes_to_mark_annotated)

    def validate(self, model: torch.fx.GraphModule) -> None:
        """validate if the annotated graph is supported by the backend"""

    @classmethod
    def get_supported_operators(cls) -> list[OperatorConfig]:
        return []


# ======================================================================================
# Habana Quant Config definition
# ======================================================================================
def habana_quant_config_symmetric(quant_dtype, weight_qscheme="ptq"):
    logger.debug(
        f"habana_quant_config_symmetric: quantized data type is {quant_dtype}, weight quant scheme is {weight_qscheme}"
    )
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
    qscheme = torch.per_tensor_symmetric

    if weight_qscheme == "pcq" or bc.get_pt_hpu_pt2eq_use_weight_pcq():
        weight_observer_or_fake_quant_ctr: _ObserverOrFakeQuantizeConstructor = SimplePerChannelAbsMaxObserver
        qscheme = torch.per_channel_symmetric

    weight_observer_or_fake_quant_args = extra_args_weight.get("for_observer").copy()
    weight_quantization_spec = QuantizationSpec(
        dtype=quant_dtype,
        quant_min=quant_min,
        quant_max=quant_max,
        qscheme=qscheme,
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
