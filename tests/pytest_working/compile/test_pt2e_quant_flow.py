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


import random

import habana_frameworks.torch.internal.bridge_config as bc
import numpy as np
import pytest
import torch
from habana_frameworks.torch.core.quantizer import (
    _mark_nodes_as_annotated,
    _update_input_qspec_map,
    _update_output_qspec,
    habana_quant_config_symmetric,
)
from habana_frameworks.torch.hpex.kernels import FusedSDPA
from habana_frameworks.torch.utils.debug.dynamo_utils import FxGraphAnalyzer
from test_utils import (
    fga_assert_helper,
    inference_env_fixture,  # noqa F401
    is_gaudi1,
)
from torch.ao.quantization.observer import MinMaxObserver
from torch.ao.quantization.qconfig import _ObserverOrFakeQuantizeConstructor
from torch.ao.quantization.quantizer import QuantizationSpec, Quantizer
from torch.ao.quantization.quantizer.xnnpack_quantizer_utils import (
    QuantizationConfig,
    get_input_act_qspec,
    get_weight_qspec,
)
from torch.fx.passes.utils.source_matcher_utils import get_source_partitions


class SimpleModel(torch.nn.Module):
    def __init__(self, dtype):
        super().__init__()
        self.gemm1 = torch.nn.Linear(4, 2, bias=False, dtype=dtype)
        self.relu1 = torch.nn.ReLU()

    def forward(self, x):
        out = self.gemm1(x)
        out = self.relu1(out)
        return out


class SimpleModelWithMultipleGraphs(torch.nn.Module):
    def __init__(self, dtype):
        super().__init__()
        self.gemm1 = torch.nn.Linear(4, 2, bias=False, dtype=dtype)
        self.relu1 = torch.nn.ReLU()
        self.gemm2 = torch.nn.Linear(2, 2, dtype=dtype)
        self.relu2 = torch.nn.ReLU()

    def forward(self, x):
        out = self.gemm1(x)
        out = self.relu1(out)
        torch._dynamo.graph_break()
        out = self.gemm2(out)
        out = self.relu2(out)
        return out


def get_sample_model(test_case, quant_dtype, graph_breaks=False):
    dtype = torch.float32 if quant_dtype == torch.int8 else torch.bfloat16
    if test_case == "linear_relu":
        return SimpleModelWithMultipleGraphs(dtype) if graph_breaks else SimpleModel(dtype)


def get_sample_input(test_case, quant_dtype):
    CPU = torch.device("cpu")
    dtype = torch.float32 if quant_dtype == torch.int8 else torch.bfloat16
    if test_case == "linear_relu":
        return torch.randn(2, 4, device=CPU, dtype=dtype)


test_case_list = [
    "linear_relu",
]
quant_int_dtype_list = [
    torch.int8,
]
quant_float_dtype_list = [
    torch.float8_e4m3fn,
]
quant_float_dtype_list_extended = quant_float_dtype_list + [
    torch.float8_e5m2,
]


def verify_nodes(ops_summary, expected_op_count):
    for op, count_list in expected_op_count.items():
        if not op.startswith("skip_"):
            fga_assert_helper(ops_summary=ops_summary, op=op, count_list=count_list)


class custom_quantizer(Quantizer):
    def __init__(self, quantization_config):
        super().__init__()
        self.global_config: QuantizationConfig = quantization_config

    def validate(self, model: torch.fx.GraphModule) -> None:
        pass

    def annotate(self, model: torch.fx.GraphModule) -> torch.fx.GraphModule:
        def _annotate_linear(gm: torch.fx.GraphModule, quantization_config: QuantizationConfig) -> None:
            module_partitions = get_source_partitions(gm.graph, [torch.nn.Linear, torch.nn.functional.linear])
            if len(module_partitions) == 0:
                return

            act_qspec = get_input_act_qspec(quantization_config)
            weight_qspec = get_weight_qspec(quantization_config)
            for module_or_fn_type, partitions in module_partitions.items():
                if module_or_fn_type == torch.nn.Linear or module_or_fn_type == torch.nn.functional.linear:
                    for p in partitions:
                        act_node = p.input_nodes[0]
                        linear_node = p.output_nodes[0]
                        assert linear_node.op == "call_function"
                        weight_node = None
                        if linear_node.target in [
                            torch.ops.aten.view.default,
                            torch.ops.aten._unsafe_view.default,
                        ]:
                            linear_node = linear_node.args[0]
                        if linear_node.target in [
                            torch.ops.aten.linear.default,
                        ]:
                            weight_node = linear_node.args[1]
                        if linear_node.target in [
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

                        if weight_node is None:
                            continue

                        _update_input_qspec_map(p, act_node, act_qspec)
                        _update_input_qspec_map(p, weight_node, weight_qspec)

                        nodes_to_mark_annotated = list(p.nodes)
                        _mark_nodes_as_annotated(nodes_to_mark_annotated)

        def _annotate_sdpa(gm: torch.fx.GraphModule, quantization_config: QuantizationConfig) -> None:
            module_partitions = get_source_partitions(
                gm.graph,
                [
                    torch.ops.hpu.sdpa_recomp_fwd_non_dropout.default,
                    torch.ops.hpu.sdpa_recomp_fwd,
                ],
            )

            if len(module_partitions) == 0:
                return

            input_act_qspec = get_input_act_qspec(quantization_config)
            output_act_qspec = get_input_act_qspec(quantization_config)
            for module_or_fn_type, partitions in module_partitions.items():
                if (
                    module_or_fn_type == torch.ops.hpu.sdpa_recomp_fwd
                    or module_or_fn_type == torch.ops.hpu.sdpa_recomp_fwd_non_dropout.default
                ):
                    for p in partitions:
                        output_node = p.output_nodes[0]
                        _update_input_qspec_map(p, p.input_nodes[0], input_act_qspec)
                        _update_input_qspec_map(p, p.input_nodes[1], input_act_qspec)
                        _update_input_qspec_map(p, p.input_nodes[2], input_act_qspec)
                        _update_output_qspec(output_node, output_act_qspec)

                        nodes_to_mark_annotated = list(p.nodes)
                        _mark_nodes_as_annotated(nodes_to_mark_annotated)

        _annotate_linear(model, self.global_config)
        _annotate_sdpa(model, self.global_config)
        return model


def custom_quant_config_symmetric(quant_dtype):
    quant_min = int(torch.iinfo(quant_dtype).min)
    quant_max = int(torch.iinfo(quant_dtype).max)

    act_observer_or_fake_quant_ctr: _ObserverOrFakeQuantizeConstructor = MinMaxObserver
    act_quantization_spec = QuantizationSpec(
        dtype=quant_dtype,
        quant_min=quant_min,
        quant_max=quant_max,
        qscheme=torch.per_tensor_symmetric,
        is_dynamic=False,
        observer_or_fake_quant_ctr=act_observer_or_fake_quant_ctr,
    )

    weight_observer_or_fake_quant_ctr: _ObserverOrFakeQuantizeConstructor = MinMaxObserver
    weight_quantization_spec = QuantizationSpec(
        dtype=quant_dtype,
        quant_min=quant_min,
        quant_max=quant_max,
        qscheme=torch.per_tensor_symmetric,
        ch_axis=0,
        is_dynamic=False,
        observer_or_fake_quant_ctr=weight_observer_or_fake_quant_ctr,
    )

    quantization_config = QuantizationConfig(
        act_quantization_spec,
        None,
        weight_quantization_spec,
        None,
    )

    return quantization_config


def use_pt2e_quant_flow(
    test_case,
    quant_dtype,
    quantizer,
    expected_op_count,
    use_graph_break,
    pass_input_during_export,
):
    # Stabilizing testing.
    torch.manual_seed(0xDEADDEAD)
    random.seed(0xDEADDEAD)
    np.random.seed(0xDEADDEAD)
    torch.use_deterministic_algorithms(True)

    CPU = torch.device("cpu")
    inputs0 = get_sample_input(test_case, quant_dtype)
    inputs1 = get_sample_input(test_case, quant_dtype)
    inputs2 = get_sample_input(test_case, quant_dtype)
    example_inputs0 = [
        inputs0,
    ]
    example_inputs1 = [
        inputs1,
    ]
    example_inputs2 = [
        inputs2,
    ]

    model = get_sample_model(test_case, quant_dtype, use_graph_break)
    model.eval()

    cpu_result2 = model(*example_inputs2)
    print(cpu_result2)

    HPU = torch.device("hpu")
    inputs0 = inputs0.to(HPU)
    inputs1 = inputs1.to(HPU)
    inputs2 = inputs2.to(HPU)
    example_inputs0 = (inputs0,)
    example_inputs1 = (inputs1,)
    example_inputs2 = (inputs2,)

    model.to(device=HPU)
    model.eval()

    with torch.no_grad():
        from torch.export import export_for_training

        if pass_input_during_export:
            model = export_for_training(model, example_inputs0)
        else:
            model = export_for_training(model)

        if isinstance(model, torch.export.exported_program.ExportedProgram):
            model = model.module()

        with FxGraphAnalyzer(reset_dynamo=False) as fga:
            from torch.ao.quantization.quantize_pt2e import prepare_pt2e

            model = prepare_pt2e(model, quantizer)

            # calibrate
            calibrate_result = model(*example_inputs0)
            calibrate_result = model(*example_inputs1)

        if use_graph_break:
            verify_nodes(fga.get_ops_summary(), expected_op_count["after_prepare_pt2e"])

        with FxGraphAnalyzer(reset_dynamo=False) as fga:
            from torch.ao.quantization.quantize_pt2e import convert_pt2e

            model = convert_pt2e(model, fold_quantize=False)

            # run inference with quantized model
            hpu_result2 = model(*example_inputs2)
            print(hpu_result2)

        if use_graph_break:
            expected_op_count_dict = (
                expected_op_count["after_convert_pt2e_with_fx_pm"]
                if bc.get_pt_hpu_pt2eq_fx_graph_pattern_matching()
                else expected_op_count["after_convert_pt2e"]
            )
            verify_nodes(fga.get_ops_summary(), expected_op_count_dict)
            assert torch.allclose(
                cpu_result2[0].float(),
                hpu_result2[0].to(CPU).float(),
                rtol=1e-2,
                atol=1e-2,
            )
        else:
            assert torch.allclose(
                cpu_result2[0].float(),
                hpu_result2[0].to(CPU).float(),
                rtol=2e-2,
                atol=2e-2,
            )


@pytest.mark.skipif(is_gaudi1(), reason="skip pt2e-quant feature testing on gaudi1")
@pytest.mark.parametrize("test_case", test_case_list)
@pytest.mark.parametrize("quant_dtype", quant_float_dtype_list_extended)
@pytest.mark.parametrize("use_graph_break", [False, True])
@pytest.mark.parametrize("pass_input_during_export", [False, True])
def test_pt2e_quant_float(
    test_case,
    quant_dtype,
    use_graph_break,
    pass_input_during_export,
    inference_env_fixture,
):
    with (
        bc.env_setting("PT_HPU_PT2EQ_FX_GRAPH_PATTERN_MATCHING", True),
        bc.env_setting("PT_HPU_PT2EQ_FX_GRAPH_FREEZING", False),
    ):
        quant_config = habana_quant_config_symmetric(quant_dtype)
        quantizer = custom_quantizer(quant_config)

        expected_op_count = {
            "after_prepare_pt2e": {
                "torch.ops.aten.relu.default": [(1, 0), (1, 0)],
                "torch.ops.aten.minimum.default": [(2, 0), (2, 0)],
                "torch.ops.aten.maximum.default": [(2, 0), (2, 0)],
                "torch.ops.aten.copy_.default": [(4, 0), (4, 0)],
                "skip_torch.ops.hpu.linear.default": [(1, 0), (1, 0)],
                "skip_torch.ops.aten.linear": [(1, 0), (1, 0)],
                "torch.ops.aten.transpose.int": [(1, 0), (1, 0)],
                "torch.ops.aten.mm.default": [(1, 0), (0, 0)],
                "torch.ops.aten.addmm.default": [(0, 0), (1, 0)],
            },
            "after_convert_pt2e_with_fx_pm": {
                "torch.ops.hpu.cast_to_fp8_v2.scalar": [(2, 0), (2, 0)],
                "torch.ops.hpu.fp8_gemm_v2.default": [(1, 0), (1, 0)],
                "torch.ops.aten.relu.default": [(1, 0), (1, 0)],
            },
            "after_convert_pt2e": {
                "torch.ops.quantized_decomposed.quantize_per_tensor.default": [
                    (2, 0),
                    (2, 0),
                ],
                "torch.ops.hpu.fp8_gemm_v2.default": [(1, 0), (1, 0)],
                "torch.ops.aten.relu.default": [(1, 0), (1, 0)],
            },
        }

        use_pt2e_quant_flow(
            test_case,
            quant_dtype,
            quantizer,
            expected_op_count,
            use_graph_break,
            pass_input_during_export,
        )


@pytest.mark.skipif(is_gaudi1(), reason="skip pt2e-quant feature testing on gaudi1")
@pytest.mark.parametrize("test_case", test_case_list)
@pytest.mark.parametrize("quant_dtype", quant_int_dtype_list)
@pytest.mark.parametrize("use_graph_break", [False, True])
@pytest.mark.parametrize("pass_input_during_export", [False, True])
def test_pt2e_quant_int(
    test_case,
    quant_dtype,
    use_graph_break,
    pass_input_during_export,
    inference_env_fixture,
):
    with (
        bc.env_setting("PT_HPU_PT2EQ_FX_GRAPH_PATTERN_MATCHING", False),
        bc.env_setting("PT_HPU_PT2EQ_FX_GRAPH_FREEZING", False),
    ):
        quant_config = custom_quant_config_symmetric(quant_dtype)
        quantizer = custom_quantizer(quant_config)

        expected_op_count = {
            "after_prepare_pt2e": {
                "torch.ops.aten.relu.default": [(1, 0), (1, 0)],
                "torch.ops.aten.minimum.default": [(2, 0), (2, 0)],
                "torch.ops.aten.maximum.default": [(2, 0), (2, 0)],
                "torch.ops.aten.copy_.default": [(4, 0), (4, 0)],
                "skip_torch.ops.hpu.linear.default": [(1, 0), (1, 0)],
                "skip_torch.ops.aten.linear": [(1, 0), (1, 0)],
                "torch.ops.aten.transpose.int": [(1, 0), (1, 0)],
                "torch.ops.aten.mm.default": [(1, 0), (0, 0)],
                "torch.ops.aten.addmm.default": [(0, 0), (1, 0)],
            },
            "after_convert_pt2e": {
                "torch.ops.quantized_decomposed.quantize_per_tensor.default": [
                    (2, 0),
                    (2, 0),
                ],
                "torch.ops.quantized_decomposed.dequantize_per_tensor.default": [
                    (2, 0),
                    (2, 0),
                ],
                "skip_torch.ops.hpu.linear.default": [(1, 0), (1, 0)],
                "skip_torch.ops.aten.linear": [(1, 0), (1, 0)],
                "torch.ops.aten.transpose.int": [(1, 0), (1, 0)],
                "torch.ops.aten.mm.default": [(1, 0), (0, 0)],
                "torch.ops.aten.addmm.default": [(0, 0), (1, 0)],
                "torch.ops.aten.relu.default": [(1, 0), (1, 0)],
            },
        }

        use_pt2e_quant_flow(
            test_case,
            quant_dtype,
            quantizer,
            expected_op_count,
            use_graph_break,
            pass_input_during_export,
        )


@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_fp8_fsdpa_with_pt2e(dtype, inference_env_fixture):
    # Stabilizing testing.
    torch.manual_seed(0xDEADDEAD)
    random.seed(0xDEADDEAD)
    np.random.seed(0xDEADDEAD)
    torch.use_deterministic_algorithms(True)

    class SimpleModel(torch.nn.Module):
        def __init__(self, dtype):
            super().__init__()
            self.relu = torch.nn.ReLU()

        def forward(self, query, key, value):
            out = FusedSDPA.apply(query, key, value, None, 0.0, False)
            out = self.relu(out)
            return out

    cpu_query0 = torch.randn(1, 1, 64, 64, dtype=torch.bfloat16).to("hpu")
    cpu_key0 = torch.randn(1, 1, 64, 64, dtype=torch.bfloat16).to("hpu")
    cpu_value0 = torch.randn(1, 1, 64, 64, dtype=torch.bfloat16).to("hpu")
    cpu_query1 = torch.randn(1, 1, 64, 64, dtype=torch.bfloat16).to("hpu")
    cpu_key1 = torch.randn(1, 1, 64, 64, dtype=torch.bfloat16).to("hpu")
    cpu_value1 = torch.randn(1, 1, 64, 64, dtype=torch.bfloat16).to("hpu")
    cpu_query2 = torch.randn(1, 1, 64, 64, dtype=torch.bfloat16).to("hpu")
    cpu_key2 = torch.randn(1, 1, 64, 64, dtype=torch.bfloat16).to("hpu")
    cpu_value2 = torch.randn(1, 1, 64, 64, dtype=torch.bfloat16).to("hpu")

    example_inputs_0 = [
        cpu_query0,
        cpu_key0,
        cpu_value0,
    ]
    example_inputs_1 = [
        cpu_query1,
        cpu_key1,
        cpu_value1,
    ]
    example_inputs_2 = [
        cpu_query2,
        cpu_key2,
        cpu_value2,
    ]

    HPU = torch.device("hpu")
    model = SimpleModel(dtype)
    model.to(device=HPU)
    model.eval()

    ref_result = model(*example_inputs_2)

    quant_dtype = torch.float8_e4m3fn
    quant_config = habana_quant_config_symmetric(quant_dtype)
    quantizer = custom_quantizer(quant_config)

    with torch.no_grad():
        from torch.export import export_for_training

        model = export_for_training(model)
        if isinstance(model, torch.export.exported_program.ExportedProgram):
            model = model.module()

        with FxGraphAnalyzer(reset_dynamo=False) as fga:
            from torch.ao.quantization.quantize_pt2e import prepare_pt2e

            model = prepare_pt2e(model, quantizer)
            # calibrate
            from compile.test_dynamo_utils import use_eager_fallback

            with use_eager_fallback():
                calibrate_result = model(*example_inputs_0)
                calibrate_result = model(*example_inputs_1)

        with FxGraphAnalyzer(reset_dynamo=False) as fga:
            from torch.ao.quantization.quantize_pt2e import convert_pt2e

            model = convert_pt2e(model)
            # run inference with quantized model
            from compile.test_dynamo_utils import use_eager_fallback

            with use_eager_fallback():
                hpu_result = model(*example_inputs_2)
            CPU = torch.device("cpu")
            print(
                "max diff:",
                torch.max(torch.abs(ref_result.to(CPU) - hpu_result.to(CPU))),
            )
            assert torch.allclose(
                ref_result.to(CPU).float(),
                hpu_result.to(CPU).float(),
                rtol=0.001,
                atol=0.4,
            )
