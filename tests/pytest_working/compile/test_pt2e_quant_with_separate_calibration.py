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
import torch.ao.quantization.quantize_pt2e as quantize_pt2e  # noqa F401
from habana_frameworks.torch.core.quantizer import (
    habana_quant_config_symmetric,
    habana_quantizer,
)
from habana_frameworks.torch.utils.debug.dynamo_utils import FxGraphAnalyzer
from test_pt2e_quant_flow import (
    SimpleModel,
    SimpleModelWithMultipleGraphs,
    custom_quant_config_symmetric,
    custom_quantizer,
    get_sample_input,
    get_sample_model,
    quant_float_dtype_list,
    quant_int_dtype_list,
    test_case_list,
    verify_nodes,
)
from test_utils import (
    inference_env_fixture,  # noqa F401
)
from torch.ao.quantization.observer import MinMaxObserver
from torch.ao.quantization.quantizer import QuantizationSpec
from torch.ao.quantization.quantizer.xnnpack_quantizer_utils import (
    QuantizationConfig,
)

test_mode = ["save", "load"]


def use_pt2e_quant_flow_with_separate_calibration(
    test_case, quant_dtype, quantizer, expected_op_count, use_graph_break, pass_input_during_export, save_or_load="save"
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

    with torch.no_grad():
        if save_or_load == "save":
            model.to(device=HPU)
            model.eval()

            if pass_input_during_export:
                model = torch.export.export_for_training(model, example_inputs0)
            else:
                model = torch.export.export_for_training(model)

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

                if pass_input_during_export:
                    model = torch.export.export(model, example_inputs0)
                else:
                    model = torch.export.export(model)

                torch.export.save(model, "./mymodel.pt2")

        elif save_or_load == "load":
            # Since PT2.6, torch.load (called in a torch.export.load function) has a 'weights_only' parameter set to True by default.
            # Therefore, to load the model with custom functions/classes, they must be added to the list of safe_globals beforehand.
            with torch.serialization.safe_globals(
                [
                    SimpleModelWithMultipleGraphs,
                    SimpleModel,
                    custom_quantizer,
                    QuantizationConfig,
                    QuantizationSpec,
                    MinMaxObserver,
                    torch.nn.Linear,
                    torch.nn.ReLU,
                ]
            ):
                loaded_model = torch.export.load("./mymodel.pt2")

            if isinstance(model, torch.export.exported_program.ExportedProgram):
                loaded_model = loaded_model.module()

            with FxGraphAnalyzer(reset_dynamo=False) as fga:
                # run inference with quantized model
                hpu_result2 = loaded_model(*example_inputs2)
                print(hpu_result2)

        else:
            pass

        if use_graph_break:
            verify_nodes(fga.get_ops_summary(), expected_op_count["after_convert_pt2e"])
            assert torch.allclose(cpu_result2[0].float(), hpu_result2[0].to(CPU).float(), rtol=1e-2, atol=1e-2)
        else:
            assert torch.allclose(cpu_result2[0].float(), hpu_result2[0].to(CPU).float(), rtol=2e-2, atol=2e-2)


@pytest.mark.skip("SW-203403 To Do Enable it once FP8 data type is added at torch.export serialization")
@pytest.mark.parametrize("save_or_load", test_mode)
@pytest.mark.parametrize("test_case", test_case_list)
@pytest.mark.parametrize("quant_dtype", quant_float_dtype_list)
@pytest.mark.parametrize("use_graph_break", [True])
@pytest.mark.parametrize("pass_input_during_export", [True, False])
def test_pt2e_quant_float(
    test_case,
    quant_dtype,
    use_graph_break,
    pass_input_during_export,
    save_or_load,
    inference_env_fixture,
):
    quantizer = habana_quantizer()
    quant_config = habana_quant_config_symmetric(quant_dtype)
    quantizer.set_global(quant_config)

    expected_op_count = {
        "after_prepare_pt2e": {
            "torch.ops.aten.relu.default": [(1, 0), (1, 0)],
            "torch.ops.aten.minimum.default": [(2, 0), (2, 0)],
            "torch.ops.aten.maximum.default": [(2, 0), (2, 0)],
            "skip_torch.ops.hpu.linear.default": [(1, 0), (1, 0)],
            "skip_torch.ops.aten.linear": [(1, 0), (1, 0)],
            "torch.ops.aten.transpose.int": [(1, 0), (1, 0)],
            "torch.ops.aten.mm.default": [(1, 0), (0, 0)],
            "torch.ops.aten.addmm.default": [(0, 0), (1, 0)],
        },
        "after_convert_pt2e": {
            "torch.ops.hpu.cast_to_fp8_v2.scalar": [(2, 0), (2, 0)],
            "torch.ops.hpu.fp8_gemm_v2.default": [(1, 0), (1, 0)],
            "torch.ops.aten.relu.default": [(1, 0), (1, 0)],
        },
    }

    with bc.env_setting("PT_HPU_PT2EQ_FX_GRAPH_PATTERN_MATCHING", True):
        use_pt2e_quant_flow_with_separate_calibration(
            test_case,
            quant_dtype,
            quantizer,
            expected_op_count,
            use_graph_break,
            pass_input_during_export,
            save_or_load,
        )


@pytest.mark.skip(reason="PT2E-Quantization with pattern matching supports fp8 dtype only")
@pytest.mark.parametrize("save_or_load", test_mode)
@pytest.mark.parametrize("test_case", test_case_list)
@pytest.mark.parametrize("quant_dtype", quant_int_dtype_list)
@pytest.mark.parametrize("use_graph_break", [True])
@pytest.mark.parametrize("pass_input_during_export", [True, False])
def test_pt2e_quant_int(
    test_case,
    quant_dtype,
    use_graph_break,
    pass_input_during_export,
    save_or_load,
    inference_env_fixture,
):
    with bc.env_setting("PT_HPU_PT2EQ_FX_GRAPH_FREEZING", False):
        quant_config = custom_quant_config_symmetric(quant_dtype)
        quantizer = custom_quantizer(quant_config)

        expected_op_count = {
            "after_prepare_pt2e": {
                "torch.ops.aten.relu.default": [(1, 0), (1, 0)],
                "torch.ops.aten.minimum.default": [(2, 0), (2, 0)],
                "torch.ops.aten.maximum.default": [(2, 0), (2, 0)],
                "skip_torch.ops.hpu.linear.default": [(1, 0), (1, 0)],
                "skip_torch.ops.aten.linear": [(1, 0), (1, 0)],
                "torch.ops.aten.transpose.int": [(1, 0), (1, 0)],
                "torch.ops.aten.mm.default": [(1, 0), (0, 0)],
                "torch.ops.aten.addmm.default": [(0, 0), (1, 0)],
            },
            "after_convert_pt2e": {
                "torch.ops.quantized_decomposed.quantize_per_tensor.default": [(2, 0), (2, 0)],
                "torch.ops.quantized_decomposed.dequantize_per_tensor.default": [(2, 0), (2, 0)],
                "skip_torch.ops.hpu.linear.default": [(1, 0), (1, 0)],
                "skip_torch.ops.aten.linear": [(1, 0), (1, 0)],
                "torch.ops.aten.transpose.int": [(1, 0), (1, 0)],
                "torch.ops.aten.mm.default": [(1, 0), (0, 0)],
                "torch.ops.aten.addmm.default": [(0, 0), (1, 0)],
                "torch.ops.aten.relu.default": [(1, 0), (1, 0)],
            },
        }

        with bc.env_setting("PT_HPU_PT2EQ_FX_GRAPH_PATTERN_MATCHING", False):
            use_pt2e_quant_flow_with_separate_calibration(
                test_case,
                quant_dtype,
                quantizer,
                expected_op_count,
                use_graph_break,
                pass_input_during_export,
                save_or_load,
            )


"""
if __name__ == "__main__":
    test_pt2e_quant_int(
        test_case="linear_relu",
        quant_dtype=torch.int8,
        use_graph_break=True,
        pass_input_during_export=True,
        save_or_load="save",
    )
    test_pt2e_quant_int(
        test_case="linear_relu",
        quant_dtype=torch.int8,
        use_graph_break=True,
        pass_input_during_export=True,
        save_or_load="load",
    )
"""
