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

import copy
import os
import random
import sys

# This is required for habana compile backends to register
import habana_frameworks.torch.core as htcore
import numpy as np
import pytest
import torch
from habana_frameworks.torch.core.quantize_pt2e import (
    convert_pt2e,
    export,
    get_weight_scale_history,
    habana_quant_config_symmetric,
    habana_quantizer,
    prepare_pt2e,
    set_activation_backoff_margin,
    set_weight_backoff_margin,
)
from habana_frameworks.torch.utils.debug.dynamo_utils import FxGraphAnalyzer
from test_dynamo_utils import assert_helper
from test_utils import is_gaudi1


class SimpleModel(torch.nn.Module):
    def __init__(self, dtype):
        super(SimpleModel, self).__init__()
        self.gemm1 = torch.nn.Linear(4, 2, bias=False, dtype=dtype)
        self.relu1 = torch.nn.ReLU()

    def forward(self, x):
        out = self.gemm1(x)
        out = self.relu1(out)
        return out


class SimpleModelWithMultipleGraphs(torch.nn.Module):
    def __init__(self, dtype):
        super(SimpleModelWithMultipleGraphs, self).__init__()
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
    torch.float8_e5m2,
]


def use_pt2e_quant_flow(test_case, quant_dtype):
    htcore.hpu_set_env()

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

    model = get_sample_model(test_case, quant_dtype, True)
    model.eval()

    cpu_result2 = model(*example_inputs2)
    print(cpu_result2)

    HPU = torch.device("hpu")
    inputs0 = inputs0.to(HPU)
    inputs1 = inputs1.to(HPU)
    inputs2 = inputs2.to(HPU)
    example_inputs0 = [
        inputs0,
    ]
    example_inputs1 = [
        inputs1,
    ]
    example_inputs2 = [
        inputs2,
    ]

    model.to(device=HPU)
    model.eval()

    with torch.no_grad():
        quantizer = habana_quantizer()
        quant_config = habana_quant_config_symmetric(quant_dtype)
        quantizer.set_global(quant_config)

        model, _ = export(model)
        model = prepare_pt2e(model, quantizer)
        with FxGraphAnalyzer(reset_dynamo=False) as fga:
            calibrate_result = model(*example_inputs0)
            calibrate_result = model(*example_inputs1)

        ops_summary = fga.get_ops_summary()
        assert_helper(ops_summary=ops_summary, op="torch.ops.aten.alias.default", count_list=[(9, 0), (9, 0)])
        assert_helper(ops_summary=ops_summary, op="torch.ops.aten.relu.default", count_list=[(1, 0), (1, 0)])
        assert_helper(ops_summary=ops_summary, op="torch.ops.aten.aminmax.default", count_list=[(3, 0), (3, 0)])
        assert_helper(ops_summary=ops_summary, op="operator.getitem", count_list=[(6, 7), (6, 7)])
        assert_helper(ops_summary=ops_summary, op="torch.ops.aten.minimum.default", count_list=[(3, 0), (3, 0)])
        assert_helper(ops_summary=ops_summary, op="torch.ops.aten.maximum.default", count_list=[(3, 0), (3, 0)])
        assert_helper(ops_summary=ops_summary, op="torch.ops.aten.copy.default", count_list=[(6, 0), (6, 0)])
        if "torch.ops.hpu.linear.default" in ops_summary:
            assert_helper(ops_summary=ops_summary, op="torch.ops.hpu.linear.default", count_list=[(1, 0), (1, 0)])
        elif "torch.ops.aten.linear" in ops_summary:
            assert_helper(ops_summary=ops_summary, op="torch.ops.aten.linear", count_list=[(1, 0), (1, 0)])
        else:
            assert_helper(ops_summary=ops_summary, op="torch.ops.aten.transpose.int", count_list=[(1, 0), (1, 0)])
            assert_helper(ops_summary=ops_summary, op="torch.ops.aten.mm.default", count_list=[(1, 0), (0, 0)])
            assert_helper(ops_summary=ops_summary, op="torch.ops.aten.addmm.default", count_list=[(0, 0), (1, 0)])

        model = convert_pt2e(model)
        with FxGraphAnalyzer(reset_dynamo=False) as fga:
            hpu_result2 = model(*example_inputs2)

        ops_summary = fga.get_ops_summary()
        print(hpu_result2)

    assert_helper(
        ops_summary=ops_summary,
        op="torch.ops.quantized_decomposed.quantize_per_tensor.default",
        count_list=[(3, 0), (3, 0)],
    )
    assert_helper(
        ops_summary=ops_summary,
        op="torch.ops.quantized_decomposed.dequantize_per_tensor.default",
        count_list=[(3, 0), (3, 0)],
    )
    if "torch.ops.hpu.linear.default" in ops_summary:
        assert_helper(ops_summary=ops_summary, op="torch.ops.hpu.linear.default", count_list=[(1, 0), (1, 0)])
    elif "torch.ops.aten.linear" in ops_summary:
        assert_helper(ops_summary=ops_summary, op="torch.ops.aten.linear", count_list=[(1, 0), (1, 0)])
    else:
        assert_helper(ops_summary=ops_summary, op="torch.ops.aten.transpose.int", count_list=[(1, 0), (1, 0)])
        assert_helper(ops_summary=ops_summary, op="torch.ops.aten.mm.default", count_list=[(1, 0), (0, 0)])
        assert_helper(ops_summary=ops_summary, op="torch.ops.aten.addmm.default", count_list=[(0, 0), (1, 0)])
    assert_helper(ops_summary=ops_summary, op="torch.ops.aten.relu.default", count_list=[(1, 0), (1, 0)])
    assert torch.allclose(cpu_result2[0].float(), hpu_result2[0].to(CPU).float(), rtol=1e-2, atol=1e-2)

    htcore.hpu_reset_env()


@pytest.mark.skipif(is_gaudi1(), reason="skip pt2e-quant feature testing on gaudi1")
@pytest.mark.parametrize("test_case", test_case_list)
@pytest.mark.parametrize("quant_dtype", quant_int_dtype_list)
def test_pt2e_quant_int(test_case, quant_dtype):
    use_pt2e_quant_flow(test_case, quant_dtype)


@pytest.mark.skipif(is_gaudi1(), reason="skip pt2e-quant feature testing on gaudi1")
@pytest.mark.parametrize("test_case", test_case_list)
@pytest.mark.parametrize("quant_dtype", quant_float_dtype_list)
def test_pt2e_quant_float(test_case, quant_dtype):
    use_pt2e_quant_flow(test_case, quant_dtype)


@pytest.mark.skipif(is_gaudi1(), reason="skip pt2e-quant feature testing on gaudi1")
def test_pt2e_quant_flow_with_backoff_margin(test_case="linear_relu", quant_dtype=torch.float8_e4m3fn):
    htcore.hpu_set_env()

    # Stabilizing testing.
    torch.manual_seed(0xDEAD0BAD)
    random.seed(0xDEAD0BAD)
    np.random.seed(0xDEAD0BAD)
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

    model_to_test = get_sample_model(test_case, quant_dtype)
    model_to_test.eval()

    HPU = torch.device("hpu")
    inputs0 = inputs0.to(HPU)
    inputs1 = inputs1.to(HPU)
    inputs2 = inputs2.to(HPU)
    example_inputs0 = [
        inputs0,
    ]
    example_inputs1 = [
        inputs1,
    ]
    example_inputs2 = [
        inputs2,
    ]

    model = copy.deepcopy(model_to_test)
    model.to(device=HPU)
    model.eval()

    torch._dynamo.reset()
    with torch.no_grad():
        quantizer = habana_quantizer()
        set_activation_backoff_margin(2)
        set_weight_backoff_margin(1)
        quant_config = habana_quant_config_symmetric(quant_dtype)
        quantizer.set_global(quant_config)

        model, _ = export(model)
        model = prepare_pt2e(model, quantizer)
        calibrate_result = model(*example_inputs0)
        calibrate_result = model(*example_inputs1)

        model = convert_pt2e(model)
        hpu_result2_1 = model(*example_inputs2)
        weight_scale_history_1 = get_weight_scale_history()

    model = copy.deepcopy(model_to_test)
    model.to(device=HPU)
    model.eval()

    torch._dynamo.reset()
    with torch.no_grad():
        quantizer = habana_quantizer()
        set_activation_backoff_margin(3)
        set_weight_backoff_margin(2)
        quant_config = habana_quant_config_symmetric(quant_dtype)
        quantizer.set_global(quant_config)

        model, _ = export(model)
        model = prepare_pt2e(model, quantizer)
        calibrate_result = model(*example_inputs0)
        calibrate_result = model(*example_inputs1)

        model = convert_pt2e(model)
        hpu_result2_2 = model(*example_inputs2)
        weight_scale_history_2 = get_weight_scale_history()

    assert weight_scale_history_1["convert_pt2e_scale"] == weight_scale_history_2["convert_pt2e_scale"]
    assert weight_scale_history_1["backed_off_scale"] != weight_scale_history_2["backed_off_scale"]
    assert weight_scale_history_1["final_hw_scale"] != weight_scale_history_2["final_hw_scale"]
    assert torch.allclose(hpu_result2_1[0].to(CPU).float(), hpu_result2_2[0].to(CPU).float(), rtol=1e-2, atol=1e-2)

    htcore.hpu_reset_env()
