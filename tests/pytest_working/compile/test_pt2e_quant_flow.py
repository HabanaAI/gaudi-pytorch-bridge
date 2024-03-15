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
    habana_quant_config_symmetric,
    habana_quantizer,
    prepare_pt2e,
)


class SimpleModelWithMultipleGraphs1(torch.nn.Module):
    def __init__(self):
        super(SimpleModelWithMultipleGraphs1, self).__init__()
        self.gemm1 = torch.nn.Linear(16, 8)
        self.relu1 = torch.nn.ReLU()
        self.gemm2 = torch.nn.Linear(8, 4)
        self.relu2 = torch.nn.ReLU()
        self.gemm3 = torch.nn.Linear(4, 2)
        self.relu3 = torch.nn.ReLU()

    def forward(self, x):
        out = self.gemm1(x)
        out = self.relu1(out)
        torch._dynamo.graph_break()
        out = self.gemm2(out)
        out = self.relu2(out)
        torch._dynamo.graph_break()
        out = self.gemm3(out)
        out = self.relu3(out)
        return out


def get_sample_model(test_case):
    if test_case == "linear_relu":
        return SimpleModelWithMultipleGraphs1()


def get_sample_input(test_case):
    CPU = torch.device("cpu")
    if test_case == "linear_relu":
        return torch.randn(2, 16, device=CPU)


quant_dtype_list = [
    torch.int8,
    # torch.float8_e4m3fn, [To do: SW-165190]
    # torch.float8_e5m2, [To do: SW-165190]
]


test_case_list = [
    "linear_relu",
]


@pytest.mark.parametrize("quant_dtype", quant_dtype_list)
@pytest.mark.parametrize("test_case", test_case_list)
def test_pt2e_quant_flow(quant_dtype, test_case):
    htcore.hpu_set_env()

    # Stabilizing testing.
    torch.manual_seed(0xBADC0FEE)
    random.seed(0xBADC0FEE)
    np.random.seed(0xBADC0FEE)
    torch.use_deterministic_algorithms(True)

    CPU = torch.device("cpu")
    inputs0 = get_sample_input(test_case)
    inputs1 = get_sample_input(test_case)
    inputs2 = get_sample_input(test_case)
    example_inputs0 = [
        inputs0,
    ]
    example_inputs1 = [
        inputs1,
    ]
    example_inputs2 = [
        inputs2,
    ]

    model = get_sample_model(test_case)
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
        calibrate_result = model(*example_inputs0)
        calibrate_result = model(*example_inputs1)
        model = convert_pt2e(model)

        hpu_result2 = model(*example_inputs2)
        print(hpu_result2)

    assert torch.allclose(cpu_result2[0].float(), hpu_result2[0].to(CPU).float(), rtol=1e-2, atol=1e-2)
    htcore.hpu_reset_env()
