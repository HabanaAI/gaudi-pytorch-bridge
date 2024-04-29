###############################################################################
# Copyright (C) 2023 Habana Labs, Ltd. an Intel Company
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

import habana_frameworks.torch.core as htcore
import numpy as np
import pytest
import torch


@pytest.fixture
def set_env_variable():
    variable_name_experimental_flags = "ENABLE_EXPERIMENTAL_FLAGS"
    original_value_experimental_flags = os.environ.get(variable_name_experimental_flags)

    # Set the environment variable to the desired value
    os.environ[variable_name_experimental_flags] = "1"

    # Yield to provide the value for the test
    yield "1"

    # Teardown: Restore the original value after the test
    if original_value_experimental_flags is not None:
        os.environ[variable_name_experimental_flags] = original_value_experimental_flags
    else:
        del os.environ[variable_name_experimental_flags]


FP8_MAX_152 = torch.tensor(57344 * 0.9, dtype=torch.float)
FP8_MAX_143 = torch.tensor(240 * 0.9, dtype=torch.float)
FP8_MAX = {"152": FP8_MAX_152, "143": FP8_MAX_143}


def variant_from_dtype(dtype):
    return "152" if (dtype is None or dtype is torch.float8_e5m2) else "143"


from test_utils import hpu, is_gaudi2

pytestmark = pytest.mark.skipif(not is_gaudi2(), reason="Only Gaudi2 supports fp8")


def test_fp8_quant_model(set_env_variable):
    scaling_param = "act_maxabs_pts_weight_maxabs_pts_pow2_hw"
    quant_mod = "linear"
    fp8_dtype = torch.float8_e4m3fn
    dtype = torch.float
    out_tensor = False
    scale_as_tensor_list = True
    outside_parameter = False
    shapeA = (8, 4096)
    shapeB = (4096, 4096)

    htcore.hpu_set_inference_env()
    from habana_frameworks.torch.core.quantization import _check_params_as_const, _mark_params_as_const

    class TestModel(torch.nn.Module):
        def __init__(self, input_scale, input_scale_inv, other_scale, other_scale_inv):
            super(TestModel, self).__init__()
            if outside_parameter:
                print(f"TestModel::__init__ {type(input_scale)=} {type(input_scale_inv)=}")
                self.input_scale = input_scale if input_scale is not None else None
                self.other_scale = other_scale if other_scale is not None else None
                self.input_scale_inv = input_scale_inv
                self.other_scale_inv = other_scale_inv
            else:
                self.input_scale = torch.nn.Parameter(input_scale) if input_scale is not None else None
                self.other_scale = torch.nn.Parameter(other_scale) if other_scale is not None else None
                self.input_scale_inv = torch.nn.Parameter(input_scale_inv)
                self.other_scale_inv = torch.nn.Parameter(other_scale_inv)
            self.input_matmul_scale = self.input_scale
            if self.input_matmul_scale.dim() > 0:
                # # print(f"{self.input_scale.dim()=} changing rank {self.input_scale.shape=} {self.other_scale.shape=}")
                self.input_matmul_scale = None

        def forward(self, input, other):
            casted_input, _ = torch.ops.hpu.cast_to_fp8_v2(input, self.input_scale_inv, False, False, fp8_dtype)
            casted_other, _ = torch.ops.hpu.cast_to_fp8_v2(other, self.other_scale_inv, False, False, fp8_dtype)

            result = torch.ops.hpu.fp8_gemm_v2(
                casted_input,
                False,
                casted_other,
                False,
                None,
                torch.bfloat16,
                self.input_matmul_scale,
                self.other_scale,
                None,
                False,
            )

            return result

    torch.manual_seed(12345)

    hpu = torch.device("hpu")
    A = torch.rand(shapeA, dtype=dtype) * 10 + 30.0
    A_hpu = A.to(hpu)
    max_A = torch.max(torch.abs(A_hpu)).to(torch.float)

    B = torch.rand(shapeB, dtype=dtype) * 10 + 30.0
    B_hpu = B.to(hpu)
    max_B = torch.max(torch.abs(B_hpu)).to(torch.float)

    scaleA_hpu = None
    scaleB_hpu = None
    scaleAInv = None
    scaleBInv = None

    variant = variant_from_dtype(fp8_dtype)

    if scaling_param == "without_scale":
        scaleA_hpu = torch.tensor(1.0, dtype=dtype, device=hpu)
        scaleB_hpu = torch.tensor(1.0, dtype=dtype, device=hpu)
    else:
        print(f"{max_A.device=}")
        scaleA_hpu = (torch.tensor(FP8_MAX[variant]).to(hpu) / max_A).to(hpu)
        scaleB_hpu = (torch.tensor(FP8_MAX[variant]).to(hpu) / max_B).to(hpu)

    if scale_as_tensor_list:
        scaleA_hpu = torch.ones(A_hpu.shape).to(hpu) * scaleA_hpu
        scaleB_hpu = torch.ones([B_hpu.shape[-1]]).to(hpu) * scaleB_hpu.reshape([1])

    scaleAInv = torch.reciprocal(scaleA_hpu)
    scaleBInv = torch.reciprocal(scaleB_hpu)

    htcore.mark_step()
    if outside_parameter:
        print("Outside parameter")
        scaleA_hpu = torch.nn.Parameter(scaleA_hpu)
        scaleAInv = torch.nn.Parameter(scaleAInv)
        scaleB_hpu = torch.nn.Parameter(scaleB_hpu)
        scaleBInv = torch.nn.Parameter(scaleBInv)
    else:
        print("Not outside parameter")
    model = TestModel(scaleA_hpu, scaleAInv, scaleB_hpu, scaleBInv)

    _mark_params_as_const(model)
    _check_params_as_const(model)

    result_model = model.forward(A_hpu, B_hpu)
    result = result_model.cpu()
    result_model = model.forward(A_hpu, B_hpu)
    result = result_model.cpu()

    htcore.hpu_teardown_inference_env()
    # result_ref = torch.matmul(A, B).cpu()

    # print(f"{A=}\n{A_hpu=}")
    # print(f"{B=}\n{B_hpu=}")
    # print(f"{result=}\n{result_ref=}")
    # percentage_diff = torch.abs(
    #     (((result - result_ref) / result_ref) * 100).to(torch.int)
    # )
    # assert np.amax(percentage_diff.numpy()) <= 15
