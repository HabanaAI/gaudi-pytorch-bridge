###############################################################################
# Copyright (c) 2025 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
###############################################################################

from enum import Enum

import pytest
import torch
from test_utils import (
    check_ops_executed_in_jit_ir,
    compare_tensors,
    compile_function_if_compile_mode,
    format_tc,
    is_pytest_mode_compile,
)

dtypes = [torch.float32, torch.bfloat16, torch.half]


class MaxMode(Enum):
    NO_MAX = 0
    MAX_ABS_PTS = 1
    MAX_ABS_PCS = 2


class ScaleMode(Enum):
    NO_SCALE = 0
    SCALE_TO_POW2 = 1


def cpu_fn(input, maxMode, scaleMode, reduceAxis=0, reduceKeepdim=False, fullscale=1.0, backoff=1.0):
    v = input

    if maxMode != MaxMode["NO_MAX"]:
        v = torch.abs(v)
        if maxMode == MaxMode["MAX_ABS_PTS"]:
            v = torch.max(v)
        elif maxMode == MaxMode["MAX_ABS_PCS"]:
            v = torch.amax(v, dim=reduceAxis, keepdim=reduceKeepdim)
        v = v / (fullscale * backoff)

    if scaleMode == ScaleMode["SCALE_TO_POW2"]:
        v = 2.0 ** torch.ceil(torch.log2(v))

    return v


def hpu_fn(input, maxMode, scaleMode, **kwargs):
    return torch.ops.hpu.calculate_scale_for_cast(input, maxMode.value, scaleMode.value, **kwargs)


def run_test(shape, dtype, addAbs, *args, **kwargs):
    cpu_input = torch.randn(shape).to(dtype=dtype)

    if addAbs:
        cpu_input.abs_()

    hpu_input = cpu_input.to("hpu")
    cpu_output = cpu_fn(cpu_input, *args, **kwargs)

    fn = compile_function_if_compile_mode(hpu_fn)

    hpu_output = fn(hpu_input, *args, **kwargs)

    tol = 1e-8 if dtype == torch.float32 else 1e-3
    compare_tensors(hpu_output, cpu_output, rtol=tol, atol=tol)

    if is_pytest_mode_compile():
        check_ops_executed_in_jit_ir("calculate_scale_for_cast")


@pytest.mark.parametrize("shape", [(10, 7)], ids=format_tc)
@pytest.mark.parametrize("dtype", dtypes, ids=format_tc)
@pytest.mark.parametrize("maxMode", ["NO_MAX", "MAX_ABS_PTS"], ids=format_tc)
@pytest.mark.parametrize("scaleMode", ["NO_SCALE", "SCALE_TO_POW2"], ids=format_tc)
def test_hpu_calculate_scale_for_cast_max_no_pts(shape, dtype, maxMode, scaleMode):
    kwargs = {}
    if scaleMode != "NO_SCALE":
        kwargs.update({"fullscale": 2.0, "backoff": 5.0})
    run_test(shape, dtype, maxMode == "NO_MAX", MaxMode[maxMode], ScaleMode[scaleMode], **kwargs)


@pytest.mark.parametrize("shape", [(10, 7)], ids=format_tc)
@pytest.mark.parametrize("dtype", dtypes, ids=format_tc)
@pytest.mark.parametrize("maxMode", ["MAX_ABS_PCS"], ids=format_tc)
@pytest.mark.parametrize("reduceAxis", [0, 1], ids=format_tc)
@pytest.mark.parametrize("reduceKeepdim", [True, False], ids=format_tc)
@pytest.mark.parametrize("scaleMode", ["NO_SCALE", "SCALE_TO_POW2"], ids=format_tc)
def test_hpu_calculate_scale_for_cast_max_pcs(shape, dtype, maxMode, reduceAxis, reduceKeepdim, scaleMode):
    kwargs = {"reduceAxis": reduceAxis, "reduceKeepdim": reduceKeepdim}
    if scaleMode != "NO_SCALE":
        kwargs.update({"fullscale": 2.0, "backoff": 5.0})
    run_test(shape, dtype, maxMode == "NO_MAX", MaxMode[maxMode], ScaleMode[scaleMode], **kwargs)
