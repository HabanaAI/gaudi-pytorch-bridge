###############################################################################
#
#  Copyright (c) 2025 Intel Corporation
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

import pytest
import torch
from test_utils import (
    check_ops_executed_in_jit_ir,
    compile_function_if_compile_mode,
    format_tc,
    hpu,
    is_gaudi2,
    is_gaudi3,
    is_pytest_mode_compile,
)

param_dtype = [torch.float32, torch.int]
param_input_size = [5]
param_bins = [3, torch.tensor([0.0, 3.0]), torch.tensor([0.0, 4.0, 6.0])]
param_range = [None, (0, 3)]
param_density = [False, True]
params_use_weight = [False, True]

atol_for_dtype = {torch.int: 1, torch.float32: 1e-08}


@pytest.mark.parametrize("input_size", param_input_size, ids=format_tc)
@pytest.mark.parametrize("bins", param_bins, ids=format_tc)
@pytest.mark.parametrize("range", param_range, ids=format_tc)
@pytest.mark.parametrize("density", param_density, ids=format_tc)
@pytest.mark.parametrize("use_weight", params_use_weight, ids=format_tc)
@pytest.mark.parametrize("dtype", param_dtype, ids=format_tc)
@pytest.mark.skipif(not (is_gaudi2() or is_gaudi3()), reason="Histogram kernel is available for G2 and G3 only")
def test_histogram(input_size, bins, range, density, use_weight, dtype):
    if torch.is_tensor(bins) and range is not None:
        pytest.skip("Invalid combination of parameters")

    if torch.is_tensor(bins) and torch.numel(bins) > 2:
        pytest.xfail("TPC kernel does not support tensor input for bins. Simplified case with only 1 bin is possible.")

    if dtype == torch.int and density:
        pytest.xfail("Histogram density calculation is not supported for int dtype.")

    def fn(input, bins, range, weight, density):
        if range is None:
            return torch.histogram(input, bins=bins, weight=weight, density=density)
        else:
            return torch.histogram(input, bins=bins, range=range, weight=weight, density=density)

    fn_cpu = fn
    fn_hpu = compile_function_if_compile_mode(fn)
    input_cpu = None
    if dtype == torch.int:
        input_cpu = torch.randint(10, (input_size,))
    else:
        input_cpu = torch.randn((input_size), dtype=dtype)
    input_hpu = input_cpu.to(hpu)

    bins_cpu = bins
    bins_hpu = bins.to(hpu) if torch.is_tensor(bins) else bins

    weight_cpu = None
    if use_weight:
        if dtype == torch.int:
            weight_cpu = torch.randint(10, (input_size,))
        else:
            weight_cpu = torch.randn((input_size), dtype=dtype)

    weight_hpu = weight_cpu.to(hpu) if weight_cpu is not None else None

    if dtype == torch.int:
        # cpu doesn't support int dtype, so cast to float before calling reference
        input_cpu = input_cpu.to(dtype=torch.float32)
        weight_cpu = weight_cpu.to(dtype=torch.float32) if weight_cpu is not None else None

    res_cpu = fn_cpu(input_cpu, bins_cpu, range, weight_cpu, density)
    res_hpu = fn_hpu(input_hpu, bins_hpu, range, weight_hpu, density)

    hist_cpu, bin_edges_cpu = None, None
    if dtype == torch.int:
        hist_cpu, bin_edges_cpu = res_cpu.hist.to(torch.long), res_cpu.bin_edges.to(torch.long)
    else:
        hist_cpu, bin_edges_cpu = res_cpu.hist, res_cpu.bin_edges
    hist_hpu, bin_edges_hpu = res_hpu.hist.cpu(), res_hpu.bin_edges.cpu()

    assert torch.allclose(hist_cpu, hist_hpu)
    assert torch.allclose(bin_edges_cpu, bin_edges_hpu, atol=atol_for_dtype[dtype])

    if is_pytest_mode_compile():
        check_ops_executed_in_jit_ir("histogram")


@pytest.mark.parametrize("input_size", param_input_size, ids=format_tc)
@pytest.mark.parametrize("bins", param_bins, ids=format_tc)
@pytest.mark.parametrize("dtype", param_dtype, ids=format_tc)
@pytest.mark.skipif(not (is_gaudi2() or is_gaudi3()), reason="Histogram kernel is available for G2 and G3 only")
def test_histc(input_size, bins, dtype):
    if torch.is_tensor(bins):
        pytest.skip("Invalid combination of parameters")

    def fn(input, bins, min, max):
        return torch.histc(input, bins=bins, min=min, max=max)

    fn_cpu = fn
    fn_hpu = compile_function_if_compile_mode(fn)
    input_cpu = None
    if dtype == torch.int:
        input_cpu = torch.randint(10, (input_size,))
    else:
        input_cpu = torch.randn((input_size), dtype=dtype)
    input_hpu = input_cpu.to(hpu)

    min = -3
    max = 3

    if dtype == torch.int:
        # cpu doesn't support int dtype, so cast to float before calling reference
        input_cpu = input_cpu.to(dtype=torch.float32)

    res_cpu = fn_cpu(input_cpu, bins, min, max)
    if dtype == torch.int:
        res_cpu = res_cpu.to(torch.long)
    res_hpu = fn_hpu(input_hpu, bins, min, max)

    assert torch.allclose(res_cpu, res_hpu.cpu())

    if is_pytest_mode_compile():
        check_ops_executed_in_jit_ir("histc")
