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
import pytest
import torch
from test_utils import (
    check_ops_executed_in_jit_ir,
    compile_function_if_compile_mode,
    format_tc,
    is_gaudi1,
    is_pytest_mode_compile,
    print_tensors,
)

Verbose = False


def mode_to_mode(mode):
    return mode


def mode_to_int(mode):
    modes = ["bilinear", "nearest", "bicubic"]
    return modes.index(mode)


def pmode_to_int(pmode):
    pmodes = ["zeros", "border", "reflection"]
    return pmodes.index(pmode)


dtypes = [torch.float, torch.bfloat16]
if not is_gaudi1():
    dtypes.append(torch.float16)


@pytest.mark.parametrize("input_shape", [(2, 4, 4, 2), (1, 2, 3, 4, 5)], ids=format_tc)
@pytest.mark.parametrize("mode", ["bilinear", "nearest", "bicubic"], ids=format_tc)
@pytest.mark.parametrize("padding_mode", ["zeros", "border", "reflection"], ids=format_tc)
@pytest.mark.parametrize("align_corners", [False, True], ids=format_tc)
@pytest.mark.parametrize("dtype", dtypes, ids=format_tc)
@pytest.mark.parametrize("api", ["torch", "torch_2d_3d", "torch.nn.functional"], ids=format_tc)
@pytest.mark.parametrize("fwdbwd", ["fwd", "bwd"], ids=format_tc)
def test_hpu_grid_sampler(input_shape, mode, padding_mode, align_corners, dtype, api, fwdbwd):
    dim = len(input_shape)
    nd = dim - 2

    if fwdbwd == "bwd":
        if is_gaudi1():
            pytest.skip("grid_sampler_bwd is not supported on Gaudi1")
        if dim == 5:
            pytest.skip("SW-217785: grid_sampler_bwd is not yet supported for 5D inputs")
        if dtype not in [torch.float]:
            pytest.skip(f"SW-217785: grid_sampler_bwd is not yet supported for {dtype}")

        SW_217785_fails = set()
        SW_217785_fails.add(("nearest", "zeros", False))
        SW_217785_fails.add(("bicubic", "zeros", False))
        SW_217785_fails.add(("nearest", "border", False))
        SW_217785_fails.add(("bicubic", "border", False))
        SW_217785_fails.add(("nearest", "reflection", False))
        SW_217785_fails.add(("nearest", "zeros", True))
        SW_217785_fails.add(("bicubic", "zeros", True))
        SW_217785_fails.add(("nearest", "border", True))
        SW_217785_fails.add(("bicubic", "border", True))
        SW_217785_fails.add(("nearest", "reflection", True))
        SW_217785_fails.add(("bicubic", "reflection", True))

        whole_mode = (mode, padding_mode, align_corners)
        if whole_mode in SW_217785_fails:
            pytest.skip(f"SW_217785: {whole_mode} not yet supported for bwd")

    if dim == 5:
        if mode == "bicubic":
            pytest.skip("Bicubic interpolation only supports 4D input")

        if dtype == torch.bfloat16:
            SW_217586_fails = set()
            SW_217586_fails.add(("nearest", "zeros", True))
            SW_217586_fails.add(("nearest", "border", True))
            SW_217586_fails.add(("nearest", "reflection", True))

            whole_mode = (mode, padding_mode, align_corners)
            if whole_mode in SW_217586_fails:
                pytest.skip(f"SW-217586: {whole_mode} not yet supported for 5D bfloat16")

    SW_217586_fails = set()
    SW_217586_fails.add(("bilinear", "border", False))
    SW_217586_fails.add(("bilinear", "reflection", False))
    SW_217586_fails.add(("bicubic", "reflection", False))

    whole_mode = (mode, padding_mode, align_corners)
    if whole_mode in SW_217586_fails:
        pytest.skip(f"SW-217586: {whole_mode} not yet supported")

    fn_selector = {}
    fn_selector[("torch", 4)] = torch.grid_sampler
    fn_selector[("torch", 5)] = torch.grid_sampler
    fn_selector[("torch_2d_3d", 4)] = torch.grid_sampler_2d
    fn_selector[("torch_2d_3d", 5)] = torch.grid_sampler_3d
    fn_selector[("torch.nn.functional", 4)] = torch.nn.functional.grid_sample
    fn_selector[("torch.nn.functional", 5)] = torch.nn.functional.grid_sample

    conv_selector = {}
    conv_selector["torch"] = [mode_to_int, pmode_to_int]
    conv_selector["torch_2d_3d"] = [mode_to_int, pmode_to_int]
    conv_selector["torch.nn.functional"] = [mode_to_mode, mode_to_mode]

    tols = {}
    if fwdbwd == "fwd":
        tols[torch.float] = 1e-7
        tols[torch.bfloat16] = 0
        tols[torch.float16] = 0
    else:
        tols[torch.float] = 1e-6
        tols[torch.bfloat16] = 0
        tols[torch.float16] = 0

    def fn(input, grid):
        return fn_selector[(api, dim)](
            input, grid, conv_selector[api][0](mode), conv_selector[api][1](padding_mode), align_corners
        )

    grid_shape = list(input_shape)
    grid_shape.pop(1)
    grid_shape.append(nd)
    cpu_input = torch.rand(input_shape, dtype=dtype).requires_grad_()
    cpu_grid = torch.rand(grid_shape, dtype=dtype).requires_grad_()
    hpu_input = cpu_input.detach().to("hpu").requires_grad_()
    hpu_grid = cpu_grid.detach().to("hpu").requires_grad_()

    hpu_wrapped_fn = compile_function_if_compile_mode(fn)

    if Verbose:
        print_tensors(["cpu_input"], [cpu_input])
        print_tensors(["cpu_grid"], [cpu_grid])

    cpu_cast = False
    if dtype in [torch.bfloat16, torch.float16]:
        cpu_input = cpu_input.to(torch.float)
        cpu_grid = cpu_grid.to(torch.float)
        cpu_cast = True
    cpu_output = fn(cpu_input, cpu_grid)
    if cpu_cast:
        cpu_output = cpu_output.to(dtype)

    hpu_output = hpu_wrapped_fn(hpu_input, hpu_grid)
    hpu_output_cpu = hpu_output.cpu()

    if is_pytest_mode_compile():
        check_ops_executed_in_jit_ir("grid_sampler_2d" if nd == 2 else "grid_sampler_3d")

    if Verbose:
        print_tensors(["cpu_output", "hpu_output_cpu"], [cpu_output, hpu_output_cpu])

    tol = tols[dtype]
    if fwdbwd == "fwd":
        if Verbose:
            print_tensors(["cpu_output", "hpu_output_cpu"], [cpu_output, hpu_output_cpu], atol=tol, rtol=tol)

        assert torch.allclose(cpu_output, hpu_output_cpu, rtol=tol, atol=tol)
    else:
        grad_cpu = torch.ones_like(cpu_output) * 2
        grad_hpu = grad_cpu.detach().to("hpu")

        hpu_output.backward(grad_hpu)
        cpu_output.backward(grad_cpu)

        hpu_input_grad_cpu = hpu_input.grad.cpu()
        hpu_grid_grad_cpu = hpu_grid.grad.cpu()

        if Verbose:
            print_tensors(
                ["input_grad_cpu", "input_grad_hpu"], [cpu_input.grad, hpu_input_grad_cpu], atol=tol, rtol=tol
            )
            print_tensors(["grid_grad_cpu", "grid_grad_hpu"], [cpu_grid.grad, hpu_grid_grad_cpu], atol=tol, rtol=tol)

        assert torch.allclose(cpu_input.grad, hpu_input_grad_cpu, rtol=tol, atol=tol)
        assert torch.allclose(cpu_grid.grad, hpu_grid_grad_cpu, rtol=tol, atol=tol)
