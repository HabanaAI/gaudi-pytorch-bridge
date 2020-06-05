import torch
import torch.nn as nn
import pytest
from test_utils import reset_seed, compare_tensors
import hb_torch

@torch.jit.script
def matmul_jit(mat1, mat2):
    result = torch.mm(mat1, mat2)
    result = torch.mm(mat1, result)
    return result

def test_hpu_linear():
    x_cpu = torch.tensor([[1., -2.], [3., -4.]], dtype=torch.float32)
    y_cpu = torch.tensor([[1., -1.], [2., -2.]], dtype=torch.float32)

    with torch.jit.optimized_execution(True):
        hpu = torch.device("habana")
        cpu = torch.device("cpu")

        torch._C._jit_override_can_fuse_on_cpu(False)
        torch._C._jit_set_profiling_executor(False)
        torch._C._jit_set_profiling_mode(False)
        out_cpu = matmul_jit(x_cpu, y_cpu)
        print(f"Input\n{x_cpu, y_cpu}")
        print(f"Result CPU\n{out_cpu}")

    try:
        hb_torch.enable()
        print("Moving Tensors to HPU")
        torch._C._jit_set_profiling_mode(False)
        torch._C._jit_set_profiling_executor(False)
        x_hpu = x_cpu.to(hpu)
        y_hpu = y_cpu.to(hpu)
        out_hpu=matmul_jit(x_hpu, y_hpu)
        print(f"output.len = {len(out_hpu)}")
        result = out_hpu.to(cpu)
        print(f"Result HPU:\n{result}")
        compare_tensors(result, out_cpu, atol=0.001, rtol=1.e-3)
    except(RuntimeError):
        print ("Exiting after printing Fused Graph post fusion pass")

if __name__ == '__main__':
    test_hpu_linear()
