import torch
import torch.nn as nn
import pytest
from test_utils import reset_seed, compare_tensors
import hb_torch

# We are running this test in JIT script mode instead of JIT trace mode
# because in JIT trace mode, 2nd argument is received as an 
# c10::optional::Tensorlist instead of TensorList and bridge code cannot 
# handle a c10::optional::Tensorlist. Bridge code will need a fix in 
# future if this OP is used in any model.
@torch.jit.script
def multiple_funcs(x1, x2, x3):
    x1.index_put_([x3], x2, True)
    return x1


def test_jit_index_put():
    #trace_file_name = "test_jit_index_put_cpu_trace.pt"
    hpu = torch.device("habana")
    cpu = torch.device("cpu")

    x_cpu = torch.randn(4, 2, 2, 3)
    y_cpu = torch.randn(2, 2, 2, 3)
    # Indices are of type Int because HPU cannot handle Long type tensors
    z_cpu = torch.tensor([0, 2], dtype=torch.int)

    # Clone original tensor for running in-place OP on CPU
    cpu_op = torch.clone(x_cpu)
    # Cast indices to Long because CPU cannot handle Int
    cpu_op.index_put_([z_cpu.to(torch.int64)], y_cpu, True)

    hb_torch.enable()
    # Run pass to convert in-place op to out-place version
    hb_torch.remove_inplace_ops()
    # "Moving Tensors to HPU"
    torch._C._jit_set_profiling_mode(False)
    torch._C._jit_set_profiling_executor(False)
    x_hpu = x_cpu.to(hpu)
    y_hpu = y_cpu.to(hpu)
    z_hpu = z_cpu.to(hpu)
    # "HPU IR Graph optimized"
    w_hpu = multiple_funcs(x_hpu, y_hpu, z_hpu)
    result = w_hpu.to(cpu)
    # print("--------------------")
    # print(f"input:\n{x_cpu, y_cpu, z_cpu}")
    # print("--------------------")
    # print(f"Result HPU:\n{result}")
    # print("--------------------")
    compare_tensors(result, cpu_op, atol=0.001, rtol=1.0e-3)
    # print("--- Comparison Done ---")

if __name__ == "__main__":
    test_jit_index_put()

