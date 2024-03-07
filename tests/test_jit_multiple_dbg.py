import habana_frameworks.torch.core as htcore
import pytest
import torch
from test_utils import compare_tensors


@torch.jit.script
def multiple_funcs(tensor_a, tensor_b, tensor_c):
    tensor_y = torch.mul(tensor_a, tensor_b)
    tensor_z = torch.div(tensor_y, tensor_c)
    tensor_p = torch.sigmoid(tensor_z)
    tensor_q = torch.relu(tensor_p)
    return tensor_q


@pytest.mark.xfail(reason="AttributeError: module 'habana_frameworks.torch.core' has no attribute 'enable'")
def test_jit_multiple_dbg():
    trace_file_name = "test_jit_multiple_dbg_cpu_trace.pt"
    hpu = torch.device("hpu")
    cpu = torch.device("cpu")

    x_cpu = torch.tensor([[1.0, -2.0], [3.0, -4.0]], dtype=torch.float32)
    y_cpu = torch.tensor([[1.0, -1.0], [2.0, -2.0]], dtype=torch.float32)
    # z_cpu = #torch.tensor([[3.]], dtype=torch.float32) #This should also work
    z_cpu = torch.tensor([[3.0, 3.0], [3.0, 3.0]], dtype=torch.float32)
    """
    x_cpu = torch.randn(2, 3, 4, 4)
    y_cpu = torch.randn(2, 3, 4, 4)
    z_cpu = torch.randn(2, 3, 4, 4)
    """

    with torch.jit.optimized_execution(True):
        htcore.disable()
        print("--------------------")
        print("CPU IR Graph optimized")
        torch._C._jit_override_can_fuse_on_cpu(False)
        torch._C._jit_set_profiling_executor(False)
        torch._C._jit_set_profiling_mode(False)
        print(multiple_funcs.graph_for(x_cpu, y_cpu, z_cpu))
        model_trace = torch.jit.trace(multiple_funcs, (x_cpu, y_cpu, z_cpu))
        torch.jit.save(model_trace, trace_file_name)
        o_cpu = multiple_funcs(x_cpu, y_cpu, z_cpu)
        print("--------------------")
        print(f"Input\n{x_cpu, y_cpu, z_cpu}")
        print("--------------------")
        print(f"Result CPU\n{o_cpu}")
        print("--------------------")

    htcore.enable()
    print("--------------------")
    print("Moving Tensors to HPU")
    torch._C._jit_set_profiling_mode(False)
    torch._C._jit_set_profiling_executor(False)
    x_hpu = x_cpu.to(hpu)
    y_hpu = y_cpu.to(hpu)
    z_hpu = z_cpu.to(hpu)
    print("--------------------")
    print("HPU IR Graph optimized")
    model_hpu = torch.jit.load(trace_file_name, map_location=hpu)
    w_hpu = model_hpu(x_hpu, y_hpu, z_hpu)
    print(f"output.len = {len(w_hpu)}")
    result = w_hpu.to(cpu)
    print("--------------------")
    print(f"input:\n{x_cpu, y_cpu, z_cpu}")
    print("--------------------")
    print(f"Result HPU:\n{result}")
    print("--------------------")
    compare_tensors(result, o_cpu, atol=0.001, rtol=1.0e-3)
    print("--- Comparison Done ---")


if __name__ == "__main__":
    test_jit_multiple_dbg()
