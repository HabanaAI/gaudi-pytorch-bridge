import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from test_utils import reset_seed, compare_tensors
import hb_torch

@torch.jit.script
def conv_relu_func(in_t, ft_t):
    conv_out = F.conv2d(in_t, ft_t)
    return F.relu(conv_out)

hpu = torch.device("hpu")
cpu = torch.device("cpu")
in_t = torch.randn(1, 1, 9, 9)
ft_t = torch.randn(1, 1, 3, 3)

@pytest.mark.parametrize("in_t, ft_t", [(in_t, ft_t)])
def test_jit_conv_perm(in_t, ft_t):
    m = torch.jit.trace(conv_relu_func, (in_t, ft_t))
    print(m.graph_for(in_t, ft_t))
    print("Eager Mode..")
    print(m(in_t.to('hpu'), ft_t.to('hpu')).to('cpu'))

    with torch.jit.optimized_execution(True):
        print("--------------------")
        print ("CPU IR Graph optimized")
        hb_torch.disable()
        torch._C._jit_override_can_fuse_on_cpu(False)
        torch._C._jit_set_profiling_executor(False)
        torch._C._jit_set_profiling_mode(False)
        print(conv_relu_func.graph_for(in_t, ft_t))
        model_trace = torch.jit.trace(conv_relu_func, (in_t, ft_t))
        torch.jit.save(model_trace, "cpu_trace.pt")
        model = conv_relu_func(in_t, ft_t)
        cpu_result = model
        print("Result CPU: " + str(model))
        print("--------------------")

    hb_torch.enable()
    print("--------------------")
    print("Moving Tensors to HPU")
    torch._C._jit_set_profiling_mode(False)
    torch._C._jit_set_profiling_executor(False)
    hpu_in_t = in_t.to(hpu)
    hpu_ft_t = ft_t.to(hpu)
    print("--------------------")
    print ("HPU IR Graph optimized")
    model_trace_hpu = torch.jit.load("cpu_trace.pt", map_location=torch.device("hpu"))
    print(model_trace_hpu.graph_for(hpu_in_t, hpu_ft_t))
    out = model_trace_hpu(hpu_in_t, hpu_ft_t)
    hpu_result = out.to(cpu)
    compare_tensors(hpu_result, cpu_result, atol=0.001, rtol=1.e-3)
    print("Result HPU: " + str(hpu_result))
    print("--------------------")

if __name__ == '__main__':
    test_jit_conv_perm(*[in_t, ft_t])
