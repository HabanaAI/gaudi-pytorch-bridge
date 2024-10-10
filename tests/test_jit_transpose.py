import habana_frameworks.torch.core as htcore
import pytest
import torch
from test_utils import compare_tensors


@torch.jit.script
def transpose(x):
    return torch.t(x)


hpu = torch.device("hpu")
cpu = torch.device("cpu")
in_t = torch.randn(8, 10)


@pytest.mark.xfail(reason="AttributeError: module 'habana_frameworks.torch.core' has no attribute 'enable'")
def test_jit_transpose():
    with torch.jit.optimized_execution(True):
        htcore.disable()
        torch._C._jit_override_can_fuse_on_cpu(False)
        torch._C._jit_set_profiling_executor(False)
        torch._C._jit_set_profiling_mode(False)
        model_trace = torch.jit.trace(transpose, in_t)
        torch.jit.save(model_trace, "cpu_trace.pt")
        model = transpose(in_t)
        cpu_result = model

    htcore.enable()
    torch._C._jit_set_profiling_mode(False)
    torch._C._jit_set_profiling_executor(False)
    hpu_t = in_t.to(hpu)
    model_trace_hpu = torch.jit.load("cpu_trace.pt", map_location=torch.device("hpu"))
    print(model_trace_hpu.graph_for(hpu_t))
    out = model_trace_hpu(hpu_t)
    hpu_result = out.to(cpu)

    compare_tensors(hpu_result, cpu_result, atol=0.001, rtol=1.0e-3)
