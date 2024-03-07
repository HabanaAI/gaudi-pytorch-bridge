import habana_frameworks.torch.core as htcore
import pytest
import torch
import torch.nn.functional as F
from test_utils import compare_tensors
from torch.testing import FileCheck


@torch.jit.script
def log_softmax_func(x):
    return F.log_softmax(x, 1)


test_case_list = [(8, 10)]


@pytest.mark.xfail(reason="AttributeError: module 'habana_frameworks.torch.core' has no attribute 'enable'")
@pytest.mark.parametrize("D1, D2", test_case_list)
def test_log_softmax(D1, D2):
    hpu = torch.device("hpu")
    cpu = torch.device("cpu")
    in_t = torch.randn(8, 10)
    in_t.requires_grad_(True)
    hpu_t = in_t.to(hpu)
    hpu_t.retain_grad()

    # Verify eager mode
    cpu_eager_result = log_softmax_func(in_t)
    hpu_eager_result = log_softmax_func(hpu_t).to(cpu)
    compare_tensors(hpu_eager_result, cpu_eager_result, atol=0.001, rtol=1.0e-3)

    with torch.jit.optimized_execution(True):
        htcore.disable()
        torch._C._jit_override_can_fuse_on_cpu(False)
        torch._C._jit_set_profiling_executor(False)
        torch._C._jit_set_profiling_mode(False)
        model_trace = torch.jit.trace(log_softmax_func, in_t)
        torch.jit.save(model_trace, "cpu_trace.pt")
        cpu_result = model_trace(in_t)
        cpu_result.sum().backward()
        cpu_grad = in_t.grad.detach()
        print("CPU grad value")
        print(cpu_grad)
        in_t.grad.zero_()

        htcore.enable()
        torch._C._jit_set_profiling_mode(False)
        torch._C._jit_set_profiling_executor(False)
        model_trace_hpu = torch.jit.load("cpu_trace.pt", map_location=torch.device("hpu"))
        model_trace_hpu_graph = model_trace_hpu.graph_for(hpu_t)
        print("Fused graph on HPU: ")
        print(model_trace_hpu_graph)
        FileCheck().check_count("= prim::HabanaFusedOp_0", 2, exactly=True).run(str(model_trace_hpu_graph))
        hpu_out = model_trace_hpu(hpu_t)
        hpu_result = hpu_out.to(cpu)
        compare_tensors(hpu_result, cpu_result, atol=0.001, rtol=1.0e-3)
        hpu_out.sum().backward()
        print("HPU grad value")
        hpu_grad = hpu_t.grad.to(cpu)
        print(hpu_grad)
        compare_tensors(hpu_grad, cpu_grad, atol=0.001, rtol=1.0e-3)


if __name__ == "__main__":
    test_log_softmax(*test_case_list[0])
