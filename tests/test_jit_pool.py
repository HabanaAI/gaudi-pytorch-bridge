import habana_frameworks.torch.core as htcore
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from test_utils import compare_tensors
from torch.testing import FileCheck

test_case_list = [
    # D1, D2, D3, D4
    (20, 16, 50, 32),
]


class MaxPool2dTestModule(nn.Module):
    def __init__(self):
        super(MaxPool2dTestModule, self).__init__()

    def forward(self, x):
        return F.max_pool2d(x, kernel_size=3, stride=2)


@pytest.mark.skip("Fails in docker tests")
@pytest.mark.parametrize("D1, D2, D3, D4", test_case_list)
def test_maxpool_2d(D1, D2, D3, D4):
    hpu = torch.device("hpu")
    cpu = torch.device("cpu")
    in_t = torch.randn(D1, D2, D3, D4)
    hpu_t = in_t.to(hpu)
    # in_t.requires_grad_(True)
    # hpu_t.requires_grad_(True)

    # Verify eager mode
    eager_model = MaxPool2dTestModule()
    cpu_eager_result = eager_model(in_t)
    hpu_eager_result = eager_model(hpu_t).to(cpu)
    compare_tensors(hpu_eager_result, cpu_eager_result, atol=0.001, rtol=1.0e-3)

    with torch.jit.optimized_execution(True):
        htcore.disable()
        torch._C._jit_override_can_fuse_on_cpu(False)
        torch._C._jit_set_profiling_executor(False)
        torch._C._jit_set_profiling_mode(False)
        model_trace = torch.jit.trace(MaxPool2dTestModule(), (in_t))
        cpu_result = model_trace(in_t)
        print("Result CPU: ")
        print(cpu_result)
        # cpu_result.sum().backward()

        htcore.enable()
        torch._C._jit_set_profiling_mode(False)
        torch._C._jit_set_profiling_executor(False)
        model_trace_hpu = torch.jit.trace(MaxPool2dTestModule(), (hpu_t), check_trace=False)
        model_trace_hpu_graph = model_trace_hpu.graph_for(hpu_t)
        print("Fused graph on HPU: ")
        print(model_trace_hpu_graph)
        FileCheck().check_count("= prim::HabanaFusedOp_0", 2, exactly=True).run(str(model_trace_hpu_graph))
        hpu_result = model_trace_hpu(hpu_t).to(cpu)
        # Backward test is disabled as permute() of byte type in forward is not supported
        # in graph mode yet
        # hpu_result.sum().to(hpu).backward()
        compare_tensors(hpu_result, cpu_result, atol=0.001, rtol=1.0e-3)


if __name__ == "__main__":
    test_maxpool_2d(*test_case_list[0])
