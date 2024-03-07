import habana_frameworks.torch.core as htcore
import pytest
import torch
import torch.nn as nn
from test_utils import compare_tensors
from torch.testing import FileCheck

hpu = torch.device("hpu")
cpu = torch.device("cpu")

data_list = [(torch.randn(1, 1, 2, 2))]


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

    def forward(self, x):
        z = torch.neg(x)
        return nn.functional.relu(z)


@pytest.mark.xfail(reason="AttributeError: module 'habana_frameworks.torch.core' has no attribute 'enable'")
@pytest.mark.parametrize("in_tensor", data_list)
def test_jit_neg(in_tensor):
    with torch.jit.optimized_execution(True):
        htcore.disable()
        torch._C._jit_override_can_fuse_on_cpu(False)
        torch._C._jit_set_profiling_executor(False)
        torch._C._jit_set_profiling_mode(False)
        model = Net()
        model_trace = torch.jit.trace(model, in_tensor)
        torch.jit.save(model_trace, "cpu_trace.pt")
        cpu_result = model(in_tensor)

    htcore.enable()
    torch._C._jit_set_profiling_mode(False)
    torch._C._jit_set_profiling_executor(False)
    hpu_t = in_tensor.to(hpu)
    model_trace_hpu = torch.jit.load("cpu_trace.pt", map_location=torch.device("hpu"))
    model_trace_hpu_graph = model_trace_hpu.graph_for(hpu_t)
    FileCheck().check_count("= prim::HabanaFusedOp_0", 2, exactly=True).run(str(model_trace_hpu_graph))
    out = model_trace_hpu(hpu_t)
    hpu_result = out.to(cpu)
    compare_tensors(hpu_result, cpu_result, atol=0.001, rtol=1.0e-3)
