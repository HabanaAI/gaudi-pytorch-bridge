import habana_frameworks.torch.core as htcore
import pytest
import torch
import torch.nn as nn
from test_utils import compare_tensors
from torch.testing import FileCheck

hpu = torch.device("hpu")
cpu = torch.device("cpu")

data_list = [
    (torch.randn(8, 10), torch.randn(4, 10), torch.randn(2, 10)),
    (torch.randn(2, 4, 4), torch.randn(4, 4, 4), torch.randn(3, 4, 4)),
]


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

    def forward(self, x1, x2, x3):
        z = torch.cat((x1, x2, x3), 0)
        y = nn.functional.relu(z)
        return y


@pytest.mark.xfail(reason="AttributeError: module 'habana_frameworks.torch.core' has no attribute 'enable'")
@pytest.mark.parametrize("in_tensors", data_list)
def test_jit_cat(in_tensors):
    with torch.jit.optimized_execution(True):
        torch._C._jit_override_can_fuse_on_cpu(False)
        torch._C._jit_set_profiling_executor(False)
        torch._C._jit_set_profiling_mode(False)
        model = Net()
        model_trace = torch.jit.trace(model, [in_tensors[0], in_tensors[1], in_tensors[2]], check_trace=False)
        torch.jit.save(model_trace, "cpu_trace.pt")
        cpu_result = model(in_tensors[0], in_tensors[1], in_tensors[2])

        htcore.enable()
        torch._C._jit_set_profiling_mode(False)
        torch._C._jit_set_profiling_executor(False)
        hpu_t1 = in_tensors[0].to(hpu)
        hpu_t2 = in_tensors[1].to(hpu)
        hpu_t3 = in_tensors[2].to(hpu)
        model_trace_hpu = torch.jit.load("cpu_trace.pt", map_location=torch.device("hpu"))
        FileCheck().check_count("= prim::HabanaFusedOp_0", 2, exactly=True).run(
            str(model_trace_hpu.graph_for(hpu_t1, hpu_t2, hpu_t3))
        )
        # print(model_trace_hpu.graph_for(hpu_t1, hpu_t2, hpu_t3))
        out = model_trace_hpu(hpu_t1, hpu_t2, hpu_t3)
        hpu_result = out.to(cpu)
        compare_tensors(hpu_result, cpu_result, atol=0.001, rtol=1.0e-3)
