import torch
import torch.nn as nn
from test_utils import reset_seed, compare_tensors
import hb_torch
import pytest

hpu = torch.device("habana")
cpu = torch.device("cpu")

data_list = [
    (torch.randn(8, 10))
]

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

    def forward(self, x):
        z = nn.functional.relu(x)
        z = z.to(torch.bfloat16)
        y = torch.t(z)
        return y


@pytest.mark.parametrize("in_t", data_list)
def test_jit_to(in_t):
  with torch.jit.optimized_execution(True):
    torch._C._jit_override_can_fuse_on_cpu(False)
    torch._C._jit_set_profiling_executor(False)
    torch._C._jit_set_profiling_mode(False)
    model = Net()
    model_trace = torch.jit.trace(model, in_t)
    torch.jit.save(model_trace, "cpu_trace.pt")
    cpu_result = model(in_t)

  try:
    hb_torch.enable()
    torch._C._jit_set_profiling_mode(False)
    torch._C._jit_set_profiling_executor(False)
    hpu_t = in_t.to(hpu)
    model_trace_hpu = torch.jit.load("cpu_trace.pt", map_location=torch.device("habana"))
    print(model_trace_hpu.graph_for(hpu_t))
    out = model_trace_hpu(hpu_t)
    hpu_result = out.to(cpu)
    compare_tensors(hpu_result, cpu_result, atol=0.001, rtol=1.e-3)
  except RuntimeError as err:
    print ("Exiting after printing Fused Graph post fusion pass")
    print("OS error: {0}".format(err))