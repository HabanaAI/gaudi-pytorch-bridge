import os
import sys
import torch

sys.path.insert(0, os.path.join(os.environ['BUILD_ROOT_LATEST']))
try:
  import hb_torch
except ImportError:
  assert False,"Could Not import hb_torch"

hpu = torch.device('hpu')
cpu = torch.device('cpu')

@torch.jit.script
def gelu_test(a, b):
  c = torch.mul(a, b)
  d = torch.mul(c, b)
  e = torch.nn.functional.gelu(d)
  return e

if __name__ == '__main__':
  import os
  from inspect import currentframe, getframeinfo
  fi = getframeinfo(currentframe())
  src = fi.filename
  base = os.path.splitext(src)[0]
  trace_file_name = base + '_trace.pt'
  hpu = torch.device("hpu")
  cpu = torch.device("cpu")

  u_cpu = torch.tensor([[ 5.,  5., -6.,  7. ]], dtype=torch.float32, requires_grad=True)
  v_cpu = torch.tensor([[-3., -3.,  4.,  4. ]], dtype=torch.float32, requires_grad=True)
  z_cpu = torch.tensor([[-5., -7.,  1., -4. ]], dtype=torch.float32, requires_grad=True)

  with torch.jit.optimized_execution(True):
    hb_torch.disable()
    torch._C._jit_override_can_fuse_on_cpu(False)
    torch._C._jit_set_profiling_executor(False)
    torch._C._jit_set_profiling_mode(False)

    model_trace = torch.jit.trace(gelu_test, (u_cpu, v_cpu))
    torch.jit.save(model_trace, trace_file_name)

    print(f"input1 shape\n{u_cpu.shape}, val\n{u_cpu}")
    print(f"input2 shape\n{v_cpu.shape}, val\n{v_cpu}")

    rx_by_cpu = gelu_test(u_cpu, v_cpu)

    print("--------------------")
    print(f"Result CPU:\n{rx_by_cpu}")
    print("--------------------")

    hb_torch.enable()
    torch._C._jit_set_profiling_mode(False)
    torch._C._jit_set_profiling_executor(False)

    # load the model
    model_hpu = torch.jit.load(trace_file_name, map_location=hpu)

    # iteration 1
    u_hpu = u_cpu.to(hpu)
    v_hpu = v_cpu.to(hpu)
    ru_hpu = model_hpu(u_hpu, v_hpu)

    ru_by_hpu = ru_hpu.to(cpu)

    print("--------------------")
    print(f"Result HPU:\n{ru_by_hpu}")
    print("--------------------")

    ru_hpu.backward(torch.ones_like(ru_hpu));

    print("Successful termination")
# ------------------------------------------------------------------------------
