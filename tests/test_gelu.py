import os
from inspect import currentframe, getframeinfo

import pytest
import torch
from test_utils import cpu, hpu

try:
    import habana_frameworks.torch.core as htcore
except ImportError:
    raise AssertionError("Could Not import habana_frameworks.torch.core")


@torch.jit.script
def gelu_test(a, b):
    c = torch.mul(a, b)
    d = torch.mul(c, b)
    e = torch.nn.functional.gelu(d)
    return e


@pytest.mark.xfail(reason="AttributeError: module 'habana_frameworks.torch.core' has no attribute 'disable'")
def test_gelu():
    fi = getframeinfo(currentframe())
    src = fi.filename
    base = os.path.splitext(src)[0]
    trace_file_name = base + "_trace.pt"
    hpu = torch.device("hpu")
    cpu = torch.device("cpu")

    u_cpu = torch.tensor([[5.0, 5.0, -6.0, 7.0]], dtype=torch.float32, requires_grad=True)
    v_cpu = torch.tensor([[-3.0, -3.0, 4.0, 4.0]], dtype=torch.float32, requires_grad=True)

    with torch.jit.optimized_execution(True):
        htcore.disable()
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

        htcore.enable()
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

        ru_hpu.backward(torch.ones_like(ru_hpu))

        print("Successful termination")
