import torch
import pytest
from test_utils import reset_seed, compare_tensors
import hb_torch


@torch.jit.script
def relu(tensor_a):
    tensor_b = torch.relu(tensor_a)
    tensor_r = torch.relu(tensor_b)
    return tensor_r


@pytest.mark.skip(reason="under development : might trigger unexpected breakage in CI")
def test_jit_relu_dbg():
    import os
    from inspect import currentframe, getframeinfo
    fi = getframeinfo(currentframe())
    src = fi.filename
    base = os.path.splitext(src)[0]
    trace_file_name = base + '_trace.pt'
    hpu = torch.device("habana")
    cpu = torch.device("cpu")

    u_cpu = torch.tensor([[ 5.,  5., -6.           ]], dtype=torch.float32)
    v_cpu = torch.tensor([[-3., -3.,  4.,  4., -5. ]], dtype=torch.float32)
    w_cpu = torch.tensor([[-4., -4.,  7.           ]], dtype=torch.float32)
    x_cpu = torch.tensor([[-3.,  2.                ]], dtype=torch.float32)
    y_cpu = torch.tensor([[-8., -3.                ]], dtype=torch.float32)
    z_cpu = torch.tensor([[-5., -7.,  1., -4.,  4. ]], dtype=torch.float32)

    with torch.jit.optimized_execution(True):
        hb_torch.disable()
        torch._C._jit_override_can_fuse_on_cpu(False)
        torch._C._jit_set_profiling_executor(False)
        torch._C._jit_set_profiling_mode(False)

        #print("--------------------")
        #print ("CPU IR Graph optimized")
        #print(relu.graph_for(x_cpu))
        #print("--------------------")

        model_trace = torch.jit.trace(relu, (x_cpu))
        torch.jit.save(model_trace, trace_file_name)
        rx_by_cpu = relu(x_cpu)
        print("--------------------")
        print(f"Input shape\n{x_cpu.shape}")
        print(f"Input\n{x_cpu}")
        print("--------------------")
        print(f"Result HPU:\n{rx_by_cpu}")
        print("--------------------")

    hb_torch.enable()
    torch._C._jit_set_profiling_mode(False)
    torch._C._jit_set_profiling_executor(False)

    ##import pdb; pdb.set_trace()
    # load the model
    #model_cpu = torch.jit.load(trace_file_name, map_location=cpu)
    #model_hpu = model_cpu.to(hpu)
    model_hpu = torch.jit.load(trace_file_name, map_location=hpu)

    # iteration 1
    u_hpu = u_cpu.to(hpu)
    ru_hpu = model_hpu(u_hpu)

    ru_by_hpu = ru_hpu.to(cpu)
    print("--------------------")
    print(f"Input\n{u_cpu}")
    print("--------------------")
    print(f"Result HPU:\n{ru_by_hpu}")
    print("--------------------")

    # iteration 2
    v_hpu = v_cpu.to(hpu)
    rv_hpu = model_hpu(v_hpu)

    rv_bv_hpu = rv_hpu.to(cpu)
    print("--------------------")
    print(f"Input\n{v_cpu}")
    print("--------------------")
    print(f"Result HPU:\n{rv_bv_hpu}")
    print("--------------------")

    #iteration 3
    w_hpu = w_cpu.to(hpu)
    rw_hpu = model_hpu(w_hpu)

    rw_by_hpu = rw_hpu.to(cpu)
    print("--------------------")
    print(f"Input\n{w_cpu}")
    print("--------------------")
    print(f"Result HPU:\n{rw_by_hpu}")
    print("--------------------")

    # iteration 4
    x_hpu = x_cpu.to(hpu)
    rx_hpu = model_hpu(x_hpu)

    rx_by_hpu = rx_hpu.to(cpu)
    print("--------------------")
    print(f"Input\n{x_cpu}")
    print("--------------------")
    print(f"Result HPU:\n{rx_by_hpu}")
    print("--------------------")

    # iteration 5
    y_hpu = y_cpu.to(hpu)
    ry_hpu = model_hpu(y_hpu)

    ry_by_hpu = ry_hpu.to(cpu)
    print("--------------------")
    print(f"Input\n{y_cpu}")
    print("--------------------")
    print(f"Result HPU:\n{ry_by_hpu}")
    print("--------------------")

    #iteration 6
    z_hpu = z_cpu.to(hpu)
    rz_hpu = model_hpu(z_hpu)

    rz_by_hpu = rz_hpu.to(cpu)
    print("--------------------")
    print(f"Input\n{z_cpu}")
    print("--------------------")
    print(f"Result HPU:\n{rz_by_hpu}")
    print("--------------------")

    print("Successful termination")
# ------------------------------------------------------------------------------

if __name__ == '__main__':
  test_jit_relu_dbg()
