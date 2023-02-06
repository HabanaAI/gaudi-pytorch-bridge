import os

os.environ["PT_HPU_EAGER_OPS"] = "1"  # enable eager mode
import torch
import habana_frameworks.torch.core as htcore
import numpy as np
import pytest


def test_relu():
    cpu_tensor = torch.Tensor(np.arange(-10.0, 10.0, 0.1))
    hpu_tensor = cpu_tensor.to("hpu")

    result_hpu = torch.relu(hpu_tensor).to("cpu")
    result_cpu = torch.relu(cpu_tensor)

    assert torch.equal(result_hpu, result_cpu)


def test_relu_inplace():
    cpu_tensor = torch.Tensor(np.arange(-10.0, 10.0, 0.1))
    hpu_tensor = cpu_tensor.to("hpu")

    torch.relu_(hpu_tensor)
    torch.relu_(cpu_tensor)

    result_hpu = hpu_tensor.to("cpu")
    result_cpu = cpu_tensor

    assert torch.equal(result_hpu, result_cpu)


def test_sin_out():
    cpu_tensor = torch.Tensor(np.arange(-10.0, 10.0, 0.1))
    hpu_tensor = cpu_tensor.to("hpu")
    cpu_out = torch.zeros(cpu_tensor.shape)
    hpu_out = torch.zeros(hpu_tensor.shape).to("hpu")

    torch.sin(hpu_tensor, out=hpu_out)
    torch.sin(cpu_tensor, out=cpu_out)

    result_hpu = hpu_out.to("cpu")
    result_cpu = cpu_out

    assert torch.allclose(result_hpu, result_cpu, atol=0.001, rtol=0.001)


def test_pow_variants():
    cpu_tensor = torch.Tensor(np.arange(-10.0, 10.0, 0.5))
    hpu_tensor = cpu_tensor.to("hpu")
    cpu_out = torch.zeros(cpu_tensor.shape)
    hpu_out = torch.zeros(hpu_tensor.shape).to("hpu")

    torch.pow(hpu_tensor, 2.0, out=hpu_out)
    torch.pow(cpu_tensor, 2.0, out=cpu_out)
    result_hpu = hpu_out.to("cpu")
    result_cpu = cpu_out
    assert torch.allclose(result_hpu, result_cpu, atol=0.001, rtol=0.001)

    torch.pow(2.0, hpu_tensor, out=hpu_out)
    torch.pow(2.0, cpu_tensor, out=cpu_out)
    result_hpu = hpu_out.to("cpu")
    result_cpu = cpu_out
    assert torch.allclose(result_hpu, result_cpu, atol=0.001, rtol=0.001)

    cpu_tensor = torch.Tensor(np.arange(-10.0, 10.0, 1))
    hpu_tensor = cpu_tensor.to("hpu")
    cpu_out = torch.zeros(cpu_tensor.shape)
    hpu_out = torch.zeros(hpu_tensor.shape).to("hpu")
    hpu_tensor_exp = cpu_tensor.to("hpu")  # duplicate input is specific case tested separately
    torch.pow(hpu_tensor, hpu_tensor_exp, out=hpu_out)
    torch.pow(cpu_tensor, cpu_tensor, out=cpu_out)

    result_hpu = hpu_out.to("cpu")
    result_cpu = cpu_out
    assert torch.allclose(result_hpu, result_cpu, atol=0.001, rtol=0.001)


# in case of out op, only empty HPU tensor are resized
# non-empty different shapes or CPU out tensors are causing an exception
def test_out_empty_or_throw():
    cpu_tensor = torch.Tensor(np.arange(-10.0, 10.0, 0.5))
    hpu_tensor = cpu_tensor.to("hpu")
    hpu_out_wrong_shape = torch.zeros(4).to("hpu")
    cpu_out_empty = torch.zeros(0)
    hpu_out_empty = torch.zeros(0).to("hpu")

    with pytest.raises(RuntimeError):
        torch.pow(hpu_tensor, 2.0, out=hpu_out_wrong_shape)

    with pytest.raises(RuntimeError):
        torch.pow(hpu_tensor, 2.0, out=cpu_out_empty)

    torch.pow(hpu_tensor, 2.0, out=hpu_out_empty)
    torch.pow(cpu_tensor, 2.0, out=cpu_out_empty)
    result_hpu = hpu_out_empty.to("cpu")
    result_cpu = cpu_out_empty
    assert torch.allclose(result_hpu, result_cpu, atol=0.001, rtol=0.001)


# duplicate input is specific case handled by backend
# here we do duplicate input to pow_out op
def test_duplicate_input_pow():
    cpu_tensor = torch.Tensor(np.arange(-10.0, 10.0, 1))
    hpu_tensor = cpu_tensor.to("hpu")
    cpu_out = torch.zeros(cpu_tensor.shape)
    hpu_out = torch.zeros(hpu_tensor.shape).to("hpu")

    torch.pow(hpu_tensor, hpu_tensor, out=hpu_out)
    torch.pow(cpu_tensor, cpu_tensor, out=cpu_out)

    result_hpu = hpu_out.to("cpu")
    result_cpu = cpu_out

    assert torch.allclose(result_hpu, result_cpu, atol=0.001, rtol=0.001)

def test_eager_backend_pool():
    cpu_tensor = torch.Tensor(np.arange(-10.0, 10.0, 0.1))
    result_cpu = torch.relu(cpu_tensor)
    # Launch the op in a loop to check the backend pool
    for i in range(2000):
        hpu_tensor = cpu_tensor.to("hpu")
        result_hpu = torch.relu(hpu_tensor).to("cpu")
        assert torch.equal(result_hpu, result_cpu)

