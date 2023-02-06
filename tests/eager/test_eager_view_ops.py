import os

os.environ["PT_HPU_EAGER_OPS"] = "1"  # enable eager mode
import torch
import habana_frameworks.torch.core as htcore
import numpy as np
import pytest
torch.manual_seed(0)

def test_relu_contiguous_view():
    cpu_tensor = torch.Tensor(np.arange(-10.0, 10.0, 0.1)).view(-1)
    hpu_tensor = cpu_tensor.to("hpu").view(-1)

    result_hpu = torch.relu(hpu_tensor).to("cpu")
    result_cpu = torch.relu(cpu_tensor)

    assert torch.allclose(result_hpu, result_cpu, atol = 0.001, rtol = 0.001)

def test_relu_contiguous_slice():
    cpu_tensor = torch.randn([4])
    hpu_tensor = cpu_tensor.to("hpu")

    cpu_tensor_slice = cpu_tensor[2:]
    hpu_tensor_slice = hpu_tensor[2:]

    result_hpu = torch.relu(hpu_tensor_slice).to("cpu")
    result_cpu = torch.relu(cpu_tensor_slice)

    assert torch.allclose(result_hpu, result_cpu, atol = 0.001, rtol = 0.001)

def test_relu_contiguous_as_strided():
    cpu_tensor = torch.randn([2,3])
    hpu_tensor = cpu_tensor.to("hpu")

    cpu_tensor.as_strided_([5], [1], 1)
    hpu_tensor.as_strided_([5], [1], 1)

    result_hpu = torch.relu(hpu_tensor).to("cpu")
    result_cpu = torch.relu(cpu_tensor)

    assert torch.allclose(result_hpu, result_cpu, atol = 0.001, rtol = 0.001)

def test_relu_contiguous_multilevel_view():
    cpu_tensor = torch.randn([2,3])
    hpu_tensor = cpu_tensor.to("hpu")

    cpu_tensor = cpu_tensor[:].view(-1)
    hpu_tensor = hpu_tensor[:].view(-1)

    result_hpu = torch.relu(hpu_tensor).to("cpu")
    result_cpu = torch.relu(cpu_tensor)

    assert torch.allclose(result_hpu, result_cpu, atol = 0.001, rtol = 0.001)

def test_relu_inplace_view():
    cpu_tensor = torch.randn([4]).view(-1)
    hpu_tensor = cpu_tensor.to("hpu").view(-1)

    cpu_tensor = cpu_tensor[2::]
    hpu_tensor = hpu_tensor[2::]

    result_hpu = torch.relu_(hpu_tensor).to("cpu")
    result_cpu = torch.relu_(cpu_tensor)

    assert torch.allclose(result_hpu, result_cpu, atol = 0.001, rtol = 0.001)

# TODO Enable the below tests after JIT IR pass is implemented SW-119307
@pytest.mark.xfail(reason="SW-119307")
def test_relu_discontiguous_slice():
    cpu_tensor = torch.randn([4])
    hpu_tensor = cpu_tensor.to("hpu")

    cpu_tensor_slice = cpu_tensor[::2]
    hpu_tensor_slice = hpu_tensor[::2]

    result_hpu = torch.relu(hpu_tensor_slice).to("cpu")
    result_cpu = torch.relu(cpu_tensor_slice)
    assert torch.allclose(result_hpu, result_cpu, atol = 0.001, rtol = 0.001)

@pytest.mark.xfail(reason="SW-119307")
def test_relu_inplace_noncontiguous_view():
    cpu_tensor = torch.randn([4])
    hpu_tensor = cpu_tensor.to("hpu")

    cpu_tensor = cpu_tensor[::2]
    hpu_tensor = hpu_tensor[::2]

    result_hpu = torch.relu_(hpu_tensor).to("cpu")
    result_cpu = torch.relu_(cpu_tensor)

    assert torch.allclose(result_hpu, result_cpu, atol = 0.001, rtol = 0.001)