import os

import torch

assert torch.__version__.startswith("2.0"), "Test suite only for PT2.0"

import habana_frameworks.torch.core as htcore
import numpy as np
import pytest

@pytest.mark.parametrize("shape_in, shape_out", [((2,3), (4,6)),
                                                 ((4,6), (2,3)),
                                                   ((2,3,4,5), (3,4,5,6))
                                                   ])
def test_resize_inplace(shape_in, shape_out):
    num_elements = np.multiply.reduce(shape_in)
    cpu_tensor = torch.Tensor(np.reshape(np.arange(num_elements, dtype=np.int32), shape_in)).type(torch.int32)
    hpu_tensor = cpu_tensor.to("hpu")
    result_cpu = cpu_tensor.resize_(shape_out).numpy().flatten()[:num_elements]
    result_hpu = hpu_tensor.resize_(shape_out).to("cpu").numpy().flatten()[:num_elements]

    assert np.array_equal(result_hpu, result_cpu)

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

    # another case of having empty tensor is with torch.empty([])
    # the difference to torch.zeros(0) is that in here out tensor
    # has shape [] but numel==1 - we need to support it as well
    cpu_out_empty = torch.empty([])
    hpu_out_empty = torch.empty([]).to("hpu")
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


def test_eager_std_mean():
    # test for EagerOp<std::tuple<Tensor, Tensor>>
    cpu_tensor = torch.Tensor(np.arange(-10.0, 10.0, 0.1))
    hpu_tensor = cpu_tensor.to("hpu")

    cpu_out = torch.std_mean(cpu_tensor)
    hpu_out0 = torch.std_mean(hpu_tensor)[0].to("cpu")
    hpu_out1 = torch.std_mean(hpu_tensor)[1].to("cpu")

    assert torch.allclose(hpu_out0, cpu_out[0], atol=0.1, rtol=0.1)
    assert torch.allclose(hpu_out1, cpu_out[1], atol=0.001, rtol=0.001)


def test_eager_frexp_out():
    # test for EagerOp<std::tuple<Tensor&, Tensor&>>
    cpu_tensor = torch.Tensor(np.arange(-10.0, 10.0, 0.1))
    hpu_tensor = cpu_tensor.to("hpu")

    cpu_outtensor = (torch.empty([200], dtype=torch.float32), torch.empty([200], dtype=torch.int32))
    hpu_outtensor = (torch.empty([200], dtype=torch.float32).to("hpu"), torch.empty([200], dtype=torch.int32).to("hpu"))

    torch.frexp(cpu_tensor, out=cpu_outtensor)
    torch.frexp(hpu_tensor, out=hpu_outtensor)

    assert torch.allclose(cpu_outtensor[0], hpu_outtensor[0].to("cpu"), atol=0.001, rtol=0.001)
    assert torch.equal(cpu_outtensor[1], hpu_outtensor[1].to("cpu"))


def test_eager_max_out():
    # test for EagerOp<std::tupel<Tensor&, Tensor&>>
    cpu_tensor = torch.Tensor(np.random.randint(-1, 1, (20, 20)))
    hpu_tensor = cpu_tensor.to("hpu")

    cpu_outtensor = (torch.empty([], dtype=torch.float32), torch.empty([], dtype=torch.int64))
    hpu_outtensor = (torch.empty([], dtype=torch.float32).to("hpu"), torch.empty([], dtype=torch.int64).to("hpu"))

    torch.max(cpu_tensor, 0, out=cpu_outtensor)
    torch.max(hpu_tensor, 0, out=hpu_outtensor)

    assert torch.allclose(cpu_outtensor[0], hpu_outtensor[0].to("cpu"), atol=0.001, rtol=0.001)
    assert torch.equal(cpu_outtensor[1], hpu_outtensor[1].to("cpu"))


def test_relu2d_contiguous():
    cpu_tensor = torch.Tensor(np.random.randint(-1, 1, (20, 20)))
    hpu_tensor = cpu_tensor.to("hpu")
    result_hpu = torch.relu(hpu_tensor).to("cpu")
    result_cpu = torch.relu(cpu_tensor)
    assert torch.allclose(result_hpu, result_cpu, atol=0.001, rtol=0.001)


def test_pow2d_contiguous():
    cpu_tensor = torch.Tensor(np.random.randint(-1, 1, (20, 20)))
    hpu_tensor = cpu_tensor.to("hpu")
    result_hpu = torch.pow(hpu_tensor, 2).to("cpu")
    result_cpu = torch.pow(cpu_tensor, 2)
    assert torch.allclose(result_hpu, result_cpu, atol=0.001, rtol=0.001)

def test_clamp_variants():
    min = None
    max = 0
    cpu_tensor = torch.Tensor(10 * np.random.random((20, 20)) - 5)
    hpu_tensor = cpu_tensor.to("hpu")
    result_hpu = torch.clamp(hpu_tensor, min, max).to("cpu")
    result_cpu = torch.clamp(cpu_tensor, min, max)
    assert torch.allclose(result_hpu, result_cpu, atol=0, rtol=0)

    min = 0
    max = None
    result_hpu = torch.clamp(hpu_tensor, min, max).to("cpu")
    result_cpu = torch.clamp(cpu_tensor, min, max)
    assert torch.allclose(result_hpu, result_cpu, atol=0, rtol=0)

    min = -2
    max = 2
    result_hpu = torch.clamp(hpu_tensor, min, max).to("cpu")
    result_cpu = torch.clamp(cpu_tensor, min, max)
    assert torch.allclose(result_hpu, result_cpu, atol=0, rtol=0)