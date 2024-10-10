###############################################################################
# Copyright (C) 2023-2024 Habana Labs, Ltd. an Intel Company
# All Rights Reserved.
#
# Unauthorized copying of this file or any element(s) within it, via any medium
# is strictly prohibited.
# This file contains Habana Labs, Ltd. proprietary and confidential information
# and is subject to the confidentiality and license agreements under which it
# was provided.
#
###############################################################################


import habana_frameworks.torch.utils.debug as htdebug
import numpy as np
import pytest
import torch
from test_utils import format_tc, is_gaudi1

Verbose = False


@pytest.mark.parametrize(
    "data1, data2",
    [
        ((2,), (1,)),
        ((2, 3), (2, 3)),
        ((6,), (6, 0)),
    ],
    ids=format_tc,
)
def test_equal(data1, data2):
    cpu_tensor1 = torch.Tensor(data1).type(torch.float32)
    hpu_tensor1 = cpu_tensor1.to("hpu")

    cpu_tensor2 = torch.Tensor(data2).type(torch.float32)
    hpu_tensor2 = cpu_tensor2.to("hpu")

    cpu_result = torch.equal(cpu_tensor1, cpu_tensor2)
    hpu_result = torch.equal(hpu_tensor1, hpu_tensor2)

    assert hpu_result == cpu_result


@pytest.mark.parametrize(
    "shape_in, shape_out", [((2, 3), (4, 6)), ((4, 6), (2, 3)), ((2, 3, 4, 5), (3, 4, 5, 6))], ids=format_tc
)
@pytest.mark.parametrize("blocking_flag", [True, False])
def test_resize_inplace(shape_in, shape_out, blocking_flag):
    num_elements = np.multiply.reduce(shape_in)
    cpu_tensor = torch.Tensor(np.reshape(np.arange(num_elements, dtype=np.int32), shape_in)).type(torch.int32)
    hpu_tensor = cpu_tensor.to("hpu", non_blocking=blocking_flag)
    result_cpu = cpu_tensor.resize_(shape_out).numpy().flatten()[:num_elements]
    result_hpu = hpu_tensor.resize_(shape_out).to("cpu").numpy().flatten()[:num_elements]

    assert np.array_equal(result_hpu, result_cpu)


def test_empty_resize():
    hpu_tensor = torch.empty([], device="hpu")
    hpu_tensor.resize_(10)
    cpu_tensor = hpu_tensor.to("cpu")
    assert np.equal(cpu_tensor.size()[0], 10)


def test_non_blocking_copy_inplace_op():
    cpu_tensor = torch.rand([100])
    hpu_tensor = cpu_tensor.to("hpu", non_blocking=True)

    cpu_tensor.add_(1)
    hpu_tensor.add_(1)

    assert torch.equal(hpu_tensor.to("cpu"), cpu_tensor)


def test_clone():
    cpu_tensor = torch.rand([2])
    hpu_tensor = cpu_tensor.to("hpu")

    result_cpu = cpu_tensor.clone()
    result_hpu = hpu_tensor.clone().to("cpu")

    assert torch.equal(result_hpu, result_cpu)


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
@pytest.mark.skip(reason="DID NOT RAISE <class 'RuntimeError'>")
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
    for _ in range(2000):
        hpu_tensor = cpu_tensor.to("hpu")
        result_hpu = torch.relu(hpu_tensor).to("cpu")
        assert torch.equal(result_hpu, result_cpu)


@pytest.mark.skip(reason="Results mismatch")
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

    cpu_outtensor = (
        torch.empty([200], dtype=torch.float32),
        torch.empty([200], dtype=torch.int32),
    )
    hpu_outtensor = (
        torch.empty([200], dtype=torch.float32).to("hpu"),
        torch.empty([200], dtype=torch.int32).to("hpu"),
    )

    torch.frexp(cpu_tensor, out=cpu_outtensor)
    torch.frexp(hpu_tensor, out=hpu_outtensor)

    assert torch.allclose(cpu_outtensor[0], hpu_outtensor[0].to("cpu"), atol=0.001, rtol=0.001)
    assert torch.equal(cpu_outtensor[1], hpu_outtensor[1].to("cpu"))


@pytest.mark.skip(reason="runtime error")
def test_eager_max_out():
    # test for EagerOp<std::tupel<Tensor&, Tensor&>>
    cpu_tensor = torch.Tensor(np.random.randint(-1, 1, (20, 20)))
    hpu_tensor = cpu_tensor.to("hpu")

    cpu_outtensor = (
        torch.empty([], dtype=torch.float32),
        torch.empty([], dtype=torch.int64),
    )
    hpu_outtensor = (
        torch.empty([], dtype=torch.float32).to("hpu"),
        torch.empty([], dtype=torch.int64).to("hpu"),
    )

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


def test_batch_norm():
    N = 2
    C = 2
    H = 2
    W = 2

    # 1. Compute BN Fwd + Bwd on CPU
    cpu_inputs = {
        "input": torch.randn(N, C, H, W, requires_grad=True),
        "running_mean": torch.randn(C),
        "running_var": torch.randn(C) + 1,
        "weight": torch.randn(C, requires_grad=True) + 1,
        "bias": torch.randn(C, requires_grad=True),
    }
    cpu_res = torch.nn.functional.batch_norm(**cpu_inputs)
    cpu_grad_outputs = torch.randn(N, C, H, W)
    (cpu_input_grad, cpu_weight_grad, cpu_bias_grad) = torch.autograd.grad(
        outputs=cpu_res,
        inputs=[v for k, v in cpu_inputs.items() if k in ["input", "bias", "weight"]],
        grad_outputs=cpu_grad_outputs,
    )

    # 2. Compute BN Fwd + Bwd on HPU
    hpu_inputs = {k: v.to("hpu") for k, v in cpu_inputs.items()}

    hpu_res = torch.nn.functional.batch_norm(**hpu_inputs)
    hpu_grad_outputs = cpu_grad_outputs.to("hpu")
    (hpu_input_grad, hpu_weight_grad, hpu_bias_grad) = torch.autograd.grad(
        outputs=hpu_res,
        inputs=[v for k, v in hpu_inputs.items() if k in ["input", "bias", "weight"]],
        grad_outputs=hpu_grad_outputs,
    )

    hpu_res = hpu_res.to("cpu")
    hpu_input_grad = hpu_input_grad.to("cpu")
    hpu_weight_grad = hpu_weight_grad.to("cpu")
    hpu_bias_grad = hpu_bias_grad.to("cpu")

    assert torch.allclose(hpu_res, cpu_res, atol=1e-3)
    assert torch.allclose(hpu_input_grad, cpu_input_grad, atol=1e-3)
    assert torch.allclose(hpu_weight_grad, cpu_weight_grad, atol=1e-3)
    assert torch.allclose(hpu_bias_grad, cpu_bias_grad, atol=1e-3)


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


def test_add():
    cpu_tensor = torch.randn(9, 9, dtype=torch.float32)
    hpu_tensor = cpu_tensor.to("hpu")

    result_hpu = torch.add(hpu_tensor, 2).to("cpu")
    result_cpu = torch.add(cpu_tensor, 2)

    assert torch.equal(result_hpu, result_cpu)


def test_addcdiv():
    cpu_tensor1 = torch.rand(4, 2)
    cpu_tensor2 = torch.rand(4, 2)
    cpu_tensor3 = torch.rand(1)
    hpu_tensor1 = cpu_tensor1.to("hpu")
    hpu_tensor2 = cpu_tensor2.to("hpu")
    hpu_tensor3 = cpu_tensor3.to("hpu")
    val = 1.5
    result_cpu = torch.addcdiv(cpu_tensor1, cpu_tensor2, cpu_tensor3, value=val)
    result_hpu = torch.addcdiv(hpu_tensor1, hpu_tensor2, hpu_tensor3, value=val).to("cpu")
    assert torch.allclose(result_hpu, result_cpu, atol=1e-6, rtol=1e-6)


def test_addcmul():
    cpu_tensor1 = torch.rand(4, 2)
    cpu_tensor2 = torch.rand(4, 2)
    cpu_tensor3 = torch.rand(1)
    hpu_tensor1 = cpu_tensor1.to("hpu")
    hpu_tensor2 = cpu_tensor2.to("hpu")
    hpu_tensor3 = cpu_tensor3.to("hpu")
    val = 1.5
    result_cpu = torch.addcmul(cpu_tensor1, cpu_tensor2, cpu_tensor3, value=val)
    result_hpu = torch.addcmul(hpu_tensor1, hpu_tensor2, hpu_tensor3, value=val).to("cpu")
    assert torch.allclose(result_hpu, result_cpu, atol=1e-6, rtol=1e-6)


def test_add_with_alpha():
    cpu_tensor = torch.randn(9, 9, dtype=torch.float32)
    hpu_tensor = cpu_tensor.to("hpu")

    result_hpu = hpu_tensor.add_(3, 2).to("cpu")
    result_cpu = cpu_tensor.add_(3, 2)

    assert torch.equal(result_hpu, result_cpu)


def test_div():
    cpu_tensor = torch.randn(9, 9, dtype=torch.float32)
    hpu_tensor = cpu_tensor.to("hpu")

    result_hpu = torch.div(hpu_tensor, 3).to("cpu")
    result_cpu = torch.div(cpu_tensor, 3)

    assert torch.allclose(result_hpu, result_cpu, atol=0.001, rtol=0.001)


def test_eq():
    cpu_tensor = torch.randint(0, 2, (10,), dtype=torch.int32)
    hpu_tensor = cpu_tensor.to("hpu")

    result_hpu = torch.eq(hpu_tensor, 1).to("cpu")
    result_cpu = torch.eq(cpu_tensor, 1)

    assert torch.equal(result_hpu, result_cpu)


def test_sub():
    cpu_tensor = torch.randn(2, 3)
    hpu_tensor = cpu_tensor.to("hpu")

    result_cpu = torch.randn(2, 3)
    result_hpu = result_cpu.to("hpu")

    torch.sub(1.0, cpu_tensor, alpha=2, out=result_cpu)
    torch.sub(1.0, hpu_tensor, alpha=2, out=result_hpu)

    result_hpu = result_hpu.to("cpu")

    assert torch.allclose(result_hpu, result_cpu, atol=0.001, rtol=0.001)


def test_wrapped_number_tensors():
    cpu_tensor = torch.randn(9, 9, dtype=torch.float32)
    hpu_tensor = cpu_tensor.to("hpu")

    result_hpu = torch.mul(hpu_tensor, 1.0)
    result_hpu = result_hpu.to("cpu")
    result_cpu = torch.mul(cpu_tensor, 1.0)

    assert torch.allclose(result_hpu, result_cpu, atol=0.001, rtol=0.001)

    result_hpu = torch.mul(hpu_tensor, 2.0).to("cpu")
    result_cpu = torch.mul(cpu_tensor, 2.0)

    assert torch.allclose(result_hpu, result_cpu, atol=0.001, rtol=0.001)

    result_hpu = torch.mul(hpu_tensor, 1.0).to("cpu")
    result_cpu = torch.mul(cpu_tensor, 1.0)

    assert torch.allclose(result_hpu, result_cpu, atol=0.001, rtol=0.001)


def test_where_variants():
    self = torch.randn(3, 5, 7, dtype=torch.float32)
    other = torch.randn(7, dtype=torch.float32)
    condition = torch.randn(1, 7) > 0

    where_cpu = torch.where(condition, self, other)
    where_hpu = torch.where(condition.to("hpu"), self.to("hpu"), other.to("hpu")).to("cpu")
    assert torch.equal(where_hpu, where_cpu)

    where_out_cpu = torch.zeros(self.shape)
    where_out_hpu = torch.zeros(self.shape).to("hpu")
    torch.where(condition, self, other, out=where_out_cpu)
    torch.where(condition.to("hpu"), self.to("hpu"), other.to("hpu"), out=where_out_hpu)
    assert torch.equal(where_out_hpu.to("cpu"), where_out_cpu)


def test_index_put_long():
    cpu_tensor = torch.arange(24).to(torch.float).view(3, 2, 4)
    hpu_tensor = cpu_tensor.to("hpu")
    hpu_tensor[
        torch.tensor([0, 2]).to("hpu"),
        torch.tensor([0, 1]).to("hpu"),
        torch.tensor([0, 1]).to("hpu"),
    ] = -100
    cpu_tensor[torch.tensor([0, 2]), torch.tensor([0, 1]), torch.tensor([0, 1])] = -100
    assert torch.equal(hpu_tensor.to("cpu"), cpu_tensor)


def test_index_put_bool():
    cpu_tensor = torch.arange(12).to(torch.float).view(3, 2, 2)
    hpu_tensor = cpu_tensor.to("hpu")

    cpu_tensor[
        torch.tensor(
            [
                [[True, False], [False, True]],
                [[True, False], [False, True]],
                [[True, False], [False, True]],
            ]
        )
    ] = -100.0
    hpu_tensor[
        torch.tensor(
            [
                [[True, False], [False, True]],
                [[True, False], [False, True]],
                [[True, False], [False, True]],
            ]
        )
    ] = -100.0
    assert torch.equal(hpu_tensor.to("cpu"), cpu_tensor)


def test_index_put_bool_different_rank():
    cpu_tensor = torch.arange(24).to(torch.float).view(3, 2, 2, 1, 2)
    hpu_tensor = cpu_tensor.to("hpu")

    cpu_tensor[
        torch.tensor(
            [
                [[True, False], [False, True]],
                [[True, False], [False, True]],
                [[True, False], [False, True]],
            ]
        )
    ] = -100.0
    hpu_tensor[
        torch.tensor(
            [
                [[True, False], [False, True]],
                [[True, False], [False, True]],
                [[True, False], [False, True]],
            ]
        )
    ] = -100.0
    assert torch.equal(hpu_tensor.to("cpu"), cpu_tensor)


def test_index_mixed():
    cpu_tensor = torch.arange(144).to(torch.float).view(4, 4, 3, 3)
    hpu_tensor = cpu_tensor.to("hpu")
    bmask = torch.tensor(
        [
            [False, True, False],
            [False, False, True],
            [True, True, False],
            [False, True, True],
        ]
    )
    ind_t = torch.tensor([0, 1, 2, 0, 1, 2]).to(torch.int64)
    res_cpu = cpu_tensor[:, bmask, ind_t]
    res_hpu = hpu_tensor[:, bmask.to("hpu"), ind_t.to("hpu")]
    assert torch.equal(res_hpu.to("cpu"), res_cpu)


def test_index_single_elem_index():
    cpu_tensor = torch.arange(48).to(torch.float).view(2, 4, 3, 2)
    hpu_tensor = cpu_tensor.to("hpu")
    ind_t = torch.tensor([-2]).to(torch.int64)
    res_cpu = cpu_tensor[..., ind_t, :]
    res_hpu = hpu_tensor[..., ind_t.to("hpu"), :]
    assert torch.equal(res_hpu.to("cpu"), res_cpu)


@pytest.mark.parametrize("shape_in", [(), (2,), (4, 4), (2, 3, 4, 4, 4)])
@pytest.mark.parametrize("zero_input", [True, False])
def test_nonzero(shape_in, zero_input):
    if zero_input:
        self = torch.zeros(shape_in)
    else:
        self = torch.randint(10, shape_in) > 5
    nonzero_cpu = torch.nonzero(self)
    nonzero_hpu = torch.nonzero(self.to("hpu")).to("cpu")
    assert torch.equal(nonzero_hpu, nonzero_cpu)


@pytest.mark.parametrize(
    "tensor_in",
    [
        torch.tensor(
            [[11, 33, 22], [44, 55, 66], [77, 99, 99], [77, 99, 99]],
            dtype=torch.int32,
        ),
        torch.tensor([[11, 33, 11]], dtype=torch.int32),
        torch.tensor(
            [
                [11.0, 33.0, 22.0],
                [44.0, 55.0, 66.0],
                [44.0, 55.0, 66.0],
                [44.0, 55.0, 66.0],
            ],
            dtype=torch.float32,
        ),
        torch.tensor([[11.0, 33.0, 11.0]], dtype=torch.float32),
        torch.empty((0, 4), dtype=torch.float32),
        torch.empty((0, 4), dtype=torch.int32),
        torch.randn([2, 4, 5, 7], dtype=torch.float32),
        torch.randint(-1000, 1000, (2, 4, 5, 7), dtype=torch.int32),
        torch.tensor(
            [[11.0, 33.0, 12.0], [44.0, 55.0, 66.0], [77.0, 99.0, 99.0]],
            dtype=torch.float32,
        ),
        torch.tensor([[44.0, 55.0, 66.0], [77.0, 99.0, 99.0]], dtype=torch.float32),
    ],
)
@pytest.mark.parametrize("return_inverse", [True, False])
@pytest.mark.parametrize("return_sorted", [True, False])
@pytest.mark.parametrize("return_counts", [True, False])
def test_unique2(tensor_in, return_inverse, return_sorted, return_counts):
    self = tensor_in
    unique_cpu = torch._unique2(
        self,
        return_inverse=return_inverse,
        sorted=return_sorted,
        return_counts=return_counts,
    )
    t1, t2, t3 = unique_cpu
    feature_map, inverse, counts = torch._unique2(
        self.to("hpu"),
        return_inverse=return_inverse,
        sorted=return_sorted,
        return_counts=return_counts,
    )
    feature_map = feature_map.to("cpu")
    if return_inverse:
        inverse = inverse.to("cpu")
    if return_counts:
        counts = counts.to("cpu")
    # # NOTE - unique is nondeterministic when returning an unsorted result,
    # # the tensors are sorted for further comparison to succeed
    if not return_sorted:
        t1 = t1.sort()[0]
        feature_map = feature_map.sort()[0]
    assert torch.allclose(feature_map, t1)
    if return_inverse:
        #     # NOTE - unique is nondeterministic when returning an unsorted result,
        #     # the inverse tensor will not be valid in such case, hence the disabled assertion
        if return_sorted:
            assert torch.equal(inverse, t2)
    else:
        assert isinstance(inverse, type(None))
    if return_counts:
        #     # NOTE - unique is nondeterministic when returning an unsorted result,
        #     # the inverse tensor will not be valid in such case, hence the disabled assertion
        if return_sorted:
            assert torch.equal(counts, t3)
    else:
        assert isinstance(counts, type(None))


@pytest.mark.parametrize(
    "tensor_in",
    [
        torch.tensor(
            [[11, 33, 22], [44, 55, 66], [77, 99, 99], [77, 99, 99]],
            dtype=torch.int32,
        ),
        torch.tensor([[11, 33, 11]], dtype=torch.int32),
        torch.tensor(
            [
                [11.0, 33.0, 22.0],
                [44.0, 55.0, 66.0],
                [44.0, 55.0, 66.0],
                [44.0, 55.0, 66.0],
            ],
            dtype=torch.float32,
        ),
        torch.tensor([[11.0, 33.0, 11.0]], dtype=torch.float32),
        torch.empty((0, 4), dtype=torch.float32),
        torch.empty((0, 4), dtype=torch.int32),
        torch.randn([2, 4, 5, 7], dtype=torch.float32),
        torch.randint(-1000, 1000, (2, 4, 5, 7), dtype=torch.int32),
        torch.tensor(
            [[11.0, 33.0, 12.0], [44.0, 55.0, 66.0], [77.0, 99.0, 99.0]],
            dtype=torch.float32,
        ),
        torch.tensor([[44.0, 55.0, 66.0], [77.0, 99.0, 99.0]], dtype=torch.float32),
    ],
)
@pytest.mark.parametrize("return_inverse", [True, False])
@pytest.mark.parametrize("return_sorted", [True, False])
@pytest.mark.parametrize("return_counts", [True, False])
def test_unique2_delegate(tensor_in, return_inverse, return_sorted, return_counts):
    self = tensor_in
    unique_cpu = torch.unique(
        self,
        return_inverse=return_inverse,
        sorted=return_sorted,
        return_counts=return_counts,
    )
    unique_hpu = torch.unique(
        self.to("hpu"),
        return_inverse=return_inverse,
        sorted=return_sorted,
        return_counts=return_counts,
    )

    if return_counts and return_inverse:
        feature_map_cpu, inverse_cpu, counts_cpu = unique_cpu
        feature_map, inverse, counts = unique_hpu
    if return_counts and not return_inverse:
        feature_map_cpu, counts_cpu = unique_cpu
        feature_map, counts = unique_hpu
    if not return_counts and return_inverse:
        feature_map_cpu, inverse_cpu = unique_cpu
        feature_map, inverse = unique_hpu
    if not return_counts and not return_inverse:
        feature_map_cpu = unique_cpu
        feature_map = unique_hpu

    feature_map = feature_map.to("cpu")
    if return_inverse:
        inverse = inverse.to("cpu")
    if return_counts:
        counts = counts.to("cpu")
    # # NOTE - unique is nondeterministic when returning an unsorted result,
    # # the tensors are sorted for further comparison to succeed
    if not return_sorted:
        feature_map_cpu = feature_map_cpu.sort()[0]
        feature_map = feature_map.sort()[0]
    assert torch.allclose(feature_map, feature_map_cpu)
    if return_inverse:
        #     # NOTE - unique is nondeterministic when returning an unsorted result,
        #     # the inverse tensor will not be valid in such case, hence the disabled assertion
        if return_sorted:
            assert torch.equal(inverse, inverse_cpu)
    if return_counts:
        #     # NOTE - unique is nondeterministic when returning an unsorted result,
        #     # the inverse tensor will not be valid in such case, hence the disabled assertion
        if return_sorted:
            assert torch.equal(counts, counts_cpu)


@pytest.mark.parametrize(
    "tensor_in",
    [
        torch.tensor(
            [[11, 33, 22], [44, 55, 66], [77, 99, 99], [77, 99, 99]],
            dtype=torch.int32,
        ),
        torch.tensor([[11, 33, 11]], dtype=torch.int32),
        torch.tensor(
            [
                [11.0, 33.0, 22.0],
                [44.0, 55.0, 66.0],
                [44.0, 55.0, 66.0],
                [44.0, 55.0, 66.0],
            ],
            dtype=torch.float32,
        ),
        torch.tensor([[11.0, 33.0, 11.0]], dtype=torch.float32),
        torch.empty((0, 4), dtype=torch.float32),
        torch.empty((0, 4), dtype=torch.int32),
        torch.randn([2, 4, 5, 7], dtype=torch.float32),
        torch.randint(-1000, 1000, (2, 4, 5, 7), dtype=torch.int32),
        torch.tensor(
            [[11.0, 33.0, 12.0], [44.0, 55.0, 66.0], [77.0, 99.0, 99.0]],
            dtype=torch.float32,
        ),
        torch.tensor([[44.0, 55.0, 66.0], [77.0, 99.0, 99.0]], dtype=torch.float32),
    ],
)
@pytest.mark.parametrize("return_inverse", [True, False])
@pytest.mark.parametrize("return_sorted", [True, False])
@pytest.mark.parametrize("return_counts", [True, False])
def test_unique2_tensor_delegate(tensor_in, return_inverse, return_sorted, return_counts):
    self = tensor_in
    unique_cpu = torch.Tensor.unique(
        self,
        return_inverse=return_inverse,
        sorted=return_sorted,
        return_counts=return_counts,
    )
    unique_hpu = torch.Tensor.unique(
        self.to("hpu"),
        return_inverse=return_inverse,
        sorted=return_sorted,
        return_counts=return_counts,
    )

    if return_counts and return_inverse:
        feature_map_cpu, inverse_cpu, counts_cpu = unique_cpu
        feature_map, inverse, counts = unique_hpu
    if return_counts and not return_inverse:
        feature_map_cpu, counts_cpu = unique_cpu
        feature_map, counts = unique_hpu
    if not return_counts and return_inverse:
        feature_map_cpu, inverse_cpu = unique_cpu
        feature_map, inverse = unique_hpu
    if not return_counts and not return_inverse:
        feature_map_cpu = unique_cpu
        feature_map = unique_hpu

    feature_map = feature_map.to("cpu")
    if return_inverse:
        inverse = inverse.to("cpu")
    if return_counts:
        counts = counts.to("cpu")
    # # NOTE - unique is nondeterministic when returning an unsorted result,
    # # the tensors are sorted for further comparison to succeed
    if not return_sorted:
        feature_map_cpu = feature_map_cpu.sort()[0]
        feature_map = feature_map.sort()[0]
    assert torch.allclose(feature_map, feature_map_cpu)
    if return_inverse:
        #     # NOTE - unique is nondeterministic when returning an unsorted result,
        #     # the inverse tensor will not be valid in such case, hence the disabled assertion
        if return_sorted:
            assert torch.equal(inverse, inverse_cpu)
    if return_counts:
        #     # NOTE - unique is nondeterministic when returning an unsorted result,
        #     # the inverse tensor will not be valid in such case, hence the disabled assertion
        if return_sorted:
            assert torch.equal(counts, counts_cpu)


@pytest.mark.parametrize(
    "tensor_in",
    [
        torch.tensor([[11, 33, 22], [44, 55, 66], [77, 99, 99], [77, 99, 99]], dtype=torch.int32),
        torch.tensor([[11, 33, 11]], dtype=torch.int32),
        torch.tensor(
            [[11.0, 33.0, 22.0], [44.0, 55.0, 66.0], [44.0, 55.0, 66.0], [44.0, 55.0, 66.0]], dtype=torch.float32
        ),
        torch.tensor([[11.0, 33.0, 11.0]], dtype=torch.float32),
        torch.empty((0, 4), dtype=torch.float32),
        torch.empty((0, 4), dtype=torch.int32),
        torch.randn([2, 4, 5, 7], dtype=torch.float32),
        torch.randint(-1000, 1000, (2, 4, 5, 7), dtype=torch.int32),
        torch.tensor([[11.0, 33.0, 12.0], [44.0, 55.0, 66.0], [77.0, 99.0, 99.0]], dtype=torch.float32),
        torch.tensor([[44.0, 55.0, 66.0], [77.0, 99.0, 99.0]], dtype=torch.float32),
    ],
)
@pytest.mark.parametrize("return_inverse", [True, False])
@pytest.mark.parametrize("return_sorted", [True, False])
def test_unique(tensor_in, return_inverse, return_sorted):
    self = tensor_in
    unique_cpu = torch._unique(self, return_inverse=return_inverse, sorted=return_sorted)
    t1, t2 = unique_cpu
    feature_map, inverse = torch._unique(self.to("hpu"), return_inverse=return_inverse, sorted=return_sorted)
    feature_map = feature_map.to("cpu")
    # NOTE - unique is nondeterministic when returning an unsorted result,
    # the tensors are sorted for further comparison to succeed
    if not return_sorted:
        t1 = t1.sort()[0]
        feature_map = feature_map.sort()[0]
    assert torch.allclose(feature_map, t1)
    if return_inverse:
        # NOTE - unique is nondeterministic when returning an unsorted result,
        # the inverse tensor will not be valid in such case, hence the disabled assertion
        if return_sorted:
            assert torch.equal(inverse.to("cpu"), t2)
    else:
        assert isinstance(inverse, type(None))


# For Scalars to() operator and item() are going with different paths for scalars
# copy h2d is done via copy_from_ operator, but item() is calling local_scalar_dense
# both should support INT64 downcasting
@pytest.mark.parametrize("init_val, dtype", [(1234567, torch.int64), (12345.678, torch.double)])
def test_local_scalar_dense(init_val, dtype):
    hpu_tensor = torch.Tensor([init_val]).type(dtype).to("hpu")
    if dtype == torch.double:
        assert np.allclose([hpu_tensor.item()], [init_val], atol=0.001, rtol=0.001)
    else:
        assert hpu_tensor.item() == init_val


@pytest.mark.skip(reason="Tests in this file are chaning env variables")
@pytest.mark.parametrize(
    "setup_teardown_env_fixture",
    [{"PT_HPU_LAZY_EAGER_SHAPE_AGNOSTIC_GRAPH": "1"}],
    indirect=True,
)
def test_sag_permute_add(setup_teardown_env_fixture):
    cpu_tensor = torch.randn(3, 3, 3, dtype=torch.float32)
    permute_cpu = torch.permute(cpu_tensor, (2, 0, 1)).contiguous()
    result_cpu = torch.add(permute_cpu, 2)

    hpu_tensor = cpu_tensor.to("hpu")
    permute_hpu = torch.permute(hpu_tensor, (2, 0, 1)).contiguous()
    result_hpu = torch.add(permute_hpu, 2)

    cpu_tensor2 = torch.randn(3, 3, 3, dtype=torch.float32)
    permute_cpu2 = torch.permute(cpu_tensor2, (2, 0, 1)).contiguous()
    result_cpu2 = torch.add(permute_cpu2, 3)

    hpu_tensor2 = cpu_tensor2.to("hpu")
    permute_hpu2 = torch.permute(hpu_tensor2, (2, 0, 1)).contiguous()
    result_hpu2 = torch.add(permute_hpu2, 3)

    assert torch.allclose(result_hpu.to("cpu"), result_cpu, atol=0.001, rtol=0.001)
    assert torch.allclose(result_hpu2.to("cpu"), result_cpu2, atol=0.001, rtol=0.001)


@pytest.mark.skip(reason="Tests in this file are chaning env variables")
@pytest.mark.parametrize(
    "setup_teardown_env_fixture",
    [{"PT_HPU_LAZY_EAGER_SHAPE_AGNOSTIC_GRAPH": "1"}],
    indirect=True,
)
def test_sag_conv_relu(setup_teardown_env_fixture):
    input_a = torch.arange(27, dtype=torch.float32, requires_grad=False).reshape(1, 3, 3, 3)
    input_b = torch.arange(64, dtype=torch.float32, requires_grad=False).reshape(1, 4, 4, 4)

    weight_a = torch.arange(27, dtype=torch.float32, requires_grad=False).reshape(3, 3, 3, 1)
    weight_b = torch.arange(64, dtype=torch.float32, requires_grad=False).reshape(4, 4, 4, 1)

    # cpu
    conv_a = torch.nn.functional.conv2d(input_a, weight_a, bias=None, stride=1, padding=0, dilation=1, groups=1)
    out_a = torch.relu(conv_a)

    conv_b = torch.nn.functional.conv2d(input_b, weight_b, bias=None, stride=1, padding=0, dilation=1, groups=1)
    out_b = torch.relu(conv_b)

    # hpu
    hpu_input_a = input_a.to("hpu")
    hpu_weight_a = weight_a.to("hpu")
    hpu_conv_a = torch.nn.functional.conv2d(
        hpu_input_a,
        hpu_weight_a,
        bias=None,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
    )
    hpu_out_a = torch.relu(hpu_conv_a)

    hpu_input_b = input_b.to("hpu")
    hpu_weight_b = weight_b.to("hpu")
    hpu_conv_b = torch.nn.functional.conv2d(
        hpu_input_b,
        hpu_weight_b,
        bias=None,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
    )
    hpu_out_b = torch.relu(hpu_conv_b)

    assert torch.allclose(hpu_out_a.to("cpu"), out_a, atol=0.001, rtol=0.001)
    assert torch.allclose(hpu_out_b.to("cpu"), out_b, atol=0.001, rtol=0.001)


# To validate permute information as part of JIT/SAG key calculation
# 1st and 2nd relu has input with real permute while 3rd relu does not
# have any permute on the input so 3rd relu should cause a JIT/SAG cache
# miss.


@pytest.mark.skip(reason="Tests in this file are chaning env variables")
@pytest.mark.parametrize(
    "setup_teardown_env_fixture",
    [{"PT_HPU_LAZY_EAGER_SHAPE_AGNOSTIC_GRAPH": "1"}],
    indirect=True,
)
def test_sag_conv_relu_relu(setup_teardown_env_fixture):
    input_a = torch.arange(27, dtype=torch.float32, requires_grad=False).reshape(1, 3, 3, 3)
    input_b = torch.arange(64, dtype=torch.float32, requires_grad=False).reshape(1, 4, 4, 4)
    input_c = torch.arange(27, dtype=torch.float32, requires_grad=False).reshape(1, 3, 3, 3)

    weight_a = torch.arange(27, dtype=torch.float32, requires_grad=False).reshape(3, 3, 3, 1)
    weight_b = torch.arange(64, dtype=torch.float32, requires_grad=False).reshape(4, 4, 4, 1)

    # cpu
    conv_a = torch.nn.functional.conv2d(input_a, weight_a, bias=None, stride=1, padding=0, dilation=1, groups=1)
    out_a = torch.relu(conv_a)

    conv_b = torch.nn.functional.conv2d(input_b, weight_b, bias=None, stride=1, padding=0, dilation=1, groups=1)
    out_b = torch.relu(conv_b)

    out_c = torch.relu(input_c)

    # hpu
    hpu_input_a = input_a.to("hpu")
    hpu_weight_a = weight_a.to("hpu")
    hpu_conv_a = torch.nn.functional.conv2d(
        hpu_input_a,
        hpu_weight_a,
        bias=None,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
    )
    hpu_out_a = torch.relu(hpu_conv_a)

    hpu_input_b = input_b.to("hpu")
    hpu_weight_b = weight_b.to("hpu")
    hpu_conv_b = torch.nn.functional.conv2d(
        hpu_input_b,
        hpu_weight_b,
        bias=None,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
    )
    hpu_out_b = torch.relu(hpu_conv_b)

    hpu_input_c = input_c.to("hpu")
    hpu_out_c = torch.relu(hpu_input_c)

    assert torch.allclose(hpu_out_a.to("cpu"), out_a, atol=0.001, rtol=0.001)
    assert torch.allclose(hpu_out_b.to("cpu"), out_b, atol=0.001, rtol=0.001)
    assert torch.allclose(hpu_out_c.to("cpu"), out_c, atol=0.001, rtol=0.001)


def test_sag_batch_norm():
    torch.manual_seed(0)

    batch_norm_cpu = torch.nn.BatchNorm2d(
        num_features=4,
        eps=1e-05,
        momentum=0.1,
        affine=True,
        track_running_stats=True,
    )
    batch_norm_hpu = torch.nn.BatchNorm2d(
        num_features=4,
        eps=1e-05,
        momentum=0.1,
        affine=True,
        track_running_stats=True,
        device="hpu",
    )

    input1 = torch.randn((2, 4, 8, 8), dtype=torch.bfloat16)
    input1_hpu = input1.to("hpu")

    input2 = torch.randn((3, 4, 16, 16), dtype=torch.bfloat16)
    input2_hpu = input2.to("hpu")

    output1 = batch_norm_cpu(input1)
    output2 = batch_norm_cpu(input2)

    output1_hpu = batch_norm_hpu(input1_hpu)
    output2_hpu = batch_norm_hpu(input2_hpu)

    assert torch.allclose(output1_hpu.cpu(), output1, atol=0.01, rtol=0.01)
    assert torch.allclose(output2_hpu.cpu(), output2, atol=0.01, rtol=0.01)


def test_sag_cat_view():
    params = [(10, 3), (20, 5), (30, 7)]
    for element_count, sliced_count in params:
        a = torch.arange(element_count, dtype=torch.int32)
        m = a[:sliced_count]
        n = a[-sliced_count:]
        out = torch.cat([m, n])

        a_hpu = a.to("hpu")
        m_hpu = a_hpu[:sliced_count]
        n_hpu = a_hpu[-sliced_count:]
        out_hpu = torch.cat([m_hpu, n_hpu])

        assert torch.equal(out_hpu.cpu(), out)


def test_sag_add_view():
    params = [(16, 4), (32, 8)]
    for shape, offset in params:
        input = torch.randn((64), dtype=torch.bfloat16)
        input_view = input.as_strided((shape,), (1,), offset)

        input_hpu = input.to("hpu")
        input_hpu_view = input_hpu.as_strided((shape,), (1,), offset)

        out = torch.add(input_view, input_view)
        out_hpu = torch.add(input_hpu_view, input_hpu_view)

        assert torch.allclose(out_hpu.cpu(), out, atol=0.01, rtol=0.01)


def test_set_op():
    tensor1 = torch.randn([64]).to("hpu")
    tensor2 = torch.randn([64]).to("hpu")
    storage1 = tensor1.untyped_storage()
    assert tensor1.data_ptr() != tensor2.data_ptr()
    tensor1.set_(tensor2)
    assert tensor1.data_ptr() == tensor2.data_ptr()
    tensor1_data_ptr = tensor1.data_ptr()
    tensor1.set_()
    assert tensor1.data_ptr() != tensor1_data_ptr
    tensor1_data_ptr = tensor1.data_ptr()
    tensor1.set_(storage1)
    assert (
        tensor1.data_ptr() == storage1.data_ptr()
        and storage1.data_ptr() != tensor2.data_ptr()
        and tensor1.data_ptr() != tensor1_data_ptr
    )
    tensor1.set_(storage1, 10, [54])
    assert tensor1.data_ptr() == (storage1.data_ptr() + 40)


def test_sag_zst_1d():
    a = torch.empty(0).to("hpu")
    b = torch.mul(a, 2)
    c = b.cpu()

    d = torch.empty(1).to("hpu")
    d.fill_(3)
    e = torch.mul(d, 2)

    assert torch.equal(e.cpu(), torch.mul(d.cpu(), 2))


@pytest.mark.skipif(is_gaudi1(), reason="G1 unsupported test")
def test_sag_conv_bwd_view():
    for N, C, H, W, C2 in [
        [2, 4, 7, 7, 3],
        [2, 4, 10, 10, 3],
    ]:
        grad_output = torch.rand([N, C2, H, W]).to("hpu")
        input = torch.rand([N, C, H, W]).to("hpu")
        input_strided = input.as_strided((N, C, H, W), (N * C * H, C * H, 1, W))
        weight = torch.rand([C2, C, 1, 1]).to("hpu")

        # convolution_backward_overrideable is not implemented on CPU
        # just check if it works without validating the results
        result = torch.ops.aten.convolution_backward_overrideable(
            grad_output,
            input_strided,
            weight,
            stride=[1, 1],
            padding=[0, 0],
            dilation=[1, 1],
            transposed=False,
            output_padding=[0, 0],
            groups=1,
            output_mask=[True, True, False],
        )

        r_cpu = result[0].cpu()
        assert r_cpu.dim() == 4


@pytest.mark.parametrize(
    "op",
    [
        torch.Tensor.floor_divide_,
        torch.Tensor.clamp_min_,
        torch.Tensor.clamp_max_,
        torch.Tensor.div_,
        torch.Tensor.mul_,
        torch.Tensor.add_,
        torch.Tensor.xlogy_,
    ],
)
def test_inplace_binary_op_channel_last_different_dtypes(op):
    shape = [2, 2, 2, 10]

    input_cpu = torch.randn(shape, device="cpu").to(dtype=torch.bfloat16)
    other_cpu = torch.rand(shape, device="cpu").to(dtype=torch.float32)

    input_hpu = input_cpu.to("hpu").contiguous(memory_format=torch.channels_last)
    other_hpu = other_cpu.to("hpu")
    input_cpu = input_cpu.contiguous(memory_format=torch.channels_last)

    op(input_cpu, other_cpu)
    op(input_hpu, other_hpu)

    torch.testing.assert_close(input_cpu, input_hpu.cpu())


def test_inplace_clamp_channel_last_different_dtypes():
    shape = [2, 2, 2, 10]

    input_cpu = torch.randn(shape, device="cpu").to(dtype=torch.bfloat16)
    min_cpu = torch.randn(shape, device="cpu").to(dtype=torch.float32)
    max_cpu = torch.randn(shape, device="cpu").to(dtype=torch.float32)

    input_hpu = input_cpu.to("hpu").contiguous(memory_format=torch.channels_last)
    input_cpu = input_cpu.contiguous(memory_format=torch.channels_last)
    min_hpu = min_cpu.to("hpu")
    max_hpu = max_cpu.to("hpu")

    input_cpu.clamp_(min_cpu, max_cpu)
    input_hpu.clamp_(min_hpu, max_hpu)

    torch.testing.assert_close(input_cpu, input_hpu.cpu())


# test node params patching for strided_view and strided_insert in same graph
def test_sag_for_each_zero():
    params = [(2, 4), (4, 8)]

    for shape1, shape2 in params:
        a = torch.randint(-5, 5, (shape1,), dtype=torch.int32)
        b = torch.randint(-5, 5, (shape2,), dtype=torch.int32)

        a_hpu = a.to("hpu")
        b_hpu = b.to("hpu")

        torch._foreach_zero_([a, b])
        torch._foreach_zero_([a_hpu, b_hpu])

        assert torch.equal(a, a_hpu.cpu())
        assert torch.equal(b, b_hpu.cpu())


# test node params patching for constant fill node
def test_sag_fill_node_params():
    a = torch.rand((2, 3), dtype=torch.bfloat16)
    a_hpu = a.to("hpu")

    n = 5
    for x in range(1, n + 1):
        a.fill_(x)
        a_hpu.fill_(x)
        assert torch.equal(a, a_hpu.cpu())


# test node params patching for strided_view and strided_insert in same graph
def test_sag_view_node_params_1():
    params = [((2, 2), (1, 2), 2), ((2, 4), (1, 4), 4), ((4, 8), (1, 8), 8)]

    for shapes, strides, offset in params:
        a = torch.rand(256, dtype=torch.float32)
        a_hpu = a.to("hpu")

        b = a.as_strided(shapes, strides, offset)
        b_hpu = a_hpu.as_strided(shapes, strides, offset)

        b.mul_(2)
        b_hpu.mul_(2)

        assert torch.allclose(b, b_hpu.cpu(), atol=0.001, rtol=0.001)


# test node params patching for strided_view op
def test_sag_view_node_params_2():
    params = [((2, 2), (1, 2), 0), ((4, 4), (1, 4), 4)]

    for shapes, strides, offset in params:
        a = torch.rand(64, dtype=torch.bfloat16)
        a_hpu = a.to("hpu")

        b = a.as_strided(shapes, strides, offset)
        b_hpu = a_hpu.as_strided(shapes, strides, offset)

        assert torch.allclose(b, b_hpu.cpu(), atol=0.001, rtol=0.001)


# test node params patching for strided_insert op
def test_sag_view_node_params_3():
    params = [(32, (2, 2), (1, 2), 2), (64, (4, 4), (1, 4), 4)]

    for base_shape, shapes, strides, offset in params:
        a = torch.arange(base_shape, dtype=torch.int32).as_strided(shapes, strides, offset)
        a_hpu = a.to("hpu")

        assert torch.equal(a, a_hpu.cpu())


# test node params patching for topk op
def test_sag_topk_node_params():
    params = [(2, 2), (4, 4)]

    for shapes in params:
        input = torch.randn(shapes, dtype=torch.bfloat16)
        input_hpu = input.to("hpu")

        sorted_sequence, _ = torch.sort(input)
        sorted_sequence_hpu, _ = torch.sort(input_hpu)

        assert torch.allclose(sorted_sequence, sorted_sequence_hpu.cpu(), atol=0.001, rtol=0.001)


def test_tensor_containing_scalar():
    input1 = torch.ones([3])
    input2 = torch.tensor(3, dtype=torch.float32)
    input1_hpu = input1.to("hpu")

    output = input1 + input2
    output_hpu = input1_hpu + input2

    assert torch.equal(output, output_hpu.cpu())


dtypes = [torch.bfloat16, torch.float, torch.int]
if not is_gaudi1():
    dtypes.append(torch.long)


@pytest.mark.parametrize("dtype", dtypes, ids=format_tc)
def test_sag_copy_bool(dtype):
    shapes = [(2, 3), (5, 8)]

    for shape in shapes:
        if dtype in (torch.int, torch.long):
            input = torch.randint(size=shape, low=0, high=2, dtype=dtype, device="cpu")
        else:
            input = torch.rand(shape, dtype=dtype, device="cpu")

        input_hpu = input.to("hpu")

        # copy to bool
        input_bool = input.to(torch.bool)
        input_hpu_bool = input_hpu.to(torch.bool)
        assert torch.equal(input_bool, input_hpu_bool.cpu())

        # copy from bool
        output = input_bool.to(dtype)
        output_hpu = input_hpu_bool.to(dtype)
        assert torch.equal(output, output_hpu.cpu())


def test_shape_agnostic_helper():
    cpu_tensor = torch.Tensor(np.arange(-10.0, 10.0, 0.1))
    hpu_tensor = cpu_tensor.to("hpu")

    result_hpu = torch.relu(hpu_tensor).to("cpu")
    result_cpu = torch.relu(cpu_tensor)

    assert torch.equal(result_hpu, result_cpu)
    shape_agnostic_not_supported_ops = htdebug._get_shape_agnostic_unsupported_ops()
    eager_compiler_not_supported_ops = htdebug._get_eager_compiler_unsupported_op_prefixes()
    if Verbose:
        print(f"Shape agnostic not supported ops:: {shape_agnostic_not_supported_ops}")
        print(f"Eager compiler not supported op prefixes:: {eager_compiler_not_supported_ops}")


# test node params patching for cat op
@pytest.mark.skipif(is_gaudi1(), reason="G1 unsupported test")
def test_sag_cat_node_params():
    params = [0, 2]

    iteration = 0
    htdebug._clear_jit_cache()
    for dim in params:
        input1 = torch.randn((2, 3, 4), dtype=torch.bfloat16)
        input1_hpu = input1.to("hpu")

        input2 = torch.randn((2, 3, 4), dtype=torch.bfloat16)
        input2_hpu = input2.to("hpu")

        output = torch.cat((input1, input2), dim)
        output_hpu = torch.cat((input1_hpu, input2_hpu), dim)

        assert torch.equal(output_hpu.cpu(), output)

        if iteration == 0:
            num_cache_entries_start = htdebug._get_jit_cache_size()

        iteration += 1

    num_cache_entries_end = htdebug._get_jit_cache_size()
    assert num_cache_entries_end == num_cache_entries_start


# test node params patching for topk op
@pytest.mark.skipif(is_gaudi1(), reason="G1 unsupported test")
def test_sag_topk_node_params_2():
    params = [3, 5]

    iteration = 0
    htdebug._clear_jit_cache()
    for k in params:
        input = torch.randn((10), dtype=torch.bfloat16)
        input_hpu = input.to("hpu")

        output = torch.topk(input, 3)
        output_hpu = torch.topk(input_hpu, 3)

        assert torch.equal(output_hpu[0].cpu(), output[0])

        if iteration == 0:
            num_cache_entries_start = htdebug._get_jit_cache_size()

        iteration += 1

    num_cache_entries_end = htdebug._get_jit_cache_size()
    assert num_cache_entries_end == num_cache_entries_start


# test node params patching for arange op
@pytest.mark.skipif(is_gaudi1(), reason="G1 unsupported test")
def test_sag_arange_node_params():
    params = [1, 2]

    iteration = 0
    htdebug._clear_jit_cache()
    for step in params:
        input = torch.randn((10, 20, 30), dtype=torch.bfloat16)
        input_hpu = input.to("hpu")

        output = torch.arange(2, 6, step)
        output_hpu = torch.arange(2, 6, step, device=input_hpu.device)

        assert torch.equal(output_hpu.cpu(), output)

        if iteration == 0:
            num_cache_entries_start = htdebug._get_jit_cache_size()

        iteration += 1

    num_cache_entries_end = htdebug._get_jit_cache_size()
    assert num_cache_entries_end == num_cache_entries_start


# test node params patching for upsample nearest 2d op
@pytest.mark.skipif(is_gaudi1(), reason="G1 unsupported test")
def test_sag_upsample_nearest_2d_node_params():
    params = [2, 3]

    iteration = 0
    htdebug._clear_jit_cache()
    for scale in params:
        input = torch.randn((1, 1, 2, 3), dtype=torch.bfloat16)
        input_hpu = input.to("hpu")

        m = torch.nn.Upsample(scale_factor=scale, mode="nearest")
        output = m(input)
        output_hpu = m(input_hpu)

        assert torch.equal(output_hpu.cpu(), output)

        if iteration == 0:
            num_cache_entries_start = htdebug._get_jit_cache_size()

        iteration += 1

    num_cache_entries_end = htdebug._get_jit_cache_size()
    assert num_cache_entries_end == num_cache_entries_start


# test node params patching for upsample nearest 1d op
@pytest.mark.skipif(is_gaudi1(), reason="G1 unsupported test")
def test_sag_upsample_nearest_1d_node_params():
    params = [2, 3]

    iteration = 0
    htdebug._clear_jit_cache()
    for scale in params:
        input = torch.randn((1, 1, 1, 3), dtype=torch.bfloat16)
        input_hpu = input.to("hpu")

        m = torch.nn.Upsample(scale_factor=scale, mode="nearest")
        output = m(input)
        output_hpu = m(input_hpu)

        assert torch.equal(output_hpu.cpu(), output)

        if iteration == 0:
            num_cache_entries_start = htdebug._get_jit_cache_size()

        iteration += 1

    num_cache_entries_end = htdebug._get_jit_cache_size()
    assert num_cache_entries_end == num_cache_entries_start


# test node params patching for upsample nearest 3d op
@pytest.mark.skipif(is_gaudi1(), reason="G1 unsupported test")
def test_sag_upsample_nearest_3d_node_params():
    params = [2, 3]

    iteration = 0
    htdebug._clear_jit_cache()
    for scale in params:
        input = torch.randn((1, 2, 2, 3), dtype=torch.bfloat16)
        input_hpu = input.to("hpu")

        m = torch.nn.Upsample(scale_factor=scale, mode="nearest")
        output = m(input)
        output_hpu = m(input_hpu)

        assert torch.equal(output_hpu.cpu(), output)

        if iteration == 0:
            num_cache_entries_start = htdebug._get_jit_cache_size()

        iteration += 1

    num_cache_entries_end = htdebug._get_jit_cache_size()
    assert num_cache_entries_end == num_cache_entries_start


# test node params patching for upsample bilinear 2d op
@pytest.mark.skipif(is_gaudi1(), reason="G1 unsupported test")
def test_sag_upsample_bilinear_2d_node_params():
    params = [2, 3]

    iteration = 0
    htdebug._clear_jit_cache()
    for scale in params:
        input = torch.randn((1, 1, 2, 3), dtype=torch.bfloat16)
        input_hpu = input.to("hpu")

        m = torch.nn.Upsample(scale_factor=scale, mode="bilinear")
        output = m(input)
        output_hpu = m(input_hpu)

        assert torch.allclose(output_hpu.cpu(), output, atol=0.005, rtol=0.005)

        if iteration == 0:
            num_cache_entries_start = htdebug._get_jit_cache_size()

        iteration += 1

    num_cache_entries_end = htdebug._get_jit_cache_size()
    assert num_cache_entries_end == num_cache_entries_start


# test node params patching for upsample bicubic 2d op
@pytest.mark.skipif(is_gaudi1(), reason="G1 unsupported test")
def test_sag_upsample_bicubic_2d_node_params():
    params = [2, 3]

    iteration = 0
    htdebug._clear_jit_cache()
    for scale in params:
        input = torch.randn((1, 1, 2, 3), dtype=torch.bfloat16)
        input_hpu = input.to("hpu")

        m = torch.nn.Upsample(scale_factor=scale, mode="bicubic")
        output = m(input)
        output_hpu = m(input_hpu)

        assert torch.allclose(output_hpu.cpu(), output, atol=0.01, rtol=0.01)

        if iteration == 0:
            num_cache_entries_start = htdebug._get_jit_cache_size()

        iteration += 1

    num_cache_entries_end = htdebug._get_jit_cache_size()
    assert num_cache_entries_end == num_cache_entries_start


# test node params patching for upsample linear 1d op
@pytest.mark.skipif(is_gaudi1(), reason="G1 unsupported test")
def test_sag_upsample_linear_1d_node_params():
    pytest.xfail("[SW-198691] Param agnostic flow disabled for UpsampleLinear1D, needs correction")
    params = [2, 3]

    iteration = 0
    htdebug._clear_jit_cache()
    for scale in params:
        input = torch.randn((1, 1, 3))
        input_hpu = input.to("hpu")

        m = torch.nn.Upsample(scale_factor=scale, mode="linear")
        output = m(input)
        output_hpu = m(input_hpu)

        assert torch.allclose(output_hpu.cpu(), output, atol=0.005, rtol=0.005)

        if iteration == 0:
            num_cache_entries_start = htdebug._get_jit_cache_size()

        iteration += 1

    num_cache_entries_end = htdebug._get_jit_cache_size()
    assert num_cache_entries_end == num_cache_entries_start


# test node params patching for upsample bilinear 2d backward op
@pytest.mark.skipif(is_gaudi1(), reason="G1 unsupported test")
def test_sag_upsample_bilinear_2d_backward_node_params():
    params = [True, False]

    iteration = 0
    htdebug._clear_jit_cache()
    for align_corners in params:
        grad_output = torch.randn((1, 1, 4, 6), dtype=torch.bfloat16)
        grad_output_hpu = grad_output.to("hpu")

        output_size = torch.Tensor([4, 6])
        output_size_hpu = output_size.to("hpu")

        input_size = torch.Tensor([1, 1, 2, 3])
        input_size_hpu = input_size.to("hpu")

        result = torch.ops.aten.upsample_bilinear2d_backward(grad_output, output_size, input_size, align_corners)

        result_hpu = torch.ops.aten.upsample_bilinear2d_backward(
            grad_output_hpu, output_size_hpu, input_size_hpu, align_corners
        )

        assert torch.allclose(result_hpu.cpu(), result, atol=0.02, rtol=0.02)

        if iteration == 0:
            num_cache_entries_start = htdebug._get_jit_cache_size()

        iteration += 1

    num_cache_entries_end = htdebug._get_jit_cache_size()
    assert num_cache_entries_end == num_cache_entries_start


# test node params patching for upsample bicubic 2d backward op
@pytest.mark.skipif(is_gaudi1(), reason="G1 unsupported test")
def test_sag_upsample_bicubic_2d_backward_node_params():
    params = [True, False]

    iteration = 0
    htdebug._clear_jit_cache()
    for align_corners in params:
        grad_output = torch.randn((1, 1, 4, 6), dtype=torch.bfloat16)
        grad_output_hpu = grad_output.to("hpu")

        output_size = torch.Tensor([4, 6])
        output_size_hpu = output_size.to("hpu")

        input_size = torch.Tensor([1, 1, 2, 3])
        input_size_hpu = input_size.to("hpu")

        result = torch.ops.aten.upsample_bicubic2d_backward(grad_output, output_size, input_size, align_corners)

        result_hpu = torch.ops.aten.upsample_bicubic2d_backward(
            grad_output_hpu, output_size_hpu, input_size_hpu, align_corners
        )

        assert torch.allclose(result_hpu.cpu(), result, atol=0.05, rtol=0.05)

        if iteration == 0:
            num_cache_entries_start = htdebug._get_jit_cache_size()

        iteration += 1

        num_cache_entries_end = htdebug._get_jit_cache_size()
        assert num_cache_entries_end == num_cache_entries_start


# test node params patching for resize op
@pytest.mark.skipif(is_gaudi1(), reason="G1 unsupported test")
def test_empty_resize_node_params():
    params = [10, 20]

    iteration = 0
    htdebug._clear_jit_cache()
    for size in params:
        hpu_tensor = torch.empty([], device="hpu")
        hpu_tensor.resize_(size)
        cpu_tensor = hpu_tensor.to("cpu")
        assert np.equal(cpu_tensor.size()[0], size)

        if iteration == 0:
            num_cache_entries_start = htdebug._get_jit_cache_size()

        iteration += 1

        num_cache_entries_end = htdebug._get_jit_cache_size()
        assert num_cache_entries_end == num_cache_entries_start


def test_sag_section_validation_issue():
    params = [(16, (2, 2), (1, 2)), (2732 * 258, (2732, 257), (1, 257))]

    for base_size, shape, strides in params:
        a = torch.rand(base_size, dtype=torch.bfloat16)
        a_h = a.to("hpu")

        a_strided = torch.as_strided(a, shape, strides)
        a_h_strided = torch.as_strided(a_h, shape, strides)

        a_strided.zero_()
        a_h_strided.zero_()

        assert torch.equal(a, a_h.cpu())


@pytest.mark.skipif(is_gaudi1(), reason="G1 unsupported test")
def test_sag_lerp():
    params = [((2, 2), 10), ((4, 4), 20)]

    htdebug._clear_jit_cache()
    for size, end_val in params:
        start = torch.ones(size)
        end = torch.empty(size).fill_(end_val)

        start_hpu = start.to("hpu")
        end_hpu = end.to("hpu")
        out = torch.lerp(start, end, 0.5)
        out_hpu = torch.lerp(start_hpu, end_hpu, 0.5)
        assert torch.allclose(out, out_hpu.cpu(), atol=0.001, rtol=0.001)

    shape_agnostic_not_supported_ops = htdebug._get_shape_agnostic_unsupported_ops()
    assert len(shape_agnostic_not_supported_ops) == 0


# test node params patching for masked_fill op
@pytest.mark.skipif(is_gaudi1(), reason="G1 unsupported test")
def test_sag_masked_fill_node_params():
    a = torch.rand((2, 3), dtype=torch.bfloat16)
    a_hpu = a.to("hpu")
    mask = torch.tensor([[True, False, False], [True, True, False]])
    mask_hpu = mask.to("hpu")
    n = 5
    iteration = 0
    for x in range(1, n + 1):
        output = a.masked_fill(mask, x)
        output_hpu = a_hpu.masked_fill(mask_hpu, x)
        assert torch.equal(output, output_hpu.cpu())
        if iteration == 0:
            num_cache_entries_start = htdebug._get_jit_cache_size()
        iteration += 1
    num_cache_entries_end = htdebug._get_jit_cache_size()
    assert num_cache_entries_end == num_cache_entries_start


# test node params patching for efficientzerotensor op
@pytest.mark.skipif(is_gaudi1(), reason="G1 unsupported test")
def test_efficientzerotensor_node_params():
    params = [(10), (20)]

    iteration = 0
    htdebug._clear_jit_cache()
    for size in params:
        hpu_tensor = torch._efficientzerotensor(size, device="hpu")
        cpu_tensor = torch._efficientzerotensor(size)

        assert torch.equal(cpu_tensor, hpu_tensor.cpu())

        if iteration == 0:
            num_cache_entries_start = htdebug._get_jit_cache_size()

        iteration += 1

    num_cache_entries_end = htdebug._get_jit_cache_size()
    assert num_cache_entries_end == num_cache_entries_start


# test node params patching for efficientzerotensor op
@pytest.mark.skipif(is_gaudi1(), reason="G1 unsupported test")
def test_efficientzerotensor_node_params_2():
    params = [torch.float32, torch.int32]

    iteration = 0
    htdebug._clear_jit_cache()
    for data_type in params:
        hpu_tensor = torch._efficientzerotensor((5), dtype=data_type, device="hpu")
        cpu_tensor = torch._efficientzerotensor((5), dtype=data_type)

        assert torch.equal(cpu_tensor, hpu_tensor.cpu())

        iteration += 1

    num_cache_entries_end = htdebug._get_jit_cache_size()
    assert num_cache_entries_end == iteration


# test for fix in SW-192192
@pytest.mark.skipif(is_gaudi1(), reason="G1 unsupported test")
def test_h2d_copy_race_condition_fix():

    t1 = torch.arange(1, 5, dtype=torch.bfloat16)
    t2 = torch.arange(1, 5, dtype=torch.bfloat16)
    for iteration in range(5):
        t1_hpu = t1.to("hpu", non_blocking=True)
        t2_hpu = t2.to("hpu", non_blocking=True)
        output_hpu = torch.add(t1_hpu, t2_hpu)
        output = torch.add(t1, t2)

    assert torch.equal(output, output_hpu.cpu())
