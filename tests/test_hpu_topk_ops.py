import numpy as np
import pytest
import torch
from test_utils import cpu, evaluate_fwd_kernel, hpu

# N - batch
# H - input height
# W - input width
# C - input channels
test_case_list = [
    #  N, C
    (8, 1024)
]

test_case_list2 = [
    #  N, C
    (10, 10, 10, 10)
]

topk_op_list = [
    # op
    torch.topk
]

topk_values_list = [
    # k, dim
    [8, -1],
    [3, 1],
    [10, 1],
]

sort_op_list = [
    # op
    torch.sort
]

sort_values_list = [
    # dim, descending
    [-1, True],
    [1, True],
    [-1, False],
    [1, False],
]


@pytest.mark.parametrize("N, C", test_case_list)
@pytest.mark.parametrize("topk_op", topk_op_list)
@pytest.mark.parametrize("k, dim", topk_values_list)
def test_hpu_topk_op(N, C, topk_op, k, dim):
    kernel_params = {"input": torch.randn(N, C), "k": k, "dim": dim}
    evaluate_fwd_kernel(kernel=topk_op, kernel_params=kernel_params)


@pytest.mark.parametrize("N, C, H, W", test_case_list2)
@pytest.mark.parametrize("topk_op", topk_op_list)
@pytest.mark.parametrize("k, dim", topk_values_list)
def test_hpu_topk_op2(N, C, H, W, topk_op, k, dim):
    kernel_params = {"input": torch.randn(N, C, H, W), "k": k, "dim": dim}
    evaluate_fwd_kernel(kernel=topk_op, kernel_params=kernel_params)


@pytest.mark.parametrize("N, C", test_case_list)
@pytest.mark.parametrize("topk_op", topk_op_list)
@pytest.mark.parametrize("k, dim", topk_values_list)
def test_hpu_topk_out_op(N, C, topk_op, k, dim):
    kernel_params = {
        "input": torch.randn(N, C),
        "k": k,
        "dim": dim,
        "out": (torch.randn(N, C), torch.empty((N, C), dtype=torch.int)),
    }
    evaluate_fwd_kernel(kernel=topk_op, kernel_params=kernel_params)


@pytest.mark.parametrize("N, C", test_case_list)
@pytest.mark.parametrize("sort_op", sort_op_list)
@pytest.mark.parametrize("dim, descending", sort_values_list)
def test_hpu_sort_op(N, C, sort_op, dim, descending):
    kernel_params = {"input": torch.randn(N, C), "dim": dim, "descending": descending}
    evaluate_fwd_kernel(kernel=sort_op, kernel_params=kernel_params)


@pytest.mark.parametrize("N, C", test_case_list)
@pytest.mark.parametrize("sort_op", sort_op_list)
@pytest.mark.parametrize("dim, descending", sort_values_list)
def test_hpu_sort_fwdbwd_op(N, C, sort_op, dim, descending):
    input_tensor = torch.randn(N, C, requires_grad=True)
    input_tensor_hpu = input_tensor.to(hpu).detach()
    input_tensor_hpu.requires_grad = True

    out_cpu, out_idx_cpu = sort_op(input_tensor, dim=dim, descending=descending)
    grad_out_cpu = torch.ones_like(out_cpu)
    grad_out_hpu = grad_out_cpu.to(hpu).detach()
    grad_out_hpu.requires_grad = True
    out_cpu.backward(grad_out_cpu)
    grad_in = input_tensor.grad

    out_hpu, out_idx_hpu = sort_op(input_tensor_hpu, dim=dim, descending=descending)
    out_hpu.backward(grad_out_hpu)
    hgrad_in = input_tensor_hpu.grad

    np.testing.assert_allclose(
        out_idx_hpu.to(cpu).detach().numpy(),
        out_idx_cpu.detach().numpy(),
        atol=0.01,
        rtol=0.01,
        equal_nan=True,
    )
    np.testing.assert_allclose(
        hgrad_in.to(cpu).detach().numpy(),
        grad_in.detach().numpy(),
        atol=0.01,
        rtol=0.01,
        equal_nan=True,
    )


if __name__ == "__main__":
    test_hpu_topk_op(*test_case_list[0], topk_op_list[0], topk_values_list[0][0], topk_values_list[0][1])
    test_hpu_topk_out_op(*test_case_list[0], topk_op_list[0], topk_values_list[0][0], topk_values_list[0][1])
    test_hpu_sort_op(*test_case_list[0], sort_op_list[0], sort_values_list[0][0], sort_values_list[0][1])
    test_hpu_topk_op2(*test_case_list2[0], topk_op_list[0], topk_values_list[0][0], topk_values_list[0][1])
