import torch
import pytest
from test_utils import evaluate_fwd_kernel, evaluate_fwd_bwd_kernel, reset_seed


# N - batch
# H - input height
# W - input width
# C - input channels
test_case_list = [
    # N, C, H, W
    (8, 3, 2, 2)
]

reduction_op_list = [
    torch.sum,
    torch.mean
]

any_op_dim_list = [
    (torch.any, 0, True),
    (torch.any, 1, False),
    (torch.any, 2, True),
    (torch.any, 3, False),
]

reduction_dim_list = [
    ([1], True),
    ([1], False),
    ([1, 2, 0], True),
    ([1, 2, 0], False),
    ([2, 0], True),
    ([1, 0, 3], True),
    ([1, 0, 3], False)
]

reduction_dim_int_list = [
    (torch.prod, 1, True),
    (torch.prod, 1, False),
    (torch.prod, 2, True),
    (torch.prod, 2, False),
]

data_type_list = [
    (torch.float, 0.001)
]


@pytest.mark.parametrize("N, C, H, W", test_case_list)
@pytest.mark.parametrize("dims, keepdims", reduction_dim_list)
@pytest.mark.parametrize("reduction_op", reduction_op_list)
@pytest.mark.parametrize("dtype, tol", data_type_list)
def test_hpu_reduction_op(N, C, H, W, reduction_op, dims, keepdims, dtype, tol):
    kernel_params = {'input': torch.randn(N, C, H, W).to(dtype),
                     'dim': dims,
                     'keepdim': keepdims}
    evaluate_fwd_kernel(kernel=reduction_op, kernel_params=kernel_params, atol=tol, rtol=tol)

@pytest.mark.parametrize("N, C, H, W", test_case_list)
@pytest.mark.parametrize("reduction_op, dims, keepdims", reduction_dim_int_list)
@pytest.mark.parametrize("dtype, tol", data_type_list)
def test_hpu_reduction_op_dim_int(N, C, H, W, reduction_op, dims, keepdims, dtype, tol):
    kernel_params = {'input': torch.randn(N, C, H, W).to(dtype),
                     'dim': dims,
                     'keepdim': keepdims}
    evaluate_fwd_kernel(kernel=reduction_op, kernel_params=kernel_params, atol=tol, rtol=tol)

@pytest.mark.parametrize("N, C, H, W", test_case_list)
@pytest.mark.parametrize("dims, keepdims", reduction_dim_list)
@pytest.mark.parametrize("reduction_op", reduction_op_list)
def test_hpu_reduction_out_op(N, C, H, W, reduction_op, dims, keepdims):
    out_list = [N, C, H, W]
    kernel_params = {'out': torch.randn(tuple(out_list)),
                     'input': torch.randn(N, C, H, W),
                     'dim': dims,
                     'keepdim': keepdims}
    evaluate_fwd_kernel(kernel=reduction_op, kernel_params=kernel_params)


@pytest.mark.parametrize("N, C, H, W", test_case_list)
@pytest.mark.parametrize("reduction_op", reduction_op_list)
def test_hpu_reduction_all_op(N, C, H, W, reduction_op):
    kernel_params = {'input': torch.randn(N, C, H, W)}
    evaluate_fwd_kernel(kernel=reduction_op, kernel_params=kernel_params)


@pytest.mark.parametrize("N, C, H, W", test_case_list)
@pytest.mark.parametrize("reduction_op, dims, keepdims", any_op_dim_list)
def test_hpu_reduction_op_any_dim(N, C, H, W, reduction_op, dims, keepdims):
    kernel_params = {'input': torch.randn(N, C, H, W) < 0,
                     'dim': dims,
                     'keepdim': keepdims}
    evaluate_fwd_kernel(kernel=reduction_op, kernel_params=kernel_params)


@pytest.mark.parametrize("N, C, H, W", test_case_list)
@pytest.mark.parametrize("reduction_op, dim, keepdim", any_op_dim_list)
def test_hpu_reduction_op_any_dim_out(N, C, H, W, reduction_op, dim, keepdim):
    out_list = [N, C, H, W]
    kernel_params = {'out': torch.randn(tuple(out_list)) < 0,
                     'input': torch.randn(N, C, H, W) < 0,
                     'dim': dim,
                     'keepdim': keepdim}
    evaluate_fwd_kernel(kernel=reduction_op, kernel_params=kernel_params)


@pytest.mark.parametrize("N, C, H, W", test_case_list)
@pytest.mark.parametrize("reduction_op", [torch.any])
def test_hpu_reduction_op_any(N, C, H, W, reduction_op):
    kernel_params = {'input': torch.randn(N, C, H, W) < 0}
    evaluate_fwd_kernel(kernel=reduction_op, kernel_params=kernel_params)


if __name__ == '__main__':
    test_hpu_reduction_op(*test_case_list[0], reduction_op_list[0], reduction_dim_list[0])
    test_hpu_reduction_op_dim_int(*test_case_list[0], reduction_dim_int_list[0])
    test_hpu_reduction_out_op(*test_case_list[0], reduction_op_list[0], reduction_dim_list[0])
    test_hpu_reduction_all_op(*test_case_list[0], reduction_op_list[0])
