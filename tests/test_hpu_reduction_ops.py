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
   (torch.any, 0, False),
   (torch.any, 1, True),
   (torch.any, 1, False)
]

reduction_dim_list = [
   ([1], True),
   ([1], False),
   ([2], True),
   ([2], False),
   ([1,2,0], True),
   ([1,2,0], False),
]

@pytest.mark.parametrize("N, C, H, W", test_case_list)
@pytest.mark.parametrize("dims, keepdims", reduction_dim_list)
@pytest.mark.parametrize("reduction_op", reduction_op_list)
def test_hpu_reduction_op(N, C, H, W, reduction_op, dims, keepdims):
    kernel_params = {'input': torch.randn(N, C, H, W),
                     'dim': dims,
                     'keepdim': keepdims}
    evaluate_fwd_kernel(kernel=reduction_op, kernel_params=kernel_params)

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
def test_hpu_reduction_op_any(N, C, H, W, reduction_op, dims, keepdims):
    kernel_params = {'input': torch.randn(N, C, H, W)<0,
                     'dim': dims,
                     'keepdim': keepdims}
    evaluate_fwd_kernel(kernel=reduction_op, kernel_params=kernel_params)

if __name__ == '__main__':
    test_hpu_reduction_op(*test_case_list[0], reduction_op_list[0], reduction_dim_list[0])
    test_hpu_reduction_out_op(*test_case_list[0], reduction_op_list[0], reduction_dim_list[0])
    test_hpu_reduction_all_op(*test_case_list[0], reduction_op_list[0])