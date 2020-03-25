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

reduction_dim_list = [
   [1],
   [2],
   [1,2,0]  
]

reduction_keep_dims = [
    True,
    False
]

@pytest.mark.parametrize("N, C, H, W", test_case_list)
@pytest.mark.parametrize("dims", reduction_dim_list)
@pytest.mark.parametrize("keepdims", reduction_keep_dims)
@pytest.mark.parametrize("reduction_op", reduction_op_list)
def test_hpu_reduction_op(N, C, H, W, reduction_op, dims, keepdims):
    kernel_params = {'input': torch.randn(N, C, H, W),
                     'dim': dims,
                     'keepdim': keepdims}
    evaluate_fwd_kernel(kernel=reduction_op, kernel_params=kernel_params)

@pytest.mark.parametrize("N, C, H, W", test_case_list)
@pytest.mark.parametrize("dims", reduction_dim_list)
@pytest.mark.parametrize("keepdims", reduction_keep_dims)
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

if __name__ == '__main__':
    test_hpu_reduction_op(*test_case_list[0], reduction_op_list[0], reduction_dim_list[0], reduction_keep_dims[0])
    test_hpu_reduction_out_op(*test_case_list[0], reduction_op_list[0], reduction_dim_list[0], reduction_keep_dims[0])
    test_hpu_reduction_all_op(*test_case_list[0], reduction_op_list[0])