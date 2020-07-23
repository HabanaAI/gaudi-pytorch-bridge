import torch
import numpy as np
import pytest
from test_utils import evaluate_fwd_kernel, evaluate_fwd_bwd_kernel

cat_op_list = [
    # op, op params dict
    (torch.cat, {'tensors': (torch.randn(8, 3, 24, 24), torch.randn(8, 3, 24, 12)),
                'dim': 3}),
    (torch.cat, {'tensors': (torch.randn(8, 3, 24, 24), torch.randn(8, 3, 24, 24)),
                'dim': -1}),
    (torch.cat, {'tensors':(torch.randn(8, 3, 24, 24), torch.randn(8, 3, 24, 12), torch.randn(8, 3, 24, 4)),
                'out': torch.randn(8, 3, 24, 24),
                'dim': 3}),
    (torch.cat, {'tensors':(torch.randn(8, 3, 24, 24), torch.randn(8, 3, 24, 24)),
                'out': torch.randn(8, 3),
                'dim': 3}),
]

cat_op_list_fwd_bwd = [
    # op, op params dict
    (torch.cat, {'tensors': (torch.randn(8, 3, 24, 24, requires_grad=True), torch.randn(8, 3, 24, 12, requires_grad=True)),
                'dim': 3}),
]

@pytest.mark.parametrize("cat_op, kernel_params_fwd", cat_op_list)
def test_hpu_cat(cat_op, kernel_params_fwd):
    evaluate_fwd_kernel(kernel=cat_op, kernel_params=kernel_params_fwd)

@pytest.mark.parametrize("cat_op, kernel_params_fwd", cat_op_list_fwd_bwd)
def test_hpu_cat_fwd_bwd(cat_op, kernel_params_fwd):
    dim = kernel_params_fwd['dim']
    tensors = kernel_params_fwd['tensors']
    shape = list(tensors[0].size())
    shape[dim] = 0
    for i in range(0,len(tensors)):
        shape[dim] += tensors[i].size()[dim]
    bwd_tensors = [torch.randn(tuple(shape))]
    evaluate_fwd_bwd_kernel(kernel=cat_op, tensor_list_bwd=bwd_tensors, kernel_params_fwd=kernel_params_fwd)

if __name__ == '__main__':
    test_hpu_cat_fwd_bwd(*cat_op_list_fwd_bwd[1])
