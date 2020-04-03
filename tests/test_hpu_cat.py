import torch
import numpy as np
import pytest
from test_utils import evaluate_fwd_kernel

# N - batch
# H - input height
# W - input width
# C - input channels

test_case_list = [
    # N, H, W, C
    (8, 24, 24, 3),
]

cat_op_list = [
    # op, op params dict
    (torch.cat, {'tensors': (torch.randn(8, 3, 24, 24), torch.randn(8, 3, 24, 24)),
                'dim': 3}),
    (torch.cat, {'tensors': (torch.randn(8, 3, 24, 24), torch.randn(8, 3, 24, 24)),
                'dim': -1}),
    (torch.cat, {'tensors':(torch.randn(8, 3, 24, 24), torch.randn(8, 3, 24, 24)),
                'out': torch.randn(8, 3, 24, 24),
                'dim': 3}),
    (torch.cat, {'tensors':(torch.randn(8, 3, 24, 24), torch.randn(8, 3, 24, 24)),
                'out': torch.randn(8, 3, 24, 24),
                'dim': 3}),
]

@pytest.mark.parametrize("N, H, W, C", test_case_list)
@pytest.mark.parametrize("cat_op, kernel_params_fwd", cat_op_list)
def test_hpu_cat(N, H, W, C, cat_op, kernel_params_fwd):
    evaluate_fwd_kernel(kernel=cat_op, kernel_params=kernel_params_fwd)

if __name__ == '__main__':
    test_hpu_cat(*test_case_list[0], torch.cat, {'tensors': (torch.randn(8, 3, 24, 24), torch.randn(8, 3, 24, 24)), 'dim': 3})
