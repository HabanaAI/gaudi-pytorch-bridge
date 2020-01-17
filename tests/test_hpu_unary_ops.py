import torch
import torch.nn.functional as F
import pytest
from test_utils import evaluate_fwd_kernel, evaluate_fwd_bwd_kernel, reset_seed


# N - batch
# H - input height
# W - input width
# C - input channels
test_case_list = [
    #  N, H, W, C,
    (8, 24, 24, 3,),
]

unary_op_list = [
    F.relu,
]


@pytest.mark.parametrize("N, H, W, C", test_case_list)
@pytest.mark.parametrize("unary_op", unary_op_list)
def test_hpu_unary_op(N, H, W, C, unary_op):
    kernel_params = {'input': torch.randn(N, C, H, W)}
    evaluate_fwd_kernel(kernel=unary_op, kernel_params=kernel_params)


if __name__ == '__main__':
    test_hpu_unary_op(*test_case_list[0], unary_op_list[0])
