import torch
import pytest
from test_utils import evaluate_fwd_kernel, evaluate_fwd_bwd_kernel, reset_seed


# N - batch
# H - input height
# W - input width
# C - input channels
test_case_list = [
    # N, C,
    (8, 3,),
]

reduction_op_list = [
    torch.sum,
]


@pytest.mark.parametrize("N, C", test_case_list)
@pytest.mark.parametrize("reduction_op", reduction_op_list)
def test_hpu_reduction_op(N, C, reduction_op):
    kernel_params = {'input': torch.randn(N, C),
                     'dim': [1],
                     'keepdim': True}
    evaluate_fwd_kernel(kernel=reduction_op, kernel_params=kernel_params)


if __name__ == '__main__':
    test_hpu_reduction_op(*test_case_list[0], reduction_op_list[0])
