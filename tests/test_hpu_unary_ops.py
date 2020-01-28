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


@pytest.mark.parametrize("N, H, W, C", test_case_list)
@pytest.mark.parametrize("unary_op", unary_op_list)
def test_hpu_unary_op_fwd_bwd(N, H, W, C, unary_op):
    # TODO: extend that test to all features
    kernel_params_fwd = {'input': torch.randn(N, C, H, W, requires_grad=True)}
    bwd_tensors = [torch.randn(N, C, H, W)]
    evaluate_fwd_bwd_kernel(kernel=unary_op, tensor_list_bwd=bwd_tensors,
                            kernel_params_fwd=kernel_params_fwd)


if __name__ == '__main__':
    test_hpu_unary_op(*test_case_list[0], unary_op_list[0])
