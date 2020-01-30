import torch
import pytest
from test_utils import evaluate_fwd_inplace_kernel, reset_seed


# N - batch
# H - input height
# W - input width
# C - input channels
test_case_list = [
    #  N, H, W, C,
    (8, 24, 24, 3,),
]

binary_inplace_op_list = [
    # op, op params dict
    ('add_', {'alpha': 1}),
    ('add_', {'alpha': 0.1}),
]


@pytest.mark.parametrize("N, H, W, C", test_case_list)
@pytest.mark.parametrize("binary_op, kernel_params_fwd", binary_inplace_op_list)
def test_hpu_binary_inplace_op(N, H, W, C, binary_op, kernel_params_fwd):
    in_out_tensor = torch.randn(N, C, H, W)
    kernel_params_fwd['other'] = torch.randn(N, C, H, W)
    evaluate_fwd_inplace_kernel(in_out_tensor=in_out_tensor,
                                kernel_name=binary_op,
                                kernel_params=kernel_params_fwd)


if __name__ == '__main__':
    test_hpu_binary_op(*test_case_list[0], binary_op_list[0])
