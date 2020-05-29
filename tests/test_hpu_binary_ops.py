import torch
import pytest
import random
from test_utils import evaluate_fwd_inplace_kernel, reset_seed, evaluate_fwd_kernel

N = 8
C = 3
H = 24
W = 24

# N - batch
# H - input height
# W - input width
# C - input channels
test_case_list = [
    #  N, H, W, C,
    (N, H, W, C,),
]

binary_inplace_op_list = [
    # op, op params dict
    ('add_', {'alpha': 1}),
    ('add_', {'alpha': 0.1}),
    ('sub_', {'alpha': 1}),
    ('sub_', {'alpha': 0.1}),
    ('mul_', {}),
    ('div_', {}),
]

binary_op_list = [
    # op, op params dict
    (torch.eq, {}),
    (torch.add, {}),
    (torch.add, {'alpha': 0.1}),
    (torch.mul, {}),
    (torch.div, {}),
]

# This list is used to test tensor_out variants of operators
binary_op_out_list = [
    # op, op params dict
    (torch.div, {})
]

binary_op_out_list_bool = [
    # op, op params dict
    (torch.eq, {}),
]


@pytest.mark.parametrize("N, H, W, C", test_case_list)
@pytest.mark.parametrize("binary_op, kernel_params_fwd", binary_op_out_list)
def test_hpu_binary_op_out_intype(N, H, W, C, binary_op, kernel_params_fwd):
    kernel_params_fwd['input'] = inT = torch.randn(N, C, H, W)
    kernel_params_fwd['other'] = torch.randn(N, C, H, W)
    kernel_params_fwd['out'] = torch.empty((N, C, H, W), dtype=inT.dtype)
    evaluate_fwd_kernel(kernel=binary_op, kernel_params=kernel_params_fwd)


@pytest.mark.parametrize("N, H, W, C", test_case_list)
@pytest.mark.parametrize("binary_op, kernel_params_fwd", binary_op_out_list_bool)
def test_hpu_binary_op_out_bool(N, H, W, C, binary_op, kernel_params_fwd):
    kernel_params_fwd['input'] = torch.randn(N, C, H, W)
    kernel_params_fwd['other'] = torch.randn(N, C, H, W)
    kernel_params_fwd['out'] = torch.empty((N, C, H, W), dtype=torch.bool)
    evaluate_fwd_kernel(kernel=binary_op, kernel_params=kernel_params_fwd)


@pytest.mark.parametrize("N, H, W, C", test_case_list)
@pytest.mark.parametrize("binary_op, kernel_params_fwd", binary_inplace_op_list)
def test_hpu_binary_inplace_op(N, H, W, C, binary_op, kernel_params_fwd):
    in_out_tensor = torch.randn(N, C, H, W)
    kernel_params_fwd['other'] = torch.randn(N, C, H, W)
    evaluate_fwd_inplace_kernel(in_out_tensor=in_out_tensor,
                                kernel_name=binary_op,
                                kernel_params=kernel_params_fwd)

@pytest.mark.parametrize("N, H, W, C", test_case_list)
@pytest.mark.parametrize("binary_op, kernel_params_fwd", binary_inplace_op_list)
def test_hpu_binary_inplace_op_broadcast_case1(N, H, W, C, binary_op, kernel_params_fwd):
    in_out_tensor = torch.randn(N, C, H, W)
    kernel_params_fwd['other'] = torch.randn(H, W)
    evaluate_fwd_inplace_kernel(in_out_tensor=in_out_tensor,
                                kernel_name=binary_op,
                                kernel_params=kernel_params_fwd)

@pytest.mark.parametrize("N, H, W, C", test_case_list)
@pytest.mark.parametrize("binary_op, kernel_params_fwd", binary_inplace_op_list)
def test_hpu_binary_inplace_op_broadcast_case2(N, H, W, C, binary_op, kernel_params_fwd):
    in_out_tensor = torch.randn(N, C, H, W)
    kernel_params_fwd['other'] = torch.randn(H, 1)
    evaluate_fwd_inplace_kernel(in_out_tensor=in_out_tensor,
                                kernel_name=binary_op,
                                kernel_params=kernel_params_fwd)

@pytest.mark.parametrize("N, H, W, C", test_case_list)
def test_hpu_binary_inplace_op_pow(N, H, W, C):
    kernel_params_fwd = {}
    in_out_tensor = torch.randn(N, C, H, W)
    kernel_params_fwd['exponent'] = torch.randn(N, C, H, W)
    evaluate_fwd_inplace_kernel(in_out_tensor=in_out_tensor,
                                kernel_name='pow_',
                                kernel_params=kernel_params_fwd)


@pytest.mark.parametrize("N, H, W, C", test_case_list)
@pytest.mark.parametrize("binary_op, kernel_params_fwd", binary_op_list)
def test_hpu_binary_op(N, H, W, C, binary_op, kernel_params_fwd):
    kernel_params_fwd['input'] = torch.randn(N, C, H, W)
    kernel_params_fwd['other'] = torch.randn(N, C, H, W)
    evaluate_fwd_kernel(kernel=binary_op, kernel_params=kernel_params_fwd)

@pytest.mark.parametrize("N, H, W, C", test_case_list)
@pytest.mark.parametrize("binary_op, kernel_params_fwd", binary_op_list)
def test_hpu_binary_op_broadcast_case1(N, H, W, C, binary_op, kernel_params_fwd):
    kernel_params_fwd['input'] = torch.randn(N, C, H, W)
    kernel_params_fwd['other'] = torch.randn(H, W)
    evaluate_fwd_kernel(kernel=binary_op, kernel_params=kernel_params_fwd)

@pytest.mark.parametrize("N, H, W, C", test_case_list)
@pytest.mark.parametrize("binary_op, kernel_params_fwd", binary_op_list)
def test_hpu_binary_op_broadcast_case2(N, H, W, C, binary_op, kernel_params_fwd):
    kernel_params_fwd['input'] = torch.randn(N, C, H, W)
    kernel_params_fwd['other'] = torch.randn(H, 1)
    evaluate_fwd_kernel(kernel=binary_op, kernel_params=kernel_params_fwd)


@pytest.mark.parametrize("N, H, W, C", test_case_list)
def test_hpu_binary_op_pow(N, H, W, C):
    kernel_params_fwd = {}
    kernel_params_fwd['input'] = torch.randn(N, C, H, W)
    kernel_params_fwd['exponent'] = torch.randn(N, C, H, W)
    evaluate_fwd_kernel(kernel=torch.pow, kernel_params=kernel_params_fwd)


@pytest.mark.parametrize("N, H, W, C", test_case_list)
def test_hpu_binary_op_pow_tensor_scalar(N, H, W, C):
    kernel_params_fwd = {}
    kernel_params_fwd['input'] = torch.randn(N, C, H, W)
    kernel_params_fwd['exponent'] = random.random()
    evaluate_fwd_kernel(kernel=torch.pow, kernel_params=kernel_params_fwd)

@pytest.mark.parametrize("N, H, W, C", test_case_list)
def test_hpu_binary_op_pow_scalar_tensor(N, H, W, C):
    kernel_params_fwd = {}
    kernel_params_fwd['self'] = 3.2
    kernel_params_fwd['exponent'] = torch.randn(N, C, H, W)
    evaluate_fwd_kernel(kernel=torch.pow, kernel_params=kernel_params_fwd)

if __name__ == '__main__':
    test_hpu_binary_op(*test_case_list[0], torch.add,
                       {'input': torch.ones((N, C, H, W), dtype=torch.float), 'other': torch.ones(1) / 5.0})
