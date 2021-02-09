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


compare_op_list = [
    # op, op params dict
    (torch.eq, {}),
    (torch.lt, {}),
    (torch.ge, {}),
    (torch.ne, {})
]

compare_op_out_list_bool = [
    # op, op params dict
    (torch.eq, {}),
]


@pytest.mark.parametrize("N, H, W, C", test_case_list)
@pytest.mark.parametrize("compare_op, kernel_params_fwd", compare_op_out_list_bool)
def test_hpu_compare_op_out_bool(N, H, W, C, compare_op, kernel_params_fwd):
    kernel_params_fwd["input"] = torch.randn(N, C, H, W)
    kernel_params_fwd["other"] = torch.randn(N, C, H, W)
    kernel_params_fwd["out"] = torch.empty((N, C, H, W), dtype=torch.bool)
    evaluate_fwd_kernel(kernel=compare_op, kernel_params=kernel_params_fwd)


@pytest.mark.parametrize("N, H, W, C", test_case_list)
@pytest.mark.parametrize("compare_op, kernel_params_fwd", compare_op_list)
def test_hpu_compare_op(N, H, W, C, compare_op, kernel_params_fwd):
    kernel_params_fwd["input"] = torch.randn(N, C, H, W)
    kernel_params_fwd["other"] = torch.randn(N, C, H, W)
    evaluate_fwd_kernel(kernel=compare_op, kernel_params=kernel_params_fwd)

@pytest.mark.parametrize("N, H, W, C", test_case_list)
@pytest.mark.parametrize("compare_op, kernel_params_fwd", compare_op_list)
def test_hpu_compare_op_broadcast_case1(N, H, W, C, compare_op, kernel_params_fwd):
    kernel_params_fwd["input"] = torch.randn(N, C, H, W)
    kernel_params_fwd["other"] = torch.randn(H, W)
    evaluate_fwd_kernel(kernel=compare_op, kernel_params=kernel_params_fwd)


@pytest.mark.parametrize("N, H, W, C", test_case_list)
@pytest.mark.parametrize("compare_op, kernel_params_fwd", compare_op_list)
def test_hpu_compare_op_broadcast_case2(N, H, W, C, compare_op, kernel_params_fwd):
    kernel_params_fwd["input"] = torch.randn(N, C, H, W)
    kernel_params_fwd["other"] = torch.randn(H, 1)
    evaluate_fwd_kernel(kernel=compare_op, kernel_params=kernel_params_fwd)


@pytest.mark.parametrize("N, H, W, C", test_case_list)
@pytest.mark.parametrize("compare_op, kernel_params_fwd", compare_op_list)
def test_hpu_compare_op_scalar(N, H, W, C, compare_op, kernel_params_fwd):
    kernel_params_fwd = {}
    kernel_params_fwd["input"] = torch.randn(N, C, H, W)
    kernel_params_fwd["other"] = 0.5
    evaluate_fwd_kernel(kernel=compare_op, kernel_params=kernel_params_fwd)


if __name__ == "__main__":
    test_hpu_compare_op(*test_case_list[0], compare_op_list[0])
