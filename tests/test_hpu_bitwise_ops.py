import torch
import pytest
import random
from test_utils import evaluate_fwd_inplace_kernel, reset_seed, evaluate_fwd_kernel

test_case_list = [
    # C, H, W
    (5, 3, 4, torch.bitwise_and),
    (5, 3, 1, torch.bitwise_and),
    (5, 1, 4, torch.bitwise_and),
    (1, 3, 4, torch.bitwise_and),
    (5, 3, 4, torch.bitwise_or),
    # Or with broadcast is failing, Enable these tests after
    # https://jira.habana-labs.com/browse/SW-39944 is fixed
    # (5, 3, 1, torch.bitwise_or),
    # (5, 1, 4, torch.bitwise_or),
    # (1, 3, 4, torch.bitwise_or),
]

@pytest.mark.parametrize("C, H, W, bitwise_op", test_case_list)
def test_hpu_bitwise_op(C, H, W, bitwise_op):
    kernel_params_fwd = {}
    kernel_params_fwd["input"] = torch.randint(-10, 10, (5,3,4)) > 0
    kernel_params_fwd["other"] = torch.randint(-10, 10, (C, H, W)) > 0
    evaluate_fwd_kernel(kernel=bitwise_op, kernel_params=kernel_params_fwd)

if __name__ == "__main__":
    test_hpu_bitwise_op(*test_case_list[0])
