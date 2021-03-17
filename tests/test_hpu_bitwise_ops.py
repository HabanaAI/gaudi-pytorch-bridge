import torch
import pytest
import random
from test_utils import evaluate_fwd_inplace_kernel, reset_seed, evaluate_fwd_kernel


bitwise_op_list = [
    # op
    torch.bitwise_and
]


@pytest.mark.parametrize("bitwise_op", bitwise_op_list)
def test_hpu_bitwise_op(bitwise_op):
    kernel_params_fwd = {}
    kernel_params_fwd["input"] = torch.randint(-10, 10, (5,3,1)) > 0
    kernel_params_fwd["other"] = torch.randint(-10, 10, (5,1,4)) > 0
    evaluate_fwd_kernel(kernel=bitwise_op, kernel_params=kernel_params_fwd)

if __name__ == "__main__":
    test_hpu_bitwise_op(torch.bitwise_and)
