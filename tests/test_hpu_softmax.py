import torch
import torch.nn.functional as F
import pytest
from test_utils import evaluate_fwd_kernel, evaluate_fwd_bwd_kernel, reset_seed

mnist_test_cast_list = [
    # N, C, dim
    (64, 10, 1),
]

test_case_list = [
    # N, C, dim
    (64, 10, 0),
] + mnist_test_cast_list


@pytest.mark.parametrize("N, C, dim", test_case_list)
def test_hpu_log_softmax(N, C, dim):
    kernel_params = {'input': torch.randn(N, C),
                     'dim': dim}
    evaluate_fwd_kernel(kernel=F.log_softmax, kernel_params=kernel_params)


# @pytest.mark.xfail(reason="SW-8642")
@pytest.mark.parametrize("N, C, dim", test_case_list)
def test_hpu_log_softmax_fwd_bwd(N, C, dim):
    kernel_params = {'input': torch.randn(N, C, requires_grad=True),
                     'dim': dim}
    bwd_tensors = [torch.randn(N, C)]
    evaluate_fwd_bwd_kernel(kernel=F.log_softmax, tensor_list_bwd=bwd_tensors,
                            kernel_params_fwd=kernel_params)


if __name__ == '__main__':
    test_hpu_log_softmax(*test_case_list[0])
