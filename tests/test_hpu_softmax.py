import torch
import torch.nn.functional as F
import pytest
from test_utils import evaluate_fwd_kernel, evaluate_fwd_bwd_kernel, reset_seed, compare_tensors

mnist_test_cast_list = [
    # N, C, dim
    (64, 10, 1),
]

test_case_list = [
    # N, C, dim
    (64, 10, 0),
    (64, 10, 1),
    (64, 10, -1),
    (2,  3,  -2),
] + mnist_test_cast_list

op_list = [
    # op, op params dict
    (F.log_softmax),
    (F.softmax),
]

@pytest.mark.parametrize("N, C, dim", test_case_list)
@pytest.mark.parametrize("kernel_op", op_list)
def test_hpu_log_softmax(N, C, kernel_op, dim):
    kernel_params = {'input': torch.randn(N, C),
                     'dim': dim}
    evaluate_fwd_kernel(kernel=kernel_op, kernel_params=kernel_params)


@pytest.mark.parametrize("N, C, dim", test_case_list)
@pytest.mark.parametrize("kernel_op", op_list)
def test_hpu_log_softmax_fwd_bwd(N, C, kernel_op, dim):
    kernel_params = {'input': torch.randn(N, C, requires_grad=True),
                     'dim': dim}
    bwd_tensors = [torch.randn(N, C)]
    evaluate_fwd_bwd_kernel(kernel=kernel_op, tensor_list_bwd=bwd_tensors,
                            kernel_params_fwd=kernel_params)

@pytest.mark.parametrize("N, C, dim", test_case_list)
@pytest.mark.parametrize("dtype", [torch.int])
def test_hpu_log_softmax_int(N, C, dim,dtype):
    input = torch.randint(low=0, high=C - 1, size=(N,C),dtype=dtype)
    hpu = torch.device('habana')
    hpu_tensor_in = input.to(hpu)
    output = F.softmax(input,dim,3,torch.float)
    hpu_output = hpu_tensor_in.softmax(dim,dtype)
    compare_tensors(hpu_output, output, atol=0.001, rtol=1.e-3)


if __name__ == '__main__':
    test_hpu_log_softmax_fwd_bwd(*test_case_list[0])
    test_hpu_log_softmax_int(*test_case_list[0])
