import torch
import torch.nn.functional as F
import pytest
from test_utils import evaluate_fwd_kernel, evaluate_fwd_bwd_kernel, reset_seed, compare_tensors

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

op_list = [
    # op
    (torch.addcdiv),
]

values_list = [
    # value
    1.0,
    5.0,
    10.0,
]

@pytest.mark.parametrize("N, H, W, C", test_case_list)
@pytest.mark.parametrize("binary_op", op_list)
@pytest.mark.parametrize("value", values_list)
def test_hpu_addcdiv_op(N, H, W, C, binary_op, value):
    kernel_params = {'input': torch.randn(N, C, H, W),
                     'tensor1': torch.randn(N, C, H, W),
                     'tensor2': torch.randn(N, C, H, W),
                     'value': value}
    evaluate_fwd_kernel(kernel=binary_op, kernel_params=kernel_params)

@pytest.mark.parametrize("N, H, W, C", test_case_list)
@pytest.mark.parametrize("binary_op", op_list)
@pytest.mark.parametrize("value", values_list)
def test_hpu_addcdiv_op_fwd_bwd(N, H, W, C, binary_op,value):
    kernel_params_fwd = {'input': torch.randn(N, C, H, W, requires_grad=True),
                         'tensor1': torch.randn(N, C, H, W,requires_grad=True),
                         'tensor2': torch.randn(N, C, H, W,requires_grad=True),
                         'value': value}
    bwd_tensors = [torch.randn(N, C, H, W)]
    evaluate_fwd_bwd_kernel(kernel=binary_op, tensor_list_bwd=bwd_tensors,
                            kernel_params_fwd=kernel_params_fwd)


@pytest.mark.parametrize("N, H, W, C", test_case_list)
@pytest.mark.parametrize("value", values_list)
def test_hpu_addciv_inplace_op(N, H, W, C,value):
    input   = torch.randn(N, C, H, W)
    tensor1 = torch.randn(N, C, H, W)
    tensor2 = torch.randn(N, C, H, W)

    hpu = torch.device('hpu')

    hpu_tensor_input = input.to(hpu)
    hpu_tensor1 = tensor1.to(hpu)
    hpu_tensor2 = tensor2.to(hpu)

    output = input.addcdiv_(tensor1,tensor2,value=value)
    hpu_tensor_output = hpu_tensor_input.addcdiv_( hpu_tensor1, hpu_tensor2, value=value)

    compare_tensors(hpu_tensor_output, output, atol=0.001, rtol=1.e-3)


if __name__ == '__main__':
    test_hpu_addcdiv_op(*test_case_list[0], *op_list[0], *values_list[0])
    test_hpu_addcdiv_op_fwd_bwd(*test_case_list[0], *op_list[0], *values_list[0])
    test_hpu_addciv_inplace_op(*test_case_list[0], *values_list[0])
