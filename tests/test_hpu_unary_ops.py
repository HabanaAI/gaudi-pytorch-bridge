import torch
import torch.nn.functional as F
import pytest
from test_utils import evaluate_fwd_kernel, evaluate_fwd_bwd_kernel, reset_seed, evaluate_fwd_inplace_kernel


# N - batch
# H - input height
# W - input width
# C - input channels
mnist_test_cast_list = [
    # N, H, W, C
    (64, 24, 24, 20),
    (64, 7, 7, 50),
]

test_case_list = [
    #  N, H, W, C,
    (8, 24, 24, 3,),
] + mnist_test_cast_list

unary_op_list = [
    F.relu,
    torch.tanh,
    torch.nn.functional.gelu,
    torch.norm,
    torch.sigmoid,
    torch.sqrt,
    torch.reciprocal,
    torch.floor,
    torch.round,
    torch.rsqrt,
]

unary_inplace_op_list = [
    ('relu_'),
    ('tanh_'),
    ('erf_'),
    ('exp_'),
    ('reciprocal_'),
    ('floor_'),
    ('round_'),
    ('rsqrt_'),
]

unary_op_out_list = [
    # op, op params dict
    (torch.tanh, {}),
    (torch.reciprocal, {}),
    (torch.neg, {}),
]

data_type_list = [
    (torch.float, 0.001)
]


@pytest.mark.parametrize("N, H, W, C", test_case_list)
@pytest.mark.parametrize("unary_op", unary_op_list)
@pytest.mark.parametrize("dtype, tol", data_type_list)
def test_hpu_unary_op(N, H, W, C, unary_op, dtype, tol):
    if unary_op == torch.norm:
        kernel_params = {'input': torch.randn(N, C, H, W).to(dtype), 'p': 6.0}
        evaluate_fwd_kernel(kernel=unary_op, kernel_params=kernel_params, atol=tol, rtol=tol)
    else:
        kernel_params = {}
        if unary_op == torch.rsqrt:
            kernel_params = {'input': torch.add(torch.rand(N, C, H, W, requires_grad=True),1).to(dtype)}
        else:
            kernel_params = {'input': torch.randn(N, C, H, W, requires_grad=True).to(dtype)}
        evaluate_fwd_kernel(kernel=unary_op, kernel_params=kernel_params, atol=tol, rtol=tol)


@pytest.mark.parametrize("N, H, W, C", test_case_list)
@pytest.mark.parametrize("unary_op", unary_op_list)
@pytest.mark.parametrize("dtype, tol", data_type_list)
def test_hpu_unary_op_fwd_bwd(N, H, W, C, unary_op, dtype, tol):
    if unary_op == torch.norm:
        kernel_params_fwd = {'input': torch.randn(N, C, H, W, requires_grad=True).to(dtype),
                            'p': 6.0}
        bwd_tensors = [torch.tensor(1).to(dtype)]
        evaluate_fwd_bwd_kernel(kernel=unary_op, tensor_list_bwd=bwd_tensors,
                            kernel_params_fwd=kernel_params_fwd, atol=tol, rtol=tol)
    else:
        kernel_params_fwd = {}
        if unary_op == torch.rsqrt:
            kernel_params_fwd = {'input': torch.add(torch.rand(N, C, H, W, requires_grad=True),1).to(dtype)}
        else:
            kernel_params_fwd = {'input': torch.randn(N, C, H, W, requires_grad=True).to(dtype)}
        # TODO: extend that test to all features
        bwd_tensors = [torch.randn(N, C, H, W).to(dtype)]
        evaluate_fwd_bwd_kernel(kernel=unary_op, tensor_list_bwd=bwd_tensors,
                                kernel_params_fwd=kernel_params_fwd, atol=tol, rtol=tol)


@pytest.mark.parametrize("N, H, W, C", test_case_list)
@pytest.mark.parametrize("dtype, tol", data_type_list)
def test_hpu_gelu_op_fwd_bwd(N, H, W, C, dtype, tol):
    kernel_params_fwd = {'input': torch.randn(N, C, H, W, requires_grad=True).to(dtype)}
    bwd_tensors = [torch.randn(N, C, H, W).to(dtype)]
    evaluate_fwd_bwd_kernel(kernel=torch.nn.functional.gelu, tensor_list_bwd=bwd_tensors,
                            kernel_params_fwd=kernel_params_fwd, atol=.003, rtol=.003, grad_on_grad_enable=False)


@pytest.mark.parametrize("N, H, W, C", test_case_list)
@pytest.mark.parametrize("unary_inplace_op", unary_inplace_op_list)
def test_hpu_unary_inplace_op(N, H, W, C, unary_inplace_op):
    in_out_tensor = torch.randn(N, C, H, W)
    evaluate_fwd_inplace_kernel(in_out_tensor=in_out_tensor, kernel_name=unary_inplace_op, kernel_params=None)


@pytest.mark.parametrize("N, H, W, C", test_case_list)
@pytest.mark.parametrize("unary_op, kernel_params_fwd", unary_op_out_list)
def test_hpu_binary_op_out_intype(N, H, W, C, unary_op, kernel_params_fwd):
    kernel_params_fwd['input'] = inT = torch.randn(N, C, H, W)
    kernel_params_fwd['out'] = torch.empty((N, C, H, W))
    evaluate_fwd_kernel(kernel=unary_op, kernel_params=kernel_params_fwd)


@pytest.mark.parametrize("N, H, W, C", test_case_list)
@pytest.mark.parametrize("lp_norm_op", [torch.norm])
@pytest.mark.parametrize("value", [11.0, 6.0])
def test_hpu_lp_norm_op_fwd_bwd(N, H, W, C, lp_norm_op, value):
    kernel_params_fwd = {'input': torch.randn(N, C, H, W, requires_grad=True, dtype=torch.float),
                         'p': value}
    bwd_tensors = [torch.tensor(1, dtype=torch.float)]
    evaluate_fwd_bwd_kernel(kernel=lp_norm_op, tensor_list_bwd=bwd_tensors,
                            kernel_params_fwd=kernel_params_fwd)

@pytest.mark.parametrize("N, H, W, C", test_case_list)
@pytest.mark.parametrize("unary_op", [torch.erf, torch.exp])
@pytest.mark.parametrize("dtype, tol", data_type_list)
def test_hpu_unary_op_erf(N, H, W, C, unary_op, dtype, tol):
    kernel_params = {'input': torch.randn(N, C, H, W).to(dtype)}
    evaluate_fwd_kernel(kernel=unary_op, kernel_params=kernel_params, atol=tol, rtol=tol)

@pytest.mark.parametrize("N, H, W, C", test_case_list)
def test_hpu_unary_op_clamp(N, H, W, C):
    kernel_params = {'input': torch.randn(N, C, H, W),
                     'min': -0.25,
                     'max': 0.25}
    evaluate_fwd_kernel(kernel=torch.clamp, kernel_params=kernel_params)
    evaluate_fwd_kernel(kernel=torch.clamp, kernel_params=kernel_params)

@pytest.mark.parametrize("N, H, W, C", test_case_list)
def test_hpu_unary_op_clamp_inplace(N, H, W, C):
    in_out_tensor = torch.randn(N, C, H, W)
    kernel_params = {'min': -0.25,
                     'max': 0.25}
    evaluate_fwd_inplace_kernel(in_out_tensor=in_out_tensor, kernel_name='clamp_', kernel_params=kernel_params)
    evaluate_fwd_inplace_kernel(in_out_tensor=in_out_tensor, kernel_name='clamp_', kernel_params=kernel_params)

if __name__ == '__main__':
    test_hpu_unary_op(*test_case_list[0], unary_op_list[0])
    test_hpu_unary_op_fwd_bwd(*test_case_list[0], unary_op_list[0])
    test_hpu_unary_inplace_op(*test_case_list[0], unary_inplace_op_list[0])
    test_hpu_binary_op_out_intype(*test_case_list[0], unary_op_out_list[0])
