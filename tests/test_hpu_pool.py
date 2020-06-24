import torch
import torch.nn.functional as F
import pytest
from test_utils import evaluate_fwd_kernel, evaluate_fwd_bwd_kernel, reset_seed, compare_tensors
from numpy import floor

# N - batch
# H - input height
# W - input width
# C - input channels
# R - filter height
# S - filter width
# str - stride
# pad - padding
# type - pooltype
# inpad - include_count_pad
mnist_test_case_list = [
    # N, H, W, C, R, S, str_H, str_W, padding, type, inpad
    (64, 24, 24, 20, 3, 3, 2, 2, 0, 'maxpool2d', False),
    (64, 7, 7, 50, 3, 3, 2, 2, 0, 'maxpool2d', False)
]

resnet50_test_case_list = [
    # N, H, W, C, R, S, str_H, str_W, padding, type, inpad
    (64, 112, 112, 64, 3, 3, 2, 2, 1, 'maxpool2d', False),
    # Note: TPC does not support any other configuration as of now
    (8, 7, 7, 2048, 7, 7, 7, 7, 0, 'avgpool2d', False),
    (8, 7, 7, 2048, 7, 7, 7, 7, 0, 'avgpool2d', True)
]

pool_test_case_list = [
    # N, H, W, C, R, S, str_H, str_W, padding, type, inpad
    (8, 27, 27, 3, 3, 3, 2, 2, 0, 'maxpool2d', False),
    pytest.param(2, 8, 8, 50, 2, 2, 2, 2, 0, False, 'maxpool2d', marks=pytest.mark.xfail(
        reason="only 3x3 window with 2x2 stride is supported")),
] + mnist_test_case_list + resnet50_test_case_list

data_type_list = [
  pytest.param(torch.bfloat16, 0.001, marks=pytest.mark.xfail(
        reason="max_pool2d_with_indices_cpu not implemented for BFloat16")),
  (torch.float, 0.001)
]

def output_size(spatial_size, pad, dilation, kernel_size, stride):
    return int(floor((spatial_size + 2 * pad - dilation * (kernel_size - 1) - 1) / stride + 1))


@pytest.mark.parametrize("N, H, W, C, R, S, str_H, str_W, padding, type, inpad", pool_test_case_list)
@pytest.mark.parametrize("dtype, tol", data_type_list)
def test_hpu_pool(N, H, W, C, R, S, str_H, str_W, padding, type, inpad, dtype, tol):
    # TODO: extend that test to all features
    kernel_params = {
        'input': torch.randn(N, C, H, W).to(dtype),
        'kernel_size': [R, S],
        'stride': [str_H, str_W],
        'padding': padding
    }
    if (type == 'maxpool2d'):
         kernel = F.max_pool2d
    else:
         kernel_params['count_include_pad'] = inpad
         kernel = F.avg_pool2d

    # don't check resuluts because indices can have different values
    hpu_result, cpu_result = evaluate_fwd_kernel(
        kernel=kernel, kernel_params=kernel_params, check_results=False)
    compare_tensors(hpu_result[0], cpu_result[0], atol=tol, rtol=tol)


@pytest.mark.parametrize("N, H, W, C, R, S, str_H, str_W, padding, type, inpad", pool_test_case_list)
@pytest.mark.parametrize("dtype, tol", data_type_list)
def test_hpu_pool_fwd_bwd(N, H, W, C, R, S, str_H, str_W, padding, type, inpad, dtype, tol):
    # TODO: extend that test to all features
    kernel_params_fwd = {
        'input': torch.randn(N, C, H, W, requires_grad=True).to(dtype),
        'kernel_size': [R, S],
        'stride': [str_H, str_W],
        'padding':padding
    }
    if (type == 'maxpool2d'):
         kernel = F.max_pool2d
    else:
         kernel_params_fwd['count_include_pad'] = inpad
         kernel = F.avg_pool2d

    bwd_tensors = [torch.randn(N, C, output_size(H, padding, 1, R, str_H), output_size(W, padding, 1, S, str_W)).to(dtype)]
    # don't check fwd resuluts because indices can have different values
    (hpu_result_fwd, _), (cpu_result_fwd, _) = evaluate_fwd_bwd_kernel(kernel=kernel, tensor_list_bwd=bwd_tensors,
                                                                       kernel_params_fwd=kernel_params_fwd, check_results_fwd=False)
    compare_tensors(hpu_result_fwd[0], cpu_result_fwd[0], atol=tol, rtol=tol)

@pytest.mark.parametrize("N, H, W, C, R, S, str_H, str_W, padding, type, inpad", pool_test_case_list)
def test_hpu_chlast_pool(N, H, W, C, R, S, str_H, str_W, padding, type, inpad):
    # TODO: extend that test to all features
    in_tensor = torch.randn(N, C, H, W)
    kernel_params = {
        'input': in_tensor.contiguous(memory_format=torch.channels_last),
        'kernel_size': [R, S],
        'stride': [str_H, str_W],
        'padding': padding
    }
    if (type == 'maxpool2d'):
         kernel = F.max_pool2d
    else:
         kernel_params['count_include_pad'] = inpad
         kernel = F.avg_pool2d

    # don't check resuluts because indices can have different values
    hpu_result, cpu_result = evaluate_fwd_kernel(
        kernel=kernel, kernel_params=kernel_params, check_results=False)
    compare_tensors(hpu_result[0], cpu_result[0], atol=0.001, rtol=1.e-3)


@pytest.mark.parametrize("N, H, W, C, R, S, str_H, str_W, padding, type, inpad", pool_test_case_list)
def test_hpu_pool_chlast_fwd_bwd(N, H, W, C, R, S, str_H, str_W, padding, type, inpad):
    # TODO: extend that test to all features
    in_tensor = torch.randn(N, C, H, W, requires_grad=True)
    kernel_params_fwd = {
        'input': in_tensor.contiguous(memory_format=torch.channels_last),
        'kernel_size': [R, S],
        'stride': [str_H, str_W],
        'padding':padding
    }
    if (type == 'maxpool2d'):
         kernel = F.max_pool2d
    else:
         kernel_params_fwd['count_include_pad'] = inpad
         kernel = F.avg_pool2d

    bwd_tensor = torch.randn(N, C, output_size(H, padding, 1, R, str_H), output_size(W, padding, 1, S, str_W))
    bwd_tensors = [bwd_tensor.contiguous(memory_format=torch.channels_last)]
    # don't check fwd resuluts because indices can have different values
    (hpu_result_fwd, _), (cpu_result_fwd, _) = evaluate_fwd_bwd_kernel(kernel=kernel, tensor_list_bwd=bwd_tensors,
                                                                       kernel_params_fwd=kernel_params_fwd, check_results_fwd=False)
    compare_tensors(hpu_result_fwd[0], cpu_result_fwd[0], atol=0.001, rtol=1.e-3)

if __name__ == '__main__':
    test_hpu_pool_fwd_bwd(*resnet50_test_case_list[1])
