import pytest
import torch
import torch.nn as nn
from test_utils import evaluate_fwd_bwd_kernel, evaluate_fwd_kernel

mnist_test_cast_list = [
    # N, C, K
    (64, 450, 500),
    (64, 500, 10),
]

# Multiply matrices NxC * CxK = NxK
test_case_list = [
    # N, C, K
    # (10, 20, 30),
    (800, 500, 10),
] + mnist_test_cast_list

# mat - NxHxW, mat2 - NxWxC, out - NxHxC
test_case_list_bmm = [
    # N, H, W, C
    (8, 24, 3, 10)
]

# mat - NxC, mat2 - C, out - N
test_case_list_mv = [
    # N, C
    (8, 10)
]

# mat - C, mat2 - C, out - 1
test_case_list_dot = [
    # C
    (10)
]

data_type_list = [(torch.float, 0.001)]


@pytest.mark.xfail(reason="Results mismatch")
@pytest.mark.parametrize("N, C, K", test_case_list)
@pytest.mark.parametrize("dtype, tol", data_type_list)
def test_hpu_linear(N, C, K, dtype, tol):
    kernel = nn.Linear(in_features=C, out_features=K, bias=True).to(dtype)
    kernel_params = {"input": torch.randn(N, C).to(dtype)}
    evaluate_fwd_kernel(kernel=kernel, kernel_params=kernel_params, atol=tol, rtol=tol)


@pytest.mark.xfail(reason="AttributeError")
@pytest.mark.parametrize("N, C, K", test_case_list)
@pytest.mark.parametrize("dtype, tol", data_type_list)
def test_hpu_linear_fwd_bwd(N, C, K, dtype, tol):
    kernel = nn.Linear(in_features=C, out_features=K, bias=True).to(dtype)
    kernel_params_fwd = {"input": torch.randn(N, C).to(dtype)}
    bwd_tensors = [torch.randn(N, K).to(dtype)]
    evaluate_fwd_bwd_kernel(
        kernel=kernel,
        kernel_params_fwd=kernel_params_fwd,
        tensor_list_bwd=bwd_tensors,
        atol=tol,
        rtol=tol,
    )


@pytest.mark.xfail(reason="Results mismatch")
@pytest.mark.parametrize("N, C, K", test_case_list)
def test_hpu_linear_no_bias(N, C, K):
    kernel = nn.Linear(in_features=C, out_features=K, bias=False)
    kernel_params = {"input": torch.randn(N, C)}
    evaluate_fwd_kernel(kernel=kernel, kernel_params=kernel_params)


@pytest.mark.xfail(reason="Not equal to tolerance")
@pytest.mark.parametrize("N, C, K", test_case_list)
def test_hpu_linear_no_bias_fwd_bwd(N, C, K):
    kernel = nn.Linear(in_features=C, out_features=K, bias=False)
    kernel_params_fwd = {"input": torch.randn(N, C)}
    bwd_tensors = [torch.randn(N, K)]
    evaluate_fwd_bwd_kernel(kernel=kernel, kernel_params_fwd=kernel_params_fwd, tensor_list_bwd=bwd_tensors)


@pytest.mark.xfail
@pytest.mark.parametrize("N, H, W, C", test_case_list_bmm)
def test_hpu_linear_bmm(N, H, W, C):
    kernel_params_fwd = {
        "input": torch.randn((N, H, W), requires_grad=True),
        "mat2": torch.randn((N, W, C), requires_grad=True),
    }
    bwd_tensors = [torch.randn((N, H, C))]
    evaluate_fwd_bwd_kernel(
        kernel=torch.bmm,
        kernel_params_fwd=kernel_params_fwd,
        tensor_list_bwd=bwd_tensors,
    )


# Functions with out = arguments don't support automatic differentiation
@pytest.mark.xfail
@pytest.mark.parametrize("N, H, W, C", test_case_list_bmm)
def test_hpu_linear_bmm_out(N, H, W, C):
    kernel_params = {
        "input": torch.randn((N, H, W), requires_grad=False),
        "mat2": torch.randn((N, W, C), requires_grad=False),
        "out": torch.empty((N, H, C), requires_grad=False),
    }
    evaluate_fwd_kernel(kernel=torch.bmm, kernel_params=kernel_params)


@pytest.mark.parametrize("N, C", test_case_list_mv)
@pytest.mark.parametrize("op", [torch.mv])
def test_hpu_linear_mv(N, C, op):
    kernel_params = {"input": torch.randn(N, C), "vec": torch.randn(C)}
    evaluate_fwd_kernel(kernel=op, kernel_params=kernel_params)


@pytest.mark.parametrize("C", test_case_list_dot)
@pytest.mark.parametrize("op", [torch.dot])
def test_hpu_linear_dot(C, op):
    kernel_params = {"input": torch.randn(C), "tensor": torch.randn(C)}
    evaluate_fwd_kernel(kernel=op, kernel_params=kernel_params)


if __name__ == "__main__":
    test_hpu_linear_no_bias(*test_case_list[-1])
