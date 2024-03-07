import numpy as np
import pytest
import torch
from test_utils import cpu, evaluate_fwd_inplace_kernel, evaluate_fwd_kernel, hpu

# N - batch
# H - input height
# W - input width
# C - input channels
# 1, 2, 3, 4
test_case_list = [
    # N, H, W, C
    (8, 224, 224, 3),
]

test_case_t_list = [
    # H, W
    (2, 3),
]

t_op_list = [
    # op, op params dict
    (torch.transpose, {"input": torch.randn((8, 224, 224, 3)), "dim0": 0, "dim1": 3}),
    (
        torch.transpose,
        {
            "input": torch.randint(1, 24, (2, 3, 4), dtype=torch.int),
            "dim0": 1,
            "dim1": 2,
        },
    ),
    (
        torch.transpose,
        {
            "input": torch.randint(1, 24, (1, 3, 2), dtype=torch.int),
            "dim0": -1,
            "dim1": -3,
        },
    ),
    (torch.transpose, {"input": torch.randn(3, 4), "dim0": 1, "dim1": 0}),
    # Disable 1D tensor SW-12421
    # (torch.transpose, {'input': torch.randn(2), 'dim0': 0, 'dim1': 0}),
]

transpose_inplace_op_list = [
    ("transpose_", {}),
]

t_inplace_op_list = [("t_",)]


@pytest.mark.parametrize("t_op, kernel_params_fwd", t_op_list)
def test_hpu_transpose(t_op, kernel_params_fwd):
    evaluate_fwd_kernel(kernel=t_op, kernel_params=kernel_params_fwd)


@pytest.mark.parametrize("N, H, W, C", test_case_list)
@pytest.mark.parametrize("transpose_inplace_op, kernel_params_fwd", transpose_inplace_op_list)
def test_hpu_transpose_inplace(N, H, W, C, transpose_inplace_op, kernel_params_fwd):
    in_out_tensor = torch.randn(N, C, H, W)
    kernel_params_fwd = {"dim0": 0, "dim1": 2}
    evaluate_fwd_inplace_kernel(
        in_out_tensor=in_out_tensor,
        kernel_name=transpose_inplace_op,
        kernel_params=kernel_params_fwd,
    )
    in_out_tensor = torch.randn(N, C, H, W)
    kernel_params_fwd = {"dim0": -1, "dim1": -2}
    evaluate_fwd_inplace_kernel(
        in_out_tensor=in_out_tensor,
        kernel_name=transpose_inplace_op,
        kernel_params=kernel_params_fwd,
    )
    in_out_tensor = torch.randn(H, W)
    kernel_params_fwd = {"dim0": 0, "dim1": 1}
    evaluate_fwd_inplace_kernel(
        in_out_tensor=in_out_tensor,
        kernel_name=transpose_inplace_op,
        kernel_params=kernel_params_fwd,
    )


@pytest.mark.parametrize("H, W", test_case_t_list)
@pytest.mark.parametrize("t_inplace_op", t_inplace_op_list)
def test_hpu_t_inplace(H, W, t_inplace_op):
    in_out_tensor = torch.randint(1, 24, (H, W), dtype=torch.int)
    evaluate_fwd_inplace_kernel(in_out_tensor=in_out_tensor, kernel_name="t_", kernel_params=None)
    in_out_tensor = torch.randint(1, 24, (1,), dtype=torch.int)
    evaluate_fwd_inplace_kernel(in_out_tensor=in_out_tensor, kernel_name="t_", kernel_params=None)


@pytest.mark.parametrize("N, H, W, C", test_case_list)
def test_hpu_transpose_of_transpose(N, H, W, C):
    in_out_tensor = torch.randn(1, 2, 3, 4)
    hputensor = in_out_tensor.to(hpu)
    thputensor = torch.transpose(hputensor, 0, 2)
    tthputensor = torch.transpose(thputensor, 0, 2)
    is_eq_tensor = torch.eq(hputensor, tthputensor)
    print(is_eq_tensor.to(cpu))


# @torch.jit.script
@pytest.mark.parametrize("N, H, W, C", test_case_list)
def test_hpu_permute(N, H, W, C):

    in_tensor = torch.randn(N, C, H, W)
    hpu_result = in_tensor.to(hpu).permute((0, 2, 3, 1)).to(cpu)
    cpu_result = in_tensor.to(cpu).permute((0, 2, 3, 1))
    np.testing.assert_allclose(
        hpu_result.detach().numpy(),
        cpu_result.detach().numpy(),
        atol=0.001,
        rtol=1.0e-3,
    )

    hpu_result = in_tensor.to(hpu).permute((0, 3, 1, 2)).to(cpu)
    cpu_result = in_tensor.to(cpu).permute((0, 3, 1, 2))
    np.testing.assert_allclose(
        hpu_result.detach().numpy(),
        cpu_result.detach().numpy(),
        atol=0.001,
        rtol=1.0e-3,
    )

    in_tensor = torch.randn(N, H, W)
    hpu_result = in_tensor.to(hpu).permute((2, 0, 1)).to(cpu)
    cpu_result = in_tensor.to(cpu).permute((2, 0, 1))
    np.testing.assert_allclose(
        hpu_result.detach().numpy(),
        cpu_result.detach().numpy(),
        atol=0.001,
        rtol=1.0e-3,
    )


if __name__ == "__main__":
    test_hpu_transpose(*t_op_list[0])
    test_hpu_t_inplace(*test_case_t_list[0], t_inplace_op_list[0])
    test_hpu_transpose_inplace(*test_case_list[0], *transpose_inplace_op_list[0])
    test_hpu_transpose_of_transpose(*test_case_list[0])
    test_hpu_permute(*test_case_list[0])
