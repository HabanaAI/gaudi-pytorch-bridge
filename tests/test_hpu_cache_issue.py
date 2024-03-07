import pytest
import torch
import torch.nn.functional as F
from test_utils import compare_tensors, hpu

test_case_list = [(9, 13, 11, 9)]


@pytest.mark.parametrize("N, H, W, C", test_case_list)
def test_hpu_pad_op_cache_issue(N, H, W, C):
    in_tensor = torch.randn(N, C, H, W)

    pad1 = (-1, -1, -1, -1)
    cpu_out_pad_tensor_1 = F.pad(in_tensor, pad1, "constant", 0)
    hpu_out_pad_tensor_1 = F.pad(in_tensor.to(hpu), pad1, "constant", 0)
    compare_tensors(hpu_out_pad_tensor_1, cpu_out_pad_tensor_1, atol=0, rtol=0)

    pad2 = (-2, -2, -2, -2)
    cpu_out_pad_tensor_2 = F.pad(in_tensor, pad2, "constant", 0)
    hpu_out_pad_tensor_2 = F.pad(in_tensor.to(hpu), pad2, "constant", 0)
    compare_tensors(hpu_out_pad_tensor_2, cpu_out_pad_tensor_2, atol=0, rtol=0)


@pytest.mark.xfail(reason="RuntimeError: synNodeCreateWithId failed")
@pytest.mark.parametrize("N, H, W, C", test_case_list)
def test_hpu_cumsum_op_cache_issue(N, H, W, C):
    in_tensor = torch.randn(N, C, H, W)

    dim1 = 1
    cpu_out_cumsum_tensor_1 = torch.cumsum(in_tensor, dim1)
    hpu_out_cumsum_tensor_1 = torch.cumsum(in_tensor.to(hpu), dim1)
    compare_tensors(hpu_out_cumsum_tensor_1, cpu_out_cumsum_tensor_1, atol=0.001, rtol=0.001)

    dim2 = -1
    cpu_out_cumsum_tensor_2 = torch.cumsum(in_tensor, dim2)
    hpu_out_cumsum_tensor_2 = torch.cumsum(in_tensor.to(hpu), dim2)
    compare_tensors(hpu_out_cumsum_tensor_2, cpu_out_cumsum_tensor_2, atol=0.001, rtol=0.001)


if __name__ == "__main__":
    test_hpu_pad_op_cache_issue(*test_case_list[0])
    test_hpu_cumsum_op_cache_issue(*test_case_list[0])
