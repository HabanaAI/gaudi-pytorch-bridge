import torch
import pytest
from test_utils import evaluate_fwd_kernel, reset_seed


# N - batch
# H - input height
# W - input width
# C - input channels
test_case_list = [
    #  N, H, W, C,
    (8, 24, 21, 3,),
]

topk_op_list = [
    # op
    torch.topk
]

topk_values_list = [
   # k, dim
   [8, -1],
   [3, 2],
   [10, 3]
]

@pytest.mark.parametrize("N, H, W, C", test_case_list)
@pytest.mark.parametrize("topk_op", topk_op_list)
@pytest.mark.parametrize("k, dim", topk_values_list)
def test_hpu_topk_op(N, H, W, C, topk_op, k, dim):
    kernel_params = {'input':torch.randn(N, C, H, W),
                     'k': k,
                     'dim': dim}
    evaluate_fwd_kernel(kernel=topk_op,
                         kernel_params=kernel_params)


@pytest.mark.parametrize("N, H, W, C", test_case_list)
@pytest.mark.parametrize("topk_op", topk_op_list)
@pytest.mark.parametrize("k, dim", topk_values_list)
def test_hpu_topk_out_op(N, H, W, C, topk_op, k, dim):
    kernel_params = {'input':torch.randn(N, C, H, W),
                     'k': k,
                     'dim': dim,
                     'out': (torch.randn(N, C, H, W), torch.empty((N, C, H, W),dtype=torch.int))}
    evaluate_fwd_kernel(kernel=topk_op,
                         kernel_params=kernel_params)


if __name__ == '__main__':
    test_hpu_topk_op(*test_case_list[0], topk_op_list[0], topk_values_list[0][0], topk_values_list[0][1])
    test_hpu_topk_out_op(*test_case_list[0], topk_op_list[0], topk_values_list[0][0], topk_values_list[0][1])
