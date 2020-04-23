import torch
import pytest
from test_utils import evaluate_fwd_kernel, reset_seed


# N - batch
# H - input height
# W - input width
# C - input channels
test_case_list = [
    #  N, C
    (8, 1024)
]

topk_op_list = [
    # op
    torch.topk
]

topk_values_list = [
   # k, dim
   [8, -1],
   [3, 1],
   [10, 1]
]

sort_op_list = [
    # op
    torch.sort
]

sort_values_list = [
# dim, descending
  [-1, True],
  [1, True],
]

@pytest.mark.parametrize("N, C", test_case_list)
@pytest.mark.parametrize("topk_op", topk_op_list)
@pytest.mark.parametrize("k, dim", topk_values_list)
def test_hpu_topk_op(N, C, topk_op, k, dim):
    kernel_params = {'input':torch.randn(N, C),
                     'k': k,
                     'dim': dim}
    evaluate_fwd_kernel(kernel=topk_op,
                         kernel_params=kernel_params)


@pytest.mark.parametrize("N, C", test_case_list)
@pytest.mark.parametrize("topk_op", topk_op_list)
@pytest.mark.parametrize("k, dim", topk_values_list)
def test_hpu_topk_out_op(N, C, topk_op, k, dim):
    kernel_params = {'input':torch.randn(N, C),
                     'k': k,
                     'dim': dim,
                     'out': (torch.randn(N, C), torch.empty((N, C),dtype=torch.int))}
    evaluate_fwd_kernel(kernel=topk_op,
                         kernel_params=kernel_params)

@pytest.mark.parametrize("N, C", test_case_list)
@pytest.mark.parametrize("sort_op", sort_op_list)
@pytest.mark.parametrize("dim, descending", sort_values_list)
def test_hpu_sort_op(N, C, sort_op, dim, descending):
    kernel_params = {'input':torch.randn(N, C),
                    'dim': dim,
                    'descending': descending}
    evaluate_fwd_kernel(kernel=sort_op,
                         kernel_params=kernel_params)

if __name__ == '__main__':
    test_hpu_topk_op(*test_case_list[0], topk_op_list[0], topk_values_list[0][0], topk_values_list[0][1])
    test_hpu_topk_out_op(*test_case_list[0], topk_op_list[0], topk_values_list[0][0], topk_values_list[0][1])
    test_hpu_sort_op(*test_case_list[0], sort_op_list[0], sort_values_list[0][0], sort_values_list[0][1])
