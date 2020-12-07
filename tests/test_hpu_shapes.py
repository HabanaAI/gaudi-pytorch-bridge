import torch
import numpy as np
import pytest
from test_utils import reset_seed, compare_tensors, evaluate_fwd_kernel, evaluate_fwd_bwd_kernel

# N - batch
# H - input height
# W - input width
# C - input channels
mnist_test_case_list = [
    # N, H, W, C
    (64, 3, 3, 50),
]

test_case_list = [
    # N, H, W, C
    (8, 28, 28, 3),
]

broadcast_test_case_list = [
    [torch.randn(8, 3, 28, 28), torch.randn(1), torch.randn(1)],
    [torch.randn(1, 4), torch.randn(3, 1), torch.randn(1)],
    [torch.randn(1, 4), torch.randn(3, 1), torch.randn(2, 1, 1)]
]

arange_test_case_list = [
    #start, end, step, dtype
    (0.0, 10.0, 2.0, torch.float),
    (1, 16, 2, torch.int32),
    (20, 40, 5, torch.long),
    (0, -10, -2, torch.long),

]

test_case_scatter_add = [
    # N, H, I, S
    (512, 768, 1, 512),
]


# @torch.jit.script
@pytest.mark.parametrize("N, H, W, C", test_case_list)
def test_hpu_view(N, H, W, C):
    hpu = torch.device('habana')
    cpu = torch.device('cpu')

    in_tensor = torch.randn(N, C, H, W)

    hpu_result = in_tensor.to(hpu).view(-1, C * H * W)
    cpu_result = in_tensor.to(cpu).view(-1, C * H * W)
    compare_tensors(hpu_result, cpu_result, atol=0.001, rtol=1.e-3)


@pytest.mark.parametrize("N, H, W, C", test_case_list)
def test_hpu_slice_and_select(N, H, W, C):
    hpu = torch.device('habana')
    cpu = torch.device('cpu')

    in_tensor = torch.randn(N, C, H, W)

    hpu_result = in_tensor.to(hpu)[:, 0, 0:4:2, 0:4]
    cpu_result = in_tensor.to(cpu)[:, 0, 0:4:2, 0:4]
    compare_tensors(hpu_result, cpu_result, atol=0, rtol=0)


@pytest.mark.parametrize("N, H, W, C", test_case_list)
@pytest.mark.parametrize("dim", [0, 1, 2, 3])
def test_hpu_index_select(N, H, W, C, dim):
    kernel = torch.index_select

    dim_list = [N, C, H, W]
    kernel_params_fwd = {
        'input': torch.randn(tuple(dim_list), requires_grad=True),
        'dim': dim,
        'index': torch.tensor([0, 2]),
    }

    dim_list[dim] = 2
    bwd_tensors = [torch.randn(tuple(dim_list))]
    evaluate_fwd_bwd_kernel(kernel=kernel, tensor_list_bwd=bwd_tensors, kernel_params_fwd=kernel_params_fwd)


@pytest.mark.parametrize("N, H, W, C", test_case_list)
@pytest.mark.parametrize("acc", [True, False])
def test_hpu_index_put(N, H, W, C, acc):
    kernel = torch.index_put

    dim_list = [N, C, H, W]
    dim_list_tensor = [N, C, H, W]
    dim_list_tensor[0] = 2
    kernel_params_fwd = {
        'input': torch.randn(tuple(dim_list), requires_grad=True),
        'indices': [torch.tensor([0, 2])],
        'values': torch.randn(tuple(dim_list_tensor)),
        'accumulate': acc
    }

    bwd_tensors = [torch.randn(tuple(dim_list))]
    evaluate_fwd_bwd_kernel(kernel=kernel, tensor_list_bwd=bwd_tensors, kernel_params_fwd=kernel_params_fwd)


@pytest.mark.parametrize("N, H, W, C", test_case_list)
@pytest.mark.parametrize("dim", [0, 1, 2, 3])
def test_hpu_index_add(N, H, W, C, dim):
    kernel = torch.index_add

    dim_list = [N, C, H, W]
    dim_list_tensor = [N, C, H, W]
    dim_list_tensor[dim] = 2
    kernel_params_fwd = {
        'input': torch.randn(tuple(dim_list), requires_grad=True),
        'dim': dim,
        'index': torch.tensor([0, 2]),
        'source': torch.randn(tuple(dim_list_tensor), requires_grad=True)
    }

    bwd_tensors = [torch.randn(tuple(dim_list))]
    evaluate_fwd_bwd_kernel(kernel=kernel, tensor_list_bwd=bwd_tensors, kernel_params_fwd=kernel_params_fwd)


@pytest.mark.parametrize("N, H, I, S", test_case_scatter_add)
def test_hpu_scatter_out(N, H, I, S):
    hpu = torch.device('habana')
    indices_torch = torch.randint(0, I * S, (N, H), dtype=torch.long)
    src = torch.randn(N, H)
    self_t = torch.randn(N, H)
    self_hpu = self_t.to(hpu)

    tcpu_out = torch.scatter(self_t, 0, indices_torch, src)
    thpu_out = torch.scatter(self_hpu, 0, indices_torch.to(hpu), src.to(hpu))
    compare_tensors(thpu_out, tcpu_out, atol=0, rtol=0)


@pytest.mark.parametrize("N, H, I, S", test_case_scatter_add)
def test_hpu_scatter_inplace(N, H, I, S):
    hpu = torch.device('habana')
    indices_torch = torch.randint(0, I * S, (N, H), dtype=torch.long)
    src = torch.randn(N, H)
    self_t = torch.randn(N, H)
    self_hpu = self_t.to(hpu)

    tcpu_out = self_t.scatter_(0, indices_torch, src)
    thpu_out = self_hpu.scatter_(0, indices_torch.to(hpu), src.to(hpu))
    compare_tensors(thpu_out, tcpu_out, atol=0, rtol=0)


@pytest.mark.skip(reason=f"https://jira.habana-labs.com/browse/SW-25327")
@pytest.mark.parametrize("N, H, I, S", test_case_scatter_add)
def test_hpu_scatter_add_out(N, H, I, S):
    hpu = torch.device('habana')
    indices_torch = torch.randint(0, I * S, (N, H), dtype=torch.long)
    src = torch.randn(N, H)
    self_t = torch.randn(N, H)
    self_hpu = self_t.to(hpu)

    tcpu_out = torch.scatter_add(self_t, 0, indices_torch, src)
    thpu_out = torch.scatter_add(self_hpu, 0, indices_torch.to(hpu), src.to(hpu))
    compare_tensors(thpu_out, tcpu_out, atol=0.001, rtol=1.e-3)


@pytest.mark.skip(reason=f"https://jira.habana-labs.com/browse/SW-25327")
@pytest.mark.parametrize("N, H, I, S", test_case_scatter_add)
def test_hpu_scatter_add_inplace(N, H, I, S):
    hpu = torch.device('habana')
    indices_torch = torch.randint(0, I * S, (N, H), dtype=torch.long)
    src = torch.randn(N, H)
    self_t = torch.randn(N, H)
    self_hpu = self_t.to(hpu)

    tcpu_out = self_t.scatter_add_(0, indices_torch, src)
    thpu_out = self_hpu.scatter_add_(0, indices_torch.to(hpu), src.to(hpu))
    compare_tensors(thpu_out, tcpu_out, atol=0.001, rtol=1.e-3)


@pytest.mark.parametrize("test_case_list", broadcast_test_case_list)
def test_hpu_broadcast(test_case_list):
    hpu = torch.device('habana')
    cpu = torch.device('cpu')

    t1 = test_case_list[0]
    t2 = test_case_list[1]
    t3 = test_case_list[2]

    tcpu_out = torch.broadcast_tensors(t1, t2, t3)
    thpu_out = torch.broadcast_tensors(t1.to(hpu), t2.to(hpu), t3.to(hpu))
    compare_tensors(thpu_out, tcpu_out, atol=0, rtol=0)


@pytest.mark.parametrize("start, end, step, dtype", arange_test_case_list)
@pytest.mark.parametrize("op", [torch.arange])
def test_hpu_arange_op_out(start, end, step, dtype, op):
    kernel_params_fwd = {}
    kernel_params_fwd['start'] = start
    kernel_params_fwd['end'] = end
    kernel_params_fwd['step'] = step
    kernel_params_fwd['dtype'] = dtype
    kernel_params_fwd['out'] = torch.empty(1, dtype=dtype)
    evaluate_fwd_kernel(kernel=op, kernel_params=kernel_params_fwd)


@pytest.mark.parametrize("test_dtype", [torch.float, torch.long])
def test_hpu_expand(test_dtype):
    hpu = torch.device('habana')
    cpu = torch.device('cpu')

    hpu = torch.device('habana')
    cpu = torch.device('cpu')
    tin = torch.arange(0, 3, 1, dtype=test_dtype).view(3, 1)
    tcpu_out = tin.expand(3, 4)
    thpu_out = tin.to(hpu).expand(3, 4)
    compare_tensors(thpu_out, tcpu_out, atol=0, rtol=0)


if __name__ == '__main__':
    test_hpu_slice_and_select(*test_case_list[0])
    test_hpu_view(*test_case_list[0])
    test_hpu_index_select(*test_case_list[0], 0)
    test_hpu_index_put(*test_case_list[0])
    test_hpu_index_add(*test_case_list[0], 0)
    test_hpu_broadcast(broadcast_test_case_list[2])
