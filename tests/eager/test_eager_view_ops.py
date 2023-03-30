import os
import pytest
import torch
assert torch.__version__.startswith("2.0"), "Test suite only for PT2.0"
import habana_frameworks.torch.core as htcore
import numpy as np

torch.manual_seed(0)

def test_relu_contiguous_view():
    cpu_tensor = torch.Tensor(np.arange(-10.0, 10.0, 0.1)).view(-1)
    hpu_tensor = cpu_tensor.to("hpu").view(-1)

    result_hpu = torch.relu(hpu_tensor).to("cpu")
    result_cpu = torch.relu(cpu_tensor)

    assert torch.allclose(result_hpu, result_cpu, atol = 0.001, rtol = 0.001)

def test_relu_contiguous_slice():
    cpu_tensor = torch.randn([4])
    hpu_tensor = cpu_tensor.to("hpu")

    cpu_tensor_slice = cpu_tensor[2:]
    hpu_tensor_slice = hpu_tensor[2:]

    result_hpu = torch.relu(hpu_tensor_slice).to("cpu")
    result_cpu = torch.relu(cpu_tensor_slice)

    assert torch.allclose(result_hpu, result_cpu, atol = 0.001, rtol = 0.001)

def test_relu_contiguous_as_strided():
    cpu_tensor = torch.randn([2,3])
    hpu_tensor = cpu_tensor.to("hpu")

    cpu_tensor.as_strided_([5], [1], 1)
    hpu_tensor.as_strided_([5], [1], 1)

    result_hpu = torch.relu(hpu_tensor).to("cpu")
    result_cpu = torch.relu(cpu_tensor)

    assert torch.allclose(result_hpu, result_cpu, atol = 0.001, rtol = 0.001)

def test_relu_contiguous_multilevel_view():
    cpu_tensor = torch.randn([2,3])
    hpu_tensor = cpu_tensor.to("hpu")

    cpu_tensor = cpu_tensor[:].view(-1)
    hpu_tensor = hpu_tensor[:].view(-1)

    result_hpu = torch.relu(hpu_tensor).to("cpu")
    result_cpu = torch.relu(cpu_tensor)

    assert torch.allclose(result_hpu, result_cpu, atol = 0.001, rtol = 0.001)

def test_relu_inplace_view():
    cpu_tensor = torch.randn([4]).view(-1)
    hpu_tensor = cpu_tensor.to("hpu").view(-1)

    cpu_tensor = cpu_tensor[2::]
    hpu_tensor = hpu_tensor[2::]

    result_hpu = torch.relu_(hpu_tensor).to("cpu")
    result_cpu = torch.relu_(cpu_tensor)

    assert torch.allclose(result_hpu, result_cpu, atol = 0.001, rtol = 0.001)

def test_relu_discontiguous_slice():
    cpu_tensor = torch.randn([4])
    hpu_tensor = cpu_tensor.to("hpu")

    cpu_tensor_slice = cpu_tensor[::2]
    hpu_tensor_slice = hpu_tensor[::2]

    result_hpu = torch.relu(hpu_tensor_slice).to("cpu")
    result_cpu = torch.relu(cpu_tensor_slice)

    assert torch.allclose(result_hpu, result_cpu, atol = 0.001, rtol = 0.001)

def test_relu_2d_discontiguous_slice():
    cpu_tensor = torch.Tensor(np.random.randint(-2, 2, (20, 20)))
    hpu_tensor = cpu_tensor.to("hpu")

    cpu_tensor_slice = cpu_tensor[0::2,0::2]
    hpu_tensor_slice = hpu_tensor[0::2,0::2]

    result_hpu = torch.relu(hpu_tensor_slice).to("cpu")
    result_cpu = torch.relu(cpu_tensor_slice)

    assert torch.allclose(result_hpu, result_cpu, atol = 0.001, rtol = 0.001)

def test_relu_inplace_1d_noncontiguous_view():
    cpu_tensor = torch.randn([4])
    hpu_tensor = cpu_tensor.to("hpu")

    torch.relu_(hpu_tensor[::2])
    torch.relu_(cpu_tensor[::2])

    torch.allclose(hpu_tensor.cpu(), cpu_tensor, atol = 0.001, rtol = 0.001)

def test_relu_inplace_2d_noncontiguous_view():
    cpu_tensor = torch.randn([4,4])
    hpu_tensor = cpu_tensor.to("hpu")

    torch.relu_(hpu_tensor[0::2,0::2])
    torch.relu_(cpu_tensor[0::2,0::2])

    torch.allclose(hpu_tensor.cpu(), cpu_tensor, atol = 0.001, rtol = 0.001)

def test_relu_inplace_noncontiguous_view():
    cpu_tensor = torch.randn([4])
    hpu_tensor = cpu_tensor.to("hpu")

    cpu_tensor = cpu_tensor[::2]
    hpu_tensor = hpu_tensor[::2]

    result_hpu = torch.relu_(hpu_tensor).to("cpu")
    result_cpu = torch.relu_(cpu_tensor)

    assert torch.allclose(result_hpu, result_cpu, atol = 0.001, rtol = 0.001)

def test_aminmax_multi_output_view_row():
    cpu_tensor = torch.randn([2, 5])
    hpu_tensor = cpu_tensor.to("hpu")

    cpu_min_tensor = torch.randn([2, 5])
    hpu_min_tensor = cpu_min_tensor.to("hpu")

    cpu_max_tensor = torch.randn([2, 5])
    hpu_max_tensor = cpu_max_tensor.to("hpu")

    cpu_min_tensor[1], cpu_max_tensor[1] = cpu_tensor.aminmax(dim=0, keepdim=True)
    hpu_min_tensor[1], hpu_max_tensor[1] = hpu_tensor.aminmax(dim=0, keepdim=True)

    assert torch.allclose(hpu_min_tensor.cpu(), cpu_min_tensor, atol = 0.001, rtol = 0.001)
    assert torch.allclose(hpu_max_tensor.cpu(), cpu_max_tensor, atol = 0.001, rtol = 0.001)

def test_aminmax_multi_output_view_col():
    cpu_tensor = torch.randn([2, 5])
    hpu_tensor = cpu_tensor.to("hpu")

    cpu_min_tensor = torch.randn([10])
    hpu_min_tensor = cpu_min_tensor.to("hpu")

    cpu_max_tensor = torch.randn([10])
    hpu_max_tensor = cpu_max_tensor.to("hpu")

    cpu_min_tensor[0::2], cpu_max_tensor[0::2] = cpu_tensor.aminmax(dim=0, keepdim=True)
    hpu_min_tensor[0::2], hpu_max_tensor[0::2] = hpu_tensor.aminmax(dim=0, keepdim=True)

    assert torch.allclose(hpu_min_tensor.cpu(), cpu_min_tensor, atol = 0.001, rtol = 0.001)
    assert torch.allclose(hpu_max_tensor.cpu(), cpu_max_tensor, atol = 0.001, rtol = 0.001)

def test_d2d_noncontiguous_views_src():
    cpu_src_tensor = torch.randn([4])
    hpu_src_tensor = cpu_src_tensor.to("hpu")

    cpu_src_tensor_view = cpu_src_tensor[::2]
    hpu_src_tensor_view = hpu_src_tensor[::2]

    cpu_dst_tensor = torch.randn([2])
    hpu_dst_tensor = cpu_dst_tensor.to("hpu")

    hpu_dst_tensor.copy_(hpu_src_tensor_view)
    cpu_dst_tensor.copy_(cpu_src_tensor_view)

    assert torch.allclose(hpu_dst_tensor.cpu(), cpu_dst_tensor, atol = 0.001, rtol = 0.001)

def test_d2d_noncontiguous_views_dst():
    cpu_src_tensor = torch.randn([2])
    hpu_src_tensor = cpu_src_tensor.to("hpu")

    cpu_dst_tensor = torch.randn([4])
    hpu_dst_tensor = cpu_dst_tensor.to("hpu")

    cpu_dst_tensor_view = cpu_dst_tensor[::2]
    hpu_dst_tensor_view = hpu_dst_tensor[::2]

    hpu_dst_tensor_view.copy_(hpu_src_tensor)
    cpu_dst_tensor_view.copy_(cpu_src_tensor)

    assert torch.allclose(hpu_dst_tensor.cpu(), cpu_dst_tensor, atol = 0.001, rtol = 0.001)

def test_d2d_noncontiguous_views_src_dst():
    cpu_src_tensor = torch.randn([4])
    hpu_src_tensor = cpu_src_tensor.to("hpu")

    cpu_src_tensor_view = cpu_src_tensor[::2]
    hpu_src_tensor_view = hpu_src_tensor[::2]

    cpu_dst_tensor = torch.randn([4])
    hpu_dst_tensor = cpu_dst_tensor.to("hpu")

    cpu_dst_tensor_view = cpu_dst_tensor[::2]
    hpu_dst_tensor_view = hpu_dst_tensor[::2]

    hpu_dst_tensor_view.copy_(hpu_src_tensor_view)
    cpu_dst_tensor_view.copy_(cpu_src_tensor_view)

    assert torch.allclose(hpu_dst_tensor.cpu(), cpu_dst_tensor, atol = 0.001, rtol = 0.001)

def test_d2h_noncontiguous_views():
    cpu_src_tensor = torch.randn([4])
    hpu_src_tensor = cpu_src_tensor.to("hpu")

    cpu_src_tensor_view = cpu_src_tensor[::2]
    hpu_src_tensor_view = hpu_src_tensor[::2]

    assert torch.allclose(hpu_src_tensor_view.cpu(), cpu_src_tensor_view, atol = 0.001, rtol = 0.001)

def test_h2d_noncontiguous_views():
    cpu_src_tensor = torch.randn([4])
    cpu_src_tensor_view = cpu_src_tensor[::2]
    hpu_src_tensor = cpu_src_tensor_view.to("hpu")

    assert torch.allclose(hpu_src_tensor.cpu(), cpu_src_tensor_view, atol = 0.001, rtol = 0.001)

def test_h2d_chlast():
    a = torch.randn([2, 3, 4, 5]).to(memory_format=torch.channels_last)
    ha = a.to('hpu')

    b = torch.relu(a)
    hb = torch.relu(ha)

    assert torch.allclose(hb.cpu(), b, atol = 0.001, rtol = 0.001)

def test_h2d_dst_noncontiguous_view():
    cpu_src_tensor = torch.randn([4])
    hpu_src_tensor = cpu_src_tensor.to("hpu")

    cpu_src_tensor2 = torch.randn([2])
    hpu_src_tensor_view = hpu_src_tensor[::2]

    hpu_src_tensor_view.copy_(cpu_src_tensor2)

    cpu_src_tensor[::2].copy_(cpu_src_tensor2)

    assert torch.allclose(hpu_src_tensor.cpu(), cpu_src_tensor, atol = 0.001, rtol = 0.001)

def test_d2d_dst_view():
    cpu_src_tensor = torch.randn([4])
    hpu_src_tensor = cpu_src_tensor.to("hpu")

    cpu_src_tensor2 = torch.randn([2])
    hpu_src_tensor2 = cpu_src_tensor2.to('hpu')
    hpu_src_tensor_view = hpu_src_tensor[::2]
    cpu_src_tensor_view = cpu_src_tensor[::2]

    hpu_src_tensor_view.copy_(hpu_src_tensor2)

    cpu_src_tensor_view.copy_(cpu_src_tensor2)

    assert torch.allclose(hpu_src_tensor.cpu(), cpu_src_tensor, atol = 0.001, rtol = 0.001)

def test_topk_transpose():
    a = torch.randint(0, 10, [2,2])
    ha = a.to('hpu')

    b = torch.topk(a, k = 2)
    c = b[0].t()

    hb = torch.topk(ha, k = 2)
    hc = hb[0].t()
    assert torch.allclose(hc.cpu(), c, atol = 0.001, rtol = 0.001)