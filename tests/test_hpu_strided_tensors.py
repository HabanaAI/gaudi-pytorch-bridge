import pytest
import test_utils
import torch
from test_utils import compare_tensors

try:
    import habana_frameworks.torch.core as htcore
except ImportError:
    raise AssertionError("Could Not import habana_frameworks.torch.core")


test_case_list_1D = [
    # N, const
    (2, 2),
]

test_case_list_4D = [
    # N, C, H, W, batch size
    (2, 1, 1, 1, 1),
]
test_case_list_5D = [
    # N, C, D, H, W, batch size
    (4, 4, 3, 2, 2, 1)
]


def test_hpu_st_tensor1():
    t4 = torch.randn(20, 5)
    t4_st = t4[0::4, ...]
    t1 = torch.randn(10, 5)
    t2 = t1.abs()
    t2[0::2, ...].copy_(t4_st)
    t5 = t2.relu()
    out_cpu = t5

    t4_hpu = t4.to("hpu")
    t4_hpu_st = t4_hpu[0::4, ...]
    t1_hpu = t1.to("hpu")
    t2_hpu = t1_hpu.abs()
    t2_hpu[0::2, ...].copy_(t4_hpu_st)
    t5_hpu = t2_hpu.relu()
    out_hpu = t5_hpu.to("cpu")

    compare_tensors(out_hpu, out_cpu, atol=0, rtol=0)


def test_hpu_st_tensor2():
    t1 = torch.randn(10, 5)
    t2 = t1.abs()
    t3 = t2[0::2, ...]
    t5 = t3.relu_()
    out_cpu = t5

    t1_hpu = t1.to("hpu")
    t2_hpu = t1_hpu.abs()
    t3_hpu = t2_hpu[0::2, ...]
    t5_hpu = t3_hpu.relu_()
    out_hpu = t5_hpu.to("cpu")

    compare_tensors(out_hpu, out_cpu, atol=0.1, rtol=0.1)


def slice_func(x):
    return x[0:5:2, 0:2]


def test_hpu_st_tensor3():
    in_t = torch.randn(5, 5, requires_grad=True)
    in_t_detach = in_t.detach()
    cpu_result = slice_func(in_t)
    grad_out = torch.randn(3, 2, requires_grad=False)
    grad_out_detach = grad_out.detach()
    cpu_result.backward(grad_out)
    cpu_grad = in_t.grad.detach()

    hpu_t = in_t_detach.to("hpu")
    hpu_t.requires_grad = True
    grad_out_hpu = grad_out_detach.to("hpu")
    hpu_result = slice_func(hpu_t)
    hpu_result.backward(grad_out_hpu)
    hpu_grad = hpu_t.grad.detach()
    hpu_result = hpu_result.to("cpu")
    hpu_grad_result = hpu_grad.to("cpu")

    compare_tensors(hpu_result, cpu_result, atol=0, rtol=0)
    compare_tensors(hpu_grad_result, cpu_grad, atol=0, rtol=0)


def test_hpu_st_tensor4():
    batched_imgs = torch.randn(2, 5, 5)
    batched_imgs_hpu = batched_imgs.to("hpu")
    tensors = [torch.ones(3, 3), torch.ones(2, 2)]
    tensors_hpu = [x.to("hpu") for x in tensors]

    for img, pad_img in zip(tensors, batched_imgs):
        pad_img[..., : img.shape[-2], : img.shape[-1]].copy_(img)

    for img_hpu, pad_img_hpu in zip(tensors_hpu, batched_imgs_hpu):
        pad_img_hpu[..., : img_hpu.shape[-2], : img_hpu.shape[-1]].copy_(img_hpu)

    out_hpu = batched_imgs_hpu.to("cpu")
    compare_tensors(out_hpu, batched_imgs, atol=0, rtol=0)


@pytest.mark.parametrize("N, C, H, W, bs", test_case_list_4D)
def test_hpu_lazy_slice_fwd_4D(N, C, H, W, bs):
    t1 = torch.zeros((N, C, H, W), requires_grad=False)
    t_in = torch.ones((N, C, H, W), requires_grad=False)

    hpu = test_utils.hpu
    t1_h = t1.detach().to(hpu)
    t_h_in = t_in.detach().to(hpu)
    for i in range(N // bs):
        t = t_in[bs * i : bs * (i + 1)]
        t_h = t_h_in[bs * i : bs * (i + 1)]
        t1[bs * i : bs * (i + 1)] = t.data
        t1_h[bs * i : bs * (i + 1)] = t_h.data
        htcore.mark_step()
    compare_tensors(t1_h, t1, atol=0, rtol=0)


@pytest.mark.xfail
@pytest.mark.parametrize("N, const", test_case_list_1D)
def test_hpu_lazy_slice_fwd_1D(N, const):
    t1 = torch.arange(N - 1, dtype=torch.float32)
    t2 = torch.ones(N - 1, dtype=torch.float32)

    hpu = test_utils.hpu
    t1_h = t1.detach().to(hpu)
    t2_h = t2.detach().to(hpu)

    for i in range(1, N):
        t1[i - 1] += const
        t1_h[i - 1] += const
        # htcore.mark_step()
    t2 += t1
    t2_h += t1_h
    htcore.mark_step()

    compare_tensors(t2_h, t2, atol=0, rtol=0)


@pytest.mark.parametrize("N, C, D, H, W, bs", test_case_list_5D)
def test_hpu_lazy_slice_fwd_5D(N, C, D, H, W, bs):
    a, b, c, d, e = (N, C, D, H, W)
    rem = a % bs
    pad = 0
    if a != 0:
        pad = bs - rem
    input_tensor = (a + pad, b, c, d, e)
    t1 = torch.zeros(input_tensor, requires_grad=False)
    t_in = torch.ones(input_tensor, requires_grad=False).contiguous(memory_format=torch.channels_last_3d)

    hpu = test_utils.hpu
    t1_h = t1.detach().to(hpu)
    t_h_in = t_in.detach().to(hpu)
    N, C, D, H, W = input_tensor
    for i in range(N // bs):
        t = t_in[bs * i : bs * (i + 1)]
        t_h = t_h_in[bs * i : bs * (i + 1)]
        t1[bs * i : bs * (i + 1)] = t.data
        t1_h[bs * i : bs * (i + 1)] = t_h.data
        htcore.mark_step()
        compare_tensors(t_h, t, atol=0, rtol=0)
    t1 = t1[pad:]
    t1_h = t1_h[pad:]
    htcore.mark_step()
    compare_tensors(t1_h, t1, atol=0, rtol=0)
    t1 = torch.transpose(t1, 0, 1).unsqueeze(0)
    t1_h = torch.transpose(t1_h, 0, 1).unsqueeze(0)
    htcore.mark_step()
    compare_tensors(t1_h, t1, atol=0, rtol=0)


@pytest.mark.parametrize("N, C, D, H, W, bs", test_case_list_5D)
def test_hpu_lazy_slice_fwd_5D_bf16(N, C, D, H, W, bs):
    t1 = torch.zeros((N, C, D, H, W), requires_grad=False)
    t_in = torch.ones((N, C, D, H, W), requires_grad=False).contiguous(
        memory_format=torch.channels_last_3d
    )  # .to(torch.bfloat16)

    hpu = test_utils.hpu
    t1_h = t1.detach().to(hpu)
    t_h_in = t_in.detach().to(hpu).to(torch.bfloat16)
    htcore.mark_step()
    for i in range(N // bs):
        t = t_in[bs * i : bs * (i + 1)]
        t_h = t_h_in[bs * i : bs * (i + 1)]  # .to(torch.float32)
        t1[bs * i : bs * (i + 1)] = t.data
        t1_h[bs * i : bs * (i + 1)] = t_h.data  # .to(torch.float32)
        htcore.mark_step()
        compare_tensors(t_h, t, atol=0.001, rtol=0.001)
        compare_tensors(t1_h, t1, atol=0.001, rtol=0.001)
    compare_tensors(t1_h, t1, atol=0, rtol=0)


@pytest.mark.parametrize("N, C", test_case_list_1D)
def test_hpu_autograd_slicefunction(N, C):
    a = torch.randn([N, C, 2], requires_grad=True)
    ha = a.to("hpu")
    a2 = torch.relu(a)
    b = torch.BoolTensor([N, C])
    hb = b.to("hpu")
    a2[b, :] = 0.0

    ha2 = torch.relu(ha)
    ha2[hb, :] = 0.0

    compare_tensors(ha2, a2, atol=0.001, rtol=0.001)


@pytest.mark.parametrize("N, C", test_case_list_1D)
def test_hpu_autograd_divout(N, C):
    a = torch.randn([C])
    ha = a.to("hpu")
    b = torch.randn([C])
    hb = b.to("hpu")
    c = torch.randn([C])
    hc = c.to("hpu")

    v = b.view(-1)
    torch.div(a, v, out=c)

    hv = hb.view(-1)
    torch.div(ha, hv, out=hc)

    compare_tensors(hc, c, atol=0.001, rtol=0.001)


if __name__ == "__main__":
    test_hpu_st_tensor1()
    test_hpu_st_tensor2()
    test_hpu_st_tensor3()
    test_hpu_st_tensor4()
    test_hpu_lazy_slice_fwd_4D(*test_case_list_4D[0])
    test_hpu_lazy_slice_fwd_1D(*test_case_list_1D[0])
    test_hpu_lazy_slice_fwd_5D(*test_case_list_5D[0])
    test_hpu_lazy_slice_fwd_5D_bf16(*test_case_list_5D[0])
