import torch
from test_utils import compare_tensors

try:
    import habana_frameworks.torch.core as htcore
except ImportError:
    raise AssertionError("Could Not import habana_frameworks.torch.core")


def test_hpu_view_early_release():
    batched_imgs = torch.randn(2, 5, 5)
    batched_imgs_hpu = batched_imgs.to("hpu")
    tensors = [torch.ones(3, 3), torch.ones(2, 2)]
    tensors_hpu = [x.to("hpu") for x in tensors]

    for img, pad_img in zip(tensors, batched_imgs):
        pad_img[..., : img.shape[-2], : img.shape[-1]].copy_(img)

    for img_hpu, pad_img_hpu in zip(tensors_hpu, batched_imgs_hpu):
        pad_img_hpu[..., : img_hpu.shape[-2], : img_hpu.shape[-1]].copy_(img_hpu)

    # Issue: This isn't triggering execution as pad_img_hpu are tmp views
    #        that are already deleted along with their IR graph
    out_hpu = batched_imgs_hpu.to("cpu")
    compare_tensors(out_hpu, batched_imgs, atol=0, rtol=0)


def test_hpu_as_strided_on_as_strided():
    x = torch.randn(5, 5)
    t = torch.randn(5, 5)
    x_h = x.to("hpu")
    t_h = t.to("hpu")

    v1 = t.as_strided((1, 25), (25, 1))
    v2 = v1.as_strided((5, 5), (5, 1))

    v2.add_(x)
    t.add_(x)

    v1_h = t_h.as_strided((1, 25), (25, 1))
    v2_h = v1_h.as_strided((5, 5), (5, 1))

    v2_h.add_(x_h)
    t_h.add_(x_h)

    # Issue: This isn't triggering execution on views v2_h and t_h
    out_h = v1_h.to("cpu")
    compare_tensors(out_h, v1, atol=0, rtol=0)


def test_hpu_as_strided_on_as_strided_2():
    x = torch.randn(5, 5)
    t = torch.randn(5, 5)
    x_h = x.to("hpu")
    t_h = t.to("hpu")

    v1 = t.as_strided((1, 25), (25, 1))
    v2 = v1.as_strided((5, 5), (5, 1))

    v2.add_(x)
    t.add_(x)

    v1_h = t_h.as_strided((1, 25), (25, 1))
    v2_h = v1_h.as_strided((5, 5), (5, 1))

    v2_h.add_(x_h)
    t_h.add_(x_h)

    # Issue: This is missing control edges and fails with graph compile
    htcore.mark_step()
    out_h = v1_h.to("cpu")
    compare_tensors(out_h, v1, atol=0, rtol=0)


def test_hpu_view_of_view():
    x = torch.randn(5, 5)
    t = torch.randn(5, 5)
    x_h = x.to("hpu")
    t_h = t.to("hpu")

    v1 = t.view(1, 25)
    v2 = v1.view(5, 5)

    v2.add_(x)
    t.add_(x)

    v1_h = t_h.view(1, 25)
    v2_h = v1_h.view(5, 5)

    v2_h.add_(x_h)
    t_h.add_(x_h)

    # Issue: view is not in-place, so updating a view doesn't update src
    # Issue: This isn't triggering execution on views v2_h and t_h
    out_h = v1_h.to("cpu")
    compare_tensors(out_h, v1, atol=0, rtol=0)


if __name__ == "__main__":
    test_hpu_view_of_view()
    test_hpu_as_strided_on_as_strided_2()
    test_hpu_as_strided_on_as_strided()
    test_hpu_view_early_release()
