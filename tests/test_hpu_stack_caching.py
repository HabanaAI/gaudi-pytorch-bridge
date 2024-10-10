import torch


def test_stack_cache():
    t1 = torch.randn(3, 3).to("hpu")
    t2 = torch.randn(3, 3).to("hpu")
    t3 = torch.randn(3, 3).to("hpu")
    t4 = torch.randn(3, 3).to("hpu")

    out1 = torch.stack([t1, t2, t1, t2], dim=1)

    t5 = torch.randn(3, 3).to("hpu")
    t6 = torch.randn(3, 3).to("hpu")
    t7 = torch.randn(3, 3).to("hpu")
    t8 = torch.randn(3, 3).to("hpu")

    out2 = torch.stack([t5, t6, t7, t8], dim=1)
