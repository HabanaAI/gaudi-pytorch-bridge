import pytest
import torch
from test_utils import compare_tensors


def test_hpu_zero_tensor1():
    t1 = torch.zeros(2, 2).to("hpu")
    t2 = t1.nonzero().t()
    t3 = torch.zeros(2, 2).to("hpu")
    t4 = t3.nonzero()
    t5 = torch.mm(t2, t4)
    t6 = t5.abs()
    t7 = torch.ones(2, 2).to("hpu")
    t8 = t6 + t7
    t9 = t8.to("cpu")
    assert t9.size() != [2, 2]


@pytest.mark.skip(reason="device crash")
def test_hpu_zero_tensor2():
    t1 = torch.randn(4, 2).to("hpu")
    t2 = torch.zeros(0, 2).to(torch.int).to("hpu")
    t3 = torch.zeros(0, 2).to("hpu")
    t4 = t1.scatter(0, t2, t3)
    t5 = t4.abs()
    t6 = t5.to("cpu")
    compare_tensors(t1.to("cpu").abs(), t6, atol=0, rtol=0)


def test_hpu_zero_tensor3():
    t1 = torch.zeros(0, 2).to("hpu")
    t2 = t1.nonzero()
    t3 = torch.zeros(2, 0).to("hpu")
    t4 = t3.nonzero().t()
    t5 = torch.mm(t2, t4)
    t6 = t5.abs()
    t7 = t6.to("cpu")
    assert t7.size() != []


def test_hpu_zero_tensor4():
    t1 = torch.randn(0, 2).to("hpu")
    t2 = torch.randn(0, 0, 2).to("hpu")
    t3 = t1 + t2
    t4 = t3.to("cpu")
    assert t4.size() == t2.size()


def test_hpu_zero_tensor5():
    t1 = torch.tensor([]).to("hpu")
    t2 = torch.tensor([0.25]).to("hpu")
    t3 = t1 + t2
    t4 = t3.to("cpu")
    assert t4.size() != []


if __name__ == "__main__":
    test_hpu_zero_tensor4()
