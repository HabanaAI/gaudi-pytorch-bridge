import torch
import torch.nn as nn
from test_utils import reset_seed, compare_tensors
from hmp import hmp
import pytest

hpu = torch.device("habana")
cpu = torch.device("cpu")


@pytest.fixture(scope="session", autouse=True)
def execute_before_any_test():
    hmp.convert(isVerbose=False)


def test_add_override():
    a = torch.randn(3, 4).bfloat16().to(hpu)
    b = torch.randn(3, 4).to(hpu)
    a = a + b
    assert a.dtype == torch.float32


def test_add_inplace():

    a = torch.randn(3, 4).bfloat16().to(hpu)
    b = torch.randn(3, 4).to(hpu)
    a = a.add_(b)
    assert a.dtype == torch.bfloat16


def test_add_tensor():
    a = torch.randn(3, 4).bfloat16().to(hpu)
    b = torch.randn(3, 4).to(hpu)
    a = a.add(b)
    assert a.dtype == torch.float32


def test_add_torch():
    a = torch.randn(3, 4).bfloat16().to(hpu)
    b = torch.randn(3, 4).to(hpu)
    a = torch.add(a, b)
    assert a.dtype == torch.float32

def test_cat_torch():
    a = torch.randn(3, 4).bfloat16().to(hpu)
    b = torch.randn(4, 4).to(hpu)
    c = torch.randn(5, 4).to(hpu)
    a = torch.cat([a,b,c])
    assert a.dtype == torch.float32

if __name__ == "__main__":
    hmp.convert(isVerbose=True)
    test_cat_torch()

