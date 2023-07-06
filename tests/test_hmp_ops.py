import pytest
import torch
from habana_frameworks.torch.hpex import hmp

pytestmark = pytest.mark.xfail

hpu = torch.device("hpu")
cpu = torch.device("cpu")


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_add_override(dtype):
    hmp.convert(isVerbose=True, low_precision_type=dtype)
    a = torch.randn(3, 4, dtype=dtype).to(hpu)
    b = torch.randn(3, 4).to(hpu)
    a = a + b
    assert a.dtype == torch.float32


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_add_ioverride(dtype):
    hmp.convert(isVerbose=True, low_precision_type=dtype)
    a = torch.randn(3, 4, dtype=dtype).to(hpu)
    b = torch.randn(3, 4).to(hpu)
    a += b
    assert a.dtype == torch.float32


@pytest.mark.parametrize(
    "dtype",
    [
        pytest.param(
            torch.bfloat16, marks=[pytest.mark.xfail(reason="Results mismatch")]
        ),
        pytest.param(torch.float16),
    ],
)
def test_truediv_override(dtype):
    hmp.convert(isVerbose=True, low_precision_type=dtype)
    a = torch.randn(3, 4, dtype=dtype).to(hpu)
    torch.randn(3, 4).to(hpu)
    a = a / 5
    assert a.dtype == dtype


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_add_inplace(dtype):
    hmp.convert(isVerbose=True, low_precision_type=dtype)
    a = torch.randn(3, 4, dtype=dtype).to(hpu)
    b = torch.randn(3, 4).to(hpu)
    a = a.add_(b)
    assert a.dtype == dtype


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_add_tensor(dtype):
    hmp.convert(isVerbose=True, low_precision_type=dtype)
    a = torch.randn(3, 4, dtype=dtype).to(hpu)
    b = torch.randn(3, 4).to(hpu)
    a = a.add(b)
    assert a.dtype == torch.float32


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_add_torch(dtype):
    hmp.convert(isVerbose=True, low_precision_type=dtype)
    a = torch.randn(3, 4, dtype=dtype).to(hpu)
    b = torch.randn(3, 4).to(hpu)
    a = torch.add(a, b)
    assert a.dtype == torch.float32


def test_cat_torch():
    hmp.convert(isVerbose=True)
    a = torch.randn(3, 4).bfloat16().to(hpu)
    b = torch.randn(4, 4).to(hpu)
    c = torch.randn(5, 4).to(hpu)
    a = torch.cat([a, b, c])
    assert a.dtype == torch.float32


@pytest.mark.xfail(reason="Results mismatch")
def test_gelu_torch():
    # This test should only be run if gelu is added to bf16_list
    hmp.convert(isVerbose=True)
    b = torch.randn(3, 4).to(hpu)
    out = torch.nn.functional.gelu(b)
    assert out.dtype == torch.bfloat16


def test_layer_norm_torch():
    # This test should only be run if gelu is added to bf16_list
    hmp.convert(isVerbose=True)
    b = torch.randn(3, 4).to(hpu)
    normalized_shape = [4]
    m = torch.nn.LayerNorm(normalized_shape, elementwise_affine=True).to(hpu)
    out = m(b)
    assert out.dtype == torch.bfloat16


if __name__ == "__main__":
    hmp.convert(isVerbose=True)
    test_cat_torch()
