import habana_frameworks.torch
import pytest
import torch
from test_utils import format_tc


@pytest.mark.parametrize("shape", [(3,), (3, 3), (3, 3, 3)], ids=format_tc)
@pytest.mark.parametrize("split_dim", [0, 1, 2])
@pytest.mark.skip(reason="Waiting for https://gerrit.habana-labs.com/#/c/338049/")
def test_split_cat(shape, split_dim):
    if split_dim >= len(shape):
        pytest.skip("Invalid case")

    def fn(in_tensor, split_size_or_sections, dim):
        # Using torch.cat because there's an issue if ListUnpack is an
        # output node of the graph
        return torch.cat(torch.split(in_tensor, split_size_or_sections, dim), dim)

    compiled_fn = torch.compile(fn, backend="hpu_backend")

    cpu_in = torch.rand(size=shape, device="cpu")
    hpu_in = cpu_in.to("hpu")
    expected = fn(cpu_in, 2, split_dim)
    result = compiled_fn(hpu_in, 2, split_dim)
    assert torch.equal(expected, result.cpu())


@pytest.mark.parametrize("shape", [(3,), (3, 3), (3, 3, 3)], ids=format_tc)
@pytest.mark.parametrize("split_dim", [0, 1, 2])
@pytest.mark.skip(reason="SW-154799 when graph returns TensorList the results are incorrect")
def test_split(shape, split_dim):
    if split_dim >= len(shape):
        pytest.skip("Invalid case")

    def fn(in_tensor, split_size_or_sections, dim):
        return torch.split(in_tensor, split_size_or_sections, dim)

    compiled_fn = torch.compile(fn, backend="hpu_backend")

    cpu_in = torch.rand(size=shape, device="cpu")
    hpu_in = cpu_in.to("hpu")
    expected = fn(cpu_in, 2, split_dim)
    result = compiled_fn(hpu_in, 2, split_dim)
    for exp, res in zip(expected, result):
        assert torch.equal(exp, res.cpu())
