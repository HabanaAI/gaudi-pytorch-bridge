import habana_frameworks.torch.core as htcore
import pytest
import torch


def test_get_and_set_arange_state():
    device = torch.device("hpu")
    dtype = torch.bfloat16

    ref_tensor = torch.tensor([0, 2, 4], dtype=dtype, device=device)
    f16_tensor = torch.arange(0, 6, step=2, dtype=dtype, device=device)
    assert torch.equal(ref_tensor, f16_tensor)
