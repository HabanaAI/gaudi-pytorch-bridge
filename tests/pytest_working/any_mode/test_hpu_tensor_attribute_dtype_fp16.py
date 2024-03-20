import habana_frameworks.torch.hpu
import pytest
import torch
from test_utils import format_tc, is_gaudi1


@pytest.mark.skipif(is_gaudi1(), reason="G1 unsupported dtype")
def test_hpu():
    tmp_cpu = torch.nn.Linear(12, 12, dtype=torch.float16)
    print(f"{tmp_cpu.weight.dtype=} | {tmp_cpu.weight.requires_grad=}")
    tmp_hpu = torch.nn.Linear(12, 12, dtype=torch.float16, device="hpu")
    print(f"{tmp_hpu.weight.dtype=} | {tmp_hpu.weight.requires_grad=}")
    assert tmp_cpu.weight.requires_grad == tmp_hpu.weight.requires_grad
