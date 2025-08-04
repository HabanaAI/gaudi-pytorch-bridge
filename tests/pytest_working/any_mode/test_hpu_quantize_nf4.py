###############################################################################
# Copyright (c) 2021-2025 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
###############################################################################


import pytest
import torch
from test_utils import (
    check_ops_executed_in_jit_ir,
    clear_t_compile_logs,
    is_pytest_mode_compile,
    is_pytest_mode_lazy,
    use_eager_fallback,
)

# reference : https://github.com/bitsandbytes-foundation/bitsandbytes/blob/888788d75db8ff8e8888838307119f98d1235c24/bitsandbytes/backends/utils.py#L25
nf4_quant_table = [
    -1.0,
    -0.6961928009986877,
    -0.5250730514526367,
    -0.39491748809814453,
    -0.28444138169288635,
    -0.18477343022823334,
    -0.09105003625154495,
    0.0,
    0.07958029955625534,
    0.16093020141124725,
    0.24611230194568634,
    0.33791524171829224,
    0.44070982933044434,
    0.5626170039176941,
    0.7229568362236023,
    1.0,
]


class QuantState:
    """container for quantization state components to work with Params4bit and similar classes"""

    def __init__(
        self,
        absmax,
        shape=None,
        code=None,
        blocksize=None,
        dtype=None,
    ):
        self.absmax = absmax
        self.shape = shape
        self.code = code
        self.dtype = dtype
        self.blocksize = blocksize


# reference : https://github.com/bitsandbytes-foundation/bitsandbytes/blob/d9333aa9061662c3d6429480dbc0a55956b99db5/bitsandbytes/backends/default/ops.py#L192
def quantize_nf4_impl_for_cpu(
    A: torch.Tensor,
    blocksize: int = 64,
    quant_storage: torch.dtype = torch.uint8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Quantizes a tensor of type fp32/bf16 to packed 4-bit values.

    Parameters
    ----------
    A : torch.Tensor
        The input tensor is of float32 or bfloat16 dtype.
    blocksize : int
        The blocksize used in quantization.
    quant_storage : torch.dtype
        The dtype to store the quantized values, typically torch.uint8.

    Returns
    -------
    torch.Tensor:
        quantized tensor of packed 4-bit values.
    torch.Tensor:
        absmax tensor containing the maximum absolute values for each block.
    """

    torch._check_is_size(blocksize)
    torch._check(
        A.dtype in [torch.bfloat16, torch.float32],
        lambda: f"Blockwise 4bit quantization only supports 16/32-bit floats, but got {A.dtype}",
    )

    n = A.numel()
    full_blocks = n // blocksize
    rem = n % blocksize
    blocks = full_blocks + 1 if rem else full_blocks
    absmax = torch.zeros((blocks,), device=A.device, dtype=torch.float32)
    A_flattened = A.reshape(n)

    # Scale full blocks of the tensor to [-1, 1]
    A_full_blocks = A_flattened[: n - rem].reshape(n // blocksize, blocksize)
    absmax[:full_blocks] = torch.abs(A_full_blocks).max(dim=-1)[0]
    scaled = torch.clamp(A_full_blocks * (1 / absmax[:full_blocks].view(-1, 1)), -1, 1).reshape(-1)

    # Scale any partial block
    if rem:
        A_rem = A_flattened[-rem:]
        absmax[-1] = torch.abs(A_rem).max()
        scaled_rem = torch.clamp(A_rem * (1 / absmax[-1]), -1, 1)
        scaled = torch.cat([scaled, scaled_rem], dim=0)

    # Quantize with the lookup table
    code = torch.tensor(nf4_quant_table).to(scaled.device).to(scaled.dtype)
    quantized = torch.argmin(torch.abs(scaled.view(-1, 1) - code), dim=-1, keepdim=True).to(torch.uint8)

    # Pack two quantized values per byte
    packed = quantized[::2] << 4 | quantized[1::2]

    if quant_storage != torch.uint8:
        packed = packed.squeeze().view(quant_storage).unsqueeze(1)

    return packed, absmax.float()


@pytest.mark.skipif(is_pytest_mode_lazy(), reason="Lazy mode does not support quantization NF4")
@pytest.mark.skipif(is_pytest_mode_compile(), reason="compile mode does not support quantization NF4")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize(
    "shape",
    [
        (100,),
        (64,),
        (64, 64),
        (64, 64, 64),
        (16, 16, 64, 64),
    ],
)
@pytest.mark.parametrize(
    "blocksize",
    [
        64,
    ],
)
def test_quantize_nf4(dtype, shape, blocksize):
    def fn(input, blocksize):
        if input.dim() != 1:
            input = input.view(-1)
        result = torch.ops.hpu.quantize_nf4(input, blocksize)
        return result[0].view(-1, 1), result[1]

    if is_pytest_mode_compile():
        clear_t_compile_logs()
        torch._dynamo.reset()
        fn = torch.compile(fn, backend="hpu_backend")

    # CPU
    cpu_input = torch.randn(shape, dtype=dtype)
    cpu_output = quantize_nf4_impl_for_cpu(A=cpu_input, blocksize=blocksize)
    # HPU

    hpu_input = cpu_input.to("hpu")

    with use_eager_fallback():
        hpu_output = fn(hpu_input, blocksize)

    atol, rtol = (1e-2, 1e-2) if dtype == torch.bfloat16 else (1e-6, 1e-6)

    assert torch.allclose(cpu_output[0], hpu_output[0].cpu())
    assert torch.allclose(cpu_output[1], hpu_output[1].cpu().float(), rtol=rtol, atol=atol)

    if is_pytest_mode_compile():
        check_ops_executed_in_jit_ir("quantize_nf4")
