###############################################################################
#
#  Copyright (c) 2021-2025 Intel Corporation
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
###############################################################################

import math

import pytest
import torch
from test_utils import (
    check_ops_executed_in_jit_ir,
    clear_t_compile_logs,
    is_pytest_mode_compile,
)

# reference : bitsandbytes/bitsandbytes/functional.py
nf4_dequant_table = [
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


def dequantize_nf4_impl_for_cpu(
    A: torch.Tensor,
    quant_state=None,
    absmax: torch.Tensor = None,
    out: torch.Tensor = None,
    blocksize: int = 64,
    big_endian: bool = True,
) -> torch.Tensor:
    """
    Dequantizes FP4 blockwise quantized values.

    Dequantizes the tensor A with maximum absolute values absmax in blocks of size blocksize.

    Parameters
    ----------
    A : torch.Tensor
        The input 8-bit tensor (packed 4-bit values).
    quant_state : QuantState
        object with quantisation stats, incl. absmax values, original tensor shape and original dtype.
    absmax : torch.Tensor
        The absmax values.
    out : torch.Tensor
        Dequantized output tensor.
    blocksize : int
        The blocksize used in quantization.
    big_endian: bool
        If True, the most significant bits are stored at the even position in the packed 4-bit values and
                least significant bits are stored at the odd position in the packed 4-bit values.
        If False, the least significant bits are stored at the even position in the packed 4-bit values and
                most significant bits are stored at the odd position in the packed 4-bit values.
    Returns
    -------
    torch.Tensor:
        Dequantized tensor.
    """
    if A.shape[0] == 1:
        transpose = False
        A = A.squeeze(0)
    elif A.shape[1] == 1:
        transpose = True
        A = A.squeeze(1)

    if quant_state is None:
        assert absmax is not None and out is not None
        quant_state = QuantState(
            absmax=absmax,
            shape=out.shape,
            dtype=out.dtype,
            blocksize=blocksize,
            code=torch.tensor(nf4_dequant_table, device=A.device),
        )
    else:
        absmax = quant_state.absmax

    # Map nf4 to [-1, 1]
    # create a output tensor of double size of input
    out_dq = torch.empty(A.size(0) * 2, dtype=torch.int32, device=A.device)
    n = out_dq.numel()
    if big_endian:
        out_dq[::2] = A >> 4  # fill msb in the even index
        out_dq[1::2] = A & 0xF  # fill lsb in the odd index
    else:
        out_dq[::2] = A & 0xF  # fill lsb in the even index
        out_dq[1::2] = A >> 4  # fill msb in the odd index
    # quant_state.code is fp32, cast to quant_state dtype to avoid the mismatch issue
    quant_state.code = quant_state.code.to(quant_state.dtype)
    out_dq = quant_state.code[out_dq]

    # Apply scales
    if out_dq.numel() != n:
        assert out_dq.numel() == n + 1
        out_dq = torch.narrow(out_dq, 0, 0, n)
    blocks = n // blocksize
    blocks += 1 if n % blocksize > 0 else 0
    rem = n % blocksize
    has_rem = rem > 0

    if has_rem:
        if out is None:
            out = torch.empty(quant_state.shape, dtype=quant_state.dtype, device=A.device)
        out_reshaped = out.reshape(-1)
        out_reshaped[: n - rem] = (
            out_dq[: n - rem].view(-1, blocksize) * absmax[: blocks - has_rem].view(-1, 1)
        ).reshape(-1)
        out_reshaped[n - rem :] = out_dq[n - rem :] * absmax[-1]
    else:
        out = (out_dq.view(-1, blocksize) * absmax.view(-1, 1)).reshape(quant_state.shape).to(quant_state.dtype)

    # take transpose here because weight is transposed (again) for computation
    if transpose:
        out = out.t()

    return out


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize(
    "n",
    [
        62,
        128,
        500,
    ],
)
@pytest.mark.parametrize(
    "blocksize",
    [
        64,
    ],
)
@pytest.mark.parametrize("absmax_random", [True])
@pytest.mark.parametrize("big_endian", [True, False])
def test_dequantize_nf4(dtype, n, blocksize, absmax_random, big_endian):
    def fn(input, absmax, blocksize, out_shape, out_dtype):
        return torch.ops.hpu.dequantize_nf4(input, absmax, blocksize, out_shape, out_dtype, big_endian)

    if is_pytest_mode_compile():
        clear_t_compile_logs()
        torch._dynamo.reset()
        fn = torch.compile(fn, backend="hpu_backend")

    # CPU
    # quantized packed 4 bit tensor (i.e. 8 bit tensor)
    cpu_input = torch.randint(high=255, size=(math.ceil(n / 2), 1), dtype=torch.uint8)
    # absmax value for every block
    absMax = (
        torch.abs(torch.randn(math.ceil(n / blocksize), dtype=dtype))
        if absmax_random
        else torch.ones(math.ceil(n / blocksize), dtype=dtype)
    )
    # dequantize tensor of preferred dtype and shape(quantize_tensor.size(0)*2, 1)
    output = torch.zeros((n, 1), dtype=dtype)
    cpu_output = dequantize_nf4_impl_for_cpu(A=cpu_input, absmax=absMax, out=output, big_endian=big_endian).reshape(-1)

    # HPU
    # As HPU expects 1d tesnor only
    hpu_input = cpu_input.reshape(-1).to("hpu")
    hpu_absMax = absMax.reshape(-1).to("hpu")
    out_shape = (n,)
    hpu_output = fn(hpu_input, hpu_absMax, blocksize, out_shape, dtype)

    atol, rtol = (1e-2, 1e-2) if dtype == torch.bfloat16 else (1e-6, 1e-6)
    if absmax_random:
        assert torch.allclose(cpu_output, hpu_output.cpu(), atol=atol, rtol=rtol)
    else:
        # Currently this comparison has some precision issue, this should match exactly with the
        # cpu tensor as argMax value is 1, but it is not matching, will debug later.
        assert torch.allclose(cpu_output, hpu_output.cpu(), rtol=rtol)
    if is_pytest_mode_compile():
        check_ops_executed_in_jit_ir("dequantize_nf4")
