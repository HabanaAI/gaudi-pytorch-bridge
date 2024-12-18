###############################################################################
#
#  Copyright (c) 2024 Intel Corporation
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

import pytest
import torch
from test_utils import (
    check_ops_executed_in_jit_ir,
    compare_tensors,
    compile_function_if_compile_mode,
    format_tc,
    is_pytest_mode_compile,
)

dtypes = [torch.float32, torch.bfloat16]
modes = ["reduced", "complete", "r"]
shapes = [(6, 3, 4, 4), (2, 3, 5), (2, 4, 8, 6), (1, 3), (1, 1)]


def make_diagonal_positive_batched(Q, R, mode):
    """
    Modifies Q and R so that R has a positive diagonal.
    Handles 'reduced', 'complete', and 'r' modes of torch.linalg.qr.
    """
    # Get the diagonal elements of R across all batches
    diag_R = torch.diagonal(R, dim1=-2, dim2=-1)  # Shape: (*batch_dims, k)

    # Compute the sign of the diagonal elements
    diag_sign = torch.sign(diag_R)
    c = diag_sign.shape[-1]  # number of columns to multiply
    diag_sign[diag_sign == 0] = 1  # Ensure that zero diagonals are treated as positive (sign of 1)

    diag_sign_expanded = diag_sign.unsqueeze(-1)  # Shape (*batch_dims, k, 1)

    R_mod = R.clone()
    # Apply the sign correction to R
    R_mod[..., :c, :] *= diag_sign_expanded

    if mode == "r":
        return Q, R_mod

    # Reshape diag_sign for broadcasting purposes to the first k columns of Q
    diag_sign_expanded = diag_sign.unsqueeze(-2)  # Shape: (*batch_dims, 1, k)

    # Apply the sign correction to the first k columns of Q
    Q_mod = Q.clone()
    Q_mod[..., :c] *= diag_sign_expanded

    return Q_mod, R_mod


def allocate_qr_for_out(aShape: list, mode: str, for_hpu: bool, dtype: torch.dtype):
    qShape = aShape.copy()
    rShape = aShape.copy()
    m = aShape[-2]
    k = min(aShape[-2], aShape[-1])
    if mode == "complete":
        qShape[-1] = m
    elif mode == "reduced":
        qShape[-1] = k
        rShape[-2] = k
    else:
        qShape = [0]
        rShape[-2] = k

    Q, R = torch.empty(qShape, dtype=dtype), torch.empty(rShape, dtype=dtype)
    if for_hpu:
        Q, R = Q.to("hpu"), R.to("hpu")
    return Q, R


def run_linalg_qr_out(A, hpu_A, mode):
    def fn_out(A, mode, QR):
        torch.linalg.qr(A, mode, out=QR)
        return

    QR = allocate_qr_for_out(list(A.shape), mode, for_hpu=False, dtype=A.dtype)
    fn_out(A, mode, QR)

    fn_out = compile_function_if_compile_mode(fn_out)
    QR_hpu = allocate_qr_for_out(list(A.shape), mode, for_hpu=True, dtype=hpu_A.dtype)
    fn_out(hpu_A, mode, QR_hpu)
    return QR, QR_hpu


def run_linalg_qr(A, hpu_A, mode):
    def fn(A, mode):
        return torch.linalg.qr(A, mode)

    QR = fn(A, mode)

    fn = compile_function_if_compile_mode(fn)
    QR_hpu = fn(hpu_A, mode)
    return QR, QR_hpu


@pytest.mark.parametrize("dtype", dtypes, ids=format_tc)
@pytest.mark.parametrize("mode", modes)
@pytest.mark.parametrize("qr_function", [run_linalg_qr, run_linalg_qr_out], ids=format_tc)
@pytest.mark.parametrize("shape", shapes, ids=format_tc)
def test_hpu_linalg_qr_random(dtype, mode, qr_function, shape):
    if mode == "r" and shape[-2] == 1:
        pytest.skip("r mode with m=1 is failing on CI")
    cpu_A = torch.randn(shape, dtype=dtype)
    hpu_A = cpu_A.to("hpu")
    cpu_A = cpu_A.to(torch.float32)  # bfloat not supported on CPU

    (Q, R), (Q_hpu, R_hpu) = qr_function(cpu_A, hpu_A, mode)

    assert Q.shape == Q_hpu.shape
    assert R.shape == R_hpu.shape

    Q, R = make_diagonal_positive_batched(Q, R, mode)
    Q_hpu, R_hpu = make_diagonal_positive_batched(Q_hpu, R_hpu, mode)

    atol = 2.4 if dtype == torch.bfloat16 else 1e-5
    rtol = 0.12 if dtype == torch.bfloat16 else 1e-5

    compare_tensors(R_hpu, R, atol, rtol)
    if mode == "r":
        assert Q.numel() == 0
    else:
        c = R_hpu.shape[-1]
        compare_tensors(Q_hpu[..., :c], Q[..., :c], atol, rtol)
        compare_tensors(Q_hpu @ R_hpu, Q @ R, atol, rtol)
        k = Q.shape[-1]
        ref = torch.eye(k, dtype=dtype)
        ref = ref.expand(*Q.shape[:-2], k, k)
        compare_tensors(Q.mT @ Q, ref, atol, rtol)

    if is_pytest_mode_compile():
        check_ops_executed_in_jit_ir("linalg_qr")
