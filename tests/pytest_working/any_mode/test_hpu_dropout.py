###############################################################################
# Copyright (C) 2023 Habana Labs, Ltd. an Intel Company
# All Rights Reserved.
#
# Unauthorized copying of this file or any element(s) within it, via any medium
# is strictly prohibited.
# This file contains Habana Labs, Ltd. proprietary and confidential information
# and is subject to the confidentiality and license agreements under which it
# was provided.
#
###############################################################################

import pytest
import torch


@pytest.mark.parametrize("p", [0.0, 0.2, 0.6, 1.0])
@pytest.mark.parametrize("dtype ", [torch.float, torch.bfloat16])
def test_hpu_dropout_fwd(p, dtype):
    shape = (32, 48)
    input = torch.randn(shape, requires_grad=True, dtype=dtype).to("hpu")
    dropout_fwd = torch.nn.Dropout(p=p)
    if pytest.mode == "compile":
        dropout_fwd = torch.compile(dropout_fwd, backend="aot_hpu_training_backend")
    out = dropout_fwd(input)

    if p == 0.0:
        assert torch.equal(input, out)
    elif p == 1.0:
        assert torch.equal(out, torch.zeros(shape, dtype=dtype, device="hpu"))
    else:
        nonzeros_p = torch.count_nonzero(out) / input.numel()
        assert torch.abs(nonzeros_p - (1.0 - p)) < 0.02

        nonzeros_idx = out != 0.0
        assert torch.allclose(
            out[nonzeros_idx], input[nonzeros_idx] * (1.0 / (1.0 - p))
        )


@pytest.mark.parametrize("p", [0.0, 0.2, 0.6, 1.0])
@pytest.mark.parametrize("train ", [True, False])
@pytest.mark.parametrize("dtype ", [torch.float, torch.bfloat16])
def test_dropout_bwd(p, train, dtype):
    input = torch.randn((32, 48), dtype=dtype)
    input_hpu = input.to("hpu").requires_grad_(True)
    input = input.requires_grad_(True)

    def dropout_bwd(input, p, train):
        return torch.dropout(input, p, train).sum()

    if pytest.mode == "compile":
        dropout_bwd = torch.compile(dropout_bwd, backend="aot_hpu_training_backend")

    result = dropout_bwd(input, p, train)
    result.backward()
    input_grad = input.grad

    result_hpu = dropout_bwd(input_hpu, p, train)
    result_hpu.backward()
    input_hpu_grad_c = input_hpu.grad.cpu()
    result_hpu_c = result_hpu.cpu()

    if p in [0.0, 1.0] or not train:
        assert torch.equal(result_hpu_c, result)
        assert torch.equal(input_hpu_grad_c, input_grad)
    else:
        unique_hpu = torch.unique(input_hpu_grad_c)
        unique_cpu = torch.unique(input_grad)
        assert torch.equal(unique_hpu, unique_cpu)

        hpu_grad_p = torch.count_nonzero(input_hpu.grad) / input.numel()
        assert torch.abs(hpu_grad_p - (1.0 - p)) < 0.02
