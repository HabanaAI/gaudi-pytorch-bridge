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

import torch
import pytest
import numpy as np


@pytest.mark.parametrize("shape", [(16, 5, 5)])
@pytest.mark.parametrize("inv_scale_attn", [1.3, 1.0])
@pytest.mark.parametrize("grouped_batch_size", [16])
@pytest.mark.parametrize("use_max", [True, False])
@pytest.mark.parametrize("mode", [0, 1, 15])
@pytest.mark.parametrize("dtype", [torch.float, torch.bfloat16])
def test_scaled_masked_triangular_softmax(
    shape, inv_scale_attn, grouped_batch_size, use_max, mode, dtype
):
    torch.manual_seed(12345)
    if mode == 1 and dtype == torch.float:
        pytest.skip("No LUT version is only supported with bfloat16 datatype")

    batch = shape[0]
    dim1 = shape[1]

    self = torch.randn(shape, dtype=dtype) * 10.0

    start_end_dim = int(batch / grouped_batch_size)
    starts = np.random.randint(0, 2, (start_end_dim,))
    start = starts[0]
    ends = [4] * start_end_dim
    end = ends[0]
    start_end = torch.tensor(np.array([starts, ends])).t()

    # simulates lower triangular softmax with mask
    min_val = torch.finfo(dtype).min
    self_tril = torch.tril(self * torch.tensor(inv_scale_attn, dtype=dtype))
    self_tril[:, :, 0:start] = min_val
    self_tril[:, :, end:] = min_val

    idx = torch.triu_indices(batch, dim1, 1)
    for i in range(batch):
        self_tril[i][idx[0], idx[1]] = min_val

    result = torch.ops.hpu.scaled_masked_triangular_softmax(
        self.to("hpu"),
        start_end.to("hpu"),
        inv_scale_attn,
        grouped_batch_size,
        use_max,
        mode,
    ).cpu()

    result_ref = torch.nn.functional.softmax(self_tril, dim=-1)
    # hpu kernel leaves zeros for masked rows
    if start == 1:
        result_ref[:, 0, :] = 0.0

    atol = 1e-3 if dtype == torch.float else 1e-1
    rtol = atol

    assert torch.allclose(result_ref, result, atol=atol, rtol=rtol)


@pytest.mark.parametrize("shape", [(192, 1, 2048)])
@pytest.mark.parametrize("inv_scale_attn", [1.3, 1.0])
@pytest.mark.parametrize("grouped_batch_size", [64])
@pytest.mark.parametrize("use_max", [True, False])
@pytest.mark.parametrize("mode", [0, 1, 15])
@pytest.mark.parametrize("dtype", [torch.float, torch.bfloat16])
def test_scaled_masked_triangular_softmax_next_token(
    shape, inv_scale_attn, grouped_batch_size, use_max, mode, dtype
):
    torch.manual_seed(12345)
    if mode == 1 and dtype == torch.float:
        pytest.skip("No LUT version is only supported with bfloat16 datatype")

    batch = shape[0]
    dim2 = shape[2]

    self = torch.randn(shape, dtype=dtype)

    # simulates lower triangular softmax with mask
    min_val = torch.finfo(dtype).min
    self_scaled = self * torch.tensor(inv_scale_attn, dtype=dtype)

    starts = [1301, 286, 1292]
    end = 1924
    group_size = 64
    starts_ends = []
    for i in range(3):
        start = starts[i]
        starts_ends += [[start, end]]
        section_start = group_size * i
        self_scaled[section_start : section_start + group_size, :, 0:start] = min_val
        self_scaled[section_start : section_start + group_size, :, end:] = min_val

    start_end = torch.tensor(starts_ends).flatten()

    result = torch.ops.hpu.scaled_masked_triangular_softmax(
        self.to("hpu"),
        start_end.to("hpu"),
        inv_scale_attn,
        grouped_batch_size,
        use_max,
        mode,
    ).cpu()

    result_ref = torch.nn.functional.softmax(self_scaled, dim=-1)

    atol = 1e-2 if dtype == torch.float else 1e-1
    rtol = atol

    assert torch.allclose(result_ref, result, atol=atol, rtol=rtol)
