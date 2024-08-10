###############################################################################
# Copyright (C) 2024 Habana Labs, Ltd. an Intel Company
# All Rights Reserved.
#
# Unauthorized copying of this file or any element(s) within it, via any medium
# is strictly prohibited.
# This file contains Habana Labs, Ltd. proprietary and confidential information
# and is subject to the confidentiality and license agreements under which it
# was provided.
#
###############################################################################
import habana_frameworks.torch.core as htcore
import pytest
import torch
from test_utils import format_tc, is_gaudi1

dtypes = [torch.float32, torch.bfloat16, torch.float16] if not is_gaudi1() else [torch.float32, torch.bfloat16]

atol = {torch.float32: 0.001, torch.float16: 0.001, torch.bfloat16: 0.01}
rtol = {torch.float32: 0.001, torch.float16: 0.001, torch.bfloat16: 0.01}


@pytest.mark.parametrize("size", [(3, 9), (1, 7, 100)], ids=format_tc)
@pytest.mark.parametrize("use_weight", [False, True], ids=format_tc)
@pytest.mark.parametrize("reduction", ["none", "mean", "sum"], ids=format_tc)
@pytest.mark.parametrize("dtype", dtypes, ids=format_tc)
def test_hpu_lazy_cross_entropy_fwd(size, use_weight, reduction, dtype):

    C = size[1]

    # CPU: doesn't support Half dtype
    source = torch.rand(size, dtype=torch.float32 if dtype == torch.float16 else dtype)
    target = torch.randint(0, C, size[:1] + size[2:])
    weights = torch.rand(C, dtype=torch.float32 if dtype == torch.float16 else dtype) if use_weight else None

    reference = torch.nn.functional.cross_entropy(source, target, weight=weights, reduction=reduction)
    reference = reference.to(dtype) if dtype == torch.float16 else reference

    # HPU
    source_hpu = source.to(device="hpu", dtype=dtype)
    target_hpu = target.to("hpu")
    weights_hpu = weights.to(device="hpu", dtype=dtype) if use_weight else None

    actual = torch.nn.functional.cross_entropy(source_hpu, target_hpu, weight=weights_hpu, reduction=reduction)
    actual = actual.to("cpu")

    assert torch.allclose(reference, actual, atol=atol[dtype], rtol=rtol[dtype])
