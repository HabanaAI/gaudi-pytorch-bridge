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
import habana_frameworks.torch.hpu as ht


@pytest.mark.xfail(
    reason="https://jira.habana-labs.com/browse/SW-158296"
)
@pytest.mark.parametrize("shape", [(4, 6, 8), (8, 8, 4, 16)])
@pytest.mark.parametrize("dtype", [torch.float, torch.bfloat16])
def test_softmax_retain_fwd(shape, dtype):
    torch.manual_seed(12345)
    input = torch.randn(shape, dtype=dtype).to("hpu")
    output, max, sum_exp = torch.ops.hpu.retain_softmax_producer(input)
    result_quick = torch.ops.hpu.retain_softmax_consumer(input, max, sum_exp)

    assert torch.equal(output, result_quick)
