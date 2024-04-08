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

from functools import reduce

import numpy as np
import pytest
import torch


@pytest.mark.skip(reason="https://jira.habana-labs.com/browse/SW-167770")
@pytest.mark.parametrize("inputs_shape", [(6,), (4, 6), (3, 5, 2)])
@pytest.mark.parametrize("accumulate", [False, True])
def test_index_put_bool_mask_only(inputs_shape, accumulate):
    def fn(tensor, bool_mask, value, accumulate):
        return tensor.index_put([bool_mask], value, accumulate)

    self_numel = reduce(lambda x, y: x * y, list(inputs_shape))
    indices_numel = reduce(lambda x, y: x * y, list(inputs_shape))
    tensor = torch.arange(self_numel).view(inputs_shape)
    mask_in = torch.arange(indices_numel).view(inputs_shape)
    bool_mask = mask_in > indices_numel / 3
    values = torch.tensor([-100])

    torch._dynamo.reset()
    cpu_res = fn(tensor, bool_mask, values, accumulate)

    torch._dynamo.reset()
    compiled_hpu = torch.compile(fn, backend="hpu_backend")
    hpu_res = compiled_hpu(tensor.to("hpu"), bool_mask.to("hpu"), values.to("hpu"), accumulate)

    assert torch.allclose(cpu_res, hpu_res.to("cpu"), rtol=1e-3, atol=1e-3)


@pytest.mark.skip(reason="https://jira.habana-labs.com/browse/SW-167770")
@pytest.mark.parametrize("inputs_shape", [(3, 5, 2)])
@pytest.mark.parametrize("ind_shape", [(3, 5)])
@pytest.mark.parametrize("accumulate", [False])
def test_index_put_bool_adv_indexing(inputs_shape, ind_shape, accumulate):
    def fn(tensor, bool_mask, value, accumulate):
        # tensor.index_put([bool_mask], value, accumulate)
        tensor[bool_mask, :] = value
        return tensor

    self_numel = reduce(lambda x, y: x * y, list(inputs_shape))
    indices_numel = reduce(lambda x, y: x * y, list(ind_shape))
    tensor = torch.arange(self_numel).view(inputs_shape)
    mask_in = torch.arange(indices_numel).view(ind_shape)
    bool_mask = mask_in > indices_numel / 3
    values = torch.tensor([-100])

    torch._dynamo.reset()
    cpu_res = fn(tensor, bool_mask, values, accumulate)

    torch._dynamo.reset()
    compiled_hpu = torch.compile(fn, backend="hpu_backend")
    hpu_res = compiled_hpu(tensor.to("hpu"), bool_mask.to("hpu"), values.to("hpu"), accumulate)

    assert torch.allclose(cpu_res, hpu_res.to("cpu"), rtol=1e-3, atol=1e-3)
