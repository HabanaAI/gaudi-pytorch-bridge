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
import habana_frameworks.torch.dynamo.compile_backend
import habana_frameworks.torch.core as htcore

@pytest.mark.parametrize("dim", [0, 1, 2, 3, -1])
@pytest.mark.parametrize("descending", [True, False])
def test_sort(dim, descending):
        def fn(input, dim, descending):
            return torch.sort(input, dim, descending)

        # CPU
        x = torch.randn([12, 10, 8, 6])
        hx = x.to('hpu')

        result1, result2 = fn(x, dim, descending)

        # HPU
        compiled_fn = torch.compile(fn, backend="aot_hpu_training_backend")

        hresult1, hresult2 = compiled_fn(hx, dim, descending)

        assert torch.allclose(result1, hresult1.cpu(), atol = 0.001, rtol = 0.001)
        #https://jira.habana-labs.com/browse/SW-154110
        #assert torch.allclose(result2, hresult2.cpu(), atol = 0.001, rtol = 0.001)
