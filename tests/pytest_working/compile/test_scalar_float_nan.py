###############################################################################
# Copyright (C) 2023-2024 Habana Labs, Ltd. an Intel Company
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
import habana_frameworks.torch.dynamo.compile_backend
import pytest
import torch


@pytest.mark.skip(reason="https://jira.habana-labs.com/browse/SW-167770")
def test_scalar_float_nan():
    def fn(val):
        return torch.full((2, 2), val, dtype=torch.float, device="hpu")

    compiled_fn = torch.compile(fn, backend="hpu_backend")
    compiled_fn(float("nan"))
