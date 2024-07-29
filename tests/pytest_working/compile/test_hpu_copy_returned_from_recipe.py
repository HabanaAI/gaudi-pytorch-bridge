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
# Test for SW-188350
import habana_frameworks.torch as _
import torch


def test_hpu_copy_returned():
    @torch.compile(backend="hpu_backend")
    def compiled_alloc():
        x = torch.tensor([2], dtype=torch.int32, device="hpu")
        return x

    A = compiled_alloc()
    B = compiled_alloc()

    assert A.data_ptr() != B.data_ptr()
