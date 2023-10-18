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

import os
import pytest
import torch
from test_utils import compare_tensors


@pytest.mark.parametrize("size", [(1,), (2, 3)])
@pytest.mark.parametrize(
    "dtype, fill_value",
    [
        (torch.float, 2.5),
        (torch.bfloat16, 2.5),
        (torch.int16, 42),
        (torch.int32, 42),
        (torch.int64, 42),
        (torch.int64, -42),
        (torch.int64, 123456789123456789),
        (torch.int64, -123456789123456789),
    ],
)
def test_full(size, dtype, fill_value):
    if (
        abs(fill_value) > 0x7FFFFFFF
        and int(os.environ.get("PT_ENABLE_INT64_SUPPORT", "0")) == 0
    ):
        pytest.skip(reason="fill_value exceed int32 range which is unsupported")

    expected = torch.full(size, fill_value=fill_value, dtype=dtype, device="cpu")
    result = torch.full(size, fill_value=fill_value, dtype=dtype, device="hpu")

    compare_tensors([result], [expected], atol=0, rtol=0)
