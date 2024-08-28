# ******************************************************************************
# Copyright (C) 2024 Habana Labs, Ltd. an Intel Company
# All Rights Reserved.
#
# Unauthorized copying of this file or any element(s) within it, via any medium
# is strictly prohibited.
# This file contains Habana Labs, Ltd. proprietary and confidential information
# and is subject to the confidentiality and license agreements under which it
# was provided.
#
# ******************************************************************************

import pytest
from common_test import custom_add, custom_topk, is_lazy


@pytest.mark.parametrize("compile", [True, False])
@pytest.mark.parametrize("op", [custom_topk, custom_add])
def test_custom_op(compile, op):
    if is_lazy and compile:
        pytest.skip()

    op(compile, False)
