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

import os

import pytest

# TODO: remove after SW-175380 is fixed
os.environ["PT_HPU_STOCHASTIC_ROUNDING_MODE"] = "0"

from test_utils import generic_setup_teardown_env


def setup_teardown_env():
    if 0 == int(os.environ.get("PT_HPU_LAZY_MODE", 1)):
        pytest.skip("This test requires PT_HPU_LAZY_MODE=1")
