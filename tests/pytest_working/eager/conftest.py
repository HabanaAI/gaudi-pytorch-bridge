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

import pytest
from test_utils import generic_setup_teardown_env

@pytest.fixture(autouse=True, scope="package")
def setup_teardown_env():
    yield from generic_setup_teardown_env(
        {"PT_HPU_LAZY_MODE": 0}
    )