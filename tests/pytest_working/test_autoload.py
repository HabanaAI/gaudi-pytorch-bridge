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

import os

import pytest

is_disabled = os.getenv("TORCH_DEVICE_BACKEND_AUTOLOAD", "1") == "0"
# In order for these tests to not collide with other tests in pytest_working, there is an env flag so that
# pytest doesn't import habana_frameworks.torch during config (see pytest_configure() in conftest.py)
# If this flag isn't set, the other tests are meant to run and those in this file should be skipped
do_not_import_habana_torch_flag = os.getenv("DO_NOT_IMPORT_HABANA_TORCH", "0") == "1"

# These tests should be skipped if the autoload mechanism is disabled with env flag


# Test if our backend has been automatically imported
@pytest.mark.skipif(
    is_disabled or (do_not_import_habana_torch_flag == False),
    reason="The autoload mechanism is disabled with env flag or DO_NOT_IMPORT_HABANA_TORCH flag is set to 0",
)
def test_backend_autoload():
    import sys

    import torch

    assert "hpu" in dir(torch)
    assert "habana_frameworks.torch" in sys.modules


# Test if hpu is available as a device (torch is linked with support for hpu devices)
@pytest.mark.skipif(
    is_disabled or (do_not_import_habana_torch_flag == False),
    reason="The autoload mechanism is disabled with env flag or DO_NOT_IMPORT_HABANA_TORCH flag is set to 0",
)
def test_hpu_available_as_device():
    import torch

    hpu = torch.device("hpu")

    # This will only work if our backend has been successfully imported
    # (and if a physical device is available, which should always be the case in the CI environment)
    t_hpu = torch.tensor([1], device=hpu)
