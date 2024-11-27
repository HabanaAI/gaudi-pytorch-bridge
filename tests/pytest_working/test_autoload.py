###############################################################################
#
#  Copyright (c) 2021-2024 Intel Corporation
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
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
