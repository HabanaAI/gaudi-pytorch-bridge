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


@pytest.fixture(autouse=True, scope="package")
def setup_teardown_env():
    if 1 == int(os.environ.get("PT_HPU_LAZY_MODE", 1)):
        pytest.skip("This test requires PT_HPU_LAZY_MODE=0")
    import habana_frameworks.torch.hpu as hthpu

    ds_org_status = hthpu.get_dynamic_shape_status()
    hthpu.enable_dynamic_shape()
    hthpu.enable_optim_output_sif()

    yield
    if ds_org_status == False:
        hthpu.disable_dynamic_shape()
    hthpu.disable_optim_output_sif()
