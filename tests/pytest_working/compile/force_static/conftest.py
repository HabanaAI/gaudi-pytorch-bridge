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

import habana_frameworks.torch.internal.bridge_config as bc
import pytest


@pytest.fixture(autouse=True, scope="module")
def setup_teardown_env():
    if 1 == int(os.environ.get("PT_HPU_LAZY_MODE", 1)):
        pytest.skip("This test requires PT_HPU_LAZY_MODE=0")

    ds_org_status = bc.get_pt_hpu_enable_refine_dynamic_shapes()
    bc.set_pt_hpu_enable_refine_dynamic_shapes(True)

    org_stats_path = bc.get_pt_compilation_stats_path()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    stats_path = os.path.join(base_dir, "compile_stats")

    bc.set_pt_compilation_stats_path(stats_path)
    pytest.stats_path = stats_path

    yield
    if ds_org_status == False:
        bc.set_pt_hpu_enable_refine_dynamic_shapes(False)

    bc.set_pt_compilation_stats_path(org_stats_path)
