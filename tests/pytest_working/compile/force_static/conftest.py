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
