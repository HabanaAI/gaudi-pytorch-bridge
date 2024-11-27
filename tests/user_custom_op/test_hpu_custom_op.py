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


import pytest
from common_test import custom_add, custom_topk, is_lazy


@pytest.mark.parametrize("compile", [True, False])
@pytest.mark.parametrize("op", [custom_topk, custom_add])
def test_custom_op(compile, op):
    if is_lazy and compile:
        pytest.skip()

    op(compile, False)
