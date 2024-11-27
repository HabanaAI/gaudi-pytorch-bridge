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
from typing import Pattern

import habana_frameworks.torch as hftorch


def _check_hftorch_path():
    if not isinstance(hftorch.__path__, list) or len(hftorch.__path__) != 1:
        assert (False, "Bad __path__ of habana_frameworks.torch, expecting list with 1 element")


def get_include_dir():
    _check_hftorch_path()
    torch_path = hftorch.__path__[0]
    include_dir = os.path.join(torch_path, "include")
    return include_dir


def get_lib_dir():
    _check_hftorch_path()
    torch_path = hftorch.__path__[0]
    include_dir = os.path.join(torch_path, "lib")
    return include_dir
