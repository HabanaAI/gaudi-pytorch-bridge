###############################################################################
#
#  Copyright (c) 2025 Intel Corporation
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
import re

import get_fabi_flag as fabi


def test_get_torch_gcc_version():
    result = fabi.get_torch_gcc_version()
    # example of gcc version: 11.3, match two digits, dot and one digit
    assert re.match(r".*\d{2}\.\d.*", result)


def test_get_gcc_version():
    compiler_path = "/usr/bin/g++"
    result = fabi.get_gcc_version(compiler_path)
    # example of gcc version: 11.3, match two digits, dot and one digit
    assert re.match(r".*\d{2}\.\d.*", result)


def test_get_major_version():
    assert fabi.get_major_version("11.41") == 11
