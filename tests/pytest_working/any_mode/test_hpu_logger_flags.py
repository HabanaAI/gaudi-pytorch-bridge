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


from habana_frameworks.torch.utils import _debug_C
from habana_frameworks.torch.utils.debug.logger import enable_logging, get_log_level


def test_hpu_logger_values():
    def check(log_type):
        enable_logging("LOG_LEVEL_ALL", log_type)
        assert _debug_C.is_log_python_enabled(get_log_level(log_type))

    assert not _debug_C.is_log_python_enabled(get_log_level("info"))
    check("info")
    check("trace")
    check("critical")
