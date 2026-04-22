###############################################################################
# Copyright (c) 2025-2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
###############################################################################
import os
import shutil

from habana_frameworks.torch.dynamo.debug_utils.logger import get_compile_backend_logger
from habana_frameworks.torch.utils.debug.logger import refresh_logging_folder_path
from test_utils import env_var_in_scope


def test_logger_dir_refresh():
    logger = get_compile_backend_logger()
    test_string = "Let's log something original."
    test_logs_dir = "/tmp/logs_test_logger_dir_refresh/"
    with env_var_in_scope({"HABANA_LOGS": test_logs_dir}):
        refresh_logging_folder_path()
        logger.critical(test_string)
    refresh_logging_folder_path()

    assert os.path.isdir(test_logs_dir), "New logs dir was not created."
    assert os.path.isfile(test_logs_dir + "pytorch_log.txt"), "Log file was not created."
    with open(test_logs_dir + "pytorch_log.txt") as f:
        assert test_string in f.read(), "Log missing in log file."
    shutil.rmtree(test_logs_dir)
