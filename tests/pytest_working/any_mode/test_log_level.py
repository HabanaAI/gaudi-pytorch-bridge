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
from contextlib import contextmanager

import habana_frameworks.torch.dynamo.compile_backend
import pytest
import torch
from habana_frameworks.torch.dynamo.debug_utils.logger import get_compile_backend_logger
from habana_frameworks.torch.utils import _debug_C
from habana_frameworks.torch.utils.debug.logger import enable_logging
from test_utils import clear_t_compile_logs, is_pytest_mode_compile

log_levels = ["trace", "debug", "info", "warn", "error", "critical"]
logger = get_compile_backend_logger()


@contextmanager
def log_level_all_pt(level):
    prev = _debug_C.get_pt_logging_levels()
    if any(num < 0 or num >= 6 for num in prev.values()):
        raise Exception("Unsupported starting log level")
    enable_logging("LOG_LEVEL_ALL_PT", log_levels[level])
    yield
    for k, v in prev.items():
        enable_logging("LOG_LEVEL_" + k, log_levels[int(v)])


def fn(x):
    y = 2 * x
    x = x * 3 * x
    return x - y


# This test purpose is to test simple operations on the lowest log level(0), becouse there were problems which only occured on low log levels.
# For example https://jira.habana-labs.com/browse/SW-166117 or https://jira.habana-labs.com/browse/SW-116763.
# It does not test the logging mechanisms themselves as stated in https://jira.habana-labs.com/browse/SW-164806.
def test_log_level_all_pt_0():
    level = 0
    failed_message = ""
    with log_level_all_pt(level):
        check_mesg = "check if log level was set correctly for log level test"
        logger.trace(check_mesg)
        try:
            fn_hpu = fn
            if is_pytest_mode_compile():
                torch._dynamo.reset()
                clear_t_compile_logs()
                fn_hpu = torch.compile(fn, backend="hpu_backend")
            input_hpu = torch.ones((2, 2), dtype=torch.bfloat16, device="hpu")
            input_cpu = torch.ones((2, 2), dtype=torch.bfloat16, device="cpu")
            res_hpu = fn_hpu(input_hpu)
            res_cpu = fn(input_cpu)
            assert torch.equal(res_hpu.to("cpu"), res_cpu)
            log_dir = os.environ.get("HABANA_LOGS")
            with open(log_dir + "/pytorch_log.txt", "r") as logs:
                assert any(check_mesg in line for line in logs)
        except Exception as e:
            failed_message = e
    assert failed_message == "", f"Exception occured with log level set to {level}"
