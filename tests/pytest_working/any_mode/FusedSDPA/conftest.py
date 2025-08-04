###############################################################################
# Copyright (c) 2021-2025 Intel Corporation
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

import pandas as pd
import pytest

total_tc_list = None
csv_file_name = None


def print_env_var(env_name, default_value):
    flag = str(default_value)
    if os.getenv(env_name) is not None:
        flag = os.getenv(env_name)
    print(env_name + "=" + flag)


def print_sdpa_env_vars():
    print("\n======================= HABANA ATTENTION CONFIGURATIONS ==================================")
    print_env_var("PT_HPU_SDPA_QKV_SLICE_MODE_FWD", 0)
    print_env_var("PT_HPU_SDPA_QKV_SLICE_MODE_BWD", 0)
    print_env_var("PT_HPU_SDPA_BC_FACTOR", 1024)
    print_env_var("PT_HPU_SDPA_BR_FACTOR", 1024)
    print_env_var("PT_HPU_QKV_SLICE_SEQ_LEN_THLD", 8192)
    print_env_var("ENABLE_EXPERIMENTAL_FLAGS", False)
    print_env_var("RUN_TPC_FUSER", True)
    print_env_var("PT_HPU_USE_ATEN_SDPA", True)
    print("=========================================================================================")


def set_env_var(env_name, value):
    if os.getenv(env_name) is None:
        os.environ[env_name] = str(value)


def setup_env(cmdopt):
    fa_mode_run = cmdopt["fa_mode"]
    if fa_mode_run in ["3.0", "3"]:
        set_env_var("PT_HPU_SDPA_QKV_SLICE_MODE_FWD", 1)
        set_env_var("PT_HPU_SDPA_QKV_SLICE_MODE_BWD", 1)
        set_env_var("ENABLE_EXPERIMENTAL_FLAGS", True)
        set_env_var("RUN_TPC_FUSER", False)


# Define the setup_globals function
def setup_globals(cmdopt):
    global csv_file_name, total_tc_list
    fa_mode_run = cmdopt["fa_mode"]
    csv_file_name = "sdpa_config.csv"
    if fa_mode_run in ["3.0", "3"]:
        csv_file_name = "sdpa_3.0_config.csv"

    print(csv_file_name)
    current_dir = os.path.dirname(__file__)
    csv_file_path = os.path.join(current_dir, csv_file_name)
    print("CSV path:", csv_file_path)

    config_reader = pd.read_csv(csv_file_path, skiprows=17)

    # Filter rows where the first column is a digit
    total_tc_list = config_reader.values.tolist()
    if fa_mode_run not in ["3", "3.0"]:
        total_tc_list = None

    print("Global variables initialized:", csv_file_name, total_tc_list)


# Define a session-scoped fixture for setup and teardown
@pytest.fixture(scope="session", autouse=True)
def setup_and_teardown_session(request):
    original_env = os.environ.copy()
    # Get cmdopt from pytest config
    cmdopt = {"fa_mode": request.config.getoption("--fa_mode")}

    setup_env(cmdopt)
    print_sdpa_env_vars()

    request.config.total_tc_list = total_tc_list

    yield

    os.environ.clear()
    os.environ.update(original_env)


def pytest_addoption(parser):
    parser.addoption("--fa_mode", action="store", default="no", help="Run with fa_mode configuration")


def pytest_sessionstart(session):
    config = session.config
    cmdopt = {"fa_mode": config.getoption("--fa_mode")}
    setup_globals(cmdopt)
    config.total_tc_list = total_tc_list
