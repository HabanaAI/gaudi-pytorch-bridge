###############################################################################
#
#  Copyright (c) 2024-2025 Intel Corporation
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
import subprocess
import sys

fake_profile_args = ["--profile", f"{os.getenv('PYTORCH_MODULES_ROOT_PATH')}/.devops/tests/dummy_profiles.json"]


def _get_stripped_output(cmd):
    return subprocess.check_output(cmd).decode().strip()


def get_profile_getter_output(args):
    return _get_stripped_output([sys.executable, "profile_getter.py"] + args)


def test_get_pt_requirement():
    assert get_profile_getter_output(["--get-pt-requirement", "current"] + fake_profile_args) == "pytorch==1.2"


def test_get_strict_pt_requirement():
    assert get_profile_getter_output(["--get-strict-pt-requirement", "current"] + fake_profile_args) == "pytorch==1.2.3"


def test_get_cmakelists_supported_vers():
    # The output should be two supported versions strings separated by a semicolon
    output = get_profile_getter_output(["--get-cmakelists-supported-vers"] + fake_profile_args)
    supported_vers = output.split(";")
    assert len(supported_vers) == 2
    assert r"1\.0\..*" in supported_vers
    assert r"1\.2\..*" in supported_vers
