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

import pytest
from build_profiles import profiles


def setup_module():
    if hasattr(profiles.get_profiles_json, "PROFILES_JSON"):
        delattr(profiles.get_profiles_json, "PROFILES_JSON")
    profiles.get_profiles_json.JSON_PATH = os.path.join(
        os.getenv("PYTORCH_MODULES_ROOT_PATH"), ".devops/tests/dummy_profiles.json"
    )


def teardown_module():
    delattr(profiles.get_profiles_json, "PROFILES_JSON")
    delattr(profiles.get_profiles_json, "JSON_PATH")


def test_get_version_literal_and_source():
    assert profiles.get_version_literal_and_source("current") == profiles.VersionLiteralAndSource("1.2", "build")
    assert profiles.get_version_literal_and_source("previous") == profiles.VersionLiteralAndSource("1.0", "build")


def test_get_args_for_profile():
    test1_args = [
        "-c",
        "-r",
        "--tidy",
        "--recreate-venv=as_needed",
        "--manylinux",
        "--python-versions=3.8",
        "--pt-versions",
        "1.0",
        "1.2",
    ]
    assert profiles.get_args_for_profile("test1") == test1_args

    assert profiles.get_args_for_profile("test2") == ["--pt-versions", "1.0"]

    assert profiles.get_args_for_profile("test6") == [
        "-c",
        "--build-whl",
        "--wheel-spec",
        "habana-pytorch:1.2,1.0:standard",
    ]

    with pytest.raises(RuntimeError, match=r".*neither.*pt.versions.*wheels"):
        profiles.get_args_for_profile("test7")


def test_get_available_versions():
    assert [x.version for x in sorted(profiles.get_available_versions())] == sorted(["1.0.0", "1.2.3", "1.4.0"])


def test_get_required_pt():
    assert (
        profiles.get_required_pt(
            profiles.get_version_literal_and_source("current").version,
        )
        == "torch==1.2"
    )


def test_get_pt_version_id():
    assert profiles.get_pt_version_id("1.2.3") == "current"
    assert profiles.get_pt_version_id("1.0.0") == "previous"
    assert profiles.get_pt_version_id("1.4.0") == "next"

    with pytest.raises(KeyError, match=r'.*pt_version "1.3.3.7" is not present in profiles.json'):
        profiles.get_pt_version_id("1.3.3.7")


def test_get_cpu_index_url():
    assert profiles.get_cpu_index_url("current") == "default"
    assert profiles.get_cpu_index_url("next") == "none"

    with pytest.raises(KeyError):
        profiles.get_cpu_index_url("nonexistent")
