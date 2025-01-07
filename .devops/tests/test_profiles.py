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

import build_profiles.profiles as profiles
import pytest
from build_profiles.version import Version


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
    assert profiles.get_version_literal_and_source("previous") == profiles.VersionLiteralAndSource("1.0", "pypi")
    assert profiles.get_version_literal_and_source("rc") is None
    assert profiles.get_version_literal_and_source("nightly") == profiles.VersionLiteralAndSource(
        "nightly", r"https://download.pytorch.org/whl/nightly/cpu"
    )


def test_get_available_profiles():
    assert profiles.get_available_profiles() == [f"test{i}" for i in range(1, 8)]


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

    with pytest.raises(RuntimeError, match=r".*profile.*pt-versions"):
        profiles.get_args_for_profile("test3")

    with pytest.raises(RuntimeError, match=r".*both.*pt_versions.*wheels"):
        profiles.get_args_for_profile("test4")

    with pytest.raises(RuntimeError, match=r".*internal.*does not specify any valid pt-versions.*"):
        profiles.get_args_for_profile("test5")

    assert profiles.get_args_for_profile("test6") == [
        "-c",
        "--build-whl",
        "--wheel-spec",
        "habana-pytorch:1.2,1.0:standard",
        "habana-pytorch-internal:nightly:optional",
    ]

    with pytest.raises(RuntimeError, match=r".*neither.*pt.versions.*wheels"):
        profiles.get_args_for_profile("test7")


def test_get_available_versions():
    assert [x.version for x in sorted(profiles.get_available_versions())] == sorted(["1.0.0", "1.2.3"])


def test_get_required_pt():
    assert (
        profiles.get_required_pt(
            profiles.get_version_literal_and_source("current").version,
            profiles.RequirementPurpose.RUNTIME,
        )
        == "pytorch==1.2"
    )
    assert (
        profiles.get_required_pt(
            profiles.get_version_literal_and_source("nightly").version,
            profiles.RequirementPurpose.RUNTIME,
        )
        == "pt-nightly-cpu"
    )
    assert (
        profiles.get_required_pt(
            profiles.get_version_literal_and_source("current").version,
            profiles.RequirementPurpose.BUILD,
        )
        == "pytorch==1.2"
    )


def test_get_wheel_install_requires():
    assert profiles.get_wheel_install_requires([Version("1.2.3")]) == "pytorch >= 1.2.3, <= 1.2.3"
    assert profiles.get_wheel_install_requires([Version("1.0.3"), Version("1.2.3")]) == "pytorch >= 1.0.3, <= 1.2.3"
    assert (
        profiles.get_wheel_install_requires([Version("1.0.3"), Version("2.2.4-rc2"), Version("2.2.3")])
        == "pytorch >= 1.0.3, <= 2.2.4rc2"
    )
    assert profiles.get_wheel_install_requires([Version("1.2.3"), Version("9.9.9", label="nightly")]) == ""
