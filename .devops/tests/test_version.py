#!/usr/bin/env python
###############################################################################
#
#  Copyright (c) 2021-2025 Intel Corporation
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


import sys

from build_profiles.version import Version, is_official_stable_cpu_version


def test_version_lt():
    assert Version("1") < Version("2")
    assert Version("1.11") < Version("1.12")
    assert Version("1.12.0") < Version("1.12.1")
    assert Version("1.12.0a0") < Version("1.12.0a1")
    assert Version("1.12.0a0") < Version("1.12.1a0")
    assert Version("2.1+git7315acd") < Version("2.1.1")
    assert Version("1.11.0a0+git0000fed") < Version("1.12.0a0+git7315acd")
    assert Version("1.13.0.dev20221006+cpu") < Version("1.14.0")


def test_version_gt():
    assert Version("2.1") > Version("2")
    assert Version("2.1.1") > Version("2.1")
    assert Version("2.1+git7315acd") > Version("2.0+git0000fed")
    assert Version("2.1+git7315acd") > Version("2+git0000fed")
    assert Version("1.13.0") > Version("1.13.0a0")
    assert Version("1.13.0.dev20221006+cpu") > Version("1.12.1")


def test_version_equal_to_same_string():
    assert Version("1") == "1"
    assert Version("1.12") == "1.12"
    assert Version("1.12.0") == "1.12.0"
    assert Version("1.12.0a0") == "1.12.0a0"
    assert Version("1.12.0a0+git7315acd") == "1.12.0a0+git7315acd"


def test_version_equal_to_same_version():
    assert Version("1") == Version("1")
    assert Version("1.12") == Version("1.12")
    assert Version("1.12.0") == Version("1.12.0")
    assert Version("1.12.0a0") == Version("1.12.0a0")
    assert Version("1.12.0a0+git7315acd") == Version("1.12.0a0+git7315acd")


def test_version_unequal_to_different_git_sha():
    assert Version("1.12.0a0+git7315ac") != Version("1.12.0a0+git7890abc")
    assert not Version("1.12.0a0+git7315ac") == Version("1.12.0a0+git7890abc")  # pylint: disable=unneeded-not


def test_version_losslessly_converts_to_and_from_string():
    version_str = "1.2.3.4.5.6a0+git1abcd12"
    assert str(Version(version_str)) == version_str


def test_significant_matches():
    assert Version("2").significant_matches(Version("2"))
    assert not Version("2").significant_matches(Version("1"))
    assert Version("2").significant_matches(Version("2.5.0"))
    assert Version("2.5").significant_matches(Version("2.5.0"))
    assert not Version("2.5").significant_matches(Version("2.4.0"))
    assert not Version("2.5").significant_matches(Version("2.4.0a0+git7315acd"))
    assert Version("2.5").significant_matches(Version("2.5.0+git7315acd"))
    assert Version("2.5.0").significant_matches(Version("2.5.0"))
    assert Version("2.5.0").significant_matches(Version("2.5.0+git7315acd"))
    assert Version("2.5.0+cu90").significant_matches(Version("2.5.0+cu90"))
    assert Version("1.12.0").significant_matches(Version("1.12.0a0+git7315acd"))
    assert Version("1.12.0a0").significant_matches(Version("1.12.0a0"))
    assert Version("2.0.0").significant_matches(Version("2.0.0+cpu.cxx11.abi"))
    assert not Version("1.12.0a0").significant_matches(Version("1.12.0"))


def test_version_from_sys_version_info():
    ver = Version(sys.version_info)
    assert ver.major > 2


def test_is_official_stable_cpu_version():
    assert is_official_stable_cpu_version(Version("1.0.0+cpu"))
    assert is_official_stable_cpu_version(Version("1.0.0+cpu.cxx11.abi"))
    assert not is_official_stable_cpu_version(Version("1.0.0+gpu"))
    assert not is_official_stable_cpu_version(Version("1.0.0a0+cpu"))
    assert not is_official_stable_cpu_version(Version("1.0.0"))
