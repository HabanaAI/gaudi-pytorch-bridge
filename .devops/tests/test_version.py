#!/usr/bin/env python
###############################################################################
# Copyright (C) 2021-2023 Habana Labs, Ltd. an Intel Company
# All Rights Reserved.
#
# Unauthorized copying of this file or any element(s) within it, via any medium
# is strictly prohibited.
# This file contains Habana Labs, Ltd. proprietary and confidential information
# and is subject to the confidentiality and license agreements under which it
# was provided.
#
###############################################################################

import sys

from build_profiles.version import Version


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
