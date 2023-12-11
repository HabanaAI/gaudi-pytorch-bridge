#!/usr/bin/env python3
###############################################################################
# Copyright (C) 2023 Habana Labs, Ltd. an Intel Company
# All Rights Reserved.
#
# Unauthorized copying of this file or any element(s) within it, via any medium
# is strictly prohibited.
# This file contains Habana Labs, Ltd. proprietary and confidential information
# and is subject to the confidentiality and license agreements under which it
# was provided.
#
###############################################################################

import os

from build_profiles.version import Version

import build

FORK_BUILD_DIR = "/tmp/PYTORCH_FORK_RELEASE_BUILD/"
MODULES_BUILD_DIR = "/tmp/PYTORCH_MODULES_RELEASE_BUILD/"


def test_locating_torch_wheel(fs, monkeypatch):
    monkeypatch.setenv("PYTORCH_FORK_RELEASE_BUILD", FORK_BUILD_DIR)
    monkeypatch.setenv("PYTORCH_MODULES_RELEASE_BUILD", MODULES_BUILD_DIR)
    whl_in_fork_pkgs = (
        FORK_BUILD_DIR + "/pkgs/torch-2.1.0a0+git0ec8fb6-cp310-cp310-linux_x86_64.whl"
    )
    fs.create_file(whl_in_fork_pkgs)
    pt_ver = Version("2.1.0")

    whl = build.locate_fork_wheel(pt_ver)

    assert os.path.normpath(whl) == os.path.normpath(whl_in_fork_pkgs)
