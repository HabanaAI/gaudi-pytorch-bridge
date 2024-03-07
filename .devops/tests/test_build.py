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
import subprocess as sp

import build
import pytest
from build_profiles.version import Version

FORK_BUILD_DIR = "/tmp/PYTORCH_FORK_RELEASE_BUILD/"
MODULES_BUILD_DIR = "/tmp/PYTORCH_MODULES_RELEASE_BUILD/"


def test_locating_torch_wheel(fs, monkeypatch):
    monkeypatch.setenv("PYTORCH_FORK_RELEASE_BUILD", FORK_BUILD_DIR)
    monkeypatch.setenv("PYTORCH_MODULES_RELEASE_BUILD", MODULES_BUILD_DIR)
    whl_in_fork_pkgs = FORK_BUILD_DIR + "/pkgs/torch-2.1.0a0+git0ec8fb6-cp310-cp310-linux_x86_64.whl"
    fs.create_file(whl_in_fork_pkgs)
    pt_ver = Version("2.1.0")

    whl = build.locate_fork_wheel(pt_ver)

    assert os.path.normpath(whl) == os.path.normpath(whl_in_fork_pkgs)


def test_reporting_outof_called_process_errors(monkeypatch):
    def mock_check_output(args, **kwargs):
        raise sp.CalledProcessError(1, args[0].join(" "), "Kaboom!")

    monkeypatch.setattr(sp, "check_output", mock_check_output)

    assert "Kaboom!" in build.outof("💣")


def test_handling_torch_version_detection_errors(monkeypatch):
    def mock_outof(*args, **kwargs):
        return "libopenblas.so.0: cannot open shared object file: No such file or directory"

    monkeypatch.setattr(build, "outof", mock_outof)

    with pytest.raises(SystemExit):
        build.query_installed_pt_ver("venv", "python3")
