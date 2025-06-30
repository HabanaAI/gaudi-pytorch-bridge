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


import atexit
import os

# We ensure that all commands are constructed safely,
# without user input that could lead to command injection.
import subprocess  # nosec B404
import sys

from habana_frameworks.torch import _hpu_C

import torch

_mandatory_libs = ["libhabana_pytorch_plugin.so"]


def _check_modules_directory(directory, library_list=[]):
    if not os.path.isdir(directory):
        return False

    if not library_list:
        for module in _mandatory_libs:
            if not os.path.isfile(os.path.join(directory, module)):
                return False
    else:
        for module in library_list:
            if not os.path.isfile(os.path.join(directory, module)):
                return False

    return True


def _get_modules_directory(library_list=[]):
    """
    Returns a directory containing Habana modules, which is:
        - habana_frameworks
    """

    def get_packaged_libs():
        return os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))

    locations = []
    locations.append(get_packaged_libs())

    for directory in locations:
        if _check_modules_directory(directory, library_list):
            return directory

    return None


def is_habana_available():
    cmd = "hl-smi -v"
    status = False
    enable_console = True
    if os.environ.get("ENABLE_CONSOLE") == "false":
        enable_console = False
        os.environ["ENABLE_CONSOLE"] = "true"
    try:
        result = subprocess.check_output(cmd, stderr=subprocess.STDOUT, shell=True).decode()  # noqa S602
        if result.find("Habana") != -1:
            status = True
    except Exception:
        # Workaround to mitigate hl-smi usage on simulators
        if os.environ.get("ENABLE_EXEUTION_ON_GAUDI_SIM") in ["true", "True", "1"]:
            print("Enabling Gaudi Simulator As Habana Device !!")
            p = subprocess.Popen(["pgrep", "coral"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)  # noqa S603 # noqa S607
            num_cards = sum(1 for _ in p.stdout)
            if num_cards >= 1:
                status = True
            else:
                status = False
    if enable_console is False:
        os.environ["ENABLE_CONSOLE"] = "false"
    return status


def _load_habana_module(library_list):
    """Load habana libs"""
    habana_modules_directory = _get_modules_directory(library_list)
    if habana_modules_directory is None:
        raise Exception("Cannot find Habana modules")

    print(f"Loading Habana modules from {habana_modules_directory}")
    for module in library_list:
        torch.ops.load_library(os.path.abspath(os.path.join(habana_modules_directory, module)))
        sys.path.insert(0, habana_modules_directory)

    atexit.register(_hpu_C.cleanup)


def load_habana_module():
    _load_habana_module(_mandatory_libs)
