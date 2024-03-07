# ******************************************************************************
# Copyright (C) 2020-2021 Habana Labs, Ltd. an Intel Company
# ******************************************************************************

import atexit
import os
import subprocess
import sys

import torch
from habana_frameworks.torch import _hpu_C

_mandatory_libs = ["libhabana_pytorch_plugin.so"]
# must be preloaded before _mandatory_libs for profiler to work
_profiler_libs = ["pytorch_synapse_logger.so"]


def _check_modules_directory(directory, library_list=list()):
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


def _get_modules_directory(library_list=list()):
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


def is_habana_avaialble():
    from subprocess import STDOUT, check_output

    cmd = "hl-smi -v"
    status = False
    enable_console = True
    if os.environ.get("ENABLE_CONSOLE") == "false":
        enable_console = False
        os.environ["ENABLE_CONSOLE"] = "true"
    try:
        result = check_output(cmd, stderr=STDOUT, shell=True).decode()
        if result.find("Habana") != -1:
            status = True
    except Exception as e:
        # Workaround to mitigate hl-smi usage on simulators
        if os.environ.get("ENABLE_EXEUTION_ON_GAUDI_SIM") in ["true", "True", "1"]:
            print("Enabling Gaudi Simulator As Habana Device !!")
            p = subprocess.Popen(["pgrep", "coral"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            num_cards = sum(1 for _ in p.stdout)
            if num_cards >= 1:
                status = True
            else:
                status = False
    if enable_console == False:
        os.environ["ENABLE_CONSOLE"] = "false"
    return status


def is_habana_available():
    from subprocess import STDOUT, check_output

    cmd = "hl-smi -v"
    status = False
    enable_console = True
    if os.environ.get("ENABLE_CONSOLE") == "false":
        enable_console = False
        os.environ["ENABLE_CONSOLE"] = "true"
    try:
        result = check_output(cmd, stderr=STDOUT, shell=True).decode()
        if result.find("Habana") != -1:
            status = True
    except Exception as e:
        # Workaround to mitigate hl-smi usage on simulators
        if os.environ.get("ENABLE_EXEUTION_ON_GAUDI_SIM") in ["true", "True", "1"]:
            print("Enabling Gaudi Simulator As Habana Device !!")
            p = subprocess.Popen(["pgrep", "coral"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            num_cards = sum(1 for _ in p.stdout)
            if num_cards >= 1:
                status = True
            else:
                status = False
    if enable_console == False:
        os.environ["ENABLE_CONSOLE"] = "false"
    return status


def _load_habana_module(library_list):
    """Load habana libs"""
    habana_modules_directory = _get_modules_directory(library_list)
    if habana_modules_directory is None:
        raise Exception("Cannot find Habana modules")

    print("Loading Habana modules from {}".format(habana_modules_directory))
    for module in library_list:
        torch.ops.load_library(os.path.abspath(os.path.join(habana_modules_directory, module)))
        sys.path.insert(0, habana_modules_directory)

    atexit.register(_hpu_C.cleanup)


def load_habana_module():
    _load_habana_module(_mandatory_libs)


def load_habana_profiler():
    _load_habana_module(_profiler_libs)
