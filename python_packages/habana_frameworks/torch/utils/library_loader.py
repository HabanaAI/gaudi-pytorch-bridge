# ******************************************************************************
# Copyright (C) 2020-2021 Habana Labs, Ltd. an Intel Company
# ******************************************************************************

import os
import sys
import torch

_mandatory_libs = ["libhabana_pytorch_plugin.so"]


def _check_modules_directory(directory):
    if not os.path.isdir(directory):
        return False

    for module in _mandatory_libs:
        if not os.path.isfile(os.path.join(directory, module)):
            return False

    return True


def _get_modules_directory():
    """
    Returns a directory containing Habana modules.
    Directory containing modules is looked up as instructed by the following
    environmental variables, in order, until a location is found with all
    the needed libraries:
        $LD_LIBRARY_PATH
        habana_frameworks
    """

    def get_packaged_libs():
        return os.path.abspath(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib")
        )

    locations = []
    if "LD_LIBRARY_PATH" in os.environ:
        locations += os.environ.get("LD_LIBRARY_PATH").split(":")
    locations.append(get_packaged_libs())

    for directory in locations:
        if _check_modules_directory(directory):
            return directory

    return None


def load_habana_module():
    """Load habana libs"""
    habana_modules_directory = _get_modules_directory()
    if habana_modules_directory is None:
        raise Exception("Cannot find Habana modules")

    print("Loading Habana modules from {}".format(habana_modules_directory))
    for module in _mandatory_libs:
        torch.ops.load_library(
            os.path.abspath(os.path.join(habana_modules_directory, module))
        )
        sys.path.insert(0, habana_modules_directory)
