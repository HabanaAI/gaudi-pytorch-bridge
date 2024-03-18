###############################################################################
# Copyright (C) 2023-2024 Habana Labs, Ltd. an Intel Company
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
import random
from pathlib import Path
from typing import Mapping

import numpy as np
import pytest

# Can't import torch module because PT_HPU_LAZY_MODE is set in pytest_configure. If any function needs torch module it must be imported locally

SKIP_TESTS_LIST = "skip_tests_list.txt"


@pytest.fixture(autouse=True)
def reset_seed(seed=0xC001A1):
    import torch

    print("Using seed: ", seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # multi-GPU.
    # TODO: for future use
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True


def pytest_addoption(parser):
    parser.addoption(
        "--mode",
        action="store",
        default="eager",
        help="{eager|lazy|graph}, default eager. Choose mode to run tests",
    )


backup_env = pytest.StashKey[Mapping]()


def pytest_configure(config):
    pytest.mode = config.getoption("--mode")
    assert pytest.mode.lower() in ["eager", "lazy", "compile"]

    # CPU fallbacks are not allowed in simple tests
    os.environ["PT_HPU_PLACE_ON_CPU"] = "none"

    config.stash[backup_env] = os.environ

    # TODO: remove after SW-175380 is fixed
    os.environ["PT_HPU_STOCHASTIC_ROUNDING_MODE"] = "0"

    if pytest.mode == "eager":
        os.environ["PT_HPU_LAZY_MODE"] = "0"
    elif pytest.mode == "lazy":
        os.environ["PT_HPU_LAZY_MODE"] = "1"
    elif pytest.mode == "compile":
        os.environ["PT_HPU_LAZY_MODE"] = "0"
        os.environ["PT_HPU_USE_EAGER_FALLBACK"] = "0"

    # import torch after flag is set
    import habana_frameworks.torch  # noqa

    # TODO: assert correct lib was read


def pytest_ignore_collect(collection_path, config):
    return not bool(pytest.mode in collection_path.parts or "any_mode" in collection_path.parts)


def pytest_unconfigure(config):
    os.environ.clear()
    os.environ.update(config.stash[backup_env])


def pytest_collection_modifyitems(config, items):
    skip_list = []
    try:
        skip_path = Path(__file__).parent.joinpath(SKIP_TESTS_LIST)
        with open(skip_path, "r") as f:
            skip_list = [l.strip() for l in f]
    except FileNotFoundError:
        import warnings

        warnings.warn(
            f"Unable to find skip_tests_list under {skip_path}\nRunning tests without skip lists might result in test suite failure.",
            UserWarning,
        )

    if len(skip_list) == 0:
        print("Tests skip list is empty.")
        return

    for item in items:
        skip_marker = pytest.mark.skip("Test present in skip_tests_list.txt")
        if item.nodeid in skip_list:
            item.add_marker(skip_marker)
