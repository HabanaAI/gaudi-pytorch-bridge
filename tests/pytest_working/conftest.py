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

import json
import os
import random
from pathlib import Path
from typing import Mapping, Union

import numpy as np
import pytest

# Can't import torch module because PT_HPU_LAZY_MODE is set in pytest_configure. If any function needs torch module it must be imported locally

SKIP_TESTS_LIST = "skip_tests_list.json"
EAGER_FALLBACK_TESTS_LIST = "compile_eager_fallback_list.json"


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
    parser.addoption(
        "--dut", action="store", default="gaudi2", help="{gaudi|gaudi2|gaudi3}, default gaudi2. Choose chip version"
    )


backup_env = pytest.StashKey[Mapping]()


def pytest_runtest_setup(item):

    if (
        pytest.mode == "compile"
        and pytest.chip in pytest.eager_fallback_tests.keys()
        and (
            get_testname(item) in pytest.eager_fallback_tests[pytest.chip]
            or get_testname(item) in pytest.eager_fallback_tests["all"]
        )
        and not os.getenv("PTT_STOP_EAGER_FALLBACK", 0)
    ):
        import warnings

        from habana_frameworks.torch.dynamo.compile_backend.config import configuration_flags

        configuration_flags["use_eager_fallback"] = True
        warnings.warn(
            "Eager fallback allowed for current test. Remove test from compile_eager_fallback_list.json to disable it"
        )


def pytest_runtest_teardown(item):
    if (
        pytest.mode == "compile"
        and pytest.chip in pytest.eager_fallback_tests.keys()
        and (
            get_testname(item) in pytest.eager_fallback_tests[pytest.chip]
            or get_testname(item) in pytest.eager_fallback_tests["all"]
        )
        and not os.getenv("PTT_STOP_EAGER_FALLBACK", 0)
    ):
        from habana_frameworks.torch.dynamo.compile_backend.config import configuration_flags

        configuration_flags["use_eager_fallback"] = os.getenv("PT_HPU_USE_EAGER_FALLBACK", "0") == "1"


def pytest_configure(config):
    pytest.mode = config.getoption("--mode")
    pytest.chip = config.getoption("--dut")
    assert pytest.mode.lower() in ["eager", "lazy", "compile"]

    # CPU fallbacks are not allowed in simple tests
    os.environ["PT_HPU_PLACE_ON_CPU"] = "none"

    config.stash[backup_env] = os.environ

    if pytest.mode == "eager":
        os.environ["PT_HPU_LAZY_MODE"] = "0"
    elif pytest.mode == "lazy":
        os.environ["PT_HPU_LAZY_MODE"] = "1"
    elif pytest.mode == "compile":
        os.environ["PT_HPU_LAZY_MODE"] = "0"
        os.environ["PT_HPU_USE_EAGER_FALLBACK"] = "0"
        try:
            eager_fallback_path = Path(__file__).parent.joinpath(EAGER_FALLBACK_TESTS_LIST)
            with open(eager_fallback_path, "r") as f:
                pytest.eager_fallback_tests = json.load(f)
        except FileNotFoundError:
            import warnings

            warnings.warn(
                f"Unable to find EAGER_FALLBACK_TESTS_LIST under {eager_fallback_path}\nRunning tests without eager fallback lists might result in test suite failure.",
                UserWarning,
            )
    # import torch after flag is set
    import habana_frameworks.torch  # noqa

    # TODO: assert correct lib was read


def pytest_ignore_collect(collection_path, config):
    return not bool(pytest.mode in collection_path.parts or "any_mode" in collection_path.parts)


def pytest_unconfigure(config):
    os.environ.clear()
    os.environ.update(config.stash[backup_env])


def pytest_collection_modifyitems(config, items):
    # skip_dict has structure {"gaudi_version": {"mode": [list of failing tests on specific gaudi for specified mode]}}
    # gaudi_version accepted values: all_gaudi | gaudi | gaudi2 | gaudi3
    # mode accepted values: all | lazy | compile | eager
    skip_dict = {}
    try:
        skip_path = Path(__file__).parent.joinpath(SKIP_TESTS_LIST)
        with open(skip_path, "r") as f:
            skip_dict = json.load(f)
    except FileNotFoundError:
        import warnings

        warnings.warn(
            f"Unable to find skip_tests_list under {skip_path}\nRunning tests without skip lists might result in test suite failure.",
            UserWarning,
        )

    if len(skip_dict) == 0:
        print("Tests skip list is empty.")
        return

    skip_items = []
    for chip in [pytest.chip, "all_gaudi"]:
        for mode in [pytest.mode, "all"]:
            skip_items += skip_dict.get(chip, {}).get(mode, [])

    for item in items:
        skip_marker = pytest.mark.skip("Test present in skip_tests_list.txt")
        if item.nodeid in skip_items:
            item.add_marker(skip_marker)


def get_testname(item: Union[pytest.Function, str]) -> str:
    if isinstance(item, str):
        testname = item
    else:
        testname = item.name
    try:
        if "::" in testname:
            testname = testname.split("::")[1]
    except Exception as e:
        import warnings

        warnings.warn(f"unable to parse testname: {testname}")
    return str(testname)
