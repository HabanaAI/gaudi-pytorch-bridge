import pytest
import numpy as np
import random
from typing import Mapping
import numpy as np
import pytest
import os

# Can't import torch module because PT_HPU_LAZY_MODE is set in pytest_configure. If any function needs torch module it must be imported locally


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

    config.stash[backup_env] = os.environ

    if pytest.mode == "eager":
        os.environ["PT_HPU_LAZY_MODE"] = "0"
    elif pytest.mode == "lazy":
        os.environ["PT_HPU_LAZY_MODE"] = "1"
    elif pytest.mode == "compile":
        os.environ["PT_HPU_LAZY_MODE"] = "0"
        # According to Piotr Papierkowski PT_HPU_DETERMINISTIC_ENABLE=1 set's alfa parameter
        # in some graphs/tensors. When using torch.compile such attribute is not defined so
        # this flag shall be ignored by bridge code.
        # os.environ["PT_HPU_DETERMINISTIC_ENABLE"] = "0"

    # import torch after flag is set
    import habana_frameworks.torch  # noqa

    # TODO: assert correct lib was read


def pytest_ignore_collect(collection_path, config):
    return not bool(
        pytest.mode in collection_path.parts or "any_mode" in collection_path.parts
    )


def pytest_unconfigure(config):
    os.environ.clear()
    os.environ.update(config.stash[backup_env])
