import pytest
import numpy as np
import random

import torch

@pytest.fixture(autouse=True)
def reset_seed(seed=0xC001A1):
    print("Using seed: ", seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # multi-GPU.
    # TODO: for future use
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True