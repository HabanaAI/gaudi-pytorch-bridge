import torch
import random
import os
import pytest
from torch.utils.data import _utils, Dataset, TensorDataset, DataLoader
torch.ops.load_library(os.path.join(os.environ['PYTORCH_MODULES_RELEASE_BUILD'], "libhabana_pytorch_plugin.so"))

def test_hpu_pin_memory():
    inps = torch.arange(10 * 5, dtype=torch.float32).view(10, 5)
    tgts = torch.arange(10 * 5, dtype=torch.float32).view(10, 5)
    dataset = TensorDataset(inps, tgts)
    loader = DataLoader(dataset, pin_memory=True)
    for input, target in loader:
        print("Is pinned memory", input.is_pinned())
        print("Is pinned memory", target.is_pinned())
        input, target = input.to('hpu'), target.to('hpu')
        x = torch.randn(10).to('hpu')
        y = torch.empty(10, pin_memory=True)
        y.copy_(x)

