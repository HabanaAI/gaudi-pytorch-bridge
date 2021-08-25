import torch
import random
import os
import pytest
from torch.utils.data import _utils, Dataset, TensorDataset, DataLoader
from habana_frameworks.torch.utils.library_loader import load_habana_module
load_habana_module()

def test_hpu_pin_memory():
    x = torch.randn(10)
    print("Is pinned memory", x.is_pinned(device='hpu'))
    inps = torch.arange(10 * 5, dtype=torch.float32).view(10, 5)
    tgts = torch.arange(10 * 5, dtype=torch.float32).view(10, 5)
    dataset = TensorDataset(inps, tgts)
    loader = DataLoader(dataset, pin_memory=True, pin_memory_device='hpu')
    for input, target in loader:
        print("Is pinned memory", input.is_pinned(device='hpu'))
        print("Is pinned memory", target.is_pinned(device='hpu'))
        input, target = input.to('hpu'), target.to('hpu')

