import numpy as np
import torch
torch.ops.load_library("libhabana_pytorch_plugin.so")


def test_set_one():
    t1 = torch.rand(3)
    print(t1)
    torch.ops.habana_kernels.set_one(t1)
    print(t1)
    assert np.all(t1.numpy() == 1)


test_set_one()
