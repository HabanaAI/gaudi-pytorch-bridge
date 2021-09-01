import os
import torch
from timeit import default_timer as timer
import pytest

os.environ["HBN_SYNAPSE_LOGGER_COMMANDS"] = "use_null_backend"
from habana_frameworks.tensorflow import load_habana_module
load_habana_module()

def test_null_device():
    hpu = torch.device('hpu')
    cpu = torch.device('cpu')
    a = torch.randn(10, 10).to(hpu)
    b = torch.randn(10, 10).to(hpu)
    start = timer()
    for loop in range(10000):
        result = torch.add(a, b)
    end = timer()
    print("With null backend, execution time is {:.4f} seconds for 10000 iterations of add, "
        "input shapes are [10, 10].".format(end - start))

if __name__ == "__main__":
    test_null_device()
