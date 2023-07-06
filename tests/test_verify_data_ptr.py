import torch
from test_utils import compare_tensors

try:
    import habana_frameworks.torch.utils.experimental as htexp
except ImportError:
    raise AssertionError("Could Not import habana_frameworks.torch.core")


def test_data_ptr():
    i = 0
    while i < 2:
        t1 = torch.empty((2, 2), dtype=torch.float32, device="hpu")
        print("t1 data_ptr ", hex(htexp._data_ptr(t1)))
        t2 = torch.zeros((2, 2), dtype=torch.float32)
        t1.copy_(t2)
        print("t1 data_ptr ", hex(htexp._data_ptr(t1)))
        print(t1.to("cpu"))

        t3 = torch.ones((2, 2), dtype=torch.float32)
        t1.copy_(t3)
        print("t1 data_ptr ", hex(htexp._data_ptr(t1)))
        print(t1.to("cpu"))

        compare_tensors(t1, t3, atol=0, rtol=0)
        i = i + 1


if __name__ == "__main__":
    test_data_ptr()
