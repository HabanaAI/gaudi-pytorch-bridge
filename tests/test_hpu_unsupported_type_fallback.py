import torch, os
import habana_frameworks.torch.core as htcore

os.environ["PT_HPU_LAZY_MODE"] = "1"
os.environ["LOG_LEVEL_FALLBACK"] = "0"

hpu = torch.device("hpu")
cpu = torch.device("cpu")



def test_hpu_half_conversion():
    t1 = torch.randn([8, 1, 64, 64], dtype=torch.float).tp(hpu)
    t1 = t1.half()


if __name__ == "__main__":
    test_hpu_half_conversion()