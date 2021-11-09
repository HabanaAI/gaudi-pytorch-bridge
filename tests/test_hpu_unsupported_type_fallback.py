import torch, os
import habana_frameworks.torch.core as htcore

os.environ["PT_HPU_LAZY_MODE"] = "1"
os.environ["PT_HPU_LOG_MOD_MASK"] = "FF"
os.environ["PT_HPU_LOG_TYPE_MASK"] = "F"

hpu = torch.device("hpu")
cpu = torch.device("cpu")



def test_hpu_half_conversion():
    t1 = torch.randn([8, 1, 64, 64], dtype=torch.float).tp(hpu)
    t1 = t1.half()


if __name__ == "__main__":
    test_hpu_half_conversion()