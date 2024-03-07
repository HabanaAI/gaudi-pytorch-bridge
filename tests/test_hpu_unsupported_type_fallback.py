import pytest
import torch
from test_utils import env_var_in_scope

pytestmark = pytest.mark.skip(reason="Tests in this file are chaning env variables")

hpu = torch.device("hpu")
cpu = torch.device("cpu")


def test_hpu_half_conversion():
    with env_var_in_scope({"LOG_LEVEL_FALLBACK": 0}):
        t1 = torch.randn([8, 1, 64, 64], dtype=torch.float).to(hpu)
        t1 = t1.half()


if __name__ == "__main__":
    test_hpu_half_conversion()
