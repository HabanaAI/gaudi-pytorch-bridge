import numpy as np
import os
import math
import torch

# Our module!
from hb_custom_C import fused_norm

torch.ops.load_library(os.path.join(os.environ['BUILD_ROOT_LATEST'], "libhabana_pytorch_plugin.so"))
habana = torch.device("habana")
cpu = torch.device("cpu")

if __name__ == "__main__":
    d1, d2, num, norm_type = 2, 1024, 5, 2.0

    vec_cpu, vec_hpu = [], []
    for i in range(num):
        u = torch.rand(d1, d2)
        vec_cpu.append(torch.norm(u))
        v = u.detach().to(habana)
        vec_hpu.append(v)

    n_cpu = torch.norm(torch.stack(vec_cpu), norm_type)
    n_hpu = fused_norm(vec_hpu, norm_type)

    comp = np.allclose(
        n_hpu.to(cpu).detach().numpy(),
        n_cpu.detach().numpy(),
        atol=0.001,
        rtol=1.e-3,
        equal_nan=True)

    print('FusedNorm output match :: {}'.format(comp))
