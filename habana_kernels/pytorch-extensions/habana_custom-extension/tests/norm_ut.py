import numpy as np
import os
import math
import torch

# Our module!
#from hb_custom_C import fused_norm
from hb_custom import FusedClipNorm

torch.ops.load_library(os.path.join(os.environ['BUILD_ROOT_LATEST'], "libhabana_pytorch_plugin.so"))
habana = torch.device("hpu")
cpu = torch.device("cpu")

if __name__ == "__main__":
    d1, d2, num, norm_type = 2, 1024, 5, 2.0
    max_norm_val = 1.0
    vec_cpu, vec_n_cpu,vec_hpu = [], [], []
    for i in range(num):
        u = torch.rand(d1, d2)
        vec_cpu.append(u)
        vec_n_cpu.append(torch.norm(u))
        v = u.detach().to(habana)
        vec_hpu.append(v)
    #max_norm_t = (torch.ones((1))*max_norm_val).to(habana)
    n_cpu = torch.norm(torch.stack(vec_n_cpu), norm_type)
    fn_hpu = FusedClipNorm(max_norm_val)
    n_hpu = fn_hpu.clip_norm(vec_hpu, norm_type)
    max_norm_cpu = float(max_norm_val)
    clip_coef = max_norm_cpu / (n_cpu + 1e-6)
    if clip_coef < 1:
      for p in vec_cpu:
        p.mul_(clip_coef)
    comp = np.allclose(
        n_hpu.to(cpu).detach().numpy(),
        n_cpu.detach().numpy(),
        atol=0.001,
        rtol=1.e-3,
        equal_nan=True)
    print('FusedNorm output match :: {}'.format(comp))
    for p, q in zip(vec_hpu, vec_cpu):
      comp = np.allclose(
        p.to(cpu).detach().numpy(),
        q.detach().numpy(),
        atol=0.001,
        rtol=1.e-3,
        equal_nan=True)
      print('FusedNorm grad param match :: {}'.format(comp))
