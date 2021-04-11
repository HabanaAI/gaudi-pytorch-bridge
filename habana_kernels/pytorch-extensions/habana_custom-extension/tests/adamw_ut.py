import numpy as np
import os
import sys
import torch

from hb_custom import FusedAdamW
from transformers import AdamW

torch.ops.load_library(os.path.join(os.environ['BUILD_ROOT_LATEST'], "libhabana_pytorch_plugin.so"))
habana = torch.device("habana")
cpu = torch.device("cpu")

sys.path.insert(0, os.path.join(os.environ['PYTORCH_MODULES_RELEASE_BUILD']))
import hb_torch

if __name__ == "__main__":
    torch.manual_seed(0)
    d1, d2, lr = 2, 1024, 0.1

    u = torch.rand(d1, d2)
    v = u.clone()
    # print('input ::\n{}'.format(u))

    x = u.detach().to(habana)
    x.requires_grad = True

    # Compute loss
    loss_x = x.sum()

    # Compute gradients of the parameters w.r.t. the loss
    loss_x.backward()

    # Modify the parameters by subtracting the gradient
    optim_x = AdamW([x], lr=lr)

    # print('before adam.step x ::\n{}'.format(x.to(cpu)))
    optim_x.step()
    x_cpu = x.to(cpu)
    # print('after  adam.step x ::\n{}'.format(x_cpu))

    # Enable this env to validate lazy path
    # os.environ['PT_HPU_LAZY_MODE'] = "1"

    y = v.detach().to(habana)
    y.requires_grad = True

    optim_y = FusedAdamW([y], lr=lr)
    hb_torch.mark_step()

    # Compute loss
    loss_y = y.sum()

    # Compute gradients of the parameters w.r.t. the loss
    loss_y.backward()

    # print('before adam_habana.step y ::\n{}'.format(y.to(cpu)))
    # Modify the parameters by subtracting the gradient
    optim_y.step()

    hb_torch.mark_step()

    y_cpu = y.to(cpu)

    # print('after  adam_habana.step y ::\n{}'.format(y_cpu))

    comp = np.allclose(x_cpu.detach().numpy(), y_cpu.detach().numpy(), atol=0.001, rtol=1.e-3, equal_nan=True)

    print('Optimizer output match :: {}'.format(comp))
