import numpy as np
import os
import sys
import torch

from hb_custom import FusedAdamW
from transformers import AdamW

torch.ops.load_library(os.path.join(os.environ['BUILD_ROOT_LATEST'], "libhabana_pytorch_plugin.so"))
habana = torch.device("habana")
cpu = torch.device("cpu")

if __name__ == "__main__":
    d1, d2, lr = 2, 1024, 0.001

    u = torch.rand(d1, d2)
    v = u.clone()
    print('input ::\n{}'.format(u))

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
    print('after  adam.step x ::\n{}'.format(x.to(cpu)))

    y = v.detach().to(habana)
    y.requires_grad = True

    # Compute loss
    loss_y = y.sum()

    # Compute gradients of the parameters w.r.t. the loss
    loss_y.backward()

    # Modify the parameters by subtracting the gradient
    optim_y = FusedAdamW([y], lr=0.001)

    # print('before adam_habana.step y ::\n{}'.format(y.to(cpu)))
    optim_y.step()
    print('after  adam_habana.step y ::\n{}'.format(y.to(cpu)))

    x_cpu = x.to(cpu)
    y_cpu = y.to(cpu)

    comp = np.allclose(x_cpu.detach().numpy(), y_cpu.detach().numpy(), atol=0.001, rtol=1.e-3, equal_nan=True)

    print('Optimizer output match :: {}'.format(comp))
