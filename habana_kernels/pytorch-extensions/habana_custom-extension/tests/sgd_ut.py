import numpy as np
import os
import sys
import torch

torch.manual_seed(0)

cpu = torch.device("cpu")

if __name__ == "__main__":
    d1, d2, lr = 1, 1024, 0.1
    momentum=0.1
    weight_decay=0.1
    dampening=0.0
    nesterov=False
    cnt = 2

    u1 = torch.rand(d1, d2)
    v1 = u1.clone()
    print('input ::\n{}'.format(u1))

    x1 = u1.detach().to(cpu)
    x1.requires_grad = True

    u2 = torch.rand(d1, d2)
    v2 = u2.clone()
    print('input ::\n{}'.format(u2))

    x2 = u2.detach().to(cpu)
    x2.requires_grad = True

    # Modify the parameters by subtracting the gradient
    optim_x = torch.optim.SGD([x1, x2], lr=lr, momentum=momentum, dampening=dampening, weight_decay=weight_decay, nesterov=nesterov)

    # print('before adam.step x ::\n{}'.format(x.to(cpu)))
    for i in range(0, cnt):
        x = torch.add(x1, x2)

        # Compute loss
        loss_x = x.sum()

        # Compute gradients of the parameters w.r.t. the loss
        loss_x.backward()

        optim_x.step()
        # for group in optim_x.param_groups:
        #     for p in group["params"]:
        #         state = optim_x.state[p]
        #         print("PT mom = ", state['momentum_buffer'].to(cpu))

    #print('after  sgd.step x ::\n{}'.format(x.to(cpu)))

    torch.ops.load_library(os.path.join(os.environ['BUILD_ROOT_LATEST'], "libhabana_pytorch_plugin.so"))
    habana = torch.device("habana")

    sys.path.insert(0, os.path.join(os.environ['PYTORCH_MODULES_RELEASE_BUILD']))
    import hb_torch
    from hb_custom import FusedSGD
    hb_torch.enable_eliminate_common_subexpression(False)
    hb_torch.enable_constant_pooling(False)

    y1 = v1.detach().to(habana)
    y1.requires_grad = True

    y2 = v2.detach().to(habana)
    y2.requires_grad = True

    # Modify the parameters by subtracting the gradient
    optim_y = FusedSGD([y1, y2], lr=lr, momentum=momentum, dampening=dampening, weight_decay=weight_decay, nesterov=nesterov)

    # print('before adam_habana.step y ::\n{}'.format(y.to(cpu)))
    for i in range(0, cnt):
        y = torch.add(y1, y2)

        # Compute loss
        loss_y = y.sum()

        # Compute gradients of the parameters w.r.t. the loss
        loss_y.backward()

        optim_y.step()
        hb_torch.mark_step()
        # for group in optim_y.param_groups:
        #     for p in group["params"]:
        #         state = optim_y.state[p]
        #         print("HPU Mom = ", state['momentum_buffer'].to(cpu))

    #print('after  sgd_habana.step y ::\n{}'.format(y.to(cpu)))

    x1_cpu = x1.to(cpu)
    y1_cpu = y1.to(cpu)

    x2_cpu = x2.to(cpu)
    y2_cpu = y2.to(cpu)

    # print(x1_cpu)
    # print(y1_cpu)
    # print(x2_cpu)
    # print(y2_cpu)

    comp1 = np.allclose(x1_cpu.detach().numpy(), y1_cpu.detach().numpy(), atol=1.e-7, rtol=1.e-5, equal_nan=True)
    comp2 = np.allclose(x2_cpu.detach().numpy(), y2_cpu.detach().numpy(), atol=1.e-7, rtol=1.e-5, equal_nan=True)

    print('Optimizer output match :: {}'.format(comp1))
    print('Optimizer output match :: {}'.format(comp2))
