# ******************************************************************************
# Copyright (C) 2023 Habana Labs, Ltd. an Intel Company
# All Rights Reserved.
#
# Unauthorized copying of this file or any element(s) within it, via any medium
# is strictly prohibited.
# This file contains Habana Labs, Ltd. proprietary and confidential information
# and is subject to the confidentiality and license agreements under which it
# was provided.
#
# ******************************************************************************

import torch
import numpy as np
from torch import nn
import habana_frameworks.torch.core as htcore
from copy import deepcopy


# Test weight norm for Linear and Conv2d for each
# of the weight dims.
cpu = "cpu"
hpu = "hpu"

torch.manual_seed(123456)
bias = True
in_feat  = 10
out_feat = 20
N = 4
s_i = (4, in_feat)
s_o = (N, out_feat)

C = 10
H = 20
W = 20
K = C
R = 5 # give an odd no so that padding can be integer
S = R
padding = int((R-1)/2)

rtol=1e-03
atol=1e-03
class nNet(nn.Module):
    def __init__(self, w, layer_name, dim):
        super().__init__()
        if layer_name == "linear":
            m = nn.Linear(in_feat, out_feat, bias=bias)
        else:
            m = nn.Conv2d(C, K, R, padding = padding, bias = bias)
        m.weight = nn.Parameter(w)
        self.layer = nn.utils.weight_norm(m, dim = dim)

    def forward(self, x):
        T2 = self.layer(x)
        return T2

def test_weight_norm_fwd_bwd(ln, device, w, x, g_in, dim):
    x = x.to(device)
    g_in = g_in.to(device)

    x = x.to(device).detach().requires_grad_()
    model = nNet(w, ln, dim)

    model = model.to(device)

    t2 = model(x)
    #print(" fwd o/p ", t2.to("cpu"))
    #print(" weight_g ", model.layer.weight_g.to("cpu"))
    #print(" weight_v ", model.layer.weight_v.to("cpu"))
    t2.backward(g_in)
    if device =="hpu":
        htcore.mark_step()
    x_grad = x.grad.to("cpu")
    wg_grad = model.layer.weight_g.grad.to("cpu")
    wv_grad = model.layer.weight_v.grad.to("cpu")
    b_grad = None
    if bias:
        b_grad = model.layer.bias.grad.to("cpu")
    #print("x_grad = ", x_grad)
    #print("wg_grad = ", wg_grad)
    #print("wv_grad = ", wv_grad)
    return x_grad,wg_grad,wv_grad,b_grad

def test_weight_norm(layer_name):
    if layer_name == "linear":
        x = torch.randn(s_i)
        g_in = torch.randn(s_o)
        # Linear weight is in out_feat, in_feat order
        w = torch.randn(out_feat, in_feat)
    else:
        x = torch.randn(N, C, H, W)
        g_in = torch.randn(N, C, H, W)
        w = torch.randn(K, C, R, S)

    print(100*"*")
    for dim in range(w.dim()):
        print(100*"*")
        print(" Testing  ", layer_name, " on dim = ", dim)
        print(100*"*")
        x_grad_c, wg_grad_c ,wv_grad_c, b_grad_c = test_weight_norm_fwd_bwd(layer_name, cpu, w, x, g_in, dim)
        x_grad_h, wg_grad_h ,wv_grad_h, b_grad_h = test_weight_norm_fwd_bwd(layer_name, hpu, w, x, g_in, dim)

        print(" input grads match? = ", torch.allclose(x_grad_c, x_grad_h, rtol=rtol, atol=atol))
        print(" wg grads match? = ", torch.allclose(wg_grad_c, wg_grad_h, rtol=rtol, atol=atol))
        print(" wv grads match? = ", torch.allclose(wv_grad_c, wv_grad_h, rtol=rtol, atol=atol))
        if bias:
            print(" bias grads match? = ", torch.allclose(b_grad_c, b_grad_h, rtol=rtol, atol=atol))

        print(100*"=")

if __name__ == '__main__':
    test_weight_norm("linear")
    print(100*"+")
    test_weight_norm("convolution")
