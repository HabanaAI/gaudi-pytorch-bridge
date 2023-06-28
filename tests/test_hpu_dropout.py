###############################################################################
# Copyright (C) 2023 Habana Labs, Ltd. an Intel Company
# All Rights Reserved.
#
# Unauthorized copying of this file or any element(s) within it, via any medium
# is strictly prohibited.
# This file contains Habana Labs, Ltd. proprietary and confidential information
# and is subject to the confidentiality and license agreements under which it
# was provided.
#
###############################################################################

import torch
import pytest


def test_hpu_dropout():
    a_cpu = torch.randn(2, 2, requires_grad=True).detach()
    a = a_cpu.to("hpu")
    a.requires_grad = True
    dropoutmod = torch.nn.Dropout(p=1.0).to("hpu")
    out = dropoutmod(a)
    grad_out_cpu = torch.randn((2, 2)).detach()
    grad_out = grad_out_cpu.to("hpu")
    grad_out.requires_grad = False
    out.backward(grad_out)
    grad_in = a.grad
    # print(a.to('cpu'), out.to('cpu'), grad_out.to('cpu'), grad_in.to('cpu'))



@pytest.mark.skip(reason="Tests in this file are chaning env variables")
@pytest.mark.parametrize('setup_teardown_env_fixture', [
    {"PT_HPU_LAZY_MODE": "1"}], indirect=True)
def test_hpu_dropout_lazy(setup_teardown_env_fixture):
    a = torch.randn(2, 2, requires_grad=True).to("hpu")
    dropoutmod = torch.nn.Dropout(p=0.3)
    out = dropoutmod(a)
    grad_out = torch.randn((2, 2), requires_grad=False)
    grad_in = out.grad_fn(grad_out.to("hpu"))
    out.to("cpu")
    grad_in.to("cpu")
    # print(a.to('cpu'), out.to('cpu'), grad_out.to('cpu'), grad_in.to('cpu'))


if __name__ == "__main__":
    test_hpu_dropout()
