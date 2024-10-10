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

import habana_frameworks.torch.core as htcore
import numpy as np
import pytest
import torch

try:
    import habana_frameworks.torch.utils.experimental as exp
except ImportError:
    raise AssertionError("Could Not import habana_frameworks.torch.core")


@pytest.mark.parametrize("input_tensor", [(5, 5)])
def test_hpu_lazy_data_ptr(input_tensor):
    t1 = torch.randn(input_tensor, requires_grad=True)

    hpu = torch.device("hpu")

    t1_h = t1.detach().to(hpu)
    t1_h.requires_grad = True
    t1_h.retain_grad()
    print("t1_h data_ptr ", hex(exp._data_ptr(t1_h)))

    t2_h = torch.abs(t1_h)
    t3_h = t2_h.mul_(t1_h)
    out_h = torch.add(t2_h, t3_h)

    print("out_h data_ptr ", hex(exp._data_ptr(out_h)))

    t2 = torch.abs(t1)
    t3 = t2.mul_(t1)
    out = torch.add(t2, t3)

    t3_view = t3_h.view(-1)
    t3_h_data_ptr = exp._data_ptr(t3_h)
    t3_view_data_ptr = exp._data_ptr(t3_view)

    print("t3_h data_ptr ", hex(t3_h_data_ptr))
    print("t3_view data_ptr ", hex(t3_view_data_ptr))

    out.sum().backward()
    t1.grad.clone().detach()

    out_h.sum().backward()
    # out_h.backward(grad_out.detach().to(hpu))
    htcore.mark_step()

    print("t1_h.grad data_ptr ", hex(exp._data_ptr(t1_h.grad)))
    t1_h.grad.cpu()

    out_cpu_to_compare = out.clone().detach()
    out_h_cpu_to_compare = out_h.cpu().clone().detach()

    # TBD: This can be enabled only after as_strided patch makes views to share
    # storage
    # assert(t3_h_data_ptr == t3_view_data_ptr)
    assert np.allclose(out_cpu_to_compare, out_h_cpu_to_compare, atol=0, rtol=0), "Data mismatch"
