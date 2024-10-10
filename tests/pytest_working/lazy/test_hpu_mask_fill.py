###############################################################################
# Copyright (C) 2023-2024 Habana Labs, Ltd. an Intel Company
# All Rights Reserved.
#
# Unauthorized copying of this file or any element(s) within it, via any medium
# is strictly prohibited.
# This file contains Habana Labs, Ltd. proprietary and confidential information
# and is subject to the confidentiality and license agreements under which it
# was provided.
#
###############################################################################

# Based on SW-186390

import habana_frameworks.torch as ht
import torch


def func(dev):
    i = torch.empty(2, 4, device=dev)
    m = torch.ones(2, 4, device=dev, dtype=torch.bool)
    s = torch.nn.Softmax(-1)
    a = i.sqrt()
    a /= 2
    a.masked_fill_(m, 1)
    b1 = s(a)
    a.masked_fill_(m, 2)
    b2 = s(a)
    a.masked_fill_(m, 3)
    b3 = s(a)
    c = b1 + b2 + b3
    return c.cpu()


def test_misc_mask_filled():
    cpu_out = func("cpu")
    hpu_out = func("hpu")
    assert torch.allclose(cpu_out, hpu_out)


if __name__ == "__main__":
    test_misc_mask_filled()
