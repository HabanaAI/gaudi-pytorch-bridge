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


def test_hpu_lazy_resize():
    src_cpu = torch.tensor([10])
    src_resized_cpu = src_cpu.resize_((2,))
    src_resized_cpu[1] = 10

    src = torch.tensor([10])
    hpu = torch.device("hpu")
    src_hpu = src.detach().to(hpu)
    src_resized_hpu = src_hpu.resize_((2,))
    src_resized_hpu[1] = 10

    assert torch.allclose(src_resized_cpu, src_resized_hpu, atol=0, rtol=0), "Data mismatch"


def test_hpu_lazy_view_resize():
    src_cpu = torch.tensor([10])
    src_cpu_view = src_cpu.view(1)
    src_resized_cpu = src_cpu_view.resize_((2,))
    src_resized_cpu[1] = 10

    src = torch.tensor([10])
    hpu = torch.device("hpu")
    src_hpu = src.detach().to(hpu)
    src_hpu_view = src_hpu.view(1)
    src_resized_hpu = src_hpu_view.resize_((2,))
    src_resized_hpu[1] = 10

    assert torch.allclose(src_resized_cpu, src_resized_hpu, atol=0, rtol=0), "Data mismatch"

