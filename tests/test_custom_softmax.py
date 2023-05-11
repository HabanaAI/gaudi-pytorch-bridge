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
import numpy as np
from habana_frameworks.torch.utils.library_loader import load_habana_module
from habana_frameworks.torch.hpex.kernels import CustomSoftmax
load_habana_module()
device = torch.device("hpu")


def test_custom_softmax():
    input = torch.tensor([[0., 1., 2., 3.], [1000, 1000, 0., 1004.],
                         [-9984, -9984, -9984, -1000]], dtype=torch.bfloat16)
    ref_output = torch.tensor([[0.0320, 0.0869, 0.2373, 0.6445],
                               [0.0177, 0.0177, 0.0000, 0.9648],
                               [0.0000, 0.0000, 0.0000, 1.0000]], dtype=torch.float32)

    out = CustomSoftmax.apply(torch.clone(input).detach().to(device), 0)
    out_cpu = out.cpu().to(torch.float32)

    assert np.allclose(out_cpu.numpy(), ref_output.numpy(), atol=1e-04)


if __name__ == "__main__":
    test_custom_softmax()
