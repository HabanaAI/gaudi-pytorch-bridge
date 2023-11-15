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

import habana_frameworks.torch.dynamo.compile_backend
import pytest
import torch


@pytest.mark.parametrize("start", [0.1664, 0.6964, 4.124])
@pytest.mark.parametrize("end", [1.2032, 2.0438, 2.5345])
@pytest.mark.parametrize("steps", [0, 1, 6, 13])
def test_hpu_linspace(start, end, steps):
    def fn(start, end, steps):
        return torch.linspace(start, end, steps)

    cpu_compiled_fn = torch.compile(fn)
    hpu_compiled_fn = torch.compile(fn, backend="aot_hpu_training_backend")

    cpu_output = cpu_compiled_fn(start, end, steps)
    hpu_output = hpu_compiled_fn(start, end, steps).cpu()

    assert torch.allclose(cpu_output, hpu_output)
