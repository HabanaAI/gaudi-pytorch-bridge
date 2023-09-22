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

import pytest
import torch
from test_utils import cpu, hpu

pytestmark = pytest.mark.skip(reason="segv")

# optional on list cause fail
@pytest.mark.parametrize(
    "shape, indices",
    [
        [(5, 5), ([1, 2, 3],)],
        pytest.param((5, 5), (None, [1, 2, 3],), marks=pytest.mark.xfail(rason="SW-155102")),
        pytest.param((2, 3, 4), (None, None,), marks=pytest.mark.xfail(rason="SW-155102")),
        pytest.param(
            (2, 3, 4),
            (
                [1, 0],
                [0],
                None,
            ),
            marks=pytest.mark.xfail(rason="SW-155102"),
        ),
        [(2, 3, 8, 8), ([[[1], [0]]],)],
        pytest.param((4, 3, 8, 8), (None, [1, 2], None,), marks=pytest.mark.xfail(rason="SW-155102")),
    ],
)
def test_index(shape, indices):
    if pytest.mode == "lazy":
        pytest.xfail()

    func = torch.ops.aten.index
    if pytest.mode == "compile":
        pytest.xfail(reason="torch._dynamo.optimize is called on a non function object")
        func = torch.compile(torch.ops.aten.index, backend="aot_hpu_training_backend")

    input_tensor = torch.randn(*shape, device="hpu:0")
    indices = [torch.tensor(x, device="hpu:0") if x is not None else x for x in indices]

    y_cpu = func(
        input_tensor.to(cpu), [x.to(cpu) if x is not None else x for x in indices]
    )
    y_hpu = func(input_tensor, indices)

    assert torch.equal(y_cpu, y_hpu.to(cpu))
