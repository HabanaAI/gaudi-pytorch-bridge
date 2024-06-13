###############################################################################
# Copyright (C) 2024 Habana Labs, Ltd. an Intel Company
# All Rights Reserved.
#
# Unauthorized copying of this file or any element(s) within it, via any medium
# is strictly prohibited.
# This file contains Habana Labs, Ltd. proprietary and confidential information
# and is subject to the confidentiality and license agreements under which it
# was provided.
#
###############################################################################

import habana_frameworks.torch as htorch
import torch
import torch._dynamo.testing


def test_graph_break():
    def fn(rand):
        if htorch.hpu.is_available():
            m = htorch.hpu._utils._get_device_index(htorch.hpu.current_device())
            return m * rand
        else:
            return -1

    cnts = torch._dynamo.testing.CompileCounter()
    ones = torch.ones(1, device="hpu")
    orig_out = fn(ones)
    opt_fn = torch._dynamo.optimize(cnts, nopython=False)(fn)
    optim_out = opt_fn(ones)
    assert cnts.frame_count == 1, "Frame Count not equal to 1, check for graph breaks"
    assert orig_out == optim_out, "Output mismatch"


if __name__ == "__main__":
    test_graph_break()
