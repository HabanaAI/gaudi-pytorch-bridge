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
import habana_frameworks.torch.internal.bridge_config as bc


RANDOM_OPS = (
    {
        "aten.bernoulli.default": torch.ops.hpu.habana_bernoulli,
        "aten.rand.default": torch.ops.hpu.habana_rand,
        "aten.randn.default": torch.ops.hpu.habana_randn,
    }
    if bc.get_pt_hpu_wrap_random_ops_compile()
    else {}
)


def is_random_op(node):
    return str(node.target) in RANDOM_OPS


def random_op_inputs(node, seed):
    op = RANDOM_OPS[str(node.target)]
    args = (node.args[0], seed)
    kwargs = node.kwargs

    return (op, args, kwargs)
