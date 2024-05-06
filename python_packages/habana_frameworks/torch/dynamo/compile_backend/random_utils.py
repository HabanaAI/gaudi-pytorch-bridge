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

import habana_frameworks.torch.internal.bridge_config as bc
import torch

RANDOM_OPS = (
    {
        "aten.bernoulli.default": torch.ops.hpu.habana_bernoulli,
        "aten.poisson.default": torch.ops.hpu.habana_poisson,
        "aten.rand.default": torch.ops.hpu.habana_rand,
        "aten.randn.default": torch.ops.hpu.habana_randn,
        "aten.randint.low": torch.ops.hpu.habana_randint,
        "aten.multinomial.default": torch.ops.hpu.habana_multinomial,
        "aten.randperm.default": torch.ops.hpu.habana_randperm,
        "aten.native_dropout.default": torch.ops.hpu.habana_native_dropout,
        "hpu.sdpa_recomp_fwd_dropout.default": torch.ops.hpu.sdpa_recomp_fwd_dropout_seed,
        "hpu.sdpa_fwd_dropout.default": torch.ops.hpu.sdpa_fwd_dropout_seed,
        "aten.uniform.default": torch.ops.hpu.habana_uniform,
    }
    if bc.get_pt_hpu_wrap_random_ops_compile()
    else {}
)


def is_random_op(node):
    return str(node.target) in RANDOM_OPS


def random_op_inputs(node, seed):
    op = RANDOM_OPS[str(node.target)]
    args = (seed,) + node.args
    kwargs = node.kwargs.copy()
    kwargs.pop("generator", None)

    return (op, args, kwargs)
