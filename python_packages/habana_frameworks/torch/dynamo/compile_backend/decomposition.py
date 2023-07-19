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

from .config import configuration_flags

from torch._decomp import core_aten_decompositions, get_decompositions, register_decomposition

# List of HPU-specific decompositions.
additional_decompositions = get_decompositions(
    [
        # no entries now, example entry: torch.ops.aten.std_mean
        torch.ops.aten.new_empty_strided.default
    ]
)

# These are OPs that are decomposed by default in core aten list, but we support them natively so
# we prefer them not being decomposed.
exclusions = [
    # These are used in RN50 and we support them natively.
    "aten._native_batch_norm_legit_functional.default",
    "aten.native_batch_norm_backward.default",
    "aten._log_softmax.default",
    "aten._log_softmax_backward_data.default",
    "aten.nll_loss_forward.default",
    "aten.nll_loss_backward.default",
    "aten.threshold_backward.default",
    "aten.t.default",

    # These decompositions are causing various functional test issues.
    "aten.binary_cross_entropy.default",
    "aten.log_sigmoid_forward.default",
    "aten.log_sigmoid_backward.default",
    "aten.ones.default",
    "aten.ones_like.default",
    "aten.zeros.default",
    "aten.zeros_like.default",
    "aten.rsub.Scalar",
    "aten.sinc.default",
    "aten.trace.default",
    "aten.tril.default",
    "aten.logit.default",
    "aten.nan_to_num.default",
    "aten.hardshrink.default",
    "aten.hardshrink_backward.default",
    "aten.hardtanh.default",
    "aten.hardtanh_backward.default",
    "aten.huber_loss.default",
    "aten.huber_loss_backward.default",
    "aten.fill.Scalar",
    "aten.softshrink.default",
    "aten.softshrink_backward.default",
    "aten.threshold.default",
    "aten.embedding.default",
    "aten.embedding_dense_backward.default",
    "aten.special_entr.default",
    "aten.hardsigmoid.default",
    "aten.hardsigmoid_backward.default",
    "aten.heaviside.default",
    "aten.index_select.default",
    "aten.index_add.default",
    "aten.expand.default",
    "aten.logit.default",
    "aten.logit_backward.default",
    "aten.grid_sampler_2d.default",
    "aten.lerp.Tensor",
    "aten.triu.default",
]

decompositions = {}

if configuration_flags["use_core_aten_decomp"]:
    decompositions = {**core_aten_decompositions(), **decompositions}

if configuration_flags["use_hpu_decomp"]:
    decompositions = {**additional_decompositions, **decompositions}

if configuration_flags["use_decomp_exclusions"]:
    decompositions = {key: decompositions[key] for key in decompositions if not str(key) in exclusions}

def get_hpu_decompositions(is_training: bool):
    return decompositions
