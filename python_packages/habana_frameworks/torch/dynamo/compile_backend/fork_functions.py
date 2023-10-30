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
from typing import List

@torch.compile(dynamic=False, backend="hpu_inference_backend")
def consolidated_inplace_copies(dsts: List[torch.Tensor], srcs: List[torch.Tensor], len: int):
    """
    This function is used by the Habana pytorch fork.

    Motivation is that some parts of dynamo code, like long sets of eager copies,
    could be wrapped into graphs. This is exactly what this function does, it
    does inplace copies taking them all from the list and wraps it in our inference
    backend so they get executed as a single graph.

    This function will become obsolete at some point of time once torch.compile
    matures for these cases.
    """
    for i in range(len):
        dsts[i].copy_(srcs[i])

    # WORKAROUND START
    # this is not really needed, this is just temporary w/a for incorrect JIT IR
    # outputs when no tensor output is specified. It should force IR to have some
    # output.
    return torch.clone(dsts[0])
    # WORKAROUND END
