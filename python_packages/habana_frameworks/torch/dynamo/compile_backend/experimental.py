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
import functools

import torch
from torch._dynamo import compiled_autograd


def enable_compiled_autograd():
    """
    Helper function to enable compiled_autograd for hpu backend. For more
    info on compiled autograd see:
        https://github.com/pytorch/pytorch/pull/103822

    This should be called before any invocations of torch.compile
    """

    def compiler_fn(gm):
        return torch.compile(gm, backend="hpu_backend", fullgraph=True)

    torch._C._dynamo.compiled_autograd.set_autograd_compiler(
        functools.partial(compiled_autograd.AutogradCompilerInstance, compiler_fn)
    )

    torch._dynamo.reset()
    torch._dynamo.config.optimize_ddp = "python_reducer"
