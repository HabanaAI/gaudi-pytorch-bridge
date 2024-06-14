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
from habana_frameworks.torch.dynamo.compile_backend.logger import get_compile_backend_logger
from torch._dynamo import compiled_autograd

logger = get_compile_backend_logger()


def enable_compiled_autograd(**kwargs):
    """
    Helper function to enable compiled_autograd for hpu backend. For more
    info on compiled autograd see:
        https://github.com/pytorch/pytorch/pull/103822

    This should be called before any invocations of torch.compile
    """
    logger.warn("Enabling CompiledAutograd for hpu_backend with torch.compile")

    def compiler_fn(gm):
        return torch.compile(
            gm, backend="hpu_backend", options={"keep_input_mutations": True, "inference": False}, **kwargs
        )

    torch._C._dynamo.compiled_autograd.set_autograd_compiler(
        functools.partial(compiled_autograd.AutogradCompilerInstance, compiler_fn)
    )

    torch._dynamo.reset()
    torch._dynamo.config.optimize_ddp = "python_reducer"
