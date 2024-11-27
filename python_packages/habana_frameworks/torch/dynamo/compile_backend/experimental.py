###############################################################################
#
#  Copyright (c) 2021-2024 Intel Corporation
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
###############################################################################
import functools

import torch
from habana_frameworks.torch.dynamo.debug_utils.logger import get_compile_backend_logger
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
        return torch.compile(gm, backend="hpu_backend", options={"inference": False}, **kwargs)

    torch._C._dynamo.compiled_autograd.set_autograd_compiler(
        functools.partial(compiled_autograd.AutogradCompilerInstance, compiler_fn)
    )

    torch._dynamo.reset()
    torch._dynamo.config.optimize_ddp = "python_reducer"
    torch._C._set_autograd_fallback_mode("nothing")
