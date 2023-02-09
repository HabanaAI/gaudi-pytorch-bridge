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
import functorch

from typing import List
from torch._dynamo.backends.common import aot_autograd

from .internal import (
    preprocess_module,
    transform_cpu_fallbacks,
    cluster_module,
    compile_clusters,
)


def _hpu_compile_inner(graph_module: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
    """
    This function will be called for each input FX graph. There will be at least
    three separate graphs for FWD, BWD and optimizer. Each of these phases can
    also generate multiple graphs and calls to this function.
    """

    # Do initial preprocessing.
    preprocess_module(graph_module, example_inputs)

    # Take ops marked for CPU fallback and transform
    # them into copy_to(CPU)->OP(CPU)->copy_to(HPU)
    transform_cpu_fallbacks(graph_module)

    # Partition the module based on propagated device placement data.
    clustered_module = cluster_module(graph_module)

    # Generate compiled recipes for the HPU clusters in the module.
    compile_clusters(clustered_module)

    # Return the module in boxed format required by AOT Autograd.
    return functorch.compile.make_boxed_func(clustered_module.forward)


# This function implements Habana backend for HPU, without AOT Autograd.
hpu_backend = _hpu_compile_inner


def aot_hpu_backend(graph_module: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
    """
    This function implements Habana backend for HPU, with AOT Autograd.
    """

    # Create AOT Autograd instance and feed it with Habana compile function.
    return aot_autograd(
        fw_compiler=_hpu_compile_inner,
        bw_compiler=_hpu_compile_inner,
    )(graph_module, example_inputs)
