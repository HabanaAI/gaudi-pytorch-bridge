###############################################################################
# Copyright (c) 2021-2025 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
###############################################################################

from habana_frameworks.torch._torch_jit_C import jit
from habana_frameworks.torch.dynamo.debug_utils.logger import get_compile_backend_logger

from .unfold_tuple_on_output import pass_unfold_tuple_on_output

logger = get_compile_backend_logger()


def get_jit_fork_passes():
    passes_list = [
        pass_unfold_tuple_on_output,
        jit.getitem_folding_pass,
        jit.remove_duplicate_const_pass,
        jit.remove_mutation_pass,
    ]
    return passes_list


def run_jit_fork_passes(jit_ir: jit.Graph):
    logger.debug("running run_jit_fork_passes")
    for jit_pass in get_jit_fork_passes():
        graph_changed = jit_pass(jit_ir)
        if graph_changed:
            logger.debug(
                "####PyTorch-generated JIT IR graph after jit passes: %s ####\n%s",
                jit_pass.__name__,
                jit_ir,
            )
