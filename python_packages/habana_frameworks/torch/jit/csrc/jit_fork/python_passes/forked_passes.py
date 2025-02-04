###############################################################################
#
#  Copyright (c) 2021-2025 Intel Corporation
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

import habana_frameworks.torch._torch_jit_C.jit as jit
from habana_frameworks.torch.dynamo.debug_utils.logger import get_compile_backend_logger

from .unfold_tuple_on_output import pass_unfold_tuple_on_output

logger = get_compile_backend_logger()


# todo rename file name, dir name and the functions' names defined here https://jira.habana-labs.com/browse/SW-199903


def get_jit_fork_passes():
    passes_list = [
        pass_unfold_tuple_on_output,
        jit.getitem_folding_pass,
        jit.remove_duplicate_const_pass,
    ]
    return passes_list


def run_jit_fork_passes(jit_ir: jit.Graph):
    # todo use proper logging https://jira.habana-labs.com/browse/SW-200787
    # if config.dump_graph:
    #     logger.info(
    #         "### JIT IR graph before running passes: ###\n%s",
    #         jit_ir.str(print_source_info=True),
    #     )

    logger.debug("running run_jit_fork_passes")
    for jit_pass in get_jit_fork_passes():
        logger.debug(f"jit_pass = {jit_pass.__name__}")
        graph_changed = jit_pass(jit_ir)
        # if graph_changed:
        #     print(f"graph changed, so printing after pass:\n{jit_ir}", flush=True)
        #     logger.debug("graph changed:")
        #     logger.debug(f"{str(jit_ir).replace('%', '%%')}")
        # todo use proper logging https://jira.habana-labs.com/browse/SW-200787
        # if graph_changed and config.dump_graph:
        #     logger.debug(
        #         "### JIT Graph after %s pass: ###\n%s",
        #         jit_pass.__qualname__,
        #         jit_ir.str(print_source_info=True),
        #     )

    # todo use proper logging https://jira.habana-labs.com/browse/SW-200787
    # if config.dump_graph:
    # logger.info(
    #     "### Final JIT IR graph passed to backend: ###\n%s",
    #     jit_ir.str(print_source_info=True),
    # )
