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
from habana_frameworks.torch.utils.debug import Logger


def get_compile_backend_logger():
    return Logger("PT_COMPILE")


def get_fx_graph_logger():
    return Logger("PT_COMPILE FX GRAPH")


def dump_fx_graph(fx_module, jit_graph, recipe_id):
    logger = get_fx_graph_logger()
    logger.debug("# # # graph_recipe_%d # # #", recipe_id)
    logger.debug(fx_module.print_readable(False))
    logger.debug("IR:\n%s\n\n", fx_module.graph)
    logger.debug("Jit IR:\n%s\n\n", jit_graph)


def log_function_start_end(fn):
    def wrapper(*args, **kwargs):
        logger = get_compile_backend_logger()
        logger.debug("Function %s start %s, %s", fn.__name__, args, kwargs)
        result = fn(*args, **kwargs)
        logger.debug(
            "Function %s end",
            fn.__name__,
        )
        return result

    return wrapper
