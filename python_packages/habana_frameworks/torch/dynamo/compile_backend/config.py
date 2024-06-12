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

import os
import sys

from torch.utils._config_module import install_config_module


def _get_bool_from_env(env_var: str, default: str):
    env_str_value = os.getenv(env_var, default).lower()
    if env_str_value in ["on", "1", "yes", "true", "y", "t"]:
        return True
    if env_str_value in ["off", "0", "no", "false", "n", "f"]:
        return False
    assert False, f"Unrecognized boolean value in env config:\n\t{env_var}: {env_str_value}"


def _get_decomp_mode(env_var: str, default: str):
    env_str_value = os.getenv(env_var, default).lower()
    assert env_str_value in [
        "habana",
        "core_aten",
        "none",
    ], f'Unrecognized string value in env config:\n\t{env_var}: {env_str_value}\n\tRecognized values: "habana", "core_aten", "none"\n'
    return env_str_value


use_compiled_recipes = _get_bool_from_env("PT_HPU_COMPILE_USE_RECIPES", "1")
# decomposition_mode can take values "habana", "core_aten" and "none" and
# with each of those values, hpu backend creates AOT Autograd instance with decomposition list
# containing either Habana defined (habana), PT Framework defined (core_aten) or no decompositions.
decomposition_mode = _get_decomp_mode("PT_HPU_COMPILE_DECOMPOSITION_MODE", "habana")
verbose = _get_bool_from_env("PT_HPU_COMPILE_VERBOSE", "0")
keep_input_mutations = _get_bool_from_env("PT_HPU_KEEP_INPUT_MUTATIONS", "1")
use_eager_fallback = _get_bool_from_env("PT_HPU_USE_EAGER_FALLBACK", "1")
# enables graph freezing and constant folding for inference
# based on the method present in torch/_inductor/freezing.py
use_graph_freezing = _get_bool_from_env("PT_HPU_COMPILE_GRAPH_FREEZE", "0")
# enables discarding the module parameters for memory efficiency
# note that it does not work if module needs to be recompiled
discard_frozen_params = _get_bool_from_env("PT_HPU_COMPILE_DISCARD_FROZEN_PARAMS", "0")
# enables removing unnecessary clone ops from the joint graph
remove_unnecessary_clones = _get_bool_from_env("PT_HPU_COMPILE_REMOVE_UNNECESSARY_CLONES", "1")
use_inplace_allreduce = _get_bool_from_env("PT_HPU_USE_INPLACE_COLLECTIVE", "0")
# for compile enable autograd, so that the training compiler is chosen
inference = _get_bool_from_env("PT_HPU_USE_INFERENCE_COMPILER", "1")

# adds patch, save_config, etc
install_config_module(sys.modules[__name__])


# keep this in order not to break current API of configuration_flags
class DictLikeClass:
    def __getitem__(self, key):
        return getattr(sys.modules[__name__], key)

    def __setitem__(self, key, value):
        return setattr(sys.modules[__name__], key, value)


configuration_flags = DictLikeClass()
