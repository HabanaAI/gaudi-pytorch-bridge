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


use_compiled_recipes = _get_bool_from_env("PT_HPU_COMPILE_USE_RECIPES", "1")
use_decompositions = _get_bool_from_env("PT_HPU_COMPILE_USE_DECOMPS", "1")
verbose = _get_bool_from_env("PT_HPU_COMPILE_VERBOSE", "0")
keep_input_mutations = _get_bool_from_env("PT_HPU_KEEP_INPUT_MUTATIONS", "0")
use_eager_fallback = _get_bool_from_env("PT_HPU_USE_EAGER_FALLBACK", "1")

# adds patch, save_config, etc
install_config_module(sys.modules[__name__])


# keep this in order not to break current API of configuration_flags
class DictLikeClass:
    def __getitem__(self, key):
        return getattr(sys.modules[__name__], key)

    def __setitem__(self, key, value):
        return setattr(sys.modules[__name__], key, value)


configuration_flags = DictLikeClass()
