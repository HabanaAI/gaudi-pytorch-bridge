
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

import os
from .config import configuration_flags

def setup_env_config(env_name, config_name):
    if os.getenv(env_name, "").upper() in ["ON", "1", "YES", "TRUE", "Y"]:
        configuration_flags[config_name] = True
    elif os.getenv(env_name, "").upper() in ["OFF", "0", "NO", "FALSE", "N"]:
        configuration_flags[config_name] = False

setup_env_config("PT_HPU_COMPILE_USE_RECIPES",             "use_compiled_recipes")
setup_env_config("PT_HPU_COMPILE_USE_DECOMPS",             "use_decompositions")
setup_env_config("PT_HPU_COMPILE_VERBOSE",                 "verbose")
setup_env_config("PT_HPU_KEEP_INPUT_MUTATIONS",            "keep_input_mutations")
setup_env_config("PT_HPU_USE_EAGER_FALLBACK",              "use_eager_fallback")

from . import backends
