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

import os
import logging
from .config import configuration_flags

logging.basicConfig()

def setup_env_config(env_name, config_name):
    if os.getenv(env_name, "").upper() in ["ON", "1", "YES", "TRUE", "Y"]:
        configuration_flags[config_name] = True
    elif os.getenv(env_name, "").upper() in ["OFF", "0", "NO", "FALSE", "N"]:
        configuration_flags[config_name] = False

setup_env_config("PT_HPU_COMPILE_USE_RECIPES",             "use_compiled_recipes")
setup_env_config("PT_HPU_COMPILE_VERBOSE",                 "verbose")
setup_env_config("PT_HPU_USE_CORE_ATEN_DECOMP",            "use_core_aten_decomp")
setup_env_config("PT_HPU_USE_HPU_DECOMP",                  "use_hpu_decomp")
setup_env_config("PT_HPU_USE_DECOMP_EXCLUSIONS",           "use_decomp_exclusions")
setup_env_config("PT_HPU_DTYPE_PROP_IN_BACKEND",           "dtype_propagation_in_backend")
setup_env_config("PT_HPU_KEEP_INPUT_MUTATIONS",            "keep_input_mutations")

# TODO: fix this flag ELSE, it now defaults to False even if it is True in the config
# it should use setup_env_config function here instead
if os.getenv("PT_HPU_USE_SHARED_LAYER_FALLBACK_CHECK", "").upper() in ["ON", "1", "YES", "TRUE", "Y"]:
    configuration_flags["shared_layer_fallback_check"] = True
else:
    configuration_flags["shared_layer_fallback_check"] = False

if configuration_flags["verbose"]:
    logger = logging.getLogger("aot_hpu_backend")
    logger.setLevel(logging.DEBUG)
    logger.info("config.verbose is ON")

from . import backends
