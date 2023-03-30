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
from . import backends

logging.basicConfig()

if os.getenv("PT_HPU_COMPILE_USE_RECIPES", "").upper() in ["ON", "1", "YES", "TRUE", "Y"]:
    configuration_flags["use_compiled_recipes"] = True
elif os.getenv("PT_HPU_COMPILE_USE_RECIPES", "").upper() in ["OFF", "0", "NO", "FALSE", "N"]:
    configuration_flags["use_compiled_recipes"] = False

if os.getenv("PT_HPU_COMPILE_VERBOSE", "").upper() in ["ON", "1", "YES", "TRUE", "Y"]:
    configuration_flags["verbose"] = True
elif os.getenv("PT_HPU_COMPILE_VERBOSE", "").upper() in ["OFF", "0", "NO", "FALSE", "N"]:
    configuration_flags["verbose"] = False

if os.getenv("PT_HPU_DTYPE_PROP_IN_BACKEND", "").upper() in ["ON", "1", "YES", "TRUE", "Y"]:
    configuration_flags["dtype_propagation_in_backend"] = True
elif os.getenv("PT_HPU_DTYPE_PROP_IN_BACKEND", "").upper() in ["OFF", "0", "NO", "FALSE", "N"]:
    configuration_flags["dtype_propagation_in_backend"] = False

if os.getenv("PT_HPU_KEEP_INPUT_MUTATIONS", "").upper() in ["ON", "1", "YES", "TRUE", "Y"]:
    configuration_flags["keep_input_mutations"] = True
else:
    configuration_flags["keep_input_mutations"] = False

if configuration_flags["verbose"]:
    logger = logging.getLogger("aot_hpu_backend")
    logger.setLevel(logging.DEBUG)
    logger.info("config.verbose is ON")
