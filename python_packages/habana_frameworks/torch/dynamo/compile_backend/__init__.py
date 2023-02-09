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
from . import config

logging.basicConfig()

if "PT_COMPILE_USE_RECIPES" in os.environ and os.environ["PT_COMPILE_USE_RECIPES"] == "True":
    config.use_compiled_recipes = True

if "PT_COMPILE_VERBOSE" in os.environ and os.environ["PT_COMPILE_VERBOSE"] == "True":
    config.verbose = True

if config.verbose:
    logger = logging.getLogger("aot_hpu_backend")
    logger.setLevel(logging.DEBUG)
    logger.info("config.verbose is ON")
