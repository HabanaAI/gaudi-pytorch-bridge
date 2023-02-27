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

import logging
import os


logger = logging.getLogger(__name__)

def is_lazy():
    return os.getenv("PT_HPU_LAZY_MODE", "1") != "0"


def lazy_only(func):
    def wrapper(*args, **kwargs):
        if is_lazy():
            func(*args, **kwargs)
        else:
            logger.info(f"Call {func.__name__} function will not have any effect. It's lazy mode only functionality.")

    return wrapper
