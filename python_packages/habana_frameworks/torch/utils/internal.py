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
from time import perf_counter

logger = logging.getLogger(__name__)


def is_lazy():
    return os.getenv("PT_HPU_LAZY_MODE", "1") != "0"


def lazy_only(func):
    def wrapper(*args, **kwargs):
        if is_lazy():
            func(*args, **kwargs)
        else:
            if not wrapper.has_run:
                logger.warning(
                    f"Calling {func.__name__} function does not have any effect. It's lazy mode only functionality. (warning logged once)"
                )
                wrapper.has_run = True

    wrapper.has_run = False
    return wrapper


class Timer:
    """Utility class to measure time using a context manager."""

    def __init__(self):
        self.start_t = None
        self.end_t = None

    def __enter__(self):
        self.start_t = perf_counter()
        return self

    def __exit__(self, type, value, traceback):
        self.end_t = perf_counter()

    @property
    def elapsed(self):
        return self.end_t - self.start_t
