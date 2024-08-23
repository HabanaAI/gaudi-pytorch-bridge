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
    if is_lazy._is_lazy is None:
        is_lazy._is_lazy = os.getenv("PT_HPU_LAZY_MODE", "1") != "0"
    return is_lazy._is_lazy


is_lazy._is_lazy = None


def lazy_only(func):
    def lazy_wrapper(*args, **kwargs):
        func(*args, **kwargs)

    def non_lazy_wrapper(*args, **kwargs):
        pass

    if is_lazy():
        return lazy_wrapper
    else:
        logger.warning(
            f"Calling {func.__name__} function does not have any effect. It's lazy mode only functionality. (warning logged once)"
        )
        return non_lazy_wrapper


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
