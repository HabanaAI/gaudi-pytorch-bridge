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
