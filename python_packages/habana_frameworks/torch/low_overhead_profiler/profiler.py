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


from enum import Enum

import habana_frameworks.torch.utils._lop_profiler_C as profiler
import torch


def start():
    profiler._start_lo_host_profiler()


def stop():
    profiler._stop_lo_host_profiler()


def flush():
    profiler._flush_lo_host_profiler()
