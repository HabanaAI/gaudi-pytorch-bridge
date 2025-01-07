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


from enum import Enum

import habana_frameworks.torch.utils._lop_profiler_C as profiler
import torch


def start():
    profiler._start_lo_host_profiler()


def stop():
    profiler._stop_lo_host_profiler()


def flush():
    profiler._flush_lo_host_profiler()
