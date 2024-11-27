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


import warnings

import habana_frameworks.torch.utils.profiler as htprofiler


def setup_profiler() -> None:
    warnings.warn(
        "habana_frameworks.torch.profiler.setup_profiler is deprecated. "
        "Please use habana_frameworks.torch.utils.profiler._setup_profiler"
    )
    htprofiler._setup_profiler()


def start_profiler() -> None:
    warnings.warn(
        "habana_frameworks.torch.profiler.start_profiler is deprecated. "
        "Please use habana_frameworks.torch.utils.profiler._start_profiler"
    )
    htprofiler._start_profiler()


def stop_profiler() -> None:
    warnings.warn(
        "habana_frameworks.torch.profiler.stop_profiler is deprecated. "
        "Please use habana_frameworks.torch.utils.profiler._stop_profiler"
    )
    htprofiler._stop_profiler()
