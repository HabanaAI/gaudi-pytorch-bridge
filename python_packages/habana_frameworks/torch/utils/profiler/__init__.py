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

from habana_frameworks.torch.utils.library_loader import load_habana_profiler

load_habana_profiler()

from habana_frameworks.torch.utils import _profiler_C


def _setup_profiler():
    _profiler_C.setup_profiler()


def _start_profiler():
    _profiler_C.start_profiler()


def _stop_profiler():
    _profiler_C.stop_profiler()
