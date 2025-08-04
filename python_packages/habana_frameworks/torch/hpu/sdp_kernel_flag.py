###############################################################################
# Copyright (c) 2021-2025 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
###############################################################################

import contextlib

is_recompute_FSDPA_enabled = True


def enable_recompute_sdp(enabled: bool):
    r"""User control to enable or disable recompute based fused SDPA
    enabled = True -> Fused SDPA with recompute
    enabled = False -> Fused SDPA without recompute
    """
    global is_recompute_FSDPA_enabled
    is_recompute_FSDPA_enabled = enabled


def recompute_sdp_enabled():
    r"""User control to check if recompute based fused SDPA is enabled.
    return = True -> Fused SDPA with recompute enabled
    return = False -> Fused SDPA without recompute enabled
    """
    global is_recompute_FSDPA_enabled
    return is_recompute_FSDPA_enabled


@contextlib.contextmanager
def sdp_kernel(
    enable_recompute: bool = True,
):
    r"""Context manager to enable or disable recompute based fused SDPA
    enable_recompute = True -> Fused SDPA with recompute
    enable_recompute = False -> Fused SDPA without recompute
    """
    recompute_backup: bool = recompute_sdp_enabled()

    try:
        enable_recompute_sdp(enable_recompute)
        yield {}
    finally:
        enable_recompute_sdp(recompute_backup)
