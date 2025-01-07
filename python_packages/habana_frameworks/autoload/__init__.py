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

import os

is_loaded = False  # A member variable of habana_frameworks module to track if our module has been imported


def is_autoload_enabled():
    # If this env flag is set, behave accordingly. Otherwise move on to next conditions
    autoload_env = os.getenv("PT_HPU_AUTOLOAD", "-1")
    if autoload_env == "1":
        return True
    if autoload_env == "0":
        return False

    # If PT_HPU_AUTOLOAD isn't set, disable autoload for lazy mode
    if os.getenv("PT_HPU_LAZY_MODE", "1") == "1":
        return False
    return True


def __autoload():
    # This is an entrypoint for pytorch autoload mechanism
    # If the following confition is true, that means our backend has already been loaded, either explicitly
    # or by the autoload mechanism and importing it again should be skipped to avoid circular imports
    global is_loaded
    if is_loaded or not is_autoload_enabled():
        return
    import habana_frameworks.torch
