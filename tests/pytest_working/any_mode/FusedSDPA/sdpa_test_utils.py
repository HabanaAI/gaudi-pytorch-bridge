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

DBG_FLAG_verbose_print = False


def check_dbg_env_var(v):
    env_var_set = False
    if int(os.getenv(v, 0)) == 1:
        env_var_set = True
    return env_var_set


def vb_print(*args, **kwargs):
    if DBG_FLAG_verbose_print:
        print(*args, **kwargs)


def get_dbg_env_var_num(v):
    return int(os.getenv(v, 0))
