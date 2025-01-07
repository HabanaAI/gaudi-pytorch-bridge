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

import habana_frameworks.torch as ht
import habana_frameworks.torch.core as htcore


# Cannot be called twice in a row.
# Can be used after calling "synrec -w ..." or after synrec_stop()
def synrec_start():
    if os.getenv("SYNREC_INIT", 0) == 0:
        raise Exception("synrec_start: Synrec not initialized.")
    if os.environ["SYNREC"] == "1":
        raise Exception("synrec_start: Synrec record already started.")
    htcore.mark_step()
    ht.hpu.synchronize()
    os.environ["SYNREC"] = "1"


# Cannot be called twice in a row.
# Can be used after calling "synrec" (without -w flag) or after synrec_start()
def synrec_stop():
    if os.getenv("SYNREC_INIT", 0) == 0:
        raise Exception("synrec_stop: Synrec not initialized.")
    if os.environ["SYNREC"] == "0":
        raise Exception("synrec_start: Synrec record already stopped.")
    htcore.mark_step()
    ht.hpu.synchronize()
    os.environ["SYNREC"] = "0"
