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
