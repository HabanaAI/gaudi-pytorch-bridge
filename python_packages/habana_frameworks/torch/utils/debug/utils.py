###############################################################################
# Copyright (C) 2024 Habana Labs, Ltd. an Intel Company
# All Rights Reserved.
#
# Unauthorized copying of this file or any element(s) within it, via any medium
# is strictly prohibited.
# This file contains Habana Labs, Ltd. proprietary and confidential information
# and is subject to the confidentiality and license agreements under which it
# was provided.
#
###############################################################################

from habana_frameworks.torch.utils import _debug_C


def dump_device_state_and_terminate(msg, flags=0):
    """
    msg is printed to $HABANA_LOGS/*/dfa_api.txt and
    $HABANA_LOGS/*/synapse_runtime.log
    flags doesn't have any effect, added for future use
    this function dumps device state and terminate
    trigger DFA(Device Failure Analysis) using this API to collect device info
    """
    _debug_C.dump_state_and_terminate(msg, flags)
