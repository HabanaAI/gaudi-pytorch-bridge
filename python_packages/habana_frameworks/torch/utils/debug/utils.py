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
