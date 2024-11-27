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

from habana_frameworks.torch.dynamo.device_interface import HpuInterface
from torch._dynamo.device_interface import register_interface_for_device

# we do not support device indices >0, so eiter hpu or hpu:0
register_interface_for_device("hpu", HpuInterface)
register_interface_for_device("hpu:0", HpuInterface)

from . import trace_rules
