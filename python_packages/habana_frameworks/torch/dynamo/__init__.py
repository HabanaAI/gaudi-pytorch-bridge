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
from habana_frameworks.torch.dynamo.device_interface import HpuInterface
from torch._dynamo.device_interface import register_interface_for_device

# we do not support device indices >0, so eiter hpu or hpu:0
register_interface_for_device("hpu", HpuInterface)
register_interface_for_device("hpu:0", HpuInterface)

from . import trace_rules
