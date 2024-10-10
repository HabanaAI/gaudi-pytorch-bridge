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
from habana_frameworks.torch.hpu import device_count
from torch._dynamo.device_interface import register_interface_for_device

register_interface_for_device("hpu", HpuInterface)

for i in range(device_count()):
    register_interface_for_device(f"hpu:{i}", HpuInterface)
from . import trace_rules
