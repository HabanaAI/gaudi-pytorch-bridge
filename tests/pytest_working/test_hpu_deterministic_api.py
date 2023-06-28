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

import torch
import habana_frameworks.torch as htcore

hpu = torch.device('hpu')
cpu = torch.device('cpu')

# Check API doesn't acquire any device on invoke
def test_hpu_deterministic_api():
    device_count = htcore.hpu.device_count()
    htcore.hpu.setDeterministic(1)
    new_device_count = htcore.hpu.device_count()
    assert(device_count == new_device_count)

if __name__ == "__main__":
    test_hpu_deterministic_api()

