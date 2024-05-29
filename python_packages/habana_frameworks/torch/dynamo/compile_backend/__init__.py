# Copyright (C) 2023-2024 Habana Labs, Ltd. an Intel Company
# All Rights Reserved.
#
# Unauthorized copying of this file or any element(s) within it, via any medium
# is strictly prohibited.
# This file contains Habana Labs, Ltd. proprietary and confidential information
# and is subject to the confidentiality and license agreements under which it
# was provided.
#
###############################################################################

import habana_frameworks.torch.internal.bridge_config as bc

from . import backends

if bc.get_pt_hpu_enable_compiled_autograd():
    from .experimental import enable_compiled_autograd

    enable_compiled_autograd()
