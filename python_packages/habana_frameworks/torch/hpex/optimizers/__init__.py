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

from habana_frameworks.torch.utils.internal import is_lazy

if not is_lazy():
    raise ImportError(f"{__name__} is no yet supported in eager mode")

from .FusedAdagrad import FusedAdagrad
from .FusedAdamW import FusedAdamW
from .FusedLamb import FusedLamb
from .FusedSGD import FusedSGD
from .FusedLars import FusedLars
from .FusedResourceApplyMomentum import FusedResourceApplyMomentum
