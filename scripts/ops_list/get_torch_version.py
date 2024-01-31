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

import torch

v = torch.__version__
opt = "HPU"
ext = "hpu"

if "+cpu" in v:
    opt = "CPU"
    ext = "cpu"
elif "+cu" in v:
    opt = "CUDA"
    ext = "gpu"

print(f"{v} {opt} {ext}")
