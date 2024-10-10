#!/usr/bin/env python
################################################################################
# Copyright (C) 2022-2023 Habana Labs, Ltd. an Intel Company
# All Rights Reserved.
#
# Unauthorized copying of this file or any element(s) within it, via any medium
# is strictly prohibited.
# This file contains Habana Labs, Ltd. proprietary and confidential information
# and is subject to the confidentiality and license agreements under which it
# was provided.
#
################################################################################
#
# Prints the highest pt_ver directory found without a trailing newline
#


import os

pt_ver_dir = os.path.join(os.getenv("PYTORCH_MODULES_ROOT_PATH"), "pt_ver")

print(".".join(max(d.split(".") for d in os.listdir(pt_ver_dir))), end="")
