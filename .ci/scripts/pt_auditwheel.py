###############################################################################
# Copyright (C) 2022-2024 Habana Labs, Ltd. an Intel Company
# All Rights Reserved.
#
# Unauthorized copying of this file or any element(s) within it, via any medium
# is strictly prohibited.
# This file contains Habana Labs, Ltd. proprietary and confidential information
# and is subject to the confidentiality and license agreements under which it
# was provided.
#
###############################################################################

import sys

from auditwheel.main import main
from auditwheel.policy import _POLICIES as POLICIES

PT_WHITELIST = [
    "libtorch.so",
    "libc10.so",
    "libtorch_python.so",
    "libtorch_cpu.so",
    "libaeon.so.1",
    "libhabana_pytorch_plugin.so",  # 2 copies of library causes bridge issue
]

for p in POLICIES:
    p["lib_whitelist"].extend(PT_WHITELIST)

if __name__ == "__main__":
    sys.exit(main())
