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
