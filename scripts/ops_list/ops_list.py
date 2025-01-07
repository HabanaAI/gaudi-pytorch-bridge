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


import argparse

import torch._C as C

parser = argparse.ArgumentParser()
parser.add_argument(
    "--dispatch_key",
    "-dk",
    help="Typical keys are: CPU, CUDA, HPU",
    required=True,
)
parser.add_argument("--autograd", "-a", action="store_true")


def main():
    args = parser.parse_args()
    dispatch_key = args.dispatch_key
    if "HPU" in dispatch_key:
        import habana_frameworks.torch

    if args.autograd:
        dispatch_key = "Autograd" + dispatch_key

    C._dispatch_print_registrations_for_dispatch_key(dispatch_key)


if __name__ == "__main__":
    main()
