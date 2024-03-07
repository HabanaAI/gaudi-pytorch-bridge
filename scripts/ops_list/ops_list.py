# ******************************************************************************
# Copyright (C) 2022 Habana Labs, Ltd. an Intel Company
# All Rights Reserved.
#
# Unauthorized copying of this file or any element(s) within it, via any medium
# is strictly prohibited.
# This file contains Habana Labs, Ltd. proprietary and confidential information
# and is subject to the confidentiality and license agreements under which it
# was provided.
#
# ******************************************************************************

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
