#!/usr/bin/env python
###############################################################################
#
#  Copyright (c) 2021-2025 Intel Corporation
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
import os

import torch
from gen_op.code_generation import generate, generate_check_kernel_support


def parse_gen_op_params():
    dirname = os.path.dirname
    join = os.path.join
    realpath = os.path.realpath

    torch_pkg_path = torch.__path__[0]
    pytorch_integration_path = dirname(dirname(realpath(__file__)))

    arg_parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    arg_parser.add_argument("--output_dir", metavar="OUTPUT_DIR", type=str)
    arg_parser.add_argument(
        "--yaml",
        default=join(pytorch_integration_path, "scripts/hpu_op.yaml"),
        help="The path to the Hpu Op yaml file",
    )
    arg_parser.add_argument(
        "pt_signatures",
        nargs="?",
        default=join(torch_pkg_path, "include/ATen/RegistrationDeclarations.h"),
        type=str,
        metavar="TYPE_DEFAULT_FILE",
        help="The path to the RegistrationDeclarations.h file",
    )
    arg_parser.add_argument(
        "native_functions",
        nargs="?",
        default=join(
            torch_pkg_path,
            "../torchgen/packaged/ATen/native/native_functions.yaml",
        ),
        type=str,
        metavar="NATIVE_FUNCTIONS_FILE",
        help="The path to the native_functions.yaml file",
    )
    arg_parser.add_argument(
        "--check_kernel_support",
        default=False,
        action="store_true",
        help="Check kernel support or normal kernel generation",
    )
    arg_parser.add_argument(
        "--templates",
        default=join(pytorch_integration_path, "scripts/hpu_op_templates.yaml"),
        type=str,
        metavar="CHECK_KERNEL_SUPPORT_FILE",
        help="The path to the yaml templates file",
    )

    return arg_parser.parse_known_args()[0]


if __name__ == "__main__":
    args = parse_gen_op_params()
    if args.check_kernel_support:
        generate_check_kernel_support(args)
    else:
        generate(args)
