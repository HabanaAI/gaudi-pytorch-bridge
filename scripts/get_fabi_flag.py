###############################################################################
# Copyright (c) 2025 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
###############################################################################

import argparse
import re
import subprocess
import sys

import torch

# https://gcc.gnu.org/onlinedocs/gcc/C_002b_002b-Dialect-Options.html
FABI_VERSIONS_PER_GCC = {11: "16", 12: "17", 13: "18", 14: "19", 15: "20", 16: "21"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("compiler_path", help="Path to the C++ compiler")
    args = parser.parse_args()
    print_fabi_flag(args)


def print_fabi_flag(args: argparse.Namespace):
    fabi_flag = get_torch_fabi_flag()
    if fabi_flag:
        print(f"-fabi-version={fabi_flag}")
        return

    try:
        torch_gcc_version = get_major_version(get_torch_gcc_version())
        plugin_gcc_version = get_major_version(get_gcc_version(args.compiler_path))
    except FileNotFoundError:
        print("Invalid path to the GCC compiler", file=sys.stderr)
        return
    except ValueError:
        print("GCC version cannot be determined!", file=sys.stderr)
        return

    if torch_gcc_version < plugin_gcc_version:
        try:
            print(f"-fabi-version={FABI_VERSIONS_PER_GCC[torch_gcc_version]}")
        except KeyError:
            print("The system GCC version is not supported!", file=sys.stderr)
            return
    elif torch_gcc_version > plugin_gcc_version:
        print(
            f"The version of GCC used {plugin_gcc_version} is lower than the Torch version {torch_gcc_version}. "
            f"This may cause problems in runtime! It is recommended to upgrade GCC to version {torch_gcc_version}",
            file=sys.stderr,
        )
        return


def get_torch_fabi_flag() -> str:
    config = torch.__config__.show()
    fabi_version_compiler_flag = "-fabi-version="
    search_result = re.search(rf"{fabi_version_compiler_flag}(\d+)", config)
    if search_result:
        return search_result.group(1)
    else:
        return ""


def get_torch_gcc_version() -> str:
    config = torch.__config__.show()
    search_result = re.search(r"GCC\s+(\d+\.\d+)", config)
    if search_result:
        return search_result.group(1)
    else:
        return ""


def get_gcc_version(compiler_path: str) -> str:
    output = subprocess.check_output([compiler_path, "--version"], stderr=subprocess.STDOUT)  # noqa S603
    first_line = output.decode().split("\n")[0]
    search_result = re.search(r"\d+\.\d+(\.\d+)?", first_line)
    if search_result:
        return search_result.group(0)
    else:
        return ""


def get_major_version(gcc_version: str) -> int:
    return int(gcc_version.split(".", 1)[0])


if __name__ == "__main__":
    main()
