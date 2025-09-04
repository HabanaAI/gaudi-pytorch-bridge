###############################################################################
#
#  Copyright (c) 2025 Intel Corporation
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

import yaml

parser = argparse.ArgumentParser()
parser.add_argument("--regi_decla", "-rd", help="RegistrationDeclarations.h file", required=True)
parser.add_argument("--yaml", "-y", help="hpu_op.yaml file", required=True)


def op_with_optional_tensor(line):
    if '"schema":' in line and "optional<Tensor>" in line:
        line = line.strip()
        par = line.find("(")
        if par < 0:
            raise Exception(f"Unxepected line syntax: {line}")
        space = line.rfind(" ", 0, par)
        if space < 0:
            raise Exception(f"Unxepected line syntax: {line}")
        return line[space + 1 : par]
    return None


def read_ops_with_optional_tensor(filename):
    label = "Ops with optional tensors"
    print(f"BEGIN: {label}")
    result = []
    with open(filename) as file:
        for line in file:
            op = op_with_optional_tensor(line)
            if op:
                result.append(op)
    print(f"END: {label}, count = {len(result)}")
    return result


def read_ops_with_custom_sl_meta(filename):
    label = "Ops with custom shared layer meta"
    print(f"BEGIN: {label}")
    result = {}
    with open(filename) as file:
        op_data = yaml.load(file.read(), Loader=yaml.CSafeLoader)
        for op in op_data:
            if "op_validator" in op_data[op] and op_data[op]["op_validator"] != "check-node-with-shared-layer":
                result[op] = op_data[op]["op_validator"]
    print(f"END: {label}, count = {len(result)}")
    return result


def filter_ops(ops_with_optional_tensor, ops_with_custom_sl_meta):
    label = "Ops with both"
    print(f"BEGIN: {label}")
    result = {}
    for op in ops_with_optional_tensor:
        if op in ops_with_custom_sl_meta:
            result.setdefault(ops_with_custom_sl_meta[op], []).append(op)
    print(f"END: {label}, count = {len(result)}")
    return result


def main():
    args = parser.parse_args()
    ops_with_optional_tensor = read_ops_with_optional_tensor(args.regi_decla)
    ops_with_custom_sl_meta = read_ops_with_custom_sl_meta(args.yaml)
    ops_with_both = filter_ops(ops_with_optional_tensor, ops_with_custom_sl_meta)

    print()
    for meta, ops_list in ops_with_both.items():
        ops_list_str = ", ".join(ops_list)
        print(f"{meta} <-- {ops_list_str}")


if __name__ == "__main__":
    main()
