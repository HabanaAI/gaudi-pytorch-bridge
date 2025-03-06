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
import re


def read_file(filepath):
    with open(filepath, "r") as file:
        return file.readlines()
    return None


def get_line_idx(lines, regex):
    for i, line in enumerate(lines):
        if re.match(regex, line):
            return i
    return None


def remove_consecutive_entries(list):
    out_list = []
    for i in range(0, len(list) - 1):
        if list[i] != list[i + 1] - 1:
            out_list.append(list[i])
    out_list.append(list[-1])
    return out_list


def slice(str, range):
    return str[range[0] : range[1]]


def concat(str, *ranges):
    return "".join(slice(str, r) for r in ranges)


if __name__ == "__main__":
    qnpu_path = os.environ.get("QNPU_PATH")
    if not qnpu_path:
        print("You need to source qnpu environment.")
        exit()

    input_filepath = os.path.join(qnpu_path, "src/pytorch-integration/docs/Pytorch_Operators.rst")
    output_filepath = os.path.join(qnpu_path, "src/pytorch-integration/docs/Pytorch_Operators_Split_Dtypes.rst")

    parser = argparse.ArgumentParser(
        description="""
     This program reads Pytorch_Operators.rst file and splits the PyTorch Operators Support Summary
     table into two, for floating and integer dtypes.
     """
    )

    parser.add_argument("--input", type=str, default=input_filepath, help=f"Input file path, default: {input_filepath}")
    parser.add_argument(
        "--output", type=str, default=output_filepath, help=f"Output file path, default: {output_filepath}"
    )

    args = parser.parse_args()

    lines = read_file(args.input)
    if not lines:
        print("Unable to open file.")
        exit()

    operators_support_summary_line_idx = get_line_idx(lines, r"^PyTorch Operators Support Summary")
    if not operators_support_summary_line_idx:
        print("Could not find PyTorch Operators Support Summary section, exiting.")
        exit()

    table_header_line_idx = get_line_idx(lines, r"^\*\*PyTorch Operator\*\*") - 1
    if not table_header_line_idx:
        print("Could not find PyTorch Operators Support Summary table, exiting.")
        exit()

    header_line = lines[table_header_line_idx]
    columns = remove_consecutive_entries([i for i, c in enumerate(header_line) if c == " "])

    range_op = (0, columns[0])
    range_floats = (columns[0], columns[4])
    range_ints = (columns[4], columns[9])
    range_operator_type = (columns[9], len(header_line))

    table_orig = lines[table_header_line_idx:]
    table_floats = "".join(concat(line, range_op, range_floats, range_operator_type) for line in table_orig)
    table_ints = "".join(concat(line, range_op, range_ints, range_operator_type) for line in table_orig)

    with open(args.output, "w") as f:
        f.writelines(lines[0:operators_support_summary_line_idx])
        f.writelines("PyTorch Operators Support Summary - Floating Point types\n")
        f.writelines("========================================================\n\n")
        f.writelines(".. rst-class:: datatable\n\n")
        f.writelines(table_floats)

        f.writelines("\n")
        f.writelines("PyTorch Operators Support Summary - Integer types\n")
        f.writelines("=================================================\n\n")
        f.writelines(".. rst-class:: datatable\n\n")
        f.writelines(table_ints)
