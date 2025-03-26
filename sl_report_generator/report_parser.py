###############################################################################
#
#  Copyright (c) 2024-2025 Intel Corporation
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

"""
This script runs shared layer report generator and processes the report
into documentation in .rst format.
The documentation is saved to the file specified in --doc_path argument.
- [-h, --help] - Print help
- [-p, --doc_path] - Specifies the path (including file name) to save the
  documentation
Example:
python report_parser.py --path ${PYTORCH_MODULES_ROOT_PATH}/docs/Pytorch_Operators.rst
"""
import argparse
from collections import defaultdict
from pathlib import Path

import doc_templates
import slrg_py as slrg
import torch


def parse_args():
    parser = argparse.ArgumentParser(
        description="Example: python report_parser.py --path ${PYTORCH_MODULES_ROOT_PATH}/docs/Pytorch_Operators.rst"
    )
    parser.add_argument(
        "-p", "--path", help="Specifies the path (including file name) to save the documentation", type=str
    )
    args = parser.parse_args()
    return args


def gen_doc(args):
    report = slrg.run_report_gen()
    keys = list(report.keys())
    keys.sort()
    doc_rows_by_namespace = defaultdict(list)
    for key in keys:
        operator_report_vector = report[key]
        report_grouped_by_namespace = defaultdict(list)
        support_summary_by_namespace: dict = {}
        for operator_report_pair in operator_report_vector:
            report_grouped_by_namespace[operator_report_pair.first.op_namespace].append(operator_report_pair)
        for namespace, items in report_grouped_by_namespace.items():
            supported_types: dict = {
                torch.float: True,
                torch.bfloat16: True,
                torch.half: True,
                torch.float8_e4m3fn: True,
                torch.float8_e5m2: True,
                torch.long: True,
                torch.int: True,
                torch.short: True,
                torch.int8: True,
                torch.bool: True,
            }
            for item in items:
                for type in supported_types.keys():
                    supported_types[type] &= item.second[type]
            support_summary_by_namespace[namespace] = supported_types
        for namespace, supported_types in support_summary_by_namespace.items():
            op_name = key
            if key.endswith("_") and not key.endswith("__"):
                op_name = key[:-1] + r"\_"
            row = doc_templates._DOC_RST_ROW.format(
                op_name=doc_templates.get_operator_name_with_spacer(op_name),
                fp32=doc_templates.get_support_value(supported_types[torch.float]),
                bf16=doc_templates.get_support_value(supported_types[torch.bfloat16]),
                fp16=doc_templates.get_support_value(supported_types[torch.half]),
                fp8=doc_templates.get_support_value(
                    supported_types[torch.float8_e4m3fn] & supported_types[torch.float8_e5m2]
                ),
                int64=doc_templates.get_support_value(supported_types[torch.long]),
                int32=doc_templates.get_support_value(supported_types[torch.int]),
                int16=doc_templates.get_support_value(supported_types[torch.short]),
                int8=doc_templates.get_support_value(supported_types[torch.int8]),
                bool=doc_templates.get_support_value(supported_types[torch.bool]),
                namespace=namespace,
            )
            doc_rows_by_namespace[namespace].append(row)

    documentation = doc_templates._DOC_FILE.format(
        operators_torch_nn_functional=str.join("", doc_rows_by_namespace["torch.nn.functional"]),
        operators_torch_linalg=str.join("", doc_rows_by_namespace["torch.linalg"]),
        operators_torch=str.join("", doc_rows_by_namespace["torch"]),
        operators_torch_ops_aten=str.join("", doc_rows_by_namespace["torch.ops.aten"]),
        operators_torch_nn=str.join("", doc_rows_by_namespace["torch.nn"]),
        operators_torch_nn_utils=str.join("", doc_rows_by_namespace["torch.nn.utils"]),
        operators_torch_tensor=str.join("", doc_rows_by_namespace["torch.Tensor"]),
        operators_torch_special=str.join("", doc_rows_by_namespace["torch.special"]),
        operators_torchvision_ops=str.join("", doc_rows_by_namespace["torchvision.ops"]),
        operators_torch_ops=str.join("", doc_rows_by_namespace["torch.ops"]),
    )

    if not Path(args.path).parent.exists():
        Path(args.path).parent.mkdir(parents=True)
    print(documentation, file=open(args.path, "w"))


if __name__ == "__main__":
    args = parse_args()
    if args.path is None:
        raise Exception("You must specify the path to save the documentation file by using [-p] or [--path]")
    gen_doc(args)
