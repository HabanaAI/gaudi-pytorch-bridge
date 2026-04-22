###############################################################################
# Copyright (c) 2024-2026 Intel Corporation
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

"""
This script runs shared layer report generator and processes the report
into documentation in .rst format.
The documentation is saved in the directory specified in --doc_path argument.
- [-h, --help] - Print help
- [-p, --doc_path] - Specifies the path to save the documentation
- [-c, --gen_custom_doc] - Indicates whether Pytorch_Custom_Operators.rst should be generated
Example:
python report_parser.py --path ${PYTORCH_MODULES_ROOT_PATH}/docs --gen_custom_doc
"""

import argparse
from collections import defaultdict
from pathlib import Path

import doc_templates
import slrg_py as slrg
import torch


def parse_args():
    parser = argparse.ArgumentParser(
        description="Example: python report_parser.py --path ${PYTORCH_MODULES_ROOT_PATH}/docs --gen_custom_doc"
    )
    parser.add_argument("-p", "--path", help="Specifies the path to save the documentation", type=str)
    parser.add_argument(
        "-c",
        "--gen_custom_doc",
        help="Indicates whether Pytorch_Custom_Operators.rst should be generated",
        action="store_true",
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
                "float4": True,
                "int4": True,
            }
            for item in items:
                for type in supported_types:
                    if type not in ("float4", "int4"):
                        supported_types[type] &= item.second[type]
                supported_types["float4"] &= item.second.fp4_support
                supported_types["int4"] &= item.second.int4_support
            support_summary_by_namespace[namespace] = supported_types
        for namespace, supported_types in support_summary_by_namespace.items():
            op_name = key
            if key.endswith("_") and not key.endswith("__"):
                op_name = key[:-1] + r"\_"
            template = doc_templates.DOC_RST_ROW
            wide_spacer = False
            if namespace == "torch.hpu":
                template = doc_templates.CUSTOM_DOC_RST_ROW
                wide_spacer = True
            elif namespace == "torch.hpu.optimizer":
                template = doc_templates.CUSTOM_DOC_RST_OPTIMIZER_ROW
            row = template.format(
                op_name=doc_templates.get_operator_name_with_spacer(op_name, wide_spacer),
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
                fp4=doc_templates.get_support_value(supported_types["float4"]),
                int4=doc_templates.get_support_value(supported_types["int4"]),
                namespace=namespace,
            )
            doc_rows_by_namespace[namespace].append(row)

    documentation = doc_templates.DOC_FILE.format(
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
    documentation = "\n".join(line.rstrip() for line in documentation.splitlines())

    if not Path(args.path).parent.exists():
        Path(args.path).parent.mkdir(parents=True)
    with open(args.path + "/Pytorch_Operators.rst", "w") as f:
        print(documentation, file=f)
    if args.gen_custom_doc:
        custom_operators_documentation = doc_templates.CUSTOM_DOC_FILE.format(
            optimizer_operators=str.join("", doc_rows_by_namespace["torch.hpu.optimizer"]),
            custom_operators=str.join("", doc_rows_by_namespace["torch.hpu"]),
        )
        custom_operators_documentation = "\n".join(
            line.rstrip() for line in custom_operators_documentation.splitlines()
        )
        with open(args.path + "/Pytorch_Custom_Operators.rst", "w") as f:
            print(
                custom_operators_documentation,
                file=f,
            )


if __name__ == "__main__":
    args = parse_args()
    if args.path is None:
        raise Exception("You must specify the path to save the documentation file by using [-p] or [--path]")
    gen_doc(args)
