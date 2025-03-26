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

_DOC_FILE = """
.. _pytorch-operators:

****************************
PyTorch Operators
****************************



Overview
========

This document provides an overview of PyTorch-supported operators for the
Intel® Gaudi® AI accelerator. Note that the operators listed below support
only selected variants and limited optional parameters for Gaudi.

For details on Fused Ops, see :ref:`custom_operators`.

PyTorch Operators Support Summary
=================================

.. rst-class:: datatable

====================================  ======== ======== ======== ======= ========= ========= ========= ======== ========  ======================
**PyTorch Operator**                  **FP32** **BF16** **FP16** **FP8** **INT64** **INT32** **INT16** **INT8** **BOOL**  **Operator Type**
====================================  ======== ======== ======== ======= ========= ========= ========= ======== ========  ======================
{operators_torch_nn_functional}\
{operators_torch_linalg}\
{operators_torch}\
{operators_torch_ops_aten}\
{operators_torch_nn}\
{operators_torch_nn_utils}\
{operators_torch_tensor}\
{operators_torch_special}\
{operators_torchvision_ops}\
{operators_torch_ops}\
====================================  ======== ======== ======== ======= ========= ========= ========= ======== ========  ======================"""

_DOC_RST_ROW = """{op_name}     {fp32}      {bf16}      {fp16}     {fp8}       {int64}      {int32}       {int16}       {int8}      {bool}    {namespace}
"""

OPERATOR_NAME_MAX_LENGTH = 36


def get_operator_name_with_spacer(op_name: str) -> str:
    return op_name + " " * (OPERATOR_NAME_MAX_LENGTH - len(op_name))


def get_support_value(is_supported: bool) -> str:
    if is_supported:
        return "Yes"
    else:
        return "No "
