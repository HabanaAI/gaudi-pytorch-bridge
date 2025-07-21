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

DOC_FILE = """
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

DOC_RST_ROW = """{op_name}     {fp32}      {bf16}      {fp16}     {fp8}       {int64}      {int32}       {int16}       {int8}      {bool}    {namespace}
"""

CUSTOM_DOC_FILE = """
.. _pytorch-custom-operators:

****************************
PyTorch Custom Operators
****************************

Overview
========

This document summarizes the SynapseAI® Software PyTorch supported custom operators for
Habana® Gaudi®. Note that the operators listed below support only selected
variants and limited optional parameters for Gaudi.

The ops in the lists below are available under the torch.ops.hpu namespace.
The supported dtypes indicate the input dtypes and any given operator may support output dtypes not listed here.
For example: cast_from_fp8 supports FP8 inputs but FP32 and BF16 outputs, but the output types are not mentioned here.


Custom Fused Optimizers Support Summary
=======================================

.. rst-class:: datatable

====================================  ======== ======== ======== ========= ========= ======== ======== ========= ======== ========
**Custom Optimizers**                 **FP32** **BF16** **FP16** **INT64** **INT32** **INT8** **BOOL**  **FP8**  **FP4**  **INT4**
====================================  ======== ======== ======== ========= ========= ======== ======== ========= ======== ========
{optimizer_operators}\
====================================  ======== ======== ======== ========= ========= ======== ======== ========= ======== ========

All of the custom fused optimizers are exposed under their own wrapper functions.
For more details on their usage see :ref:`custom_operators`.
Custom optimizers are only supported in Lazy and Eager modes of execution.

Custom Operators Support Summary
=================================

.. rst-class:: datatable

====================================================  ======== ======== ======== ========= ========= ======== ======== ========= ======== ========
**Custom Operator**                                   **FP32** **BF16** **FP16** **INT64** **INT32** **INT8** **BOOL**  **FP8**  **FP4**  **INT4**
====================================================  ======== ======== ======== ========= ========= ======== ======== ========= ======== ========
{custom_operators}\
====================================================  ======== ======== ======== ========= ========= ======== ======== ========= ======== ========"""

CUSTOM_DOC_RST_OPTIMIZER_ROW = """{op_name}     {fp32}      {bf16}      {fp16}      {int64}       {int32}       {int8}     {bool}       {fp8}       {fp4}      {int4}
"""
CUSTOM_DOC_RST_ROW = """{op_name}    {fp32}      {bf16}      {fp16}      {int64}       {int32}       {int8}      {bool}      {fp8}       {fp4}      {int4}
"""

OPERATOR_NAME_MAX_LENGTH = 36
WIDE_OPERATOR_NAME_MAX_LENGTH = 52


def get_operator_name_with_spacer(op_name: str, is_wide_name: bool = False) -> str:
    max_length = WIDE_OPERATOR_NAME_MAX_LENGTH if is_wide_name else OPERATOR_NAME_MAX_LENGTH
    return op_name + " " * (max_length - len(op_name))


def get_support_value(is_supported: bool) -> str:
    if is_supported:
        return "Yes"
    else:
        return "No "
