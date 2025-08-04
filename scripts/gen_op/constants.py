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

from enum import Enum
from typing import Any, NamedTuple

from lark.tree import Tree

from .op import Op


class HabanaExecutionMode(Enum):
    EAGER = 0
    COMPILE = 1
    LAZY = 2
    INVALID = 3


def get_execution_mode_from_string(execution_mode: str) -> HabanaExecutionMode:
    return HabanaExecutionMode[execution_mode.upper()]


class FuncDef(NamedTuple):
    cpp_sig: str
    aten_sig: str
    dtdf: bool


class OpGen(NamedTuple):
    tree: Tree
    func: str
    op_frontend_eager: str | None
    op_frontend_lazy: str
    op_backend: str
    cname: str
    sig: str
    cppsig: str
    funsig: str
    aten_sig: str
    ctxop: Op
    opgroup: str
    fc_params: Any
    op_variant: str
    ns: str
    only_slrg: bool


class OpMeta(NamedTuple):
    op_variant: str
    mapsig: str
    func: str
    funsig: str
    autograd: bool


AVAILABLE_FIELDS = {
    "acc_thread",
    "autograd",
    "broadcast",
    "frontend_blocklist",
    "custom_fill_params",
    "custom_op_schema",
    "dtypes",
    "early_exit",
    "fallback_check",
    "guid",
    "handle_bool_inputs",
    "hpu_wrap",
    "inplace_ids",
    "is_custom_op_out_variant",
    "lazy",
    "no_compute_flag",
    "only_shared_layer",
    "only_slrg",
    "op_backend",
    "op_frontend",
    "op_validator",
    "out_ids",
    "output_meta",
    "override_fn",
    "promote_int_to_float",
    "promote_int_to_long",
    "promote_to_common_type",
    "safe_cast_check",
    "scalar_ids",
    "schema_args",
    "st_meta",
    "synapse_layouts",
    "tpc_input_order",
    "namespaces",
    "pytorch_module_names",
    "overwritten_op_names_in_slrg",
    "skip_slrg",
    "treat_as_dtdf",
    "op_validator_exception",
    "handle_output_mask",
    "custom_cpp_sig",
    "treat_as_non_inplace",
}


CP_TYPE_CHECK_MAP = {
    "double": "isDouble",
    "bool": "isBool",
    "int64_t": "isInt",
    "at::Scalar": "isScalar",
    "at::Tensor": "isTensor",
    "at::ITensorListRef": "isTensorList",
    "at::TensorList": "isTensorList",
    "::std::optional<at::ArrayRef>": "isList",
    "at::IntArrayRef": "isList",
}

NAMESPACE_TO_POSTFIX = {
    "hpu": "_custom",
    "quantized_decomposed": "_quant",
    "torchvision": "_torchvision",
    "torch_sparse": "_torch_sparse",
}
