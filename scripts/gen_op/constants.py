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

from enum import Enum
from typing import Any, NamedTuple

from lark.tree import Tree

from .op import Op
from .version_checker import is_pytorch_older_than


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
    xtree: Tree
    rwxtree: Tree
    func: str
    xfunc: str
    op_frontend_eager: str | None
    op_frontend_lazy: str
    op_backend: str
    cname: str
    sig: str
    rwsig: str
    cppsig: str
    funsig: str
    mapsig: str
    aten_sig: str
    dtdf: bool
    ctxop: Op
    opgroup: str
    fc_params: Any
    op_variant: str
    ns: str


class OpMeta(NamedTuple):
    op_variant: str
    mapsig: str
    func: str
    funsig: str


# List of non-leaf ops we want to override both forward + backward.
# TODO(https://github.com/pytorch/pytorch/issues/39959)
FN_AUTOGRAD_HPU = {"matmul", "softmax.int", "dropout"}

if is_pytorch_older_than("2.7.0"):
    TYPE_NSMAP = {
        "Tensor": "at::Tensor",
        "TensorList": "at::TensorList",
        "Scalar": "at::Scalar",
        "Storage": "at::Storage",
        "IntArrayRef": "at::IntArrayRef",
        "OptionalIntArrayRef": "at::OptionalIntArrayRef",
        "ArrayRef": "at::ArrayRef",
        "Generator": "at::Generator",
        "Layout": "at::Layout",
        "ScalarType": "at::ScalarType",
        "Device": "c10::Device",
        "MemoryFormat": "at::MemoryFormat",
        "QScheme": "at::QScheme",
        "Dimname": "at::Dimname",  # namedtensor-only
        "DimnameList": "at::DimnameList",  # namedtensor-only
        "ITensorListRef": "at::ITensorListRef",
        "OptionalSymIntArrayRef": "at::OptionalSymIntArrayRef",
    }
else:
    TYPE_NSMAP = {}


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
}

# List of ops that will not be checked for shared layer support.
# These exceptions are tracked in SW-213270
OP_VALIDATOR_EXCEPTIONS = {
    # op name: reason for lack of op_validator
    "mixture_of_experts.fp8_fused_weights_scalars": "custom op",
    "mixture_of_experts.fp8": "custom op",
    "mixture_of_experts.fp8_scalars": "custom op",
    "mixture_of_experts.fp8_fused_weights": "custom op",
    "mixture_of_experts.fp8_fused_weights_scalars_dynamic": "custom op",
    "mixture_of_experts.fp8_dynamic": "custom op",
    "mixture_of_experts.fp8_scalars_dynamic": "custom op",
    "mixture_of_experts.fp8_fused_weights_dynamic": "custom op",
    "cast_to_fp8": "not implemented yet",
    "fp8_gemm": "not implemented yet",
    "_native_batch_norm_legit": "not implemented yet",
    "_native_batch_norm_legit_no_training": "not implemented yet",
    "_native_batch_norm_legit.no_stats": "not implemented yet",
    "_native_batch_norm_legit_functional": "not implemented yet",
    "native_batch_norm": "not implemented yet",
    "native_batch_norm.out": "not implemented yet",
    "native_batch_norm_backward": "not implemented yet",
    "native_layer_norm": "not implemented yet",
    "native_layer_norm_backward": "not implemented yet",
    "_weight_norm_interface": "not implemented yet",
    "_weight_norm_interface_backward": "not implemented yet",
    "_prelu_kernel": "not implemented yet",
    "sdpa_bwd": "custom op",
    "quantize_per_tensor": "custom op",
    "quantize_per_tensor.tensor": "custom op",
    "quantize_per_tensor.tensor2": "custom op",
    "quantize_per_channel": "custom op",
    "dequantize_per_channel": "custom op",
    "dequantize_per_tensor": "custom op",
    "dequantize_per_tensor.tensor": "custom op",
    "dequantize_per_tensor.tensor2": "custom op",
    "deform_conv2d": "custom op",
    "_deform_conv2d_backward": "custom op",
    "ctc_loss_custom": "custom op",
    "ctc_loss_custom_backward": "custom op",
    "cast_from_fp8": "custom op",
    "cast_from_fp8.scalar": "custom op",
    "cast_from_fp8.scalar_list": "custom op",
    "cast_to_fp8_v2": "custom op",
    "cast_to_fp8_v2.scalar": "custom op",
    "cast_to_fp8_v2.scalar_list": "custom op",
    "cast_to_fp8_hybrid": "custom op",
    "conv2d_fp8": "custom op",
    "conv2d_fp8.scalar": "custom op",
    "custom_softmax": "custom op",
    "fp8_gemm_v2": "custom op",
    "fp8_gemm_v2.scalar": "custom op",
    "fp8_gemm_v2.scalar_list": "custom op",
    "kv_reorder_": "custom op",
    "scaled_masked_softmax": "custom op",
    "scaled_masked_triangular_softmax": "custom op",
    "scaled_triangular_softmax": "custom op",
    "scaled_triangular_softmax_retain": "custom op",
    "softmax_fp8": "custom op",
    "softmax_fp8.Scalar_scales": "custom op",
    "softmax_fp8.Scalar": "custom op",
    "ragged_softmax": "custom op",
    "rms_norm": "custom op",
    "rms_norm_fast": "custom op",
    "rms_norm_backward": "custom op",
    "rms_norm_fast_backward": "custom op",
    "rotary_pos_embedding": "custom op",
    "rotary_pos_embedding_backward": "custom op",
    "in_place_interleave_": "custom op",
    "sdpa_recomp_bwd": "custom op",
    "fp8_sdpa_bwd": "custom op",
}

CP_TYPE_CHECK_MAP = {
    "double": "isDouble",
    "bool": "isBool",
    "int64_t": "isInt",
}

if is_pytorch_older_than("2.7.0"):
    CP_TYPE_CHECK_MAP.update(
        {
            "Scalar": "isScalar",
            "Tensor": "isTensor",
            "ITensorListRef": "isTensorList",
            "TensorList": "isTensorList",
            "std::optional<ArrayRef>": "isList",
            "IntArrayRef": "isList",
        }
    )
else:
    CP_TYPE_CHECK_MAP.update(
        {
            "at::Scalar": "isScalar",
            "at::Tensor": "isTensor",
            "at::ITensorListRef": "isTensorList",
            "at::TensorList": "isTensorList",
            "::std::optional<at::ArrayRef>": "isList",
            "at::IntArrayRef": "isList",
        }
    )

# For PT2.0, there are non-mandatory op (from PT2.0 point of view),
# that we still need to register in the new Eager flow.
# In order to do it, we overwrite them to default=False, dispatch=True
NON_MANDATORY_OPS_ALLOWLIST = {
    "all",
    "any",
    "complex",
    "convolution_overrideable",
    "convolution_backward_overrideable",
    "is_pinned",
    "native_layer_norm",
    "native_group_norm",
    "repeat",
    "_unsafe_view",
}

AUTOCAST_REPLACEMENTS = (
    ("::std::tuple<at::Tensor,at::Tensor>", "tuple_2_tensors"),
    ("::std::tuple<at::Tensor,at::Tensor,at::Tensor>", "tuple_3_tensors"),
    (
        "::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor>",
        "tuple_4_tensors",
    ),
    (
        "::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor>",
        "tuple_5_tensors",
    ),
    (
        "::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor>",
        "tuple_6_tensors",
    ),
    (
        "::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor,int64_t>",
        "tuple_4_tensors_int64",
    ),
    (
        "::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor,int64_t,int64_t>",
        "tuple_4_tensors_2_int64",
    ),
    (
        "::std::tuple<at::Tensor,at::Tensor,double,int64_t>",
        "tuple_2_tensors_double_int64",
    ),
    (
        "::std::tuple<at::Tensor,at::Tensor,at::Tensor,::std::vector<at::Tensor>>",
        "tuple_3_tensors_vector",
    ),
    ("::std::tuple<double,int64_t>", "tuple_double_int64"),
    (
        "::std::tuple<at::Tensor,::std::vector<at::Tensor>>",
        "tuple_tensor_vector",
    ),
    (
        "::std::tuple<::std::vector<at::Tensor>,at::Tensor>",
        "tuple_vector_tensor",
    ),
    (
        "::std::tuple<at::Tensor,::std::vector<at::Tensor>,::std::vector<at::Tensor>>",
        "tuple_tensor_2_vectors",
    ),
    (
        "::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor,int64_t,int64_t,at::Tensor,at::Tensor>",
        "tuple_4_tensors_2_int64_2_tensors",
    ),
    (
        "::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor,int64_t,int64_t,int64_t,int64_t,at::Tensor>",
        "tuple_4_tensors_4_int64_tensor",
    ),
    (
        "::std::tuple<at::Tensor,at::Tensor,int64_t,int64_t,at::Tensor>",
        "tuple_2_tensors_2_int64_tensor",
    ),
    (
        "::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor,int64_t,int64_t,at::Tensor,at::Tensor,at::Tensor>",
        "tuple_4_tensors_2_int64_3_tensor",
    ),
    (
        "::std::tuple<::std::vector<at::Tensor>,::std::vector<at::Tensor>,::std::vector<at::Tensor>>",
        "tuple_3_vectors",
    ),
    (
        "::std::tuple<::std::vector<at::Tensor>,::std::vector<at::Tensor>,::std::vector<at::Tensor>,::std::vector<at::Tensor>,::std::vector<at::Tensor>>",
        "tuple_5_vectors",
    ),
    ("SymIntArrayRef", "IntArrayRef"),
    ("c10::SymInt", "int64_t"),
    (
        "::std::tuple<::std::vector<at::Tensor>,::std::vector<at::Tensor>,::std::vector<at::Tensor>,::std::vector<at::Tensor>>",
        "tuple_4_vectors",
    ),
)

AUTOCAST_BLOCKLIST = {
    "_cummax_helper",
    "_cummin_helper",
    "fused_moving_avg_obs_fake_quant",
    "_fused_moving_avg_obs_fq_helper",
    "_native_batch_norm_legit",
    "sym_size.int",
    "sym_numel",
    "sym_stride.int",
    "sym_storage_offset",
    "_scaled_dot_product_flash_attention",
    "_efficient_attention_forward",
    "_batch_norm_with_update",
    "_scaled_dot_product_fused_attention_overrideable",
}


# List of override_fn ops that are supporting eager frontend
EAGER_OPS_OVERRIDE_FNS_ALLOWLIST = {
    "_copy_from",
    "_copy_from_and_resize",
    "complex_hpu",
    "as_strided_hpu",
    "set_",
    "set_source_Storage",
    "set_source_Storage_storage_offset",
    "set_source_Tensor",
    "view_hpu",
    "_local_scalar_dense_hpu",
    "repeat_hpu",
}

EAGER_CUSTOM_FRONTENDS_ALLOWLIST = {
    "AddCOpFE",
    "ArangeFE",
    "BernoulliFE",
    "BernoulliFEOut",
    "ClampFE",
    "FillFE",
    "BinaryScalarFE",
    "GeneratorToSeed",
    "GeneratorToSeedOut",
    "IndexFE",
    "IndexOutFE",
    "NativeDropoutFE",
    "TopKFE",
    "ConvolutionOverrideableFE",
    "ConvolutionBackwardOverrideableFE",
}
