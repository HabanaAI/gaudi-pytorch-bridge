#!/usr/bin/env python
###############################################################################
#
#  Copyright (c) 2021-2024 Intel Corporation
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


from __future__ import print_function

import argparse
import collections.abc
import copy
import json
import os
import re
import sys
from abc import ABC, abstractmethod
from collections import defaultdict
from enum import Enum
from functools import wraps
from typing import Any, Dict, List, Set, Tuple

import gen_op_files.code_templates as templates
import gen_op_files.parser as parser
import torch
import yaml
from gen_op_files.custom_ops import cpp_from_schema
from packaging.version import Version
from torchgen import local
from torchgen.api.translate import translate
from torchgen.api.types import CppSignatureGroup
from torchgen.api.unboxing import convert_arguments
from yaml import CLoader as Loader


class HabanaExecutionMode(Enum):
    EAGER = 0
    COMPILE = 1
    LAZY = 2
    INVALID = 3


def get_execution_mode_from_string(execution_mode: str) -> HabanaExecutionMode:
    return HabanaExecutionMode[execution_mode.upper()]


def namedtuple_with_defaults(typename, field_names, default_values=()):
    ntuple = collections.namedtuple(typename, field_names)
    ntuple.__new__.__defaults__ = (None,) * len(ntuple._fields)
    if isinstance(default_values, collections.abc.Mapping):
        prototype = ntuple(**default_values)
    else:
        prototype = ntuple(*default_values)
    ntuple.__new__.__defaults__ = tuple(prototype)
    return ntuple


FuncDef = namedtuple_with_defaults("FuncDef", "cpp_sig, aten_sig, dtdf")

OpGen = namedtuple_with_defaults(
    "OpGen",
    "tree, xtree, rwxtree, func, xfunc, op_frontend_eager, op_frontend_lazy, op_backend, cname, sig, rwsig, cppsig, funsig, mapsig, aten_sig, dtdf, ctxop, opgroup, fc_params, op_variant, ns",
)

OpMeta = namedtuple_with_defaults("OpMeta", "op_variant, mapsig, func, funsig")


# List of non-leaf ops we want to override both forward + backward.
# TODO(https://github.com/pytorch/pytorch/issues/39959)
_FN_AUTOGRAD_HPU = set(
    [
        "matmul",
        "softmax.int",
        "dropout",
    ]
)

_TYPE_NSMAP = {
    "Tensor": "at::Tensor",
    "TensorList": "at::TensorList",
    "Scalar": "at::Scalar",
    "Storage": "at::Storage",
    "IntList": "at::IntList",
    "IntArrayRef": "at::IntArrayRef",
    "OptionalIntArrayRef": "at::OptionalIntArrayRef",
    "ArrayRef": "at::ArrayRef",
    "Generator": "at::Generator",
    "Layout": "at::Layout",
    "ScalarType": "at::ScalarType",
    "TensorOptions": "at::TensorOptions",
    "SparseTensorRef": "at::SparseTensorRef",
    "Device": "c10::Device",
    "optional": "c10::optional",
    "MemoryFormat": "at::MemoryFormat",
    "QScheme": "at::QScheme",
    "ConstQuantizerPtr": "at::ConstQuantizerPtr",
    "Dimname": "at::Dimname",  # namedtensor-only
    "DimnameList": "at::DimnameList",  # namedtensor-only
    "ITensorListRef": "at::ITensorListRef",
    "OptionalSymIntArrayRef": "at::OptionalSymIntArrayRef",
    "SymInt": "c10::SymInt",
}


_AVAILABLE_FIELDS = {
    "acc_thread",
    "broadcast",
    "custom_fill_params",
    "custom_op_schema",
    "dtypes",
    "early_exit",
    "fallback_check",
    "guid",
    "handle_bool_inputs",
    "hpu_wrap_all_versions",
    "hpu_wrap_version_list",
    "hpu_wrap_version_range",
    "inplace_ids",
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


def torch_library_fragment(custom_schema_regs):
    if custom_schema_regs:
        return "TORCH_LIBRARY_FRAGMENT(hpu, m) {\n" "  static_cast<void>(m);\n" f"{custom_schema_regs}\n" "}"
    return ""


def torch_library_impl(torch_regs, ns="aten"):
    if torch_regs:
        return f"TORCH_LIBRARY_IMPL({ns}, HPU, m) {{\n{torch_regs}\n}}\n"
    return ""


_DEVICE_STR_TO_ENUM = {
    "All": "-1",
    "Gaudi": "synDeviceGaudi",
    "Gaudi2": "synDeviceGaudi2",
    "Gaudi3": "synDeviceGaudi3",
}


NUM_SHARDS = 10


def should_write_and_go_to_next_file(idx: int, num_idxs_per_shard: int, file_idx: int, total_idxs):
    return ((file_idx + 1) < NUM_SHARDS and (idx + 1) % num_idxs_per_shard == 0) or (idx + 1) == total_idxs


class OpValidatorGenerator(ABC):
    """
    Class describing how to generate code for given op_validator
    """

    def __init__(self, ctxop: "Op", execution_mode_for_shared_layer):
        self.execution_mode_for_shared_layer = execution_mode_for_shared_layer
        self._ctxop = ctxop

    @abstractmethod
    def can_generate(self):
        """
        Returns information if generator can generate code.

        Some generators depends on additional data in YAML file.
        This method can check if all data is present and generator
        is usable.
        """

    @abstractmethod
    def get_fallback_if_prefix(self):
        """
        Returns prefix name for FALLBACK_IF macro family. This is temporary solutions allowing
        to coexists old way and new way begin developed. The proof-of-concept being
        developed uses own family of macros having POC_ prefix in its name
        """

    @abstractmethod
    def get_validator_data_def(self, isoutfn, cpp_sig=""):
        """
        Returns dtypes line to be added to CPP file
        """

    @abstractmethod
    def get_validator_inline_data_def(self):
        """
        Returns dtypes line to be added to CPP file inside function
        """


def generate_dtype_macro(dtypes, check_implicit_types=True):
    def generate_line(dd_pairs, suffix=""):
        code = []
        for dd_pair in dd_pairs:
            dev_type, dtypes = dd_pair
            assert isinstance(dtypes, list)
            dtypes_set = set(dtypes)
            assert len(dtypes) == len(dtypes_set), "Found same dtype defined more than once!"

            if check_implicit_types:
                assert not any(x in dtypes_set for x in ["Double", "Bool"]), (
                    "Double and Bool are not natively supported, they are treated as "
                    "Float and Char respectively. For instance if Float is a supported "
                    "dtype, Double is added as a supported dtype by the script."
                )

            if "Float" in dtypes and "Double" not in dtypes:
                dtypes.append("Double")
            if "Char" in dtypes and "Bool" not in dtypes:
                dtypes.append("Bool")
            code.append(
                "{{{}, {{{}}}}}".format(
                    _DEVICE_STR_TO_ENUM[dev_type],
                    ", ".join(["at::k" + d for d in dtypes]),
                )
            )
        return "  HPU_SUPPORTED_DTYPES(({{{}}}){})\n".format(",\n   ".join(code), ", " + suffix if suffix else "")

    if isinstance(dtypes, list):
        return generate_line([("All", dtypes)])

    if isinstance(dtypes, dict) and not any(x in dtypes.keys() for x in _DEVICE_STR_TO_ENUM.keys()):
        lines = ""
        for k, v in dtypes.items():
            lines += generate_line([("All", v)], k)
        return lines

    assert isinstance(dtypes, dict)
    if all(isinstance(x, list) for x in dtypes.values()):
        return generate_line(dtypes.items())
    dt_map = defaultdict(list)
    for k, v in dtypes.items():
        for suffix, dt in v.items():
            dt_map[suffix].append((k, dt))
    lines = ""
    for suffix, arg in dt_map.items():
        lines += generate_line(arg, suffix)
    return lines


class UseDtypesOpValidatorGenerator(OpValidatorGenerator):
    def get_fallback_if_prefix(self):
        return ""

    def can_generate(self):
        dtypes = self._ctxop.get_dtypes()
        return dtypes is not None

    def get_validator_data_def(self, _isoutfn, cpp_sig=""):
        return ""

    def get_validator_inline_data_def(self):
        dtypes = copy.deepcopy(self._ctxop.get_dtypes())
        return generate_dtype_macro(dtypes, self._ctxop.get_lazy() == {})


def get_promotion_ids(ctxop, cpp_sig):
    type_promotion = ctxop.promote_to_common_type()
    promote_int_to_float = ctxop.promote_int_to_float()
    promotion_ids = []

    if type_promotion or promote_int_to_float:
        promotion_inputs = type_promotion if type_promotion else promote_int_to_float
        inputs = []
        # Below regex extracts op's arguments from cpp signature.
        m = re.search(r"\(([^)]*)", cpp_sig)
        if m:
            for input in m.group(1).split(", "):
                inputs.append(input.split(" ")[-1])
        promotion_ids = [inputs.index(x) for x in promotion_inputs if x in inputs]
    return sorted(promotion_ids), bool(promote_int_to_float)


def get_out_ids(ctxop, cpp_sig, isoutfn):
    out_ids = []
    if ctxop.get_out_ids():
        out_ids = ctxop.get_out_ids()
    elif ctxop.get_inplace_ids():
        out_ids = ctxop.get_inplace_ids()
    elif isoutfn:
        out_ids = [-1]
        m = re.match(r"::std::tuple<([^>]*)>", cpp_sig)
        if m:
            output_count = m.group(1).count(",") + 1
            out_ids = list(range(-output_count, 0))

    return out_ids


class CheckNodeWithSharedLayerValidatorGenerator(OpValidatorGenerator):
    def get_fallback_if_prefix(self):
        return "VAL_"

    def can_generate(self):
        has_early_exit = self._ctxop.op.get("early_exit", False)
        has_op_frontend = self._ctxop.op.get("op_frontend", False)
        has_op_backend = self._ctxop.op.get("op_backend", False)
        has_reduction = self._ctxop.op.get("reduction", False)

        is_compatible_with_shared_layer = not any([has_op_backend, has_op_frontend, has_early_exit, has_reduction])
        assert is_compatible_with_shared_layer, f"cannot use shared layer for {self._ctxop.opname}"

        return is_compatible_with_shared_layer

    def get_validator_inline_data_def(self):
        return ""

    def get_validator_data_def(self, isoutfn, cpp_sig=""):
        def is_inplace():
            if ctxop.get_inplace_ids() != []:
                return True
            return ctxop.opname.split(".")[0].endswith("_")

        ctxop = self._ctxop
        opname = ctxop.opname
        var_opname = ctxop.opname.replace(".", "_")

        if ctxop.get_guid() is None:
            raise Exception(
                f"Invalid specification for op {opname}: selected op_validator requires `guid` in specification"
            )

        out_ids = get_out_ids(ctxop, cpp_sig, isoutfn)
        promotion_ids, promote_to_float = get_promotion_ids(ctxop, cpp_sig)

        arg_opname = f'"{opname}"'
        arg_guid = f'"{ctxop.get_guid()}"'
        arg_out_ids = f"{{{', '.join([str(o) for o in out_ids])}}}"
        arg_scalar_ids = f"{{{', '.join([str(o) for o in ctxop.get_scalar_ids()])}}}"
        arg_output_meta = ctxop.get_output_meta()
        arg_type_promotion_ids = f"{{{', '.join([str(o) for o in promotion_ids])}}}"
        arg_promote_int_to_float = str(promote_to_float).lower()
        arg_safe_cast_check = str(ctxop.safe_cast_check()).lower()
        arg_isinplace = str(is_inplace()).lower()
        arg_isoutfn = str(isoutfn).lower()

        if arg_output_meta is None:
            arg_output_meta = "nullptr"

        # Rebase
        if arg_safe_cast_check == "none":
            arg_safe_cast_check = "false"

        constructor_args = [
            arg_opname,
            arg_guid,
            arg_out_ids,
            arg_scalar_ids,
            arg_output_meta,
            arg_type_promotion_ids,
            arg_promote_int_to_float,
            arg_safe_cast_check,
            arg_isinplace,
            arg_isoutfn,
        ]
        constructor_args = ", ".join(constructor_args)

        return f"static CheckNodeWithSharedLayerValidator validator_{var_opname}({constructor_args});\n"


class CheckNodeWithCustomSharedLayerValidatorGenerator(CheckNodeWithSharedLayerValidatorGenerator):
    def get_fallback_if_prefix(self):
        return "VAL_CUSTOM_"

    def can_generate(self):
        return True

    def get_validator_data_def(self, isoutfn, cpp_sig=""):
        ctxop = self._ctxop
        opname = ctxop.opname
        var_opname = ctxop.opname.replace(".", "_")

        arg_opname = f'"{opname}"'
        arg_shared_meta = ctxop.get_op_validator()

        arg_execution_mode = f"habana_helpers::{self.execution_mode_for_shared_layer}".replace(".", "::")

        constructor_args = [arg_opname, arg_shared_meta, arg_execution_mode]
        constructor_args = ", ".join(constructor_args)

        return f"static CheckNodeWithSharedLayerValidator validator_{var_opname}({constructor_args});\n"


allowed_lazy_keys = set()


def lazy_support(func):
    lazy_property = func.__name__[4:]

    allowed_lazy_keys.add(lazy_property)

    @wraps(func)
    def wrapper(self):
        # Null op initialized with empty dict
        # just to get default values for properties
        null_op = Op("null_op", {})

        # Calling __wrapped__ to avoid recursion
        default = getattr(null_op, func.__name__).__wrapped__(null_op)

        if self.mode == "lazy":
            return self.get_lazy().get(lazy_property, default)

        return func(self)

    return wrapper


class Op:
    def __init__(self, opname, op, mode=None):
        self.op = op
        self.opname = opname
        self.mode = mode

    def get_guid(self):
        return self.op.get("guid", None)

    def get_dtypes(self):
        if self.get_op_validator():
            return None
        return self.op.get("dtypes", None)

    def get_synapse_layouts(self):
        return self.op.get("synapse_layouts", [])

    def get_op_backend_class(self):
        op_backend_class = self.op.get("op_backend", None)
        if op_backend_class:
            return op_backend_class
        return "OpBackend"

    def get_op_frontend_class(self):
        op_frontend_class = self.op.get("op_frontend", None)
        if op_frontend_class:
            return op_frontend_class
        return "LazyOp"

    def get_early_exit_fun(self):
        early_exit_fun = self.op.get("early_exit", None)
        if early_exit_fun:
            return early_exit_fun
        return None

    def get_no_compute_flag(self):
        return self.op.get("no_compute_flag", False)

    def get_custom_fill_params(self):
        return self.op.get("custom_fill_params", None)

    def get_custom_output_shape(self):
        if self.op.get("broadcast", False):
            return "BinaryOutputShape"
        return None

    def get_output_meta(self):
        return self.op.get("output_meta", None)

    def get_st_meta(self):
        return self.op.get("st_meta", None)

    def get_inplace_ids(self):
        return self.op.get("inplace_ids", [])

    def get_out_ids(self):
        return self.op.get("out_ids", [])

    def get_scalar_ids(self):
        return self.op.get("scalar_ids", [])

    def get_tpc_input_order(self):
        return self.op.get("tpc_input_order", None)

    def promote_to_common_type(self):
        return self.op.get("promote_to_common_type", [])

    def promote_int_to_float(self):
        return self.op.get("promote_int_to_float", [])

    def promote_int_to_long(self):
        return self.op.get("promote_int_to_long", [])

    def safe_cast_check(self):
        return self.op.get("safe_cast_check", None)

    def handle_bool_inputs(self):
        return self.op.get("handle_bool_inputs", None)

    def custom_schema(self):
        schema_args = self.op.get("schema_args", None)
        if schema_args:
            return self.opname + schema_args
        return None

    def get_custom_op_schema(self):
        return self.op.get("custom_op_schema", None)

    def get_hpu_wrap_all_versions(self):
        return self.op.get("hpu_wrap_all_versions", False)

    def get_hpu_wrap_version_range(self):
        return self.op.get("hpu_wrap_version_range", False)

    def get_hpu_wrap_version_list(self):
        return self.op.get("hpu_wrap_version_list", False)

    def get_only_shared_layer(self):
        return self.op.get("only_shared_layer", False)

    def get_op_validator(self):
        return self.op.get("op_validator", None)

    def get_shared_layer_meta(self):
        if self.get_op_validator() in [None, "check-node-with-shared-layer"]:
            return None
        return self.get_op_validator()

    def get_fallback_check(self):
        return self.op.get("fallback_check", [])

    @lazy_support
    def get_override_fn(self):
        return self.op.get("override_fn", None)

    def get_lazy(self):
        lazy_desc = self.op.get("lazy", {})
        assert all(
            key in allowed_lazy_keys for key in lazy_desc.keys()
        ), f"Only {allowed_lazy_keys} are supported for lazy, but {lazy_desc.keys()} are provided for {self.opname}. In order to support another property, please add proper handling in Op class in {os.path.realpath(__file__)}"

        return lazy_desc

    def set_lazy(self):
        self.mode = "lazy"

    @lazy_support
    def get_acc_thread(self):
        return self.op.get("acc_thread", False)

    def get_op_validator_generator(self, execution_mode_for_shared_layer) -> OpValidatorGenerator:
        op_validator = self.get_op_validator()
        mapping = {
            None: UseDtypesOpValidatorGenerator,
            "check-node-with-shared-layer": CheckNodeWithSharedLayerValidatorGenerator,
        }
        result = mapping.get(op_validator, CheckNodeWithCustomSharedLayerValidatorGenerator)(
            self, execution_mode_for_shared_layer
        )
        if not result.can_generate():
            return None
        return result


class YamlContext:
    def __init__(self, yamlfile):
        with open(yamlfile, "r") as ff:
            self.op_data = yaml.load(ff.read(), Loader=Loader)

    def get_op_names(self):
        return self.op_data.keys()

    def get_op_data(self):
        return self.op_data.items()


def is_out_fn(fname):
    return fname.endswith("_out")


def generate_entry_debug_code(fname, params, is_eager_frontend):
    # Emits debug code for a given intercepted function.
    if not is_eager_frontend:
        code = "  PT_LAZY_OP_TRACE;\n"
        code += "  PT_LAZY_TRACE;\n"
    else:
        code = "  PT_EAGER_TRACE;\n"

    params_names = []
    for p in params:
        params_names.append(parser.param_name(p))
    params_count = len(params)
    dump_args = "DUMP_ARG" if params_count == 1 else f"DUMP_{params_count}ARGS"
    code += f'  PT_OP_INFO("{fname}: ", {dump_args}({", ".join(params_names)}));\n\n'
    code += f"  [[maybe_unused]] bool require_h2d = false;\n"
    code += f"  [[maybe_unused]] bool require_st = false;\n\n"
    return code


def fallback_if_unsupported(
    tinputs,
    opname,
    overload,
    param_vars,
    check_per_tensor,
    prefix,
    check_st_h2d_str,
    is_check_kernel_support=False,
    is_custom=False,
):
    if is_check_kernel_support:
        fallback_string = "RETURN"
    elif is_custom and prefix == "VAL_":
        fallback_string = "FAIL_CUSTOM"
    else:
        fallback_string = "FALLBACK"
    fallback_string += "_IF_UNSUPPORTED_DTYPE"
    per_tensor_string = "_PER_TENSOR" if check_per_tensor else ""
    overload_variant = "2" if overload else ""
    is_dynamic_string = ", is_dynamic" if is_check_kernel_support else ""
    overload_string = overload + ", " if overload else ""

    def single_fallback(tensor_opt=""):
        tensor_string = tensor_opt + ", " if tensor_opt else ""
        return "  {}{}{}{}({}{}{}, {}{}{})\n".format(
            prefix,
            fallback_string,
            per_tensor_string,
            overload_variant,
            tensor_string,
            opname,
            is_dynamic_string,
            overload_string,
            check_st_h2d_str,
            ", ".join(param_vars),
        )

    if prefix:
        return single_fallback()

    code = ""
    for t in tinputs:
        code += single_fallback(t)
    return code


def bitwise_ops_alt_guid(guid):
    logical_guid = guid.replace("bitwise_", "")
    return "if (ScalarType() == at::kBool) {{\n" '      SetGuid("{}_i8");\n' "    }}".format(logical_guid)


def extract_reduction_vars_indices(param_vars, use_int=False):
    default_id = "-1" if use_int else "c10::nullopt"
    reduction_vars_indices = [default_id] * 3

    for i, var in enumerate(param_vars):
        i = str(i)
        if var == "dim":
            reduction_vars_indices[0] = i
        elif var == "keepdim":
            reduction_vars_indices[1] = i
        elif var == "dtype":
            reduction_vars_indices[2] = i

    return reduction_vars_indices


def get_op_backend_class_impl(ctxop, fname, cname, num_out_tensors, param_vars):
    guid = ctxop.get_guid()
    out_ids = ctxop.get_out_ids()
    inplace_ids = ctxop.get_inplace_ids()
    scalar_ids = ctxop.get_scalar_ids()
    no_compute_flag = ctxop.get_no_compute_flag()
    custom_fill_params = ctxop.get_custom_fill_params()
    tpc_input_order = ctxop.get_tpc_input_order()
    op_backend_class = ctxop.get_op_backend_class()
    output_shape_fn = ctxop.get_custom_output_shape()
    output_meta_fn = ctxop.get_output_meta()
    shared_layer_meta_meta_fn = ctxop.get_shared_layer_meta()
    st_meta_fn = ctxop.get_st_meta()
    promote_to_common_type = ctxop.promote_to_common_type()
    promote_int_to_float = ctxop.promote_int_to_float()
    handle_bool_inputs = ctxop.handle_bool_inputs()

    assert (not out_ids) ^ (not inplace_ids) ^ is_out_fn(fname), (
        "`out_ids` or `inplace_ids` should not be defined for {}".format(fname)
        if is_out_fn(fname)
        else "Either `out_ids` or `inplace_ids` should be defined for {}".format(fname)
    )

    scalar_ids_set = set(scalar_ids)
    err_ids = scalar_ids_set.intersection(out_ids if len(out_ids) else inplace_ids)
    assert len(err_ids) == 0, "Input(s) at {} cannot be both scalar and tensor for {}.".format(err_ids, fname)

    out_ids = ", ".join([str(o) for o in out_ids])
    inplace_ids = ", ".join([str(i) for i in inplace_ids])
    scalar_ids = ", ".join([str(s) for s in scalar_ids])

    if fname.startswith("bitwise_"):
        custom_handler = templates._CUSTOM_HANDLER.format(body=bitwise_ops_alt_guid(guid))
    else:
        custom_handler = ""

    ctor_extra_calls = []
    if no_compute_flag:
        ctor_extra_calls.append("setNoComputeFlag();")

    synapse_layouts = ctxop.get_synapse_layouts()
    if len(synapse_layouts):
        assert len(synapse_layouts) == 2, "Define both input and output layouts."
        assert len(synapse_layouts[0]), "Input layouts size should be atleast 1."
        assert len(synapse_layouts[1]), "Output layouts size should be atleast 1."
        in_layouts = ", ".join(["synapse_helpers::layouts::SynapseLayoutFormat::" + l for l in synapse_layouts[0]])
        out_layouts = ", ".join(["synapse_helpers::layouts::SynapseLayoutFormat::" + l for l in synapse_layouts[1]])
        ctor_extra_calls.append("SetSynapseLayouts({{{}}}, {{{}}});".format(in_layouts, out_layouts))

    if is_out_fn(fname) and num_out_tensors > 1:
        ctor_extra_calls.append("SetNumOutTensors({});".format(num_out_tensors))

    if output_meta_fn:
        ctor_extra_calls.append("SetOutputMetaFn({});".format(output_meta_fn))
    elif promote_to_common_type or promote_int_to_float:
        if promote_int_to_float:
            type_promo_variant = "PromoteIntToFloat"
            dtype_helper_inputs = promote_int_to_float
        else:
            type_promo_variant = "PromoteToCommon"
            dtype_helper_inputs = promote_to_common_type
        input_indices = []
        for input in dtype_helper_inputs:
            input_indices.append(param_vars.index(input))
        ctor_extra_calls.append(
            f"SetOutputMetaFn("
            f"PointwiseMeta<static_cast<int>(DTypeHelper::DtypePromoteVariant::k{type_promo_variant}), "
            f"{str(ctxop.op.get('broadcast', False)).lower()}, "
            f"{', '.join(map(str, input_indices))}>);"
        )
    elif output_shape_fn:
        ctor_extra_calls.append("SetComputeOutputShapes({});".format(output_shape_fn))

    if st_meta_fn:
        ctor_extra_calls.append("SetSTMetaFn({});".format(st_meta_fn))

    if custom_fill_params:
        ctor_extra_calls.append("SetFillParams({});".format(custom_fill_params))

    if tpc_input_order:
        tpc_input_order_str = ", ".join(map(str, tpc_input_order))
        ctor_extra_calls.append("SetTpcInputOrder({{{}}});".format(tpc_input_order_str))

    if promote_to_common_type:
        ctor_extra_calls.append("EnableTypePromotion();")
    elif promote_int_to_float:
        ctor_extra_calls.append("PromoteIntToFloat();")

    if handle_bool_inputs:
        ctor_extra_calls.append("HandleBoolInputs();")

    if shared_layer_meta_meta_fn:
        ctor_extra_calls.append("SetSharedLayerMetaFn({});".format(shared_layer_meta_meta_fn))

    return templates._OPCLASS_HEADER.format(
        op_backend_class=op_backend_class,
        cname=cname,
        guid=guid,
        out_ids=out_ids,
        inplace_ids=inplace_ids,
        scalar_ids=scalar_ids,
        is_out_fn=str(is_out_fn(fname)).lower(),
        ctor_extra_calls="".join(["\n" + " " * 8 + c for c in ctor_extra_calls]),
        custom_handler=custom_handler,
    )


def generate_impl(op_variant, overload, override_fn):
    return '  m.impl("{}", static_cast<{}>(&{}));\n'.format(op_variant, overload, override_fn)


def generate_dtype_defs(fgen, execution_mode_for_shared_layer):
    op_validator_generator = fgen.ctxop.get_op_validator_generator(execution_mode_for_shared_layer)
    if op_validator_generator is not None:
        return op_validator_generator.get_validator_data_def(is_out_fn(fgen.func), fgen.cppsig)
    return ""


def generate_frontend_functions(fgen, mode):
    # torch registrations
    override_fn = "habana::{}".format(fgen.func)
    pos = fgen.funsig.find("(")
    overload = fgen.funsig[:pos] + " (*)" + fgen.funsig[pos:]
    impl = generate_impl(get_aten_opname(fgen.aten_sig), overload, override_fn)
    assert fgen.mapsig not in _FN_AUTOGRAD_HPU
    torch_regs = impl

    dtype_defs = generate_dtype_defs(fgen, get_execution_mode_from_string(mode))

    op_frontend_functions = fgen.op_frontend_lazy if mode == "lazy" else fgen.op_frontend_eager
    op_frontend_functions += "\n\n"

    return (
        dtype_defs,
        op_frontend_functions,
        torch_regs,
    )


def generate_backend_functions(fgen, is_custom=False):
    custom_schema_regs = ""

    op_backend = f"{fgen.op_backend}\n"
    op = fgen.aten_sig.split("(")[0].split("::")[1]

    schema_template = '.REGISTER_HPU_BACKEND("{ns}::{op}", {func})\n'

    if is_custom:
        kr_regs = schema_template.format(ns="hpu", op=op, func=fgen.cname)
        custom_schema_regs += f'  m.def("{fgen.aten_sig}");\n'
    else:
        kr_regs = schema_template.format(ns=fgen.ns, op=op, func=fgen.cname)
        if fgen.ctxop.custom_schema():
            kr_regs += schema_template.format(ns="hpu", op=op, func=fgen.cname)
            custom_schema_regs = f'  m.def("{fgen.ctxop.custom_schema()}");\n'

    return (
        op_backend,
        kr_regs,
        custom_schema_regs,
    )


def is_custom_class(class_name, is_backend):
    default_class = "OpBackend" if is_backend else "LazyOp"
    return not (class_name == default_class or class_name.endswith("Template"))


def generate_op_hclasses(fgens, classes, header_file, is_backend, base_class=""):
    code = ""
    macro = "BACKEND" if is_backend else "FRONTEND"
    for fgen in fgens:
        fclass = fgen.ctxop.get_op_backend_class() if is_backend else fgen.ctxop.get_op_frontend_class()
        if (not is_custom_class(fclass, is_backend)) or fclass in classes:
            continue
        macro_params = fclass
        if not is_backend:
            macro_params = base_class + ", " + macro_params
        code += f"HPU_OP_{macro}({macro_params})\n"
        classes[fclass] = header_file
    return code


def generate_op_backend_hclasses(fgens, classes, header_file):
    return generate_op_hclasses(fgens, classes, header_file, True)


def generate_op_frontend_hclasses(fgens, classes, header_file, base_class):
    return generate_op_hclasses(fgens, classes, header_file, False, base_class)


def generate_header_decls(fgens):
    fill_params = set()
    early_exit_fns = set()
    outshape_fns = set()
    outmeta_fns = set()
    shared_layer_meta_fns = set()
    stmeta_fns = set()
    fc_fns = set()

    def build(fn, fns_set, macro, args=[]):
        if fn and fn not in fns_set:
            fns_set.add(fn)
            return "{}({});\n".format(macro, ", ".join([fn] + args))
        return ""

    reg_decls = ""
    early_exit_decls = ""
    outshape_decls = ""
    outmeta_decls = ""
    shared_layer_meta_decls = ""
    stmeta_decls = ""
    fill_params_decls = ""
    fallback_check_decls = ""
    for fgen in fgens:
        reg_decls += "{};\n".format(fgen.rwsig)

        early_exit_fun = fgen.ctxop.get_early_exit_fun()
        if early_exit_fun is not None and early_exit_fun not in early_exit_fns:
            pattern = fgen.func + "("
            func_pos = fgen.rwsig.find(pattern)
            if func_pos >= 0:
                rtype = fgen.rwsig[:func_pos]
                args = fgen.rwsig[func_pos + len(pattern) :]
                early_exit_decls += f"unsigned {early_exit_fun}Condition({args};\n"
                early_exit_decls += f"{rtype}{early_exit_fun}(unsigned eePath, {args};\n"
                early_exit_fns.add(early_exit_fun)
        outshape_decls += build(fgen.ctxop.get_custom_output_shape(), outshape_fns, "OUTSHAPE_DECL")

        outmeta_decls += build(fgen.ctxop.get_output_meta(), outmeta_fns, "OUTMETA_DECL")
        shared_layer_meta_decls += build(
            fgen.ctxop.get_shared_layer_meta(), shared_layer_meta_fns, "SHARED_LAYER_META_DECL"
        )
        stmeta_decls += build(fgen.ctxop.get_st_meta(), stmeta_fns, "STMETA_DECL")
        fill_params_decls += build(fgen.ctxop.get_custom_fill_params(), fill_params, "FILL_PARAMS_DECL")

        fc = fgen.ctxop.get_fallback_check()
        if fc:
            fallback_check_decls += build(fc[0], fc_fns, "FALLBACK_CHECK", fgen.fc_params)

    return (
        reg_decls
        + early_exit_decls
        + outshape_decls
        + outmeta_decls
        + shared_layer_meta_decls
        + stmeta_decls
        + fill_params_decls
        + fallback_check_decls
    )


def gen_output_file(args, name):
    if not args.output_dir:
        return sys.stdout
    filename = os.path.join(args.output_dir, name)
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))
    return open(filename, "w")


def gen_h_output_file(args, opgroup):
    return gen_output_file(args, "{}.h".format(opgroup))


def gen_cpp_output_file(args, opgroup):
    return gen_output_file(args, "{}.cpp".format(opgroup))


# Generate file with all potential ops for autocast. The actual ops registered
# for autocast are based on the default lists in autocast_helpers.h file or
# on the external file provided via env.
def generate_autocast_ops(op_metas, args):
    op_registration = '  Hpu_KERNEL({function_name}, "{op_name}", {signature})'
    replacements = (
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
    ) + (
        (
            (
                "::std::tuple<::std::vector<at::Tensor>,::std::vector<at::Tensor>,::std::vector<at::Tensor>,::std::vector<at::Tensor>>",
                "tuple_4_vectors",
            ),
        )
        if Version(torch.__version__) > Version("2.3")
        else ()
    )

    blocklist = [
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
    ] + (["_fused_adam", "_fused_adamw"] if Version(torch.__version__) < Version("2.1") else [])

    def op_to_skip(function_name, op_name):
        return (
            function_name.endswith(("_out", "_"))
            or any(s in function_name for s in ("_.", "cuda", "cudn", "backward"))
            or op_name in blocklist
        )

    def get_registration(op_meta, op_name):
        function_name = op_meta.func
        signature = op_meta.funsig
        for r in replacements:
            signature = signature.replace(*r)
        return op_registration.format(function_name=function_name, op_name=op_name, signature=signature)

    # only ops defined in 'at' namespace are applicable for autocast
    ops_not_in_at = set()
    with open(args.native_functions, "r") as f:
        native_functions = yaml.load(f.read(), Loader=Loader)
    for function in native_functions:
        if "variants" in function and "function" not in function["variants"]:
            ops_not_in_at.add(re.search(r"(.*?)\(", function["func"]).group(1))

    ops_registrations = []
    for op_meta in op_metas:
        op_name = op_meta.op_variant
        if op_to_skip(op_meta.func, op_name) or op_name in ops_not_in_at:
            continue

        ops_registrations.append(get_registration(op_meta, op_name))

    number_of_chunks = 10
    chunk_size = len(ops_registrations) // number_of_chunks + 1

    for i in range(number_of_chunks):
        print(
            templates._AUTOCAST_TEMPLATE.format(
                gen=os.path.basename(sys.argv[0]),
                fallback_fallthrough=templates._AUTOCAST_FALLBACK if i == 0 else "",
                ops_registrations=("\n").join(ops_registrations[chunk_size * i : chunk_size * (i + 1)]),
            ),
            file=gen_cpp_output_file(args, f"backend/hpu_autocast_ops{i}"),
        )


class TensorFetcher:
    def __init__(self):
        self.tensors = []

    def add(self, name):
        self.tensors.append(name)

    def get_tensors(self):
        return self.tensors


inplace_params_blacklist = [
    "_native_batch_norm_legit",
]


def parse_params(params, fname, rtype, fc, funsig, out_ids):
    param_vars = []
    call_args = []
    out_indices = []
    fc_params = []
    tfetcher = TensorFetcher()
    param_types = re.findall(r"([^(,)]+)(?!.*\()", funsig)

    for i, p in enumerate(params):
        ptype = parser.param_type(p)
        cptype = parser.type_core(ptype)
        pname = parser.param_name(p)

        param_vars.append(pname)

        if cptype == "Tensor":
            if parser.type_is_const(ptype):
                tfetcher.add(pname)
            else:
                tfetcher.add(pname)
                if fname not in inplace_params_blacklist:
                    call_args.append(pname)
                    out_indices.append(i)

        if rtype == "void":
            if cptype in ["TensorList", "Tensor"]:
                call_args.append(pname)
                if out_ids is not None:
                    out_indices.append(i)

        elif rtype == "const at::Tensor &" and cptype == "Tensor":
            call_args.append(pname)
            out_indices.append(i)

        if fc and pname in fc[1:]:
            fc_params.append("{} {}".format(param_types[i].strip(), pname))

    assert len(fc) == 0 or len(fc_params) + 1 == len(
        fc
    ), "Cannot find all params specified for fallback check {}.".format(fc[0])

    if rtype == "void" and out_ids is not None:
        out_indices = out_ids
        call_args = [call_args[i] for i in out_indices]
    return param_vars, call_args, out_indices, fc_params, tfetcher


def is_inplace_or_out_op(opname):
    if opname.endswith("_out"):
        return True
    if opname.startswith("__"):  # shift specific ops
        return opname.startswith("__i")  # inplace shift ops start with 'i' in name
    return opname.endswith("_")


def handle_type_promotion(ctxop, fname, fe_call_args, param_vars):
    promote_to_common_type = ctxop.promote_to_common_type()
    promote_int_to_float = ctxop.promote_int_to_float()
    safe_cast_check = ctxop.safe_cast_check()

    promote_types = bool(promote_to_common_type) + bool(promote_int_to_float)
    assert promote_types < 2, "Only one of [promote_to_common_type, promote_int_to_float] may be set to True."
    promote_types = bool(promote_types)
    use_compute_type = promote_types
    dtype_helper_inputs = []
    type_promo_variant = "None"
    code = ""

    if use_compute_type:
        if promote_types:
            if promote_int_to_float:
                type_promo_variant = "PromoteIntToFloat"
                dtype_helper_inputs = promote_int_to_float
            else:
                type_promo_variant = "PromoteToCommon"
                dtype_helper_inputs = promote_to_common_type
        else:
            dtype_helper_inputs = ["self"]
            type_promo_variant = "Reduction"

        safe_cast = is_inplace_or_out_op(fname)
        if safe_cast_check is not None:
            assert safe_cast, f"safe_cast_check cannot check for non inplace/non out variant, op={fname}"
            assert safe_cast_check is False, f"safe_cast_check is true by default for inplace/out variant, op={fname}"
            safe_cast = safe_cast_check

        code = (
            f"  auto compute_type = "
            f"DTypeHelper::get_compute_dtype({{{', '.join(dtype_helper_inputs)}}}, "
            f'{fe_call_args if fe_call_args else "c10::nullopt"}, '
            f"DTypeHelper::DtypePromoteVariant::k{type_promo_variant}, "
            f"{str(safe_cast).lower()}/*safe_cast*/"
            f'{", dtype" if "dtype" in param_vars else ""});\n'
            f"  static_cast<void>(compute_type);\n\n"
        )

    return code, promote_types, dtype_helper_inputs, type_promo_variant, use_compute_type


def handle_validator_generator(
    ctxop,
    execution_mode_for_shared_layer,
    use_compute_type,
    overload,
    opname,
    param_vars,
    tfetcher,
    fname,
    is_check_kernel_support=False,
    check_st_h2d=False,
):
    dtypes = ctxop.get_dtypes()
    op_validator_generator = ctxop.get_op_validator_generator(execution_mode_for_shared_layer)
    if not op_validator_generator:
        return ""

    fallback_if_prefix = op_validator_generator.get_fallback_if_prefix()
    code = op_validator_generator.get_validator_inline_data_def()

    # Check with compute_type when using compute_type
    fallback_string = f"{'RETURN' if is_check_kernel_support else 'FALLBACK'}_IF_UNSUPPORTED_DTYPE"

    # check shape tensor and h2d tensor string
    check_st_h2d_str = ""
    if fallback_if_prefix and not is_check_kernel_support:
        if check_st_h2d:
            check_st_h2d_str = "true, "
        else:
            check_st_h2d_str = "false, "
    if use_compute_type:
        code += "  {}{}{}({}{}{}, {}{}{})\n".format(
            fallback_if_prefix,
            fallback_string,
            "2" if overload else "",
            "" if fallback_if_prefix else "compute_type, ",
            opname,
            ", is_dynamic" if is_check_kernel_support else "",
            overload + ", " if overload else "",
            check_st_h2d_str,
            ", ".join(param_vars),
        )
    else:
        tinputs = tfetcher.get_tensors()
        if is_out_fn(fname) and len(tinputs) > 1:
            tinputs = tinputs[:-1]

        check_per_tensor = False
        if isinstance(dtypes, dict):
            if any(p in dtypes.keys() for p in param_vars):
                check_per_tensor = True
            elif any(isinstance(x, dict) for x in dtypes.values()):
                check_per_tensor = True
        code += fallback_if_unsupported(
            tinputs,
            opname,
            overload,
            param_vars,
            check_per_tensor,
            fallback_if_prefix,
            check_st_h2d_str,
            is_check_kernel_support,
            ctxop.get_custom_op_schema() is not None,
        )
    code += "\n"
    return code


def handle_fallback_check(ctxop, overload, opname, param_vars):
    fallback_check = ctxop.get_fallback_check()
    if not fallback_check:
        return ""

    fallback_check_fname = fallback_check[0]
    args = fallback_check[1:]
    return "  FALLBACK_IF_UNSUPPORTED_INPUTS{}({}({}), {}, {}{})\n".format(
        "2" if overload else "",
        fallback_check_fname,
        ", ".join(args),
        opname,
        overload + ", " if overload else "",
        ", ".join(param_vars),
    )


def handle_override_fn(ctxop, param_vars, is_eager):
    override_fn = ctxop.get_override_fn()
    if not override_fn:
        return ""

    ns = "habana::eager" if is_eager else "habana_lazy"
    code_line = f"  return {ns}::{override_fn}({', '.join(param_vars)});"
    return code_line + "\n}"


def handle_output_shape_fn(ctxop, param_vars):
    if ctxop.get_output_meta():
        return "};\n"
    code = ""
    output_shape_fn = ctxop.get_custom_output_shape()
    if output_shape_fn:
        code += f", {output_shape_fn}"
    return code + "};\n"


def handle_output_meta(ctxop, promote_types, dtype_helper_inputs, param_vars, type_promo_variant):
    code = ""
    output_meta = ctxop.get_output_meta()
    if output_meta:
        code = f"  hpu_op.SetOutputMetaFn({output_meta});\n"
    elif promote_types:
        input_indices = []
        for input in dtype_helper_inputs:
            input_indices.append(param_vars.index(input))

        code = (
            f"  hpu_op.SetOutputMetaFn("
            f"PointwiseMeta<static_cast<int>(DTypeHelper::DtypePromoteVariant::k{type_promo_variant}), "
            f"{str(ctxop.op.get('broadcast', False)).lower()}, "
            f"{', '.join(map(str, input_indices))}>);\n"
        )
    return code


def is_acc_thread_supported(ctxop, rtype, sig):
    if ctxop.get_override_fn():
        return ctxop.get_acc_thread()  # only custom lazy func ops that are supported
    return (
        rtype.startswith("at::Tensor")  # regular, in-place, _out ops
        or rtype.startswith("const at::Tensor")  # only resize_ op so far, handled as inplace/out (shape change)
        or rtype.startswith("::std::tuple<at::Tensor")  # tuple ops
        or "TensorList" in sig  # TensorList ops
    )


def handle_return_lazy(ctxop, rtype, sig, fname, fe_call_args, param_vars):
    if not is_acc_thread_supported(ctxop, rtype, sig):
        return "  {}hpu_op.call({});".format("" if rtype == "void" else "return ", fe_call_args)

    code = ""
    if is_inplace_or_out_op(fname):
        if rtype.startswith("::std::tuple<at::Tensor"):
            code += "  auto tuple = {};\n".format(fe_call_args)
            code += "  RUN_INPLACE_TUPLE_MAYBE_WITH_ACC_THREAD({}, hpu_op, tuple)".format(fname)
        elif rtype == "void" and "TensorList" in sig:
            code += "  RUN_TENSOR_LIST_INPLACE_MAYBE_WITH_ACC_THREAD({}, hpu_op, {})".format(fname, param_vars[0])
        elif rtype.startswith("const at::Tensor"):
            code += "  RUN_CONST_INPLACE_MAYBE_WITH_ACC_THREAD({}, hpu_op, {})".format(fname, fe_call_args)
        else:
            code += "  RUN_INPLACE_MAYBE_WITH_ACC_THREAD({}, hpu_op, {})".format(fname, fe_call_args)
    else:
        if rtype.startswith("::std::tuple<at::Tensor"):
            code += "  RUN_TUPLE_MAYBE_WITH_ACC_THREAD({}, hpu_op)".format(fname)
        elif rtype.startswith("void") and "TensorList" in sig:
            if sig.count("TensorList") == 1:
                code += "  RUN_TENSOR_LIST_MAYBE_WITH_ACC_THREAD({}, hpu_op, {})".format(fname, param_vars[0])
            elif sig.count("TensorList") == 2:
                code += "  RUN_TENSOR_LIST2_MAYBE_WITH_ACC_THREAD({}, hpu_op, {})".format(
                    fname, f"{param_vars[0]}, {param_vars[1]}"
                )
                pass
            else:
                raise Exception(f"Only up to 2 TensorList inputs are supported. Sig: {sig}")
        else:
            code += "  RUN_MAYBE_WITH_ACC_THREAD({}, hpu_op)".format(fname)
    return code + ";"


def lazy_frontend(
    ctxop,
    tfetcher,
    fname,
    aten_sig,
    rtype,
    param_vars,
    call_args,
    sig,
    params,
    is_check_kernel_support,
    ns,
):
    if ctxop.custom_schema():
        ns = "hpu"
    aten_opname = get_aten_opname(aten_sig)
    opname = aten_opname.split(".")[0]
    overload = aten_opname.split(".")[1] if len(aten_opname.split(".")) > 1 else None
    schema_fn = f"{ns}::{opname}"
    code = "{} {{\n".format(sig)
    if not is_check_kernel_support:
        code += generate_entry_debug_code(fname, params, False)
    if not (is_check_kernel_support or is_acc_thread_supported(ctxop, rtype, sig)):
        code += "  habana_lazy::NoAccThread no_acc_thread;\n"

    fe_call_args = ""
    if call_args:
        fe_call_args = ", ".join(call_args)
        if "std::tuple" in rtype:
            fe_call_args = f"{rtype}({fe_call_args})"

    # TODO safe cast check should be done for out variants without type promotion too
    # https://jira.habana-labs.com/browse/SW-111202
    promotion_code, promote_types, dtype_helper_inputs, type_promo_variant, use_compute_type = handle_type_promotion(
        ctxop, fname, fe_call_args, param_vars
    )

    code += promotion_code
    code += handle_validator_generator(
        ctxop,
        HabanaExecutionMode.LAZY,
        use_compute_type,
        overload,
        opname,
        param_vars,
        tfetcher,
        fname,
        is_check_kernel_support,
    )

    if is_check_kernel_support:
        return code + "  return true;\n}"

    code += handle_fallback_check(ctxop, overload, opname, param_vars)
    override_code = handle_override_fn(ctxop, param_vars, False)
    if override_code:
        return code + override_code

    early_exit_fun = ctxop.get_early_exit_fun()

    if early_exit_fun is not None:
        code += f"  if (auto eePath = {early_exit_fun}Condition({', '.join(param_vars)}))\n"
        code += f"    return {early_exit_fun}(eePath, {', '.join(param_vars)});\n\n"

    op_frontend_class = ctxop.get_op_frontend_class()

    code += '  {}<{}> hpu_op{{"{}", {{{}}}'.format(op_frontend_class, rtype, schema_fn, ", ".join(param_vars))
    code += handle_output_shape_fn(ctxop, param_vars)

    if use_compute_type:
        code += "  hpu_op.set_scalar_types({compute_type});\n"

    code += handle_output_meta(ctxop, promote_types, dtype_helper_inputs, param_vars, type_promo_variant)

    st_meta = ctxop.get_st_meta()
    if st_meta:
        code += f"  hpu_op.SetSTMetaFn({st_meta});\n"

    code += handle_return_lazy(ctxop, rtype, sig, fname, fe_call_args, param_vars)
    return code + "\n}"


def handle_eager_not_supported(param_vars, overload, opname):
    # for not supported eager ops in Eager compilation, unconditionally
    # fallback to CPU
    args_str = ", ".join(param_vars)
    if overload:
        code = f"  FALLBACK_UNSUPPORTED_OP2_O({opname}, PARAMS2({args_str}), {overload})\n"
    else:
        code = f"  FALLBACK_UNSUPPORTED_OP2({opname}, PARAMS2({args_str}))\n"
    code += "  // ANYTHING BELOW IS JUST FOR REFERENCE WHEN MIGRATING TO EAGER OP\n\n"
    return code + "  // MOVE TO EAGER: "


def handle_return_eager(rtype, fname, fe_call_args, is_eager_op_supported, call_args, inplace_op_info):
    eager_op_info_args = (
        len(call_args)
        if is_out_fn(fname)
        else f"decltype(eager::EagerOpMetaData::out_indices_){{{', '.join(map(str, inplace_op_info[2]))}}}"
    )
    code = (
        f"  hpu_op.set_eager_op_info({{"
        f"{inplace_op_info[0]}, "
        f'"{inplace_op_info[1]}", '
        f"require_h2d, "
        f"require_st, "
        f"{eager_op_info_args}}});\n"
    )
    code += "  {}hpu_op.call({})".format("" if rtype == "void" else "return ", fe_call_args)
    if not is_eager_op_supported:
        code += "  */\n"
    return code


# List of override_fn ops that are supporting eager frontend
eager_ops_override_fns_whitelist = [
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
]

eager_custom_frontends_whitelist = [
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
]


# helper function to determine if op supports eager::EagerOp
def is_eager_op(ctxop):
    if ctxop.get_op_frontend_class() in eager_custom_frontends_whitelist:
        return True
    if ctxop.get_op_frontend_class() != "LazyOp":
        return False

    if ctxop.get_override_fn():
        return ctxop.get_override_fn() in eager_ops_override_fns_whitelist

    return True


def get_eager_op_info(opname, ns):
    type = "eager::eagerOpKind::"
    if opname.endswith("_out"):
        type += "InplaceOut"
    elif opname.endswith("_grad_input"):
        type += "InplaceOut"
    elif opname.endswith("_"):
        if opname.endswith("resize_"):
            type += "OutOfPlace"
        else:
            type += "Inplace"
    else:
        type += "OutOfPlace"

    name = opname
    # replace to coresponding OutOfPlace variant.
    # generally, the OutOfPlace variant is Inplace Op without suffix "_",
    # but some ops not follow this rule.
    if type != "eager::eagerOpKind::OutOfPlace":
        if name == "__ilshift__":
            name = "__lshift__"
        elif name == "__irshift__":
            name = "__rshift__"
        elif name in ["__lshift__", "__rshift__"]:
            pass
        else:
            name = opname.rsplit("_", 1)[0]
    op_name = f"{ns}::{name}"

    return type, op_name


def eager_frontend(
    ctxop,
    tfetcher,
    fname,
    aten_sig,
    rtype,
    param_vars,
    call_args,
    sig,
    params,
    out_indices,
    ns,
):
    aten_opname = get_aten_opname(aten_sig)
    opname = aten_opname.split(".")[0]
    overload = aten_opname.split(".")[1] if len(aten_opname.split(".")) > 1 else None
    schema_fn = f"{ns}::{opname}"
    code = "{} {{\n".format(sig)
    code += generate_entry_debug_code(fname, params, True)

    fe_call_args = ""
    if call_args:
        fe_call_args = ", ".join(call_args)
        if "std::tuple" in rtype:
            fe_call_args = f"{rtype}({fe_call_args})"

    # TODO safe cast check should be done for out variants without type promotion too
    # https://jira.habana-labs.com/browse/SW-111202
    promotion_code, promote_types, dtype_helper_inputs, type_promo_variant, use_compute_type = handle_type_promotion(
        ctxop, fname, fe_call_args, param_vars
    )

    code += promotion_code
    code += handle_validator_generator(
        ctxop,
        HabanaExecutionMode.EAGER,
        use_compute_type,
        overload,
        opname,
        param_vars,
        tfetcher,
        fname,
        False,
        True,
    )
    code += handle_fallback_check(ctxop, overload, opname, param_vars)

    is_eager_op_supported = is_eager_op(ctxop)
    if not is_eager_op_supported:
        code += handle_eager_not_supported(param_vars, overload, opname)

    override_code = handle_override_fn(ctxop, param_vars, True)
    if override_code:
        return code + override_code

    if not is_eager_op_supported:
        code += "  /* MOVE TO EAGER: \n"

    early_exit_fun = ctxop.get_early_exit_fun()

    if early_exit_fun is not None:
        code += f"  if (auto eePath = {early_exit_fun}Condition({', '.join(param_vars)}))\n"
        code += f"    return {early_exit_fun}(eePath, {', '.join(param_vars)});\n\n"

    op_frontend_class = "eager::EagerOp" if ctxop.get_op_frontend_class() == "LazyOp" else ctxop.get_op_frontend_class()

    code += '  {}<{}> hpu_op{{"{}", {{{}}}'.format(op_frontend_class, rtype, schema_fn, ", ".join(param_vars))
    code += handle_output_shape_fn(ctxop, param_vars)

    if use_compute_type:
        code += "  hpu_op.set_scalar_types({compute_type});\n"

    code += handle_output_meta(ctxop, promote_types, dtype_helper_inputs, param_vars, type_promo_variant)

    op_type, op_name = get_eager_op_info(fname, ns)
    inplace_op_info = (op_type, op_name, out_indices)

    code += handle_return_eager(rtype, fname, fe_call_args, is_eager_op_supported, call_args, inplace_op_info)
    return code + ";\n}"


def prepare_sig_for_kernel_support(sig):
    pos = sig.find("(")
    pos2 = sig.find(")")
    sig = "bool impl" + sig[pos:pos2] + ", bool is_dynamic" + sig[pos2:]
    sig = sig.replace("at::OptionalSymIntArrayRef", "at::OptionalIntArrayRef")
    sig = sig.replace("c10::SymIntArrayRef", "at::IntArrayRef")
    sig = sig.replace("c10::SymInt", "int64_t")
    return sig


def get_op_group(opname):
    opgroup = opname.split(".")[0]
    is_dunder = opgroup.startswith("__") and opgroup.endswith("__")
    if not is_dunder and opgroup.endswith("_"):
        opgroup = opgroup[:-1]
    return opgroup


# List of ops that shouldn't be generated in lazy mode
lazy_frontend_blacklist = [
    # convolution and convolution_backward are handled in lazy mode with
    # convolution_overrideable and convolution_backward_overrideable
    "convolution",
    "convolution_backward",
]

# List of ops that shouldn't be generated in eager mode
eager_frontend_blacklist = []


def generate_op(fndef, op_name, ctxop, op_params, is_check_kernel_support=False, ns="aten"):
    dtdf = fndef.dtdf
    tree = parser.parse(fndef.cpp_sig)
    xtree = parser.xparse(fndef.cpp_sig)
    mapsig = parser.create_map_sig(xtree, fndef.cpp_sig)
    rwsig = parser.rewrite_signature(fndef.cpp_sig, _TYPE_NSMAP)
    rwxtree = parser.xparse(rwsig)
    params = parser.get_parameters(tree)
    aten_sig = fndef.aten_sig
    funsig = parser.create_stdfunc_sig(rwxtree, rwsig)
    opgroup = get_op_group(op_name)
    rtype = parser.get_return_type_str(rwxtree, rwsig)

    sig, fname, xfname = parser.get_function_signature(rwxtree, rwsig, lambda x: "{}".format(x))

    if is_check_kernel_support:
        sig = prepare_sig_for_kernel_support(sig)

    out_indices = (
        op_params["inplace_ids"] if isinstance(op_params, dict) and "inplace_ids" in op_params.keys() else None
    )
    param_vars, call_args, out_indices, fc_params, tfetcher = parse_params(
        params, fname, rtype, ctxop.get_fallback_check(), funsig, out_indices
    )

    op_backend = None
    op_backend_class = None
    if ctxop.get_override_fn():
        assert (
            ctxop.get_op_frontend_class() == "LazyOp" and ctxop.get_op_backend_class() == "OpBackend"
        ), f"{op_name} has defined override_fn, it cannot take op_frontend or op_backend"
    elif not ctxop.get_only_shared_layer():
        op_backend_class = "Gen{}".format(op_name.replace(".", "_"))
        op_backend = get_op_backend_class_impl(ctxop, fname, op_backend_class, len(call_args), param_vars)

    op_frontend_eager = None
    if dtdf and opgroup not in eager_frontend_blacklist:
        op_frontend_eager = eager_frontend(
            ctxop,
            tfetcher,
            fname,
            aten_sig,
            rtype,
            param_vars,
            call_args,
            sig,
            params,
            out_indices,
            ns,
        )

    if ctxop.get_lazy():
        ctxop.set_lazy()

    op_frontend_lazy = None
    if is_check_kernel_support or opgroup not in lazy_frontend_blacklist:
        op_frontend_lazy = lazy_frontend(
            ctxop,
            tfetcher,
            fname,
            aten_sig,
            rtype,
            param_vars,
            call_args,
            sig,
            params,
            is_check_kernel_support,
            ns,
        )

    return OpGen(
        tree=tree,
        xtree=xtree,
        rwxtree=rwxtree,
        func=fname,
        xfunc=xfname,
        op_frontend_eager=op_frontend_eager,
        op_frontend_lazy=op_frontend_lazy,
        op_backend=op_backend,
        cname=op_backend_class,
        sig=fndef.cpp_sig,
        rwsig=rwsig,
        cppsig=sig,
        mapsig=mapsig,
        funsig=funsig,
        aten_sig=aten_sig,
        dtdf=dtdf,
        ctxop=ctxop,
        opgroup=opgroup,
        fc_params=fc_params,
        op_variant=op_name,
        ns=ns,
    )


def generate_op_meta(cpp_sig, op_name):
    xtree = parser.xparse(cpp_sig)
    mapsig = parser.create_map_sig(xtree, cpp_sig)
    rwsig = parser.rewrite_signature(cpp_sig, _TYPE_NSMAP)
    rwxtree = parser.xparse(rwsig)
    funsig = parser.create_stdfunc_sig(rwxtree, rwsig)

    _, fname, _ = parser.get_function_signature(rwxtree, rwsig, lambda x: "{}".format(x))
    return OpMeta(op_variant=op_name, mapsig=mapsig, funsig=funsig, func=fname)


def gen_hpu_wrap_ops(op_metas, args, out_dir):
    aten_code = ""
    autogradhpu_code = ""
    aten_impls = []
    autograd_impls = []
    header_impls = []
    for fgen in op_metas:
        override_fn = "hpu_wrap::{}".format(fgen.func)

        pos = fgen.funsig.find("(")
        overload = fgen.funsig[:pos] + " (*)" + fgen.funsig[pos:]
        impl = generate_impl(fgen.op_variant, overload, override_fn)
        header_impls.append(f"{fgen.funsig[:pos]} {fgen.func}{fgen.funsig[pos:]};")
        if fgen.op_variant in _FN_AUTOGRAD_HPU:
            autograd_impls.append(impl)
        else:
            aten_impls.append(impl)

    if aten_impls:
        aten_code = f"TORCH_LIBRARY_IMPL(aten, HPU, m) {{\n{''.join(aten_impls)}\n}}\n"
    if autograd_impls:
        autogradhpu_code = f"TORCH_LIBRARY_IMPL(aten, AutogradHPU, m) {{\n{''.join(autograd_impls)}\n}}\n"

    regs = aten_code + autogradhpu_code

    print(
        templates._HPU_WRAP_REGS.format(
            gen=os.path.basename(sys.argv[0]),
            regs=regs,
        ),
        file=gen_cpp_output_file(args, out_dir + "/" + "wrap_kernels_registrations"),
    )

    print(
        templates._HPU_WRAP_HEADERS.format(
            gen=os.path.basename(sys.argv[0]),
            headers="\n".join(header_impls),
        ),
        file=gen_h_output_file(args, out_dir + "/" + "wrap_kernels_declarations"),
    )


def get_hpu_wrap(ctxop, minor_pt_ver):
    hpu_wrap = ctxop.get_hpu_wrap_all_versions()
    hpu_wrap_list = ctxop.get_hpu_wrap_version_list()
    hpu_wrap_range = ctxop.get_hpu_wrap_version_range()

    if not (hpu_wrap or hpu_wrap_list or hpu_wrap_range):
        return False, False

    if isinstance(hpu_wrap_list, list):
        hpu_wrap_list = minor_pt_ver in hpu_wrap_list
    if isinstance(hpu_wrap_range, list):
        hpu_wrap_range = (hpu_wrap_range[0] == 0 or Version(hpu_wrap_range[0]) <= Version(minor_pt_ver)) and (
            hpu_wrap_range[1] == 0 or Version(hpu_wrap_range[1]) >= Version(minor_pt_ver)
        )
    return True, (hpu_wrap or hpu_wrap_list or hpu_wrap_range)


# For PT2.0, there are non-mandatory op (from PT2.0 point of view),
# that we still need to register in the new Eager flow.
# In order to do it, we overwrite them to default=False, dispatch=True
non_mandatory_ops_whitelist = [
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
]


def get_dtdf(fields):
    if any(op in fields["schema"] for op in non_mandatory_ops_whitelist):
        return True

    return fields.get("dispatch", "False") == "True" and fields.get("default", "False") == "False"


def create_funcdef(fndef, jdata):
    fields = json.loads(jdata)
    return FuncDef(
        cpp_sig=fndef,
        aten_sig=fields["schema"],
        dtdf=get_dtdf(fields),
    )


def get_aten_opname(aten_sig):
    return aten_sig.split("(")[0].split("::")[1]


def is_tensor_api(fndef):
    fndef = fndef.replace("at::", "")
    fndef = fndef.replace("c10::Device", "Device")
    m = re.search(r"\bTensor\b", fndef)
    return m is not None, fndef


def extract_pt_ops(path, hpu_ops):
    errors = []
    pt_ops = dict()
    all_ops_metas = []

    for line in open(path, "r"):
        m = re.match(r"\s*([^\s].*); //\s+(.*)", line)
        if not m:
            continue

        cpp_sig = m.group(1)

        try:
            parser.xparse(cpp_sig)
            op_variant = re.search(r"aten::([^(]*)", m.group(2))[1]
            all_ops_metas.append(generate_op_meta(cpp_sig, op_variant))
            if op_variant not in hpu_ops:
                continue

            op_def = create_funcdef(cpp_sig, m.group(2))
            pt_ops[get_aten_opname(op_def.aten_sig)] = op_def
        except Exception as e:
            if is_tensor_api(cpp_sig)[0]:
                errors.append((cpp_sig, str(e)))
                print('Error parsing "{}": {}'.format(cpp_sig, e), file=sys.stderr)
    return pt_ops, errors, all_ops_metas


def fndef_from_schema(schema):
    cpp_sig = cpp_from_schema(schema)
    return FuncDef(
        cpp_sig=cpp_sig,
        aten_sig=schema,
        dtdf=True,
    )


def print_backend_to_file(op_groups, op_backend, kr_regs, custom_schema_regs, gen_file_idx, args):
    backend_inclusions = """
#include "hpu_ops/op_validator.h"
"""
    header_inclusions = ""
    for op_group in sorted(op_groups):
        header_inclusions += f'#include "{op_group}.h"\n'
    print(
        templates._CPP_HEADER.format(
            gen=os.path.basename(sys.argv[0]),
            header_inclusions=backend_inclusions + header_inclusions,
            dtype_defs="",
            funcs="",
            op_backend=op_backend,
            kr_regs=kr_regs,
            torch_regs="",
            custom_schema_regs=torch_library_fragment(custom_schema_regs),
            file_idx=gen_file_idx,
        ),
        file=gen_cpp_output_file(args, "backend/hpu_op{}".format(gen_file_idx)),
    )


def handle_reused_class(fgen, class_per_op_group, op_groups, is_backend):
    class_name = fgen.ctxop.get_op_backend_class() if is_backend else fgen.ctxop.get_op_frontend_class()
    if is_custom_class(class_name, is_backend):
        if class_name in class_per_op_group:
            op_groups.add(class_per_op_group[class_name])
        else:
            class_per_op_group[class_name] = fgen.opgroup


def handle_reused_backend(fgen, backend_per_op_group, op_groups):
    return handle_reused_class(fgen, backend_per_op_group, op_groups, True)


def handle_reused_frontend(fgen, frontend_per_op_group, op_groups):
    return handle_reused_class(fgen, frontend_per_op_group, op_groups, False)


def generate_backend(args, fgens, is_custom=False):
    num_fgens_per_shard = len(fgens) // NUM_SHARDS
    gen_file_idx = 0
    op_backend = ""
    kr_regs = ""
    custom_schema_regs = ""
    fgen_files = defaultdict(list)
    op_groups = set()
    backend_per_op_group = dict()
    for idx, fgen in enumerate(fgens):
        fgen_files[fgen.opgroup].append(fgen)
        if fgen.op_backend is None:
            continue

        handle_reused_backend(fgen, backend_per_op_group, op_groups)
        op_groups.add(fgen.opgroup)

        (
            _op_backend,
            _kr_regs,
            _custom_schema_regs,
        ) = generate_backend_functions(fgen, is_custom)
        op_backend += _op_backend
        kr_regs += _kr_regs
        custom_schema_regs += _custom_schema_regs

        if not is_custom and should_write_and_go_to_next_file(idx, num_fgens_per_shard, gen_file_idx, len(fgens)):
            print_backend_to_file(op_groups, op_backend, kr_regs, custom_schema_regs, gen_file_idx, args)
            gen_file_idx += 1
            op_backend = ""
            kr_regs = ""
            custom_schema_regs = ""
            op_groups = set()

    if is_custom:
        print_backend_to_file(op_groups, op_backend, kr_regs, custom_schema_regs, "_custom", args)

    backend_class_headers = {}

    for fgen_file, ffgens in fgen_files.items():
        op_backend_classes = generate_op_backend_hclasses(ffgens, backend_class_headers, fgen_file)
        header_decls = generate_header_decls(ffgens)

        # Create output file ...
        print(
            templates._H_HEADER.format(
                op=fgen_file,
                gen=os.path.basename(sys.argv[0]),
                op_backend_classes=op_backend_classes,
                op_frontend_classes="",
                header_decls=header_decls,
            ),
            file=gen_h_output_file(args, "backend/" + fgen_file),
        )


def get_frontend_inclusions(mode):
    common_inclusions = (
        '#include "hpu_ops/cpu_fallback.h"\n'
        '#include "hpu_ops/op_validator.h"\n'
        '#include "hpu_ops/op_logger.h"\n'
        '#include "common/dump_args.h"\n'
    )

    eager_inclusions = (
        '#include "habana_eager/eager_exec.h"\n'
        '#include "habana_eager/ops/eager_op.h"\n'
        '#include "habana_eager/ops/override_fns.h"\n'
    )

    lazy_inclusions = (
        '#include "habana_kernels/lazy_kernels_declarations.h"\n'
        '#include "habana_kernels/lazy_kernels.h"\n'
        '#include "habana_lazy/hpu_stage_submission.h"\n'
        "using habana_lazy::LazyOp;\n"
        "using habana_lazy::GraphHashBuilder;\n"
    )

    if mode == "eager":
        return common_inclusions + eager_inclusions
    return "\n" + common_inclusions + lazy_inclusions + "\n"


def print_frontend_to_file(op_groups, dtype_defs, functions, torch_regs, gen_file_idx, out_dir, args, ns):
    frontend_inclusions = get_frontend_inclusions(out_dir)
    header_inclusions = ""
    for op_group in sorted(op_groups):
        header_inclusions += f'#include "{op_group}.h"\n'

    print(
        templates._CPP_HEADER.format(
            gen=os.path.basename(sys.argv[0]),
            header_inclusions=frontend_inclusions + header_inclusions,
            dtype_defs=dtype_defs,
            funcs=functions,
            op_backend="",
            kr_regs="",
            torch_regs=torch_library_impl(torch_regs, ns),
            custom_schema_regs="",
            file_idx=gen_file_idx,
        ),
        file=gen_cpp_output_file(args, "{}/hpu_op{}".format(out_dir, gen_file_idx)),
    )


def generate_frontend(args, fgens, out_dir, namespace="aten"):
    frontend_func = f"op_frontend_{out_dir}"
    fgens_filtered = [x for x in fgens if getattr(x, frontend_func) is not None]
    ops_count = len(fgens_filtered)
    num_fgens_per_shard = ops_count // NUM_SHARDS
    gen_file_idx = 0
    dtype_defs = ""
    functions = ""
    torch_regs = ""
    is_custom = namespace != "aten"

    fgen_files = defaultdict(list)
    op_groups = set()
    frontend_per_op_group = dict()

    for idx, fgen in enumerate(fgens_filtered):
        fgen_files[fgen.opgroup].append(fgen)

        handle_reused_frontend(fgen, frontend_per_op_group, op_groups)
        op_groups.add(fgen.opgroup)

        (
            _dtype_defs,
            _functions,
            _torch_regs,
        ) = generate_frontend_functions(fgen, mode=out_dir)
        dtype_defs += _dtype_defs
        functions += _functions
        torch_regs += _torch_regs

        if not is_custom and should_write_and_go_to_next_file(idx, num_fgens_per_shard, gen_file_idx, ops_count):
            print_frontend_to_file(op_groups, dtype_defs, functions, torch_regs, gen_file_idx, out_dir, args, namespace)
            gen_file_idx += 1
            dtype_defs = ""
            functions = ""
            torch_regs = ""
            op_groups = set()

    if is_custom:
        namespace_to_postfix = {"hpu": "_custom", "quantized_decomposed": "_quant", "torchvision": "_torchvision"}
        file_postfix = namespace_to_postfix[namespace]
        print_frontend_to_file(op_groups, dtype_defs, functions, torch_regs, file_postfix, out_dir, args, namespace)

    frontend_class_headers = {}

    for fgen_file, ffgens in fgen_files.items():
        op_frontend_classes = generate_op_frontend_hclasses(
            ffgens,
            frontend_class_headers,
            fgen_file,
            "habana_lazy::LazyOp" if out_dir == "lazy" else "eager::EagerOp",
        )

        header_decls = generate_header_decls(ffgens)

        # Create output files ...
        print(
            (templates._H_HEADER if out_dir == "lazy" else templates._H_HEADER_EAGER).format(
                op=fgen_file,
                gen=os.path.basename(sys.argv[0]),
                op_backend_classes="",
                op_frontend_classes=op_frontend_classes,
                header_decls=header_decls,
            ),
            file=gen_h_output_file(args, out_dir + "/" + fgen_file),
        )


def check_op_params(op_name, op_params):
    for field in op_params.keys():
        if field not in _AVAILABLE_FIELDS:
            raise Exception(f"Invalid field for {op_name}: {field}")


def generate(args):
    yaml_ctx = YamlContext(args.yaml)
    pt_ops, errors, all_ops_metas = extract_pt_ops(args.pt_signatures, yaml_ctx.get_op_names())
    assert len(errors) == 0

    minor_pt_ver = ".".join(torch.__version__.split(".")[:2])
    fgens_native = []
    fgens_hpu_wrap_lazy = []
    fgens_hpu_wrap_eager = []
    fgens_custom = []
    fgens_quant = []
    fgens_torchvision = []

    for op_name, op_params in yaml_ctx.get_op_data():
        check_op_params(op_name, op_params)
        ctxop = Op(op_name, op_params)
        is_hpu_wrap, is_current_version = get_hpu_wrap(ctxop, minor_pt_ver)
        if is_hpu_wrap:
            if not is_current_version:
                continue
            fndef = pt_ops.get(op_name, None)
            assert fndef is not None, f"Op {op_name} doesn't exist in aten namespace, consider removing it from yaml."
            op_meta = generate_op_meta(fndef.cpp_sig, op_name)
            fgens_hpu_wrap_lazy.append(op_meta)
            if fndef.dtdf:
                fgens_hpu_wrap_eager.append(op_meta)
        elif ctxop.get_custom_op_schema():
            fndef = fndef_from_schema(ctxop.get_custom_op_schema())
            namespace = re.search(r"^(.*)::", ctxop.get_custom_op_schema()).group(1)
            if namespace == "torchvision":
                fgens_torchvision.append(generate_op(fndef, op_name, ctxop, op_params, ns=namespace))
            elif namespace == "quantized_decomposed":
                fgens_quant.append(generate_op(fndef, op_name, ctxop, op_params, ns=namespace))
            else:
                fgens_custom.append(generate_op(fndef, op_name, ctxop, op_params, ns=namespace))
        elif not ctxop.get_only_shared_layer():
            fndef = pt_ops.get(op_name, None)
            if fndef is None:
                print(f"Op {op_name} doesn't exist in aten namespace, consider removing it from yaml.")
                continue
            fgens_native.append(generate_op(fndef, op_name, ctxop, op_params))

    gen_hpu_wrap_ops(fgens_hpu_wrap_lazy, args, "lazy")
    gen_hpu_wrap_ops(fgens_hpu_wrap_eager, args, "eager")

    generate_autocast_ops(all_ops_metas, args)

    generate_backend(args, fgens_native + fgens_quant + fgens_torchvision)
    generate_backend(args, fgens_custom, is_custom=True)

    for mode in ["eager", "lazy"]:
        generate_frontend(args, fgens_native, mode)
        generate_frontend(args, fgens_custom, mode, namespace="hpu")
        generate_frontend(args, fgens_quant, mode, namespace="quantized_decomposed")
        generate_frontend(args, fgens_torchvision, mode, namespace="torchvision")


def generate_check_kernel_support_sigs(fgen):
    dtype_defs = ""
    op_frontend_functions = ""

    # torch registrations
    assert fgen.mapsig not in _FN_AUTOGRAD_HPU

    op_validator_generator = fgen.ctxop.get_op_validator_generator(HabanaExecutionMode.COMPILE)
    if op_validator_generator is not None:
        dtype_defs += op_validator_generator.get_validator_data_def(is_out_fn(fgen.func), fgen.cppsig)

    if fgen.op_frontend_lazy:
        # Lazy functions
        op_frontend_functions += "{}\n\n".format(fgen.op_frontend_lazy)

    # Lowering Kernel code

    return (
        dtype_defs,
        op_frontend_functions,
    )


def get_cp_type_check(cptype):
    cp_type_check_map = {
        "Scalar": "isScalar",
        "Tensor": "isTensor",
        "double": "isDouble",
        "bool": "isBool",
        "int64_t": "isInt",
        "ITensorListRef": "isTensorList",
        "TensorList": "isTensorList",
        "c10::optional<ArrayRef>": "isList",
        "IntArrayRef": "isList",
    }
    if cptype not in cp_type_check_map:
        return None
    return cp_type_check_map[cptype]


def generate_native_functions_from_yaml(native_yaml_path, tags_yaml_path):
    from torchgen.gen import parse_native_yaml

    parsed_yaml = parse_native_yaml(native_yaml_path, tags_yaml_path)
    native_func_dict = {}
    for f in parsed_yaml.native_functions:
        func_name = f.func.name.__str__()
        native_func_dict[func_name] = f
    return native_func_dict


@local.parametrize(use_const_ref_for_mutable_tensors=False, use_ilistref_for_tensor_lists=False)
def codegen_torchgen(f):
    sig_group = CppSignatureGroup.from_native_function(f, method=False)
    sig = sig_group.most_faithful_signature()
    binding_list, code_list = convert_arguments(f)
    translated_args = translate(binding_list, sig.arguments(), method=sig.method)
    code_connector = "\n      "
    arg_connector = ", "
    code_str = "  " + code_connector.join(code for code in code_list)
    args_str = f"{arg_connector.join(e.expr for e in translated_args)}"

    code = templates._TORCHGEN_POP_CODE_FORMAT_.format(code_str, args_str)
    code = re.sub(r"\t", "  ", code)
    code = re.sub(r" +\n", "\n", code)

    return code


def codegen_stackpop(param_vars, fun_args):
    arg_def = ""
    func_args = ""
    stack_args = ""
    for fun_arg, param_var in zip(fun_args, param_vars):
        arg = fun_arg.replace("&", "").replace("const", "")
        arg_def += "{} {};".format(arg, param_var)
        func_args += " ,{}".format(param_var)
        stack_args += " ,{}".format(param_var)
    return templates._STACK_POP_CODE_FORMAT_.format(arg_def, stack_args, func_args[2:])


def generate_stack_pop(fgens, fgen_pos, native_func_dict):
    struct_def = "struct shared_layer_{func_name} : SharedLayerOp {{\n".format(func_name=fgens[fgen_pos[0]].func)
    stack_unroll = "bool func(torch::jit::Stack &stack, bool is_dynamic) {\n"

    funsig = fgens[fgen_pos[0]].funsig
    fun_args = re.split(",", re.split(r"\(|\)", funsig)[1])
    param_nums = []
    for pos in fgen_pos:
        fun_args = re.split(r",(?!\d)", re.split(r"\(|\)", fgens[pos].funsig)[1])
        param_nums.append(len(fun_args))
    param_nums, fgen_pos = (list(t) for t in zip(*sorted(zip(param_nums, fgen_pos))))
    num_param_fun = 0
    first_stack_pop = True
    for pos_idx, pos in enumerate(fgen_pos):
        funsig = fgens[pos].funsig
        fun_args = re.split(r",(?!\d)", re.split(r"\(|\)", funsig)[1])
        params = parser.get_parameters(fgens[pos].tree)
        param_vars = []
        param_types = []

        for p in params:
            ptype = parser.param_type(p)
            cptype = parser.type_core(ptype)
            pname = parser.param_name(p)
            param_types.append(ptype)
            param_vars.append(pname)
        if len(param_types) != len(param_vars):
            print("Error in generating ", fgens[pos].func)
        if pos_idx == 0:
            stack_unroll += "  if (stack.size() == {num_param}) {{\n".format(num_param=len(param_vars))
        elif num_param_fun != len(param_types):
            stack_unroll += "  }}\n  if (stack.size() == {num_param}) {{\n".format(num_param=len(param_vars))
            first_stack_pop = True
        num_param_fun = len(param_types)
        stack_unroll += (
            "    auto ivalue_arr = torch::jit::last(stack, {num_elem});\n    if (".format(num_elem=len(param_types))
            if first_stack_pop
            else "    else if ("
        )
        first_stack_pop = False
        param_idx = 0

        for idx, ptype in enumerate(param_types):
            cptype = parser.type_core(ptype)
            cptype_check = get_cp_type_check(cptype)
            if cptype_check is not None:
                stack_unroll += "&& " if param_idx > 0 else ""
                stack_unroll += "ivalue_arr[{idx}].{cptype_check}() ".format(idx=idx, cptype_check=cptype_check)
                param_idx += 1
        # generates if (true) in case there was no other condition
        if param_idx == 0:
            stack_unroll += "true"
        stack_unroll += ") {\n"

        aten_sig = fgens[pos].aten_sig
        aten_sig_name = aten_sig[0 : aten_sig.find("(")]
        aten_sig_name = aten_sig_name.split("::")[1] if "::" in aten_sig_name else aten_sig_name
        if aten_sig_name in native_func_dict:
            stack_unroll += codegen_torchgen(native_func_dict[aten_sig_name])
        else:
            stack_unroll += codegen_stackpop(param_vars, fun_args)
    stack_unroll += "  }\n  return false;\n}\n"
    struct_def += stack_unroll
    struct_def += "private:\n"
    return struct_def


CHECK_KERNEL_SUPPORT_HEADERS = """
#include <torch/extension.h>
#include <pybind11/stl.h>
#include <pybind11/pybind11.h>
#include <torch/csrc/jit/tensorexpr/tensorexpr_init.h>
#include <torch/csrc/jit/python/pybind_utils.h>
#include "cpu_fallback.h"

using habana_helpers::DTypeHelper;
using namespace torch::jit;
"""


def add_fgen_idx_to_generate(fgen: OpGen, idx: int, unique_funcs: Dict[str, List[int]], ops_added: Set[str]) -> None:
    if fgen.func in unique_funcs:
        unique_funcs[fgen.func].append(idx)
    else:
        unique_funcs[fgen.func] = [idx]
        ops_added.add(fgen.func)


def generate_functions_code(
    fgens: List[OpGen], fgen_pos: List[int], native_func_dict: Dict[str, Any], functions: str, dtype_defs: str
) -> Tuple[str, str]:
    functions += generate_stack_pop(fgens, fgen_pos, native_func_dict)
    for fgen_idx in fgen_pos:
        fgen = fgens[fgen_idx]

        (
            _dtype_defs,
            _functions,
        ) = generate_check_kernel_support_sigs(fgen)

        dtype_defs += _dtype_defs
        functions += _functions
    functions += "};\n\n"

    return functions, dtype_defs


def generate_check_kernel_support_frontend(args, fgens, fgens_hpu_wrap, fgens_custom, frontend_inclusions):
    OUT_DIR = "check_kernel_support"
    dtype_defs = ""
    functions = ""

    fgen_files = defaultdict(list)
    op_groups = set()
    ops_added = set()

    unique_func_map = {}
    for idx, fgen in enumerate(fgens):
        fgen_files[fgen.opgroup].append(fgen)
        op_groups.add(fgen.opgroup)
        add_fgen_idx_to_generate(fgen, idx, unique_func_map, ops_added)

    hpu_shared_layer_unsupported_ops = set([x.func for x in fgens_hpu_wrap])
    for op in hpu_shared_layer_unsupported_ops:
        if op in unique_func_map:
            del unique_func_map[op]
        if op in ops_added:
            ops_added.remove(op)

    unique_func_map_custom = {}
    for idx, fgen in enumerate(fgens_custom):
        if fgen.ctxop.get_op_validator():
            add_fgen_idx_to_generate(fgen, idx, unique_func_map_custom, ops_added)

    num_fgens_per_shard = len(unique_func_map) // NUM_SHARDS
    gen_file_idx = 0
    gen_hdr_file_includes = CHECK_KERNEL_SUPPORT_HEADERS
    native_yaml_path = args.native_functions
    tags_yaml_path = os.path.join(os.path.dirname(native_yaml_path), "tags.yaml")
    native_func_dict = generate_native_functions_from_yaml(native_yaml_path, tags_yaml_path)

    for idx, fgen_pos in enumerate(unique_func_map.values()):
        functions, dtype_defs = generate_functions_code(fgens, fgen_pos, native_func_dict, functions, dtype_defs)

        if should_write_and_go_to_next_file(idx, num_fgens_per_shard, gen_file_idx, len(unique_func_map)):
            frontend_inclusions = "\n" + frontend_inclusions + "\n"

            file_name = gen_h_output_file(args, f"{OUT_DIR}/hpu_op{gen_file_idx}")
            print(
                templates._CPP_HEADER_CHECK_KERNEL_SUPPORT.format(
                    gen=os.path.basename(sys.argv[0]),
                    header_inclusions=CHECK_KERNEL_SUPPORT_HEADERS,
                    dtype_defs=dtype_defs,
                    funcs=functions,
                    op_backend="",
                    file_idx=gen_file_idx,
                ),
                file=file_name,
            )
            gen_hdr_file_includes += "#include<hpu_op{}.h>\n".format(gen_file_idx)
            gen_file_idx += 1
            dtype_defs = ""
            functions = ""

    for idx, fgen_pos in enumerate(unique_func_map_custom.values()):
        functions, dtype_defs = generate_functions_code(fgens_custom, fgen_pos, native_func_dict, functions, dtype_defs)

    frontend_inclusions = "\n" + frontend_inclusions + "\n"

    file_name = gen_h_output_file(args, f"{OUT_DIR}/hpu_op_custom")
    print(
        templates._CPP_HEADER_CHECK_KERNEL_SUPPORT.format(
            gen=os.path.basename(sys.argv[0]),
            header_inclusions=CHECK_KERNEL_SUPPORT_HEADERS,
            dtype_defs=dtype_defs,
            funcs=functions,
            op_backend="",
            file_idx=gen_file_idx,
        ),
        file=file_name,
    )
    gen_hdr_file_includes += "#include<hpu_op_custom.h>\n"

    frontend_class_headers = {}

    header_inclusions = ""
    for op_group in sorted(op_groups):
        header_inclusions += f'#include "{op_group}.h"\n'

    hpu_shared_layer_unsupported_ops_def = (
        """std::set<std::string> hpu_shared_layer_unsupported_ops = {{ "{}" }};""".format(
            '",\n"'.join(sorted(hpu_shared_layer_unsupported_ops))
        )
    )
    map_def = "std::unordered_map<std::string, std::function<bool(c10::FunctionSchema&, bool, bool, const py::list&, py::args& args, const py::kwargs& kwargs)>> fallback_support_check_map = {\n"
    for op in sorted(ops_added):
        map_def += """{{"{op}", &check_support<habana::shared_layer_{op}>}},\n""".format(op=op)
    map_def += "};\n"
    print(
        header_inclusions + gen_hdr_file_includes + map_def + hpu_shared_layer_unsupported_ops_def,
        file=gen_cpp_output_file(args, f"{OUT_DIR}/op_def"),
    )

    for fgen_file, ffgens in fgen_files.items():
        op_frontend_classes = generate_op_frontend_hclasses(
            ffgens,
            frontend_class_headers,
            fgen_file,
            "eager::EagerOp",
        )

        header_decls = generate_header_decls(ffgens)
        # Create output files ...
        print(
            (templates._H_HEADER_EAGER).format(
                op=fgen_file,
                gen=os.path.basename(sys.argv[0]),
                op_backend_classes="",
                op_frontend_classes=op_frontend_classes,
                header_decls=header_decls,
            ),
            file=gen_h_output_file(args, f"{OUT_DIR}/{fgen_file}"),
        )


def generate_check_kernel_support(args):
    yaml_ctx = YamlContext(args.yaml)
    # pt_ops is dict of {op_name : (cpp_sig, aten_sig, dtdf)}
    pt_ops, errors, _ = extract_pt_ops(args.pt_signatures, yaml_ctx.get_op_names())
    assert len(errors) == 0

    minor_pt_ver = ".".join(torch.__version__.split(".")[:2])
    fgens_native = []
    fgens_hpu_wrap = []
    fgens_custom = []

    for op_name, op_params in yaml_ctx.get_op_data():
        ctxop = Op(op_name, op_params)
        is_hpu_wrap, is_current_version = get_hpu_wrap(ctxop, minor_pt_ver)
        if is_hpu_wrap:
            if not is_current_version:
                continue
            fndef = pt_ops.get(op_name, None)
            assert fndef is not None, f"Op {op_name} doesn't exist in the aten namespace."
            op_meta = generate_op_meta(fndef.cpp_sig, op_name)
            fgens_hpu_wrap.append(op_meta)
        elif ctxop.get_custom_op_schema():
            fndef = fndef_from_schema(ctxop.get_custom_op_schema())
            fgen_custom = generate_op(fndef, op_name, ctxop, op_params, True)
            fgens_custom.append(fgen_custom)
        else:
            fndef = pt_ops.get(op_name, None)
            if fndef is None:
                print(f"Op {op_name} doesn't exist in aten namespace, consider removing it from yaml.")
                continue
            fgens_native.append(generate_op(fndef, op_name, ctxop, op_params, True))

    header_inclusions = (
        '#include "habana_kernels/lazy_kernels_declarations.h"\n'
        '#include "hpu_ops/cpu_fallback.h"\n'
        '#include "hpu_ops/op_validator.h"\n'
        '#include "habana_eager/ops/eager_op.h"\n'
    )
    generate_check_kernel_support_frontend(
        args,
        fgens_native,
        fgens_hpu_wrap,
        fgens_custom,
        header_inclusions,
    )


if __name__ == "__main__":
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
        type=bool,
        default=False,
        help="Check kernel support or normal kernel generation",
    )
    args, files = arg_parser.parse_known_args()
    if args.check_kernel_support:
        generate_check_kernel_support(args)
    else:
        generate(args)
