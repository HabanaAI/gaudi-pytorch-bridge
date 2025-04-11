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

import json
import os
import re
import sys
from collections import defaultdict
from typing import Any

import yaml
from torchgen import local
from torchgen.api.translate import translate
from torchgen.api.types import CppSignatureGroup
from torchgen.api.unboxing import convert_arguments
from yaml import CLoader as Loader

from . import code_templates as templates
from . import constants, parser
from .custom_ops import cpp_from_schema
from .op import Op
from .op_validator import get_op_validator_generator
from .version_checker import is_pytorch_older_than


def torch_library_fragment(custom_schema_regs):
    if custom_schema_regs:
        return "TORCH_LIBRARY_FRAGMENT(hpu, m) {\n" "  static_cast<void>(m);\n" f"{custom_schema_regs}\n" "}"
    return ""


def torch_library_impl(torch_regs, ns="aten"):
    if torch_regs:
        return f"TORCH_LIBRARY_IMPL({ns}, HPU, m) {{\n{torch_regs}\n}}\n"
    return ""


NUM_SHARDS = 10


def should_write_and_go_to_next_file(idx: int, num_idxs_per_shard: int, file_idx: int, total_idxs):
    return ((file_idx + 1) < NUM_SHARDS and (idx + 1) % num_idxs_per_shard == 0) or (idx + 1) == total_idxs


class YamlContext:
    def __init__(self, yamlfile):
        with open(yamlfile) as ff:
            self.op_data = yaml.load(ff.read(), Loader=Loader)
            # Sometimes disabling ops for an upcoming pytorch version is necessary.
            # All ops added to skip_list won't be processed, same way if they were not present in hpu_op.yaml
            skip_ops = []
            self.op_data = {op: self.op_data[op] for op in self.op_data if op not in skip_ops}

    def get_op_names(self):
        return self.op_data.keys()

    def get_op_data(self):
        return self.op_data.items()


def is_out_fn(is_custom_op_out_variant, fname):
    if is_custom_op_out_variant:
        return True
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
    code += "  [[maybe_unused]] bool require_h2d = false;\n"
    code += "  [[maybe_unused]] bool require_st = false;\n\n"
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
    elif is_custom and (prefix == "VAL_" or prefix == "VAL_CUSTOM_"):
        prefix = "VAL_"
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
        return f'  {prefix}{fallback_string}{per_tensor_string}{overload_variant}({tensor_string}{opname}{is_dynamic_string}, {overload_string}{check_st_h2d_str}{", ".join(param_vars)})\n'

    if prefix:
        return single_fallback()

    code = ""
    for t in tinputs:
        code += single_fallback(t)
    return code


def bitwise_ops_alt_guid(guid):
    logical_guid = guid.replace("bitwise_", "")
    return "if (ScalarType() == at::kBool) {\n" f'      SetGuid("{logical_guid}_i8");\n' "    }"


def extract_reduction_vars_indices(param_vars, use_int=False):
    default_id = "-1" if use_int else "std::nullopt"
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
    is_custom_op_out_variant = ctxop.get_is_custom_op_out_variant()

    assert (not out_ids) ^ (not inplace_ids) ^ is_out_fn(is_custom_op_out_variant, fname), (
        f"`out_ids` or `inplace_ids` should not be defined for {fname}"
        if is_out_fn(is_custom_op_out_variant, fname)
        else f"Either `out_ids` or `inplace_ids` should be defined for {fname}"
    )

    scalar_ids_set = set(scalar_ids)
    err_ids = scalar_ids_set.intersection(out_ids if len(out_ids) else inplace_ids)
    assert len(err_ids) == 0, f"Input(s) at {err_ids} cannot be both scalar and tensor for {fname}."

    out_ids = ", ".join([str(o) for o in out_ids])
    inplace_ids = ", ".join([str(i) for i in inplace_ids])
    scalar_ids = ", ".join([str(s) for s in scalar_ids])

    if fname.startswith("bitwise_"):
        custom_handler = templates.CUSTOM_HANDLER.format(body=bitwise_ops_alt_guid(guid))
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
        ctor_extra_calls.append(f"SetSynapseLayouts({{{in_layouts}}}, {{{out_layouts}}});")

    if is_out_fn(is_custom_op_out_variant, fname) and num_out_tensors > 1:
        ctor_extra_calls.append(f"SetNumOutTensors({num_out_tensors});")

    if output_meta_fn:
        ctor_extra_calls.append(f"SetOutputMetaFn({output_meta_fn});")
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
        ctor_extra_calls.append(f"SetComputeOutputShapes({output_shape_fn});")

    if st_meta_fn:
        ctor_extra_calls.append(f"SetSTMetaFn({st_meta_fn});")

    if custom_fill_params:
        ctor_extra_calls.append(f"SetFillParams({custom_fill_params});")

    if tpc_input_order:
        tpc_input_order_str = ", ".join(map(str, tpc_input_order))
        ctor_extra_calls.append(f"SetTpcInputOrder({{{tpc_input_order_str}}});")

    if promote_to_common_type:
        ctor_extra_calls.append("EnableTypePromotion();")
    elif promote_int_to_float:
        ctor_extra_calls.append("PromoteIntToFloat();")

    if handle_bool_inputs:
        ctor_extra_calls.append("HandleBoolInputs();")

    if shared_layer_meta_meta_fn:
        ctor_extra_calls.append(f"SetSharedLayerMetaFn({shared_layer_meta_meta_fn});")

    return templates.OPCLASS_HEADER.format(
        op_backend_class=op_backend_class,
        cname=cname,
        guid=guid,
        out_ids=out_ids,
        inplace_ids=inplace_ids,
        scalar_ids=scalar_ids,
        is_out_fn=str(is_out_fn(is_custom_op_out_variant, fname)).lower(),
        ctor_extra_calls="".join(["\n" + " " * 8 + c for c in ctor_extra_calls]),
        custom_handler=custom_handler,
    )


def generate_impl(op_variant, funsig, override_fn):
    overload = funsig.replace("(", " (*)(", 1)
    return f'  m.impl("{op_variant}", static_cast<{overload}>(&{override_fn}));\n'


def generate_dtype_defs(fgen, execution_mode_for_shared_layer):
    op_validator_generator = get_op_validator_generator(fgen.ctxop, execution_mode_for_shared_layer)
    if op_validator_generator is not None:
        return op_validator_generator.get_validator_data_def(
            is_out_fn(fgen.ctxop.get_is_custom_op_out_variant(), fgen.func), fgen.cppsig
        )
    return ""


def generate_frontend_functions(fgen, mode):
    # torch registrations
    override_fn = f"habana::{fgen.func}"
    impl = generate_impl(get_aten_opname(fgen.aten_sig), fgen.funsig, override_fn)

    dtype_defs = generate_dtype_defs(fgen, constants.get_execution_mode_from_string(mode))

    op_frontend_functions = fgen.op_frontend_lazy if mode == "lazy" else fgen.op_frontend_eager
    op_frontend_functions += "\n\n"

    return (
        dtype_defs,
        op_frontend_functions,
        impl,
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


def generate_header_decls(fgens, gen_check_node_with_sl_val=False):
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
            return f'{macro}({", ".join([fn] + args)});\n'
        return ""

    reg_decls = ""
    early_exit_decls = ""
    outshape_decls = ""
    outmeta_decls = ""
    shared_layer_meta_decls = ""
    check_node_with_sl_decls = ""
    stmeta_decls = ""
    fill_params_decls = ""
    fallback_check_decls = ""
    forward_decls = ""
    for fgen in fgens:
        reg_decls += f"{fgen.rwsig};\n"

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

        if gen_check_node_with_sl_val and fgen.ctxop.get_op_validator() is not None:
            validator_sufix = fgen.op_variant.replace(".", "_")
            check_node_with_sl_decls += f"extern CheckNodeWithSharedLayerValidator validator_{validator_sufix};\n"

        if fgen.ctxop.get_st_meta() is not None and not fgen.ctxop.get_st_meta().startswith("Default"):
            stmeta_decls += build(fgen.ctxop.get_st_meta(), stmeta_fns, "STMETA_DECL")

        fill_params_decls += build(fgen.ctxop.get_custom_fill_params(), fill_params, "FILL_PARAMS_DECL")

        fc = fgen.ctxop.get_fallback_check()
        if fc:
            fallback_check_decls += build(fc[0], fc_fns, "FALLBACK_CHECK", fgen.fc_params)

    if len(check_node_with_sl_decls) > 0:
        forward_decls += "class CheckNodeWithSharedLayerValidator;\n"

    return (
        forward_decls
        + reg_decls
        + early_exit_decls
        + outshape_decls
        + outmeta_decls
        + shared_layer_meta_decls
        + check_node_with_sl_decls
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
    return gen_output_file(args, f"{opgroup}.h")


def gen_cpp_output_file(args, opgroup):
    return gen_output_file(args, f"{opgroup}.cpp")


# Generate file with all potential ops for autocast. The actual ops registered
# for autocast are based on the default lists in autocast_helpers.h file or
# on the external file provided via env.
def generate_autocast_ops(op_metas, args):
    def op_to_skip(function_name, op_name):
        return (
            function_name.endswith(("_out", "_"))
            or any(s in function_name for s in ("_.", "cuda", "cudn", "backward"))
            or op_name in constants.AUTOCAST_BLOCKLIST
        )

    def get_registration(op_meta, op_name):
        function_name = op_meta.func
        signature = op_meta.funsig
        for r in constants.AUTOCAST_REPLACEMENTS:
            signature = signature.replace(*r)
        return f'  Hpu_KERNEL({function_name}, "{op_name}", {signature})'

    # only ops defined in 'at' namespace are applicable for autocast
    ops_not_in_at = set()
    with open(args.native_functions) as f:
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

    chunk_size = len(ops_registrations) // NUM_SHARDS + 1

    for i in range(NUM_SHARDS):
        print(
            templates.AUTOCAST_TEMPLATE.format(
                gen=os.path.basename(sys.argv[0]),
                fallback_fallthrough=templates.AUTOCAST_FALLBACK if i == 0 else "",
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


inplace_params_blocklist = [
    "_native_batch_norm_legit",
]

# SW-212132
inplace_params_blocklist_dict = {
    "rrelu_with_noise": "noise",
    "rrelu_with_noise_": "noise",
    "rrelu_with_noise_out": "noise",
}


def should_skip_inplace_params(fname, pname):
    return fname in inplace_params_blocklist or (
        fname in inplace_params_blocklist_dict.keys() and inplace_params_blocklist_dict[fname] == pname
    )


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

        if cptype == ("Tensor" if is_pytorch_older_than("2.7.0") else "at::Tensor"):
            tfetcher.add(pname)
            if not parser.type_is_const(ptype) and not should_skip_inplace_params(fname, pname):
                call_args.append(pname)
                out_indices.append(i)

        if rtype == "void":
            if cptype in (
                ["TensorList", "Tensor"] if is_pytorch_older_than("2.7.0") else ["at::TensorList", "at::Tensor"]
            ):
                call_args.append(pname)
                if out_ids is not None:
                    out_indices.append(i)

        elif rtype == "const at::Tensor &" and cptype == ("Tensor" if is_pytorch_older_than("2.7.0") else "at::Tensor"):
            call_args.append(pname)
            out_indices.append(i)

        if fc and pname in fc[1:]:
            fc_params.append(f"{param_types[i].strip()} {pname}")

    assert len(fc) == 0 or len(fc_params) + 1 == len(
        fc
    ), f"Cannot find all params specified for fallback check {fc[0]}."

    if rtype == "void" and out_ids is not None:
        out_indices = out_ids
        call_args = [call_args[i] for i in out_indices if i < len(call_args)]
    return param_vars, call_args, out_indices, fc_params, tfetcher


def is_inplace_or_out_op(is_custom_op_out_variant, opname):
    if is_custom_op_out_variant or opname.endswith("_out"):
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

        safe_cast = is_inplace_or_out_op(ctxop.get_is_custom_op_out_variant(), fname)

        if safe_cast_check is not None:
            assert safe_cast, f"safe_cast_check cannot check for non inplace/non out variant, op={fname}"
            assert safe_cast_check is False, f"safe_cast_check is true by default for inplace/out variant, op={fname}"
            safe_cast = safe_cast_check

        code = templates.AUTO_COMPUTE_TYPE.format(
            dtype_helper_inputs=", ".join(dtype_helper_inputs),
            type_promo_variant=type_promo_variant,
            safe_cast=str(safe_cast).lower(),
            fe_call_args=fe_call_args if fe_call_args else "std::nullopt",
            param_vars=", dtype" if "dtype" in param_vars else "",
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
    op_validator_generator = get_op_validator_generator(ctxop, execution_mode_for_shared_layer)
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
        if is_out_fn(ctxop.get_is_custom_op_out_variant(), fname) and len(tinputs) > 1:
            tinputs = tinputs[:-1]

        check_per_tensor = False
        if isinstance(dtypes, dict):
            if any(p in dtypes.keys() for p in param_vars) or any(isinstance(x, dict) for x in dtypes.values()):
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
    if is_inplace_or_out_op(ctxop.get_is_custom_op_out_variant(), fname):
        if rtype.startswith("::std::tuple<at::Tensor"):
            code += f"  auto tuple = {fe_call_args};\n"
            code += f"  RUN_INPLACE_TUPLE_MAYBE_WITH_ACC_THREAD({fname}, hpu_op, tuple)"
        elif rtype == "void" and "TensorList" in sig:
            code += f"  RUN_TENSOR_LIST_INPLACE_MAYBE_WITH_ACC_THREAD({fname}, hpu_op, {param_vars[0]})"
        elif rtype.startswith("const at::Tensor"):
            code += f"  RUN_CONST_INPLACE_MAYBE_WITH_ACC_THREAD({fname}, hpu_op, {fe_call_args})"
        else:
            code += f"  RUN_INPLACE_MAYBE_WITH_ACC_THREAD({fname}, hpu_op, {fe_call_args})"
    else:
        if rtype.startswith("::std::tuple<at::Tensor"):
            code += f"  RUN_TUPLE_MAYBE_WITH_ACC_THREAD({fname}, hpu_op)"
        elif rtype.startswith("void") and "TensorList" in sig:
            raise Exception(
                f"Support for TensorLists inputs was removed, as we don't intend to develop new lazy features and no operator needed it so far. Sig: {sig}"
            )
        else:
            code += f"  RUN_MAYBE_WITH_ACC_THREAD({fname}, hpu_op)"
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
    code = f"{sig} {{\n"
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
        constants.HabanaExecutionMode.LAZY,
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

    code += f'  {op_frontend_class}<{rtype}> hpu_op{{"{schema_fn}", {{{", ".join(param_vars)}}}'
    code += handle_output_shape_fn(ctxop, param_vars)

    if use_compute_type:
        code += "  hpu_op.set_scalar_types({compute_type});\n"

    code += handle_output_meta(ctxop, promote_types, dtype_helper_inputs, param_vars, type_promo_variant)

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


def handle_return_eager(
    rtype, fname, fe_call_args, is_eager_op_supported, call_args, inplace_op_info, is_custom_op_out_variant
):
    eager_op_info_args = (
        len(call_args)
        if is_out_fn(is_custom_op_out_variant, fname)
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


def get_eager_op_info(opname, ns):
    type = "eager::eagerOpKind::"
    if opname.endswith("_out") or opname.endswith("_grad_input"):
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
    code = f"{sig} {{\n"
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
        constants.HabanaExecutionMode.EAGER,
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

    is_eager_op_supported = ctxop.is_eager_op()
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

    code += f'  {op_frontend_class}<{rtype}> hpu_op{{"{schema_fn}", {{{", ".join(param_vars)}}}'
    code += handle_output_shape_fn(ctxop, param_vars)

    if use_compute_type:
        code += "  hpu_op.set_scalar_types({compute_type});\n"

    code += handle_output_meta(ctxop, promote_types, dtype_helper_inputs, param_vars, type_promo_variant)

    op_type, op_name = get_eager_op_info(fname, ns)
    inplace_op_info = (op_type, op_name, out_indices)

    code += handle_return_eager(
        rtype,
        fname,
        fe_call_args,
        is_eager_op_supported,
        call_args,
        inplace_op_info,
        ctxop.get_is_custom_op_out_variant(),
    )
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


def generate_op(fndef, op_name, ctxop, op_params, is_check_kernel_support=False, ns="aten"):
    dtdf = fndef.dtdf or ctxop.treat_as_dtdf()
    tree = parser.parse(fndef.cpp_sig)
    xtree = parser.xparse(fndef.cpp_sig)
    mapsig = parser.create_map_sig(xtree, fndef.cpp_sig)
    rwsig = (
        parser.rewrite_signature(fndef.cpp_sig, constants.TYPE_NSMAP)
        if is_pytorch_older_than("2.7.0")
        else fndef.cpp_sig
    )
    rwxtree = parser.xparse(rwsig)
    params = parser.get_parameters(tree)
    aten_sig = fndef.aten_sig
    funsig = parser.create_stdfunc_sig(rwxtree, rwsig)
    opgroup = get_op_group(op_name)
    rtype = parser.get_return_type_str(rwxtree, rwsig)

    sig, fname, xfname = parser.get_function_signature(rwxtree, rwsig, lambda x: f"{x}")

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
        op_backend_class = f'Gen{op_name.replace(".", "_")}'
        op_backend = get_op_backend_class_impl(ctxop, fname, op_backend_class, len(call_args), param_vars)

    op_frontend_eager = None
    blocklisted_frontends = ctxop.get_frontend_blocklist()
    if dtdf and "eager" not in blocklisted_frontends:
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
    if is_check_kernel_support or "lazy" not in blocklisted_frontends:
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

    return constants.OpGen(
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


def generate_op_meta(cpp_sig, op_name, op_params):
    xtree = parser.xparse(cpp_sig)
    mapsig = parser.create_map_sig(xtree, cpp_sig)
    rwsig = parser.rewrite_signature(cpp_sig, constants.TYPE_NSMAP) if is_pytorch_older_than("2.7.0") else cpp_sig
    rwxtree = parser.xparse(rwsig)
    funsig = parser.create_stdfunc_sig(rwxtree, rwsig)

    _, fname, _ = parser.get_function_signature(rwxtree, rwsig, lambda x: f"{x}")
    return constants.OpMeta(
        op_variant=op_name, mapsig=mapsig, funsig=funsig, func=fname, autograd=op_params.get("autograd", False)
    )


def gen_hpu_wrap_ops(op_metas, args, out_dir):
    aten_code = ""
    autogradhpu_code = ""
    aten_impls = []
    autograd_impls = []
    header_impls = []
    for fgen in op_metas:
        override_fn = f"hpu_wrap::{fgen.func}"

        pos = fgen.funsig.find("(")
        impl = generate_impl(fgen.op_variant, fgen.funsig, override_fn)
        header_impls.append(f"{fgen.funsig[:pos]} {fgen.func}{fgen.funsig[pos:]};")
        if fgen.autograd:
            autograd_impls.append(impl)
        else:
            aten_impls.append(impl)

    if aten_impls:
        aten_code = f"TORCH_LIBRARY_IMPL(aten, HPU, m) {{\n{''.join(aten_impls)}\n}}\n"
    if autograd_impls:
        autogradhpu_code = f"TORCH_LIBRARY_IMPL(aten, AutogradHPU, m) {{\n{''.join(autograd_impls)}\n}}\n"

    regs = aten_code + autogradhpu_code

    print(
        templates.HPU_WRAP_REGS.format(
            gen=os.path.basename(sys.argv[0]),
            regs=regs,
        ),
        file=gen_cpp_output_file(args, out_dir + "/" + "wrap_kernels_registrations"),
    )

    print(
        templates.HPU_WRAP_HEADERS.format(
            gen=os.path.basename(sys.argv[0]),
            headers="\n".join(header_impls),
        ),
        file=gen_h_output_file(args, out_dir + "/" + "wrap_kernels_declarations"),
    )


def get_dtdf(fields):
    return fields.get("dispatch", "False") == "True" and fields.get("default", "False") == "False"


def create_funcdef(fndef, jdata):
    fields = json.loads(jdata)
    return constants.FuncDef(
        cpp_sig=fndef,
        aten_sig=fields["schema"],
        dtdf=get_dtdf(fields),
    )


def fndef_from_schema(schema):
    cpp_sig = cpp_from_schema(schema)
    return constants.FuncDef(
        cpp_sig=cpp_sig,
        aten_sig=schema,
        dtdf=True,
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
    pt_ops = {}
    all_ops_metas = []

    for line in open(path):
        m = re.match(r"\s*([^\s].*); //\s+(.*)", line)
        if not m:
            continue

        cpp_sig = m.group(1)

        try:
            parser.xparse(cpp_sig)
            op_variant = re.search(r"aten::([^(]*)", m.group(2))[1]
            all_ops_metas.append(generate_op_meta(cpp_sig, op_variant, {}))
            if op_variant not in hpu_ops:
                continue

            op_def = create_funcdef(cpp_sig, m.group(2))
            pt_ops[get_aten_opname(op_def.aten_sig)] = op_def
        except Exception as e:
            if is_tensor_api(cpp_sig)[0]:
                errors.append((cpp_sig, str(e)))
                print(f'Error parsing "{cpp_sig}": {e}', file=sys.stderr)
    return pt_ops, errors, all_ops_metas


def print_backend_to_file(op_groups, op_backend, kr_regs, custom_schema_regs, gen_file_idx, args):
    backend_inclusions = """
#include "hpu_ops/op_validator.h"
"""
    header_inclusions = ""
    for op_group in sorted(op_groups):
        header_inclusions += f'#include "{op_group}.h"\n'
    print(
        templates.CPP_HEADER.format(
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
        file=gen_cpp_output_file(args, f"backend/hpu_op{gen_file_idx}"),
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
    backend_per_op_group = {}
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
            templates.H_HEADER.format(
                op=fgen_file,
                gen=os.path.basename(sys.argv[0]),
                op_backend_classes=op_backend_classes,
                op_frontend_classes="",
                header_decls=header_decls,
            ),
            file=gen_h_output_file(args, "backend/" + fgen_file),
        )


def get_frontend_inclusions(mode, ns):
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

    lazy_declarations = "lazy_kernels_declarations" if ns == "aten" else "lazy_custom_op_declarations"
    lazy_inclusions = (
        f'#include "habana_kernels/{lazy_declarations}.h"\n'
        '#include "habana_kernels/lazy_kernels.h"\n'
        '#include "habana_lazy/hpu_stage_submission.h"\n'
        "using habana_lazy::LazyOp;\n"
        "using habana_lazy::GraphHashBuilder;\n"
    )

    if mode == "eager":
        return common_inclusions + eager_inclusions
    return "\n" + common_inclusions + lazy_inclusions + "\n"


def print_frontend_to_file(op_groups, dtype_defs, functions, torch_regs, gen_file_idx, out_dir, args, ns):
    frontend_inclusions = get_frontend_inclusions(out_dir, ns)
    header_inclusions = ""
    for op_group in sorted(op_groups):
        header_inclusions += f'#include "{op_group}.h"\n'

    print(
        templates.CPP_HEADER.format(
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
        file=gen_cpp_output_file(args, f"{out_dir}/hpu_op{gen_file_idx}"),
    )


def generate_frontend(args, fgens, op_validator_map, out_dir, namespace="aten"):
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
    frontend_per_op_group = {}

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
        op_validator = fgen.ctxop.get_op_validator()
        if (
            not is_custom
            and out_dir == "lazy"
            and not fgen.ctxop.op.get("custom_op_schema", False)
            and op_validator is not None
            and not fgen.ctxop.get_skip_slrg()
        ):
            op_name = get_aten_opname(fgen.aten_sig)
            validator_suffix = op_name.replace(".", "_")
            # extract input params
            schema = re.search(r"\((.*?)\)\s*->", fgen.aten_sig)
            if schema:
                op_name_parts = op_name.split(".", 1)
                pure_op_name = op_name_parts[0]
                schema = schema.group(1).replace(" *,", "")
                # remove (a), (a!), (b), (b!)...
                schema = re.sub(r"\([a-z]!?\)", "", schema)
                # remove (a -> *), (b -> *), ...
                schema = re.sub(r"\([a-z] -> \*\)", "", schema)
                # remove default values
                schema = re.sub(r"=\s*[^,)\s]+", "", schema)
                # fix schema with double space
                schema = re.sub(r"\s{2,}", " ", schema)
                namespaces = fgen.ctxop.op.get("namespaces", None)
                pytorch_module_names = fgen.ctxop.op.get("pytorch_module_names", None)
                is_generic_sl_meta = op_validator == "check-node-with-shared-layer"
                op_validator_map[op_name] = {
                    "op_name": pure_op_name,
                    "overload": op_name,
                    "validator_name": f"validator_{validator_suffix}",
                    "validator_header_rel_path": f"generated/lazy/{fgen.opgroup}.h",
                    "generator_name": f"stack_generator_{validator_suffix}",
                    "schema": schema,
                    "namespaces": namespaces,
                    "pytorch_module_names": pytorch_module_names,
                    "is_generic_sl_meta": is_generic_sl_meta,
                    "overwritten_op_names_in_slrg": fgen.ctxop.get_overwritten_op_names_in_slrg(),
                    "executor_name": f"shared_layer_executor_{validator_suffix}",
                }
            else:
                raise Exception(f"Couldn't extract schema input params for {op_name}")

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

        header_decls = generate_header_decls(ffgens, out_dir == "lazy")

        # Create output files ...
        print(
            (templates.H_HEADER if out_dir == "lazy" else templates.H_HEADER_EAGER).format(
                op=fgen_file,
                gen=os.path.basename(sys.argv[0]),
                op_backend_classes="",
                op_frontend_classes=op_frontend_classes,
                header_decls=header_decls,
            ),
            file=gen_h_output_file(args, out_dir + "/" + fgen_file),
        )


def check_valid_fields(op_name, op_params) -> list[str]:
    exceptions = []
    for field in op_params.keys():
        if field not in constants.AVAILABLE_FIELDS:
            exceptions.append(f"Invalid field for {op_name}: {field}\n")
    return exceptions


def check_op_validator(op_name, op_params) -> str:
    if op_params.get("only_shared_layer", False) or op_params.get("hpu_wrap", False):
        return ""
    elif "op_validator" not in op_params and "op_validator_exception" not in op_params:
        return f"{op_name} needed op_validator or op_validator_exception field\n"
    elif "op_validator" in op_params and "op_validator_exception" in op_params:
        return f"{op_name} cannot have both op_validator and op_validator_exception fields\n"
    else:
        return ""


def check_op_params(op_data):
    exceptions = []

    for op_name, op_params in op_data:
        exceptions.extend(check_valid_fields(op_name, op_params))
        op_validator_error = check_op_validator(op_name, op_params)
        if op_validator_error:
            exceptions.append(op_validator_error)

    if exceptions:
        raise Exception(f"Found errors during checking yaml params:\n{exceptions}")


def get_autograd_class_name(op_name: str) -> str:
    op_name = op_name.replace(".", "_")
    components = op_name.split("_")
    return "".join([x.title() for x in components]) + "Function"


def generate_autograd_functions_h_file(fgens_autograd: list[constants.OpGen]) -> str:
    dispatch_functions = ""
    autograd_functions = ""
    for fgen in fgens_autograd:
        autograd_class_name = get_autograd_class_name(fgen.op_variant)

        input_params = fgen.sig.split("(")[1][:-1]
        input_params = ",\n\t  ".join(input_params.split(", "))

        return_type = fgen.sig.split(" ")[0]
        inputs = fgen.rwsig.split("(")[1][:-1].replace(", ", ",\n\t")

        dispatch_op_name = f"{fgen.op_variant}_dispatch"
        dispatch_op_name = dispatch_op_name.replace(".", "_")

        dispatch_functions += f"{return_type} {dispatch_op_name}(\n\t{inputs});\n\n"

        if "std::tuple" in return_type:
            return_type = "std::vector<at::Tensor>"

        autograd_functions += templates.AUTOGRAD_CLASS_DEFINITION.format(
            autograd_class_name=autograd_class_name,
            return_type=return_type,
            op_name=fgen.op_variant,
            input_params=input_params,
        )
    return templates.AUTOGRAD_H_FILE.format(
        gen=os.path.basename(sys.argv[0]), dispatch_functions=dispatch_functions, autograd_functions=autograd_functions
    )


def create_dispatch_function(fgen: constants.OpGen, input_names: list[str], inputs: list[str]) -> str:
    return_type = fgen.sig.split(" ")[0]
    results = fgen.op_variant.split(".")
    if len(results) == 1:
        results.append("")

    dispatch_op_name = f"{fgen.op_variant}_dispatch"
    dispatch_op_name = dispatch_op_name.replace(".", "_")

    return templates.AUTOGRAD_DISPATCH_FUNCTION.format(
        return_type=return_type,
        dispatch_op_name=dispatch_op_name,
        inputs=inputs,
        op_name=results[0],
        variant_name=results[1],
        input_names=input_names,
    )


def create_autograd_frontend(
    fgen: constants.OpGen, input_names: list[str], input_names_len: int, inputs: list[str]
) -> str:
    frontend = ""

    dump_prexif = f"DUMP_{input_names_len}ARGS" if input_names_len > 1 else "DUMP_ARG"

    op_name = f"{fgen.op_variant}_autograd".replace(".", "_")
    return_type = fgen.sig.split(" ")[0]

    frontend += f"{return_type} {op_name}(\n\t{inputs}) {{\n"
    frontend += f'  PT_OP_INFO("{op_name} :", {dump_prexif}({input_names}));\n'

    autograd_call = f"{get_autograd_class_name(fgen.op_variant)}::apply({input_names});\n"
    if "std::tuple" in return_type:
        frontend += f"  auto result = {autograd_call}"
        frontend += "  return {"
        tuple_size = return_type.count("Tensor")
        for i in range(tuple_size):
            frontend += f"result[{i}]"
            if i != tuple_size - 1:
                frontend += ", "
        frontend += "};\n}"
    else:
        frontend += f"  return {autograd_call}"
    frontend += "\n\n"

    return frontend


def generate_autograd_functions_cpp_file(fgens_autograd: list[constants.OpGen]) -> str:
    frontend = ""
    dispatch_functions = ""
    impls = "TORCH_LIBRARY_IMPL(hpu, AutogradHPU, m) {\n"
    for fgen in fgens_autograd:
        inputs = fgen.rwsig.split("(")[1][:-1].replace(", ", ",\n\t")

        input_names = re.findall(r"\b(\w+)\b(?=[,)])", fgen.cppsig[fgen.cppsig.find(fgen.func) :])
        input_names_len = len(input_names)
        input_names = ", ".join(input_names)

        op_name = f"{fgen.op_variant}_autograd".replace(".", "_")
        impls += generate_impl(fgen.op_variant, fgen.funsig, op_name)

        dispatch_functions += create_dispatch_function(fgen, input_names, inputs)
        frontend += create_autograd_frontend(fgen, input_names, input_names_len, inputs)

    impls += "}\n"

    return templates.AUTOGRAD_CPP_FILE.format(
        gen=os.path.basename(sys.argv[0]), frontend=frontend, dispatch_functions=dispatch_functions, impls=impls
    )


def generate_autograd_ops(args, fgens_autograd):
    if fgens_autograd:
        print(generate_autograd_functions_h_file(fgens_autograd), file=gen_h_output_file(args, "autograd/autograd_ops"))
        print(
            generate_autograd_functions_cpp_file(fgens_autograd),
            file=gen_cpp_output_file(args, "autograd/autograd_ops"),
        )


def generate(args):
    yaml_ctx = YamlContext(args.yaml)
    pt_ops, errors, all_ops_metas = extract_pt_ops(args.pt_signatures, yaml_ctx.get_op_names())
    assert len(errors) == 0

    fgens_native = []
    fgens_hpu_wrap_lazy = []
    fgens_hpu_wrap_eager = []
    fgens_custom = []
    fgens_quant = []
    fgens_torchvision = []
    fgens_autograd = []

    check_op_params(yaml_ctx.get_op_data())

    for op_name, op_params in yaml_ctx.get_op_data():
        ctxop = Op(op_name, op_params)
        if ctxop.get_hpu_wrap():
            fndef = pt_ops.get(op_name, None)
            assert fndef is not None, f"Op {op_name} doesn't exist in aten namespace, consider removing it from yaml."
            op_meta = generate_op_meta(fndef.cpp_sig, op_name, op_params)
            fgens_hpu_wrap_lazy.append(op_meta)
            if fndef.dtdf or ctxop.treat_as_dtdf():
                fgens_hpu_wrap_eager.append(op_meta)
        elif ctxop.get_custom_op_schema():
            fndef = fndef_from_schema(ctxop.get_custom_op_schema())
            namespace = re.search(r"^(.*)::", ctxop.get_custom_op_schema()).group(1)
            generated = generate_op(fndef, op_name, ctxop, op_params, ns=namespace)
            if namespace == "torchvision":
                fgens_torchvision.append(generated)
            elif namespace == "quantized_decomposed":
                fgens_quant.append(generated)
            else:
                fgens_custom.append(generated)
            if ctxop.is_op_autograd():
                fgens_autograd.append(generated)
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

    op_validator_map = {}
    for mode in ["eager", "lazy"]:
        generate_frontend(args, fgens_native, op_validator_map, mode)
        generate_frontend(args, fgens_custom, op_validator_map, mode, namespace="hpu")
        generate_frontend(args, fgens_quant, op_validator_map, mode, namespace="quantized_decomposed")
        generate_frontend(args, fgens_torchvision, op_validator_map, mode, namespace="torchvision")

    generate_slrg_files(args, op_validator_map)
    generate_autograd_ops(args, fgens_autograd)


def generate_check_kernel_support_sigs(fgen):
    dtype_defs = ""
    op_frontend_functions = ""

    # torch registrations
    op_validator_generator = get_op_validator_generator(fgen.ctxop, constants.HabanaExecutionMode.COMPILE)
    if op_validator_generator is not None:
        dtype_defs += op_validator_generator.get_validator_data_def(
            is_out_fn(fgen.ctxop.get_is_custom_op_out_variant(), fgen.func), fgen.cppsig
        )

    if fgen.op_frontend_lazy:
        # Lazy functions
        op_frontend_functions += f"{fgen.op_frontend_lazy}\n\n"

    # Lowering Kernel code

    return (
        dtype_defs,
        op_frontend_functions,
    )


def generate_slrg_files(args, op_validator_map):
    generate_slrg_stack_generators(args, op_validator_map)
    generate_slrg_registry_h(args)
    generate_slrg_registry_cpp(args, op_validator_map)


def generate_slrg_stack_generators(args, op_validator_map):
    headers = []
    for key in op_validator_map.keys():
        op_validator = op_validator_map[key]
        headers.append(f'#include "{op_validator["validator_header_rel_path"]}"')

    print(
        templates._SLRG_VALIDATOR_HEADERS.format(gen=os.path.basename(sys.argv[0]), headers=str.join("\n", headers)),
        file=gen_h_output_file(args, "slrg/validator_headers"),
    )


def generate_slrg_registry_cpp(args, op_validator_map):
    generators = []
    executors = []
    funcs = []
    for key in op_validator_map.keys():
        op_validator = op_validator_map[key]
        generators.append(
            f'static SchemaStackGenerator {op_validator["generator_name"]}("{op_validator["schema"]}", "{op_validator["op_name"]}", "{op_validator["overload"]}");'
        )
        executor_type = (
            "GenericSharedLayerExecutor" if op_validator["is_generic_sl_meta"] else "CustomSharedLayerExecutor"
        )
        executors.append(
            f'static {executor_type} {op_validator["executor_name"]}(&{op_validator["generator_name"]}, &habana::{op_validator["validator_name"]});'
        )
        if op_validator["namespaces"] is not None:
            for namespace in op_validator["namespaces"]:
                op_names = (
                    op_validator["pytorch_module_names"] if namespace == "torch.nn" else [op_validator["op_name"]]
                )
                if op_validator["overwritten_op_names_in_slrg"] is not None and namespace != "torch.nn":
                    op_names = op_validator["overwritten_op_names_in_slrg"]
                for op_name in op_names:
                    funcs.append(
                        f'  report_generator->register_op({{"{op_name}", "{op_validator["overload"]}", "{namespace}"}}, &{op_validator["executor_name"]});'
                    )
    print(
        templates._SLRG_REGISTRY_CPP.format(
            gen=os.path.basename(sys.argv[0]),
            generators=str.join("\n", generators),
            executors=str.join("\n", executors),
            funcs=str.join("\n", funcs),
        ),
        file=gen_cpp_output_file(args, "slrg/registry"),
    )


def generate_slrg_registry_h(args):
    print(
        templates._SLRG_REGISTRY_H.format(gen=os.path.basename(sys.argv[0])),
        file=gen_h_output_file(args, "slrg/registry"),
    )


def get_cp_type_check(cptype):
    if cptype not in constants.CP_TYPE_CHECK_MAP:
        return None
    return constants.CP_TYPE_CHECK_MAP[cptype]


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

    code = templates.TORCHGEN_POP_CODE_FORMAT_.format(code_str, args_str)
    code = re.sub(r"\t", "  ", code)
    code = re.sub(r" +\n", "\n", code)

    return code


def codegen_custom_ops(param_vars, fun_args):
    func_args = ""
    ivalue_lines = []
    base_lines = []

    for i, (name, type_) in enumerate(zip(param_vars, fun_args, strict=False)):
        type_ = type_.replace("const ", "").replace("&", "").strip()
        ivalue_lines.append(f"c10::IValue {name} = std::move(peek(stack, {i}, {len(param_vars)}));")

        if type_ == "at::OptionalIntArrayRef":
            optional_str = templates.OPTIONAL_STRING.format(name=name)
            func_args += f"{name}_opt_out, "
            base_lines.append(optional_str)
        else:
            base_lines.append(f"{type_} {name}_base = {name}.to<{type_}>();")
            func_args += f"{name}_base, "

    func_args = func_args[:-2]

    connector = "\n      "
    code_str = connector.join(line for line in ivalue_lines)
    args_str = connector.join(line for line in base_lines)

    return templates.STACK_POP_CODE_FORMAT_.format(code_str, args_str, func_args)


def generate_stack_pop(fgens, fgen_pos, native_func_dict):
    struct_def = f"struct shared_layer_{fgens[fgen_pos[0]].func} : SharedLayerOp {{\n"
    stack_unroll = "bool func(torch::jit::Stack &stack, bool is_dynamic) {\n"

    funsig = fgens[fgen_pos[0]].funsig
    fun_args = re.split(",", re.split(r"\(|\)", funsig)[1])
    param_nums = []
    for pos in fgen_pos:
        fun_args = re.split(r",(?!\d)", re.split(r"\(|\)", fgens[pos].funsig)[1])
        param_nums.append(len(fun_args))
    param_nums, fgen_pos = (list(t) for t in zip(*sorted(zip(param_nums, fgen_pos, strict=False)), strict=False))
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
            stack_unroll += f"  if (stack.size() == {len(param_vars)}) {{\n"
        elif num_param_fun != len(param_types):
            stack_unroll += f"  }}\n  if (stack.size() == {len(param_vars)}) {{\n"
            first_stack_pop = True
        num_param_fun = len(param_types)
        stack_unroll += (
            f"    auto ivalue_arr = torch::jit::last(stack, {len(param_types)});\n    if ("
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
                stack_unroll += f"ivalue_arr[{idx}].{cptype_check}() "
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
            stack_unroll += codegen_custom_ops(param_vars, fun_args)
    stack_unroll += "  }\n  return false;\n}\n"
    struct_def += stack_unroll
    struct_def += "private:\n"
    return struct_def


def add_fgen_idx_to_generate(
    fgen: constants.OpGen, idx: int, unique_funcs: dict[str, list[int]], ops_added: set[str]
) -> None:
    if fgen.func in unique_funcs:
        unique_funcs[fgen.func].append(idx)
    else:
        unique_funcs[fgen.func] = [idx]
        ops_added.add(fgen.func)


def generate_functions_code(
    fgens: list[constants.OpGen], fgen_pos: list[int], native_func_dict: dict[str, Any], functions: str, dtype_defs: str
) -> tuple[str, str]:
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


def generate_check_kernel_support_frontend(args, fgens, fgens_hpu_wrap, fgens_custom):
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

    hpu_shared_layer_unsupported_ops = {x.func for x in fgens_hpu_wrap}
    for op in hpu_shared_layer_unsupported_ops:
        if op in unique_func_map:
            del unique_func_map[op]
        if op in ops_added:
            ops_added.remove(op)

    unique_func_map_custom = {}
    for idx, fgen in enumerate(fgens_custom):
        if fgen.ctxop.get_op_validator():
            fgen_files[fgen.opgroup].append(fgen)
            op_groups.add(fgen.opgroup)
            add_fgen_idx_to_generate(fgen, idx, unique_func_map_custom, ops_added)

    num_fgens_per_shard = len(unique_func_map) // NUM_SHARDS
    gen_file_idx = 0
    gen_hdr_file_includes = templates.CHECK_KERNEL_SUPPORT_HEADERS
    native_yaml_path = args.native_functions
    tags_yaml_path = os.path.join(os.path.dirname(native_yaml_path), "tags.yaml")
    native_func_dict = generate_native_functions_from_yaml(native_yaml_path, tags_yaml_path)

    for idx, fgen_pos in enumerate(unique_func_map.values()):
        functions, dtype_defs = generate_functions_code(fgens, fgen_pos, native_func_dict, functions, dtype_defs)

        if should_write_and_go_to_next_file(idx, num_fgens_per_shard, gen_file_idx, len(unique_func_map)):

            file_name = gen_h_output_file(args, f"{OUT_DIR}/hpu_op{gen_file_idx}")
            print(
                templates.CPP_HEADER_CHECK_KERNEL_SUPPORT.format(
                    gen=os.path.basename(sys.argv[0]),
                    header_inclusions=templates.CHECK_KERNEL_SUPPORT_HEADERS,
                    dtype_defs=dtype_defs,
                    funcs=functions,
                    op_backend="",
                    file_idx=gen_file_idx,
                ),
                file=file_name,
            )
            gen_hdr_file_includes += f"#include<hpu_op{gen_file_idx}.h>\n"
            gen_file_idx += 1
            dtype_defs = ""
            functions = ""

    for fgen_pos in unique_func_map_custom.values():
        functions, dtype_defs = generate_functions_code(fgens_custom, fgen_pos, native_func_dict, functions, dtype_defs)

    file_name = gen_h_output_file(args, f"{OUT_DIR}/hpu_op_custom")
    print(
        templates.CPP_HEADER_CHECK_KERNEL_SUPPORT.format(
            gen=os.path.basename(sys.argv[0]),
            header_inclusions=templates.CHECK_KERNEL_SUPPORT_HEADERS,
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
        map_def += f"""{{"{op}", &check_support<habana::shared_layer_{op}>}},\n"""
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
            (templates.H_HEADER_EAGER).format(
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

    fgens_native = []
    fgens_hpu_wrap = []
    fgens_custom = []

    for op_name, op_params in yaml_ctx.get_op_data():
        ctxop = Op(op_name, op_params)
        if ctxop.get_hpu_wrap():
            fndef = pt_ops.get(op_name, None)
            assert fndef is not None, f"Op {op_name} doesn't exist in the aten namespace."
            op_meta = generate_op_meta(fndef.cpp_sig, op_name, op_params)
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

    generate_check_kernel_support_frontend(
        args,
        fgens_native,
        fgens_hpu_wrap,
        fgens_custom,
    )
