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

import copy
import re
from abc import ABC, abstractmethod
from collections import defaultdict

from .op import Op

_DEVICE_STR_TO_ENUM = {
    "All": "-1",
    "Gaudi2": "synDeviceGaudi2",
    "Gaudi3": "synDeviceGaudi3",
}


class OpValidatorGenerator(ABC):
    """
    Class describing how to generate code for given op_validator
    """

    def __init__(self, ctxop: Op, execution_mode_for_shared_layer):
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


def get_op_validator_generator(ctxop: Op, execution_mode_for_shared_layer) -> OpValidatorGenerator:
    op_validator = ctxop.get_op_validator()
    mapping = {
        None: UseDtypesOpValidatorGenerator,
        "check-node-with-shared-layer": CheckNodeWithSharedLayerValidatorGenerator,
    }
    result = mapping.get(op_validator, CheckNodeWithCustomSharedLayerValidatorGenerator)(
        ctxop, execution_mode_for_shared_layer
    )
    if not result.can_generate():
        return None
    return result
