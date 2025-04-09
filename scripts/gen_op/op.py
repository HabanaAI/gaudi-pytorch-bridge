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
import os
from functools import wraps

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

    def is_op_autograd(self):
        return self.op.get("autograd", False)

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

    def get_is_custom_op_out_variant(self):
        return self.op.get("is_custom_op_out_variant", False)

    def get_hpu_wrap(self):
        return self.op.get("hpu_wrap", False)

    def get_only_shared_layer(self):
        return self.op.get("only_shared_layer", False)

    def get_overwritten_op_names_in_slrg(self):
        return self.op.get("overwritten_op_names_in_slrg", None)

    def get_op_validator(self):
        return self.op.get("op_validator", None)

    def get_shared_layer_meta(self):
        if self.get_op_validator() in [None, "check-node-with-shared-layer"]:
            return None
        return self.get_op_validator()

    def get_skip_slrg(self):
        return self.op.get("skip_slrg", False)

    def get_fallback_check(self):
        return self.op.get("fallback_check", [])

    def get_frontend_blocklist(self):
        return self.op.get("frontend_blocklist", [])

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

    def is_eager_op(self):
        override_fn = self.get_override_fn()
        if override_fn:
            if "lazy" in override_fn:
                return False
        return True

    def treat_as_dtdf(self):
        return self.op.get("treat_as_dtdf", False)
