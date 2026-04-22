###############################################################################
# Copyright (c) 2021-2026 Intel Corporation
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


from typing import Any, NamedTuple

from habana_frameworks.torch._torch_jit_C import jit
from habana_frameworks.torch.dynamo.debug_utils.logger import get_compile_backend_logger

import torch
from torch._ops import HigherOrderOperator
from torch._ops import OpOverload as TorchOpOverload

from ._fx_jit_lowering_utils import BUILTIN_OPS_TO_ATEN_OPS, TYPE_TO_JIT_TYPE

logger = get_compile_backend_logger()


def propagate_node_module_name(fx_node: torch.fx.Node, jit_val: jit.Value):
    if "nn_module_stack" in fx_node.meta:
        last_module_stack_elem = next(reversed(fx_node.meta["nn_module_stack"].values()))[0]
        module_list = last_module_stack_elem.split(".")
        # First element is in different format: e.x. L['self']; we convert it to L_self
        module_list[0] = module_list[0].translate(str.maketrans({"[": "_", "]": None, "'": None}))
        debug_name = f"{'/'.join(module_list)}/{jit_val.debugName()}"
        jit_val.setDebugName(debug_name)


# The whole mechanism of how an interpreter works is well explained
# in the torch repository. Please refer to torch/fx/interpreter.py
# to see details. This particular interpreter is responsible for
# lowering FX IR to JIT IR. It is not done using the torch.script
# function for a couple of reasons. These are the most important:
#   1. Performance of generating JIT representation using the
#      torch.script was poor.
#   2. We are using our own JIT fork which is available under the
#      "csrc/gpu/jit" directory. Classes which are responsible for
#      JIT IR live in a different namespace than the original ones,
#      so they are different classes from compiler's perspective.
#      The torch.script generates torch::jit::Graph and we need
#      habana_torch::jit::Graph, so even if we use the torch.script,
#      we would need a way to cast the original torch Graph to the
#      forked habana_torch Graph.
# The lowering below fulfill two functions:
#   1. Generates a JIT IR graph based on the graph described in FX.
#   2. Rewrites metadata from FX graph nodes to JIT graph nodes.
class FxToJitLowering(torch.fx.Interpreter):
    def __init__(self, graph_module: torch.fx.GraphModule):
        super().__init__(graph_module)
        self.graph_module = graph_module
        self.jit_ir = jit.Graph()
        self.const_cache: dict[tuple[type, Any], jit.Value] = {}

    ##############################################################
    # Below is the implementation of the functions from the base
    # interpreter class.
    ##############################################################

    def run_node(self, n: torch.fx.Node):
        # This function has been overwritten because we need
        # access to FX nodes, not node.target as done in the base
        # run_node function.
        with self._set_current_node(n):
            args, kwargs = self.fetch_args_kwargs_from_env(n)
            if not isinstance(args, tuple):
                raise AssertionError("Not a tuple instance")
            if not isinstance(kwargs, dict):
                raise AssertionError("Not a dict instance")
            return getattr(self, n.op)(n, args, kwargs)

    def call_module(self, node: torch.fx.Node, args, kwargs):
        raise NotImplementedError("Modules should not be present in our FX submodules. Please report a bug.")

    def get_attr(self, node: torch.fx.Node, args, kwargs):
        raise NotImplementedError("Attributes should not be present in our FX submodules. Please report a bug.")

    def call_function(self, node: torch.fx.Node, args, kwargs):
        # Prepare arguments for JIT node and schema if exists or
        # aten opcode otherwise.
        schema, jit_op_name, jit_args, jit_kwargs = self._handle_target(node, args, kwargs)

        # Create JIT nodes which are as an equivalent for the node
        # being processed.
        returned_val = self._emit_jit_node(node, schema, jit_op_name, jit_args, jit_kwargs)

        # Set jit.Value's debug name based on fx node metadata
        propagate_node_module_name(node, returned_val)

        # Check if the metadata for the node is well-formed.
        self._check_meta(node)

        # Rewrite the metadata from the FakeTensorProp execution.
        self._apply_meta(node.meta["val"], returned_val)

        return returned_val

    def call_method(self, node: torch.fx.Node, args, kwargs):
        raise NotImplementedError("Methods should not be present in our FX submodules. Please report a bug.")

    def output(self, node: torch.fx.Node, args, kwargs):
        for arg in args:
            output = self._get_jit_val(arg)
            self.jit_ir.registerOutput(output)

        if kwargs:
            raise AssertionError("Kwargs as output values are not supported. Please report a bug.")

        # Validate the graph and check if all created nodes were
        # inserted into the JIT graph.
        self.jit_ir.lint()

    def placeholder(self, node: torch.fx.Node, args, kwargs):
        input_val = self.jit_ir.addInput()

        # Check if the metadata for the node is well-formed.
        self._check_meta(node)

        # Rewrite the metadata from the FakeTensorProp execution.
        self._apply_meta(node.meta["val"], input_val)

        input_val.setDebugName(str(node))

        return input_val

    ##############################################################
    # Below are the functions responsible for obtaining JIT values
    # from the arguments passed by the interpreter.
    ##############################################################

    def _insert_list_from_jit_vals(self, jit_vals: list[jit.Value], parameter) -> jit.Value:
        element_type = None
        optional_type = False

        # Sometimes there may be a situation in which we need to create
        # a list whose element type will be an optional type. The following
        # code handles this situation.
        types = {type(elem.type()) for elem in jit_vals}
        if len(types) > 1 and jit.NoneType in types:
            types.remove(jit.NoneType)
            optional_type = True

        for elem in jit_vals:
            # If we are dealing with a situation in which the element type is
            # an optional type, we omit JIT values of the NoneType type.
            if optional_type and elem.type().kind() == "NoneType":
                continue

            element_type = elem.type()
            if element_type.kind() == "TensorType":
                # If the type of a list element refers to an exact tensor type,
                # it makes it impossible to find a suitable match to the op schema.
                element_type = jit.TensorType.get()

            # We got the expected type of the list item.
            break

        if len(types) > 1:
            type_kinds = [jit_type.get().kind() for jit_type in types]
            raise RuntimeError(f"Cannot create list for many different type kinds: {type_kinds}")

        if optional_type:
            element_type = jit.OptionalType(element_type)

        # Special cases for no elem list when stride/dim/size=[]
        # Get the element type from the parameter type
        if len(types) == 0 and hasattr(parameter, "type"):
            if parameter.type.kind() == "ListType":
                element_type = parameter.type.getElementType()
            elif parameter.type.kind() == "OptionalType":
                element_type = parameter.type.getElementType().getElementType()

        list_node = self.jit_ir.createList(element_type, jit_vals)
        return self.jit_ir.insertNode(list_node).output()

    def _insert_tuple_from_jit_vals(self, jit_vals: list[jit.Value]) -> jit.Value:
        types = [(jit.TensorType.get() if isinstance(val.type(), jit.TensorType) else val.type()) for val in jit_vals]
        tuple_type = jit.TupleType(types)
        tuple_node = self.jit_ir.createTuple(jit_vals, tuple_type)
        return self.jit_ir.insertNode(tuple_node).output()

    def _insert_namedtuple_from_jit_vals(self, fx_tuple, jit_vals: list[jit.Value]) -> jit.Value:
        names = list(fx_tuple._fields)
        types = [getattr(fx_tuple, field).type() for field in fx_tuple._fields]
        tuple_type = jit.TupleType(type(fx_tuple).__name__, names, types)

        tuple_node = self.jit_ir.createTuple(jit_vals, tuple_type)
        return self.jit_ir.insertNode(tuple_node).output()

    def _get_jit_val_from_iterable(self, iterable_arg, parameter) -> jit.Value:
        collected_vals = [self._get_jit_val(elem) for elem in iterable_arg]

        if isinstance(iterable_arg, list):
            return self._insert_list_from_jit_vals(collected_vals, parameter)
        elif isinstance(iterable_arg, tuple):
            return self._insert_tuple_from_jit_vals(collected_vals)
        elif isinstance(iterable_arg, NamedTuple):
            return self._insert_namedtuple_from_jit_vals(iterable_arg, collected_vals)

    def _get_jit_val(self, arg: Any, parameter=None) -> jit.Value:
        if isinstance(arg, jit.Value):
            return arg

        converter = TYPE_TO_JIT_TYPE.find(type(arg))
        if converter:
            jit_type = converter(arg)
            if jit_type:
                if (
                    isinstance(jit_type, jit.TupleType)
                    and hasattr(parameter, "type")
                    and parameter.type.kind() == "ListType"
                ):
                    # If jit_type mismatchs the parameter type, we need to convert it.
                    element_type = parameter.type.getElementType()
                    jit_type = jit.ListType(element_type)
                elif (
                    isinstance(jit_type, jit.BoolType | torch.BoolType)
                    and hasattr(parameter, "type")
                    and (parameter.type.kind() == "NumberType" or parameter.type.kind() == "TensorType")
                ):
                    arg = int(arg)
                    jit_type = TYPE_TO_JIT_TYPE.find(type(arg))(arg)

                cache_key = (type(arg), tuple(arg) if isinstance(arg, list) else arg)
                if cache_key in self.const_cache:
                    return self.const_cache[cache_key]
                else:
                    new_const = self.jit_ir.insertConstant(arg, jit_type)
                    self.const_cache[cache_key] = new_const
                    return new_const

        # A workaround, should be isinstance(arg, (list, tuple, namedtuple)), but lintrule force a syntax of
        # UP038 Use `X | Y` in `isinstance` call instead of `(X, Y)`
        # however, arg maybe a UnionType which cannot follow the rule UP038. You'll get
        # TypeError: unsupported operand type(s) for |: 'types.UnionType' and 'function
        if isinstance(arg, list) or isinstance(arg, tuple) or isinstance(arg, NamedTuple):  # noqa SIM101
            return self._get_jit_val_from_iterable(arg, parameter)

        raise NotImplementedError(f"The argument {arg} contains unsupported type: {type(arg)}. Please report a bug.")

    ##############################################################
    # Below are the functions responsible for emitting jit
    # nodes.
    ##############################################################

    def _handle_schema(self, schema: torch.FunctionSchema, args, kwargs) -> tuple[str, list[jit.Value]]:
        jit_args: list[jit.Value] = []

        for i, parameter in enumerate(schema.arguments):
            if i < len(args):
                jit_args.append(self._get_jit_val(args[i], parameter))
            elif parameter.name in kwargs:
                jit_args.append(self._get_jit_val(kwargs[parameter.name], parameter))
            else:
                if not parameter.has_default_value():
                    raise RuntimeError(f"The parameter {i} is not present in the argument list for {schema.name}.")
                jit_args.append(self._get_jit_val(parameter.default_value, parameter))

        return schema.name, jit_args

    def _handle_target(
        self, node: torch.fx.Node, args, kwargs
    ) -> tuple[torch.FunctionSchema, str, list[jit.Value], list[jit.NamedValue]]:
        target = node.target

        schema: torch.FunctionSchema = None
        jit_op_name: str = None
        jit_args: list[jit.Value] = []
        jit_kwargs: list[jit.NamedValue] = []

        # Each target that comes from the aten space should inherit from
        # the OpOverload class (represents C++ ATen operators). Thanks
        # to this, such a target has an assigned schema that we can read.
        # It contains the name, default values of some arguments, etc.
        # If the target has a schema, we can automatically create the
        # corresponding operation in the JIT graph.
        if isinstance(target, TorchOpOverload):
            schema = target._schema
            jit_op_name, jit_args = self._handle_schema(schema, args, kwargs)
        elif hasattr(target, "default") and isinstance(target.default, TorchOpOverload):
            schema = target.default._schema
            jit_op_name, jit_args = self._handle_schema(schema, args, kwargs)
        # Python-only operators that are unrepresentable in TorchScript.
        # Examples: cond, while loop, triton wrapper, etc.
        # You can find them under <pytorch_repo>/torch/_higher_order_ops.
        elif isinstance(target, HigherOrderOperator):
            full_fx_op_name = node._pretty_print_target(target)
            raise NotImplementedError(f"Unimplemented HigherOrderOperator: target={full_fx_op_name}")
        # Python specific operators. Defined directly in python language
        # or inside torch repository. Operators defined inside operator.py
        elif target.__name__ in BUILTIN_OPS_TO_ATEN_OPS:
            for arg in args:
                jit_args.append(self._get_jit_val(arg))
            for k, v in kwargs.items():
                op_v = self._get_jit_val(v)
                jit_kwargs.append(jit.NamedValue(k, op_v))

            jit_op_name = BUILTIN_OPS_TO_ATEN_OPS[target.__name__](jit_args, jit_kwargs)
        else:
            full_fx_op_name = node._pretty_print_target(target)
            raise NotImplementedError(f"Unimplemented call_function: target={full_fx_op_name}")

        return schema, jit_op_name, jit_args, jit_kwargs

    def _get_stack_trace(self, fx_node: torch.fx.Node):
        try:
            from pathlib import Path

            # It is available from torch 2.3
            from torch.fx.graph import _parse_stack_trace

            if fx_node.stack_trace:
                parsed_st = _parse_stack_trace(fx_node.stack_trace)
                filename = Path(parsed_st.file).name
                stack_trace = f"File: {filename}:{parsed_st.lineno} in {parsed_st.name}, code: {parsed_st.code}"
                return stack_trace
            return None
        except ImportError:
            if fx_node.stack_trace:
                lines = fx_node.stack_trace.strip().split("\n")
                if len(lines) == 2:
                    # Return code line
                    return lines[1].lstrip()
                return None

    def _emit_jit_node(
        self,
        fx_node: torch.fx.Node,
        schema: torch.FunctionSchema,
        op_name: str,
        jit_args: list[jit.Value],
        jit_kwargs: list[jit.NamedValue],
    ) -> jit.Value:
        emitted_nodes: list[jit.Node] = []
        returned_val = None
        return_name = str(fx_node)

        if not schema:
            # The 'insert' function checks the correctness of the opcode
            # against the defined schemas in PyTorch. Using schema is key to
            # generate proper nodes as default arguments are not present
            # in FX and we need them in JIT IR. This function is able to add
            # them automatically to the JIT graph. They are added in the form of
            # constant values. Moreover, if a node creates several outputs,
            # it collects them into one output value, e.g. using prim::ListConstruct
            # or prim::TupleConstruct what allow to return every time single JIT value.
            returned_val = self.jit_ir.insert(op_name, jit_args, jit_kwargs)
            emitted_nodes.append(returned_val.node())
        else:
            num_returns = len(schema.returns)
            jit_node = self.jit_ir.create(op_name, jit_args, num_returns)
            jit_node = self.jit_ir.insertNode(jit_node)
            emitted_nodes.append(jit_node)

            for i, return_arg in enumerate(schema.returns):
                if num_returns > 1:
                    arg_name = return_arg.name if return_arg.name else return_name + f".{i}"
                    jit_node.outputsAt(i).setDebugName(arg_name)
                jit_node.outputsAt(i).setType(return_arg.type)

            returned_val = self.jit_ir.packValues(list(jit_node.outputs())) if num_returns > 1 else jit_node.output()
            emitted_nodes.append(returned_val.node())

        returned_val.setDebugName(return_name)

        stack_trace = self._get_stack_trace(fx_node)
        if stack_trace:
            for jit_node in emitted_nodes:
                jit_node.setStackTrace(stack_trace)

        return returned_val

    ##############################################################
    # Below are the functions responsible for rewriting metadata
    # from a FX node to a JIT node.
    ##############################################################

    def _check_meta(self, node: torch.fx.Node):
        if "val" not in node.meta:
            raise RuntimeError(f"node.meta['val'] is not present for: {node}")

    def _apply_meta_for_simple_types(self, meta_val, jit_val: jit.Value):
        jit_type = None
        input_type = None

        converter = TYPE_TO_JIT_TYPE.find(type(meta_val))
        if converter:
            jit_type = converter(meta_val)

        if jit_type:
            if isinstance(meta_val, torch.Tensor):
                input_type = jit.TensorType.get()
            elif isinstance(meta_val, torch.SymBool | torch.SymInt | torch.SymFloat):
                input_type = jit_type
            else:
                input_type = jit_type
        else:
            raise NotImplementedError(f"The metadata contains unsupported type: {type(meta_val)}. Please report a bug.")

        if isinstance(input_type, jit.NoneType) and jit_val.type().annotation_str == "Tensor":
            logger.debug("Won't rewrite metadata from Tensor to NoneType")
        else:
            jit_val.setType(input_type)

    def _apply_meta_for_collections(self, meta_val, jit_val: jit.Value):
        # If we are dealing with ListConstruct or TupleConstruct,
        # we do not want to describe the output type of these nodes,
        # because it may break the schema. Therefore, we rewrite the
        # metadata for the input values.
        jit_type = jit_val.node().kind()
        if jit_type in ["prim::ListConstruct", "prim::TupleConstruct"]:
            num_meta_elem = len(meta_val)
            num_jit_value_inputs = jit_val.node().inputsSize()
            if num_meta_elem != num_jit_value_inputs:
                raise RuntimeError(
                    f"The number of elements:{str(num_meta_elem)} in the FX collection does not match "
                    f"with number of elements:{str(num_jit_value_inputs)} in the JIT collection."
                )

            for i, val in enumerate(meta_val):
                self._apply_meta(val, jit_val.node().inputsAt(i))

    def _create_unpack(self, meta_val, jit_val: jit.Value):
        # Additionally, we create a node for each list that is responsible
        # for unpacking the values. This will allow for easier collections processing
        # in further graph manipulation.
        unpacked_collection = None

        jit_val_type = jit_val.type()
        if isinstance(jit_val_type, jit.ListType):
            unpacked_collection = self.jit_ir.createListUnpack(jit_val, len(meta_val))
        elif isinstance(jit_val_type, jit.TupleType):
            unpacked_collection = self.jit_ir.createTupleUnpack(jit_val)
        else:
            raise NotImplementedError(
                f"The JIT IR contains unsupported collection type {jit_val_type}. Please report a bug."
            )

        unpacked_collection = self.jit_ir.insertNode(unpacked_collection)
        num_unpack_outputs = unpacked_collection.outputsSize()
        if len(meta_val) != num_unpack_outputs:
            raise RuntimeError(
                "The number of elements in the FX collection does not match "
                "with number of elements in the upacked JIT collection."
            )

        for i, val in enumerate(meta_val):
            self._apply_meta(val, unpacked_collection.outputsAt(i))

    def _apply_meta(self, meta_val, jit_val: jit.Value):
        if isinstance(meta_val, list | tuple):
            self._apply_meta_for_collections(meta_val, jit_val)
            self._create_unpack(meta_val, jit_val)
        else:
            self._apply_meta_for_simple_types(meta_val, jit_val)
