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


try:
    from types import NoneType
except ImportError:
    NoneType = type(None)

from collections.abc import Iterable
from typing import Any, Dict, List, Optional, Tuple

import habana_frameworks.torch._torch_jit_C.jit as jit
import torch


class TypeToLambdaDict:
    def __init__(self):
        self._mapping: Dict[type, Any] = {}

    def add(self, key: type, value: Any):
        self._mapping[key] = value

    def find(self, key: type) -> Optional[Any]:
        value = self._mapping.get(key)
        if not value:
            for subtype, value in self._mapping.items():
                if issubclass(key, subtype):
                    self._mapping[key] = value
                    return value
            self._mapping[key] = None
            return None
        return value

    def keys(self):
        return self._mapping.keys()


TYPE_TO_JIT_TYPE = TypeToLambdaDict()


def py_list_to_jit_list(py_list: List[Any]):
    types = set([type(elem) for elem in py_list])
    if len(types) == 1:
        converter = TYPE_TO_JIT_TYPE.find(next(iter(types)))
        if converter:
            return jit.ListType(converter(None))
    elif len(types) == 2:
        if NoneType in types:
            types.remove(NoneType)
            converter = TYPE_TO_JIT_TYPE.find(next(iter(types)))
            if converter:
                return jit.ListType(jit.OptionalType(converter(None)))
    return None


def py_tuple_to_jit_tuple(py_tuple: Tuple[Any]):
    jit_types = []
    types = set([type(elem) for elem in py_tuple])
    if len(types) > 0:
        for jit_type in types:
            converter = TYPE_TO_JIT_TYPE.find(jit_type)
            if converter:
                jit_types.append(converter(None))
            else:
                return None
        return jit.TupleType(types)
    else:
        return None


TYPE_TO_JIT_TYPE.add(bool, lambda arg: jit.BoolType.get())
TYPE_TO_JIT_TYPE.add(float, lambda arg: jit.FloatType.get())
TYPE_TO_JIT_TYPE.add(int, lambda arg: jit.IntType.get())
TYPE_TO_JIT_TYPE.add(list, py_list_to_jit_list)
TYPE_TO_JIT_TYPE.add(NoneType, lambda arg: jit.NoneType.get())
TYPE_TO_JIT_TYPE.add(str, lambda arg: jit.StringType.get())
TYPE_TO_JIT_TYPE.add(torch._subclasses.fake_tensor.FakeTensor, lambda arg: jit.TensorType.get())
TYPE_TO_JIT_TYPE.add(torch.device, lambda arg: jit.DeviceObjType.get())
TYPE_TO_JIT_TYPE.add(torch.dtype, lambda arg: jit.IntType.get())
TYPE_TO_JIT_TYPE.add(torch.layout, lambda arg: jit.LayoutType.get())
TYPE_TO_JIT_TYPE.add(torch.memory_format, lambda arg: jit.MemoryFormatType.get())
TYPE_TO_JIT_TYPE.add(torch.SymBool, lambda arg: jit.SymBoolType.get())
TYPE_TO_JIT_TYPE.add(torch.SymFloat, lambda arg: jit.SymFloatType.get())
TYPE_TO_JIT_TYPE.add(torch.SymInt, lambda arg: jit.SymIntType.get())
TYPE_TO_JIT_TYPE.add(torch.Tensor, lambda arg: jit.TensorType.get())
TYPE_TO_JIT_TYPE.add(tuple, py_tuple_to_jit_tuple)


def convert_getitem_op(args: List[jit.Value], kwargs: List[jit.Value]):
    # Schema of aten::__getitem__ does not accept
    # tuple type as the first argument, we need
    # the equivalent of aten::__getitem__, but for
    # tuple type
    if type(args[0].type()) == jit.TupleType:
        return "prim::TupleIndex"
    elif type(args[0].type()) == jit.ListType:
        return "aten::__getitem__"

    raise NotImplementedError(f"Not supported argument type: {args[0].type()} for getitem operator.")


BUILTIN_OPS_TO_ATEN_OPS: Dict[str, Any] = {
    "getitem": convert_getitem_op,
    "mul": lambda args, kwargs: "aten::mul",
    "truediv": lambda args, kwargs: "aten::div",
    "add": lambda args, kwargs: "aten::add",
    "sub": lambda args, kwargs: "aten::sub",
    "lt": lambda args, kwargs: "aten::lt",
    "le": lambda args, kwargs: "aten::le",
    "ge": lambda args, kwargs: "aten::ge",
    "ne": lambda args, kwargs: "aten::ne",
    "gt": lambda args, kwargs: "aten::gt",
    "sym_float": lambda args, kwargs: "aten::Float",
    "sym_size": lambda args, kwargs: "aten::sym_size",
    "pow": lambda args, kwargs: "aten::pow",
    "neg": lambda args, kwargs: "aten::neg",
}


def _is_node_output_symbolic(node: torch.fx.node.Node) -> bool:
    from torch._subclasses.fake_tensor import FakeTensor

    output_meta_value = node.meta.get("val", node.meta.get("tensor_meta", None))
    if isinstance(output_meta_value, FakeTensor):
        return output_meta_value._has_symbolic_sizes_strides
    if isinstance(output_meta_value, torch.SymInt):
        return True
    if isinstance(output_meta_value, torch.SymFloat):
        # todo fix me https://jira.habana-labs.com/browse/SW-199903
        # if syngraph_config.enable_dynamic_nodes:
        #    return True

        return True

        raise AssertionError(
            "Currently we do not have any specific support for sym floats."
        )  # not available until above fix delivered
    return False


def is_graph_module_dynamic(gm: torch.fx.GraphModule) -> bool:
    for node in gm.graph.nodes:
        if _is_node_output_symbolic(node):
            return True
    return False


def check_node_and_args(node: torch.fx.node.Node, predicate):
    for arg in node.args:
        if arg.__class__ == torch.fx.node.Node:
            if predicate(arg):
                return True

    if predicate(node):
        return True

    return False


def is_node_dynamic(node: torch.fx.node.Node) -> bool:
    return check_node_and_args(node, _is_node_output_symbolic)


def is_node_or_arg_float64(node: torch.fx.node.Node) -> bool:
    def pred(node):
        return "tensor_meta" in node.meta and node.meta["tensor_meta"].dtype == torch.float64

    return check_node_and_args(node, pred)


def is_node_or_arg_complex(node: torch.fx.node.Node) -> bool:
    def pred(node):
        return "tensor_meta" in node.meta and (
            node.meta["tensor_meta"].dtype in [torch.complex32, torch.complex64, torch.complex128]
        )

    return check_node_and_args(node, pred)


def flatten(nested_iterable):
    for item in nested_iterable:
        if isinstance(item, Iterable) and not isinstance(item, (str, bytes)):
            yield from flatten(item)
        else:
            yield item


def get_jit_types(args):
    jit_types = []

    for arg in args:
        converter = TYPE_TO_JIT_TYPE.find(type(arg))
        if not converter:
            raise NotImplementedError(f"Converter for given type was not provided: {type(arg)}")

        jit_type = converter(arg)
        jit_types.append(jit_type)

    return jit_types
