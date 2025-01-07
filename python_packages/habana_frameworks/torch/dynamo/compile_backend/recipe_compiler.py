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

import logging
import os
import sys

import habana_frameworks.torch.internal.bridge_config as bc
import sympy
import torch
from habana_frameworks.torch.dynamo.compile_backend import config as hpu_backend_config
from habana_frameworks.torch.dynamo.debug_utils.logger import dump_fx_graph, get_compile_backend_logger
from sympy import sympify
from torch._subclasses.fake_tensor import FakeTensor
from torch.fx.experimental.proxy_tensor import py_sym_types, unset_fake_temporarily

from .random_utils import is_random_op
from .symbolic_execution import PythonPrinter, SymbolicShapeEvaluator, substitute_sympyfn

logger = get_compile_backend_logger()
enable_dynamic_output_preallocate = bc.get_pt_hpu_enable_dynamic_output_preallocate()


def get_input_symbolic(graph_module, inputs):
    """
    Returns a list of input shapes from the graph, in the form of
    in the order in which they appear in the graph.
    """
    import numpy as np

    from ._recipe_compiler_C import RangeInfo

    # We are limiting the max size here for range max value.
    # Since we dont have clear indicaion for which all symbols
    # the max range is specifically set by user so for symbols
    # where max is not set it is coming [2, INT_MAX], now we could
    # have had this check specifically to check for INT_MAX but
    # sometimes the symbols are coming as expression in which case
    # the value max value may depend on expression. Hence as asfe side
    # we are limiting the max value to MAX_UPPER_SIZE, above which we create
    # out own default range [val, val*2]. It also help us in avoiding
    # workspace allocation failiure.
    MAX_UPPER_SIZE = 1_00_00_000
    MAX_LOWER_SIZE = -1_00_00_000

    def is_mark_dynamic(inputs):
        for input in inputs:
            if hasattr(input, "_dynamo_dynamic_range"):
                logger.debug("Enabling user min/max flow")
                return True
        return False

    def get_input(input_shape):
        min_shape = []
        max_shape = []
        shape_expr = []
        for dim in input_shape:
            if isinstance(dim, torch.SymInt):
                node = dim.node
                expr = node.expr
                shape_env = node.shape_env
                # An expr can be a independent SymInt node (eg: s0 or s1) or a composition of them eg: (48*s0 or s0*s1).
                # In the case of expr which has symbolic computation, bound_sympy evaluates them.
                # https://pytorch.org/docs/stable/generated/torch.fx.experimental.symbolic_shapes.ShapeEnv.html#torch.fx.experimental.symbolic_shapes.ShapeEnv.bound_sympy
                # expr.xreplace replaces the symbolic variables with their current values and computes the expression.
                var_range = shape_env.var_to_range.get(expr, None) or shape_env.bound_sympy(expr)
                var_val = shape_env.var_to_val.get(expr, None) or expr.xreplace(shape_env.var_to_val)
                assert var_range, var_val
                # if range upper value is greater than MAX_UPPER_SIZE
                # then allocate min as current val and max as 2*curr val
                # so that in backend can create dynamic recipe in 1 shot
                logger.debug("Initial MIN ", var_range.lower)
                logger.debug("Initial MAX ", var_range.upper)
                if var_range.upper > MAX_UPPER_SIZE or var_range.lower < MAX_LOWER_SIZE:
                    logger.debug(
                        f"WARN: Range {var_range.lower}-{var_range.upper} out of bound using [val, 2*val] as range"
                    )
                    min_shape.append(np.int64(var_val))
                    max_shape.append(np.int64(var_val) * 2)
                else:
                    min_shape.append(np.int64(var_range.lower))
                    max_shape.append(np.int64(var_range.upper))
                # Simplify the sympy expression so that can be used by Exprtk
                pexpr = PythonPrinter().doprint
                sz_str = pexpr(expr)
                sz_sympy = sympify(sz_str)
                sz_sympy = substitute_sympyfn(sz_sympy)
                shape_expr.append(sz_sympy)
            else:
                min_shape.append(dim)
                max_shape.append(dim)
                shape_expr.append(dim)
        logger.debug("Final MIN ", min_shape)
        logger.debug("Final MAX ", max_shape)
        logger.debug("Shape ", shape_expr)
        return min_shape, max_shape, shape_expr

    min_max_shapes = []
    mark_dynamic = is_mark_dynamic(inputs)
    with unset_fake_temporarily():
        input_idx = 0
        for input_node in graph_module.graph.nodes:
            if input_node.op == "placeholder":
                logger.debug("Name ", input_node.name)
                stack_input = inputs[input_idx]
                if input_node.meta:
                    if "val" in input_node.meta:
                        input_meta = input_node.meta["val"]
                        if isinstance(input_meta, (FakeTensor, torch.Tensor)):
                            input_shape = input_meta.size()
                            logger.debug(f"Getting Min/Max for Tensor {input_node.name}")
                            min, max, expr = get_input(input_shape)
                            expr_strides = [item for t in input_node.meta["output_strides"] for item in t]
                            range_info = RangeInfo(min, max, str(expr), str(expr_strides), input_idx)
                            min_max_shapes.append(range_info)
                        elif isinstance(input_meta, torch.SymInt) or isinstance(input_meta, int):
                            input_shape = [input_meta]
                            logger.debug(f"Getting Min/Max for Symbol {input_node.name}")
                            min, max, expr = get_input(input_shape)
                            range_info = RangeInfo(min, max, str(expr), "INVALID", input_idx)
                            min_max_shapes.append(range_info)
                        else:
                            logger.debug(
                                f"WARN: The meta val for input node {input_node.target} is of type : {type(input_meta)}. Supported types: torch.Tensor|FakeTensor|torch.SymInt|torch.Int"
                            )
                    elif "tensor_meta" in input_node.meta:
                        input_meta = input_node.meta["tensor_meta"]
                        input_shape = input_meta.shape
                        min, max, expr = get_input(input_shape)
                        range_info = RangeInfo(min, max, str(expr), "INVALID", input_idx)
                        min_max_shapes.append(range_info)
                    else:
                        shape = list(stack_input.size())
                        range_info = RangeInfo(shape, shape, "INVALID", "INVALID", input_idx)
                        min_max_shapes.append(range_info)
                        logger.debug(
                            f"WARN: Input does not contain val and tensor_meta fields in the metadata={input_node.meta}. Filling {shape} as min max. Please ensure you have exported the graph correctly"
                        )
                else:
                    shape = list(stack_input.size())
                    range_info = RangeInfo(shape, shape, "INVALID", "INVALID", input_idx)
                    min_max_shapes.append(range_info)
                    logger.debug(
                        f"WARN: Input {input_node.name} does not contain metadata.  Filling {shape} as min max. Please ensure you have exported the graph correctly"
                    )
                input_idx = input_idx + 1
    assert len(inputs) == len(min_max_shapes)

    return min_max_shapes, mark_dynamic


class HabanaGraphModule(torch.nn.Module):
    def __init__(
        self,
        jit_ir,
        graph_module,
        parent_graph_name,
        outputs_metadata,
        symbolic_metadata,
        pholder_symbolic_dict,
        is_training=False,
        dynamic=False,
        force_static_compile=False,
    ):
        from ._recipe_compiler_C import EmptyBatchData, RangeInfo

        logger.debug("Creating HabanaGraphModule")
        super().__init__()
        self._name = f"{parent_graph_name}_{repr(graph_module)}"[:-2]
        self._jit_ir = jit_ir
        self._fx_module = graph_module
        self._in_to_out_dups = graph_module.meta.get("in_to_out_dups", None)
        self._outputs_metadata = outputs_metadata.copy()
        self._pholder_symbolic_dict = pholder_symbolic_dict
        self._range_list = []
        self._inference = not is_training
        self._recipe_id = None
        self._dynamic = dynamic
        self._mark_dynamic = False
        self._force_static_compile = force_static_compile
        self._symbol_evaluator = SymbolicShapeEvaluator(symbolic_metadata)
        self._has_randoms = False
        self._ds_output_prealloc = self._dynamic and enable_dynamic_output_preallocate
        self._outputs_batch_data = []
        self._hash_key = 0
        if self._ds_output_prealloc:
            for md in self._outputs_metadata:
                self._outputs_batch_data.append(EmptyBatchData((), md[1], md[2]))
        else:
            for md in self._outputs_metadata:
                self._outputs_batch_data.append(EmptyBatchData(md[0], md[1], md[2]))

        # We won't allocate tensors for outputs who duplicate inputs
        if self._in_to_out_dups is not None:
            self._out_to_in_dups = {v: k for k, v in self._in_to_out_dups.items()}
            for idx in self._out_to_in_dups.keys():
                self._outputs_batch_data.remove(self._outputs_batch_data[idx])
                self._outputs_metadata.remove(self._outputs_metadata[idx])

    @property
    def fx_module(self):
        return self._fx_module

    @property
    def is_dynamic(self):
        return self._dynamic

    @property
    def name(self):
        return self._name

    @property
    def is_inference(self):
        return self._inference

    @property
    def graph_str_repr_with_source_info(self):
        return self._jit_ir.str(print_source_info=True)

    def __call__(self, *args):
        outputs = []
        inputs = tuple(args)

        from ._recipe_compiler_C import RangeInfo, batch_empty, calculate_hash_code, graph_compile, graph_launch

        if self._ds_output_prealloc:
            self._symbol_evaluator.clear_symbolic_value_dict()
            for output, metadata in zip(self._outputs_batch_data, self._outputs_metadata):
                size = self._symbol_evaluator.calculate_shape(metadata[0], inputs)
                output.size = size

        outputs = batch_empty(self._outputs_batch_data)

        # If dynamic recipe compilation is disabled
        # recompilation will happen everytime
        # except for static cache hit scenario
        if self._force_static_compile:
            new_hash_key = calculate_hash_code(inputs)
            if self._hash_key != new_hash_key:
                self._hash_key = new_hash_key
                self._recipe_id = None

        if self._recipe_id is None:
            self._range_list, self._mark_dynamic = get_input_symbolic(self._fx_module, inputs)
            self.check_for_random_ops()
            if self._has_randoms:
                inputs = (None, None) + inputs
                self._range_list.insert(0, RangeInfo([1], [1], "1", "1", 0))
                self._range_list.insert(1, RangeInfo([1], [1], "1", "1", 1))

            self._recipe_id = graph_compile(
                graph=self._jit_ir.graph,
                inputs=inputs,
                dynamic=self._dynamic,
                inference=self._inference,
                has_preallocated_outputs=bool(outputs),
                has_randoms=self._has_randoms,
                in_symbol_idx_map=self._pholder_symbolic_dict,
                range_infos=self._range_list,
                mark_dynamic=self._mark_dynamic,
            )
            dump_fx_graph(self._fx_module, self._jit_ir.graph, self._recipe_id)
        elif self._has_randoms:
            inputs = (None, None) + inputs

        out_stack = graph_launch(
            recipe_id=self._recipe_id,
            inputs=inputs,
            outputs=outputs,
        )

        # insert the inputs into the out stack
        if self._in_to_out_dups is not None:
            out_stack = (
                list(out_stack) if type(out_stack) == tuple else ([out_stack] if out_stack is not None else list())
            )
            out_idxes = list(self._out_to_in_dups.keys())
            for out_idx in out_idxes:
                out_stack.insert(out_idx, args[self._out_to_in_dups[out_idx]])
            out_stack = tuple(out_stack) if len(out_stack) > 1 else out_stack[0]

        return out_stack

    def check_for_random_ops(self):
        for n in self._fx_module.graph.nodes:
            if is_random_op(n):
                self._has_randoms = True
                return


def get_callable_recipe(
    jit_ir, graph_module: torch.fx.GraphModule, parent_graph_name, is_training=False, is_dynamic=False
):
    """
    Calls backend to create compiled recipe or just returns unchanged module to
    run it eagerly depending on config.
    """
    outputs_metadata = []
    symbolic_metadata = {}
    pholder_symbolic_dict = {}
    if not is_dynamic:
        outputs_metadata = get_outputs_metadata(graph_module)
    elif is_dynamic and enable_dynamic_output_preallocate:
        outputs_metadata = get_outputs_metadata_dynamic(graph_module)
        symbolic_metadata, pholder_symbolic_dict = get_symbolic_metadata(graph_module, outputs_metadata)

    if hpu_backend_config.use_compiled_recipes:
        return HabanaGraphModule(
            jit_ir,
            graph_module,
            parent_graph_name,
            outputs_metadata,
            symbolic_metadata,
            pholder_symbolic_dict,
            is_training=is_training,
            dynamic=is_dynamic,
            force_static_compile=hpu_backend_config.force_static_compile,
        )
    else:
        # Return unchanged module, it will be ran eagerly.
        return graph_module


def get_symbolic_metadata(graph_module, outputs_metadata):
    """
    Return metadata of symbolic variables in the graph input

    symbolic_meta:
        data (dict): A dictionary to store symbol information.
        data format: {symbol: (tensor_index, tensor_dimension, (sub symbols))}

        Add a symbol with its associated tensor index and dimension or sub symbols to
        look at launch time for the current size. Valid sub symbol is inserted when
        the full expression is not directly part of any of the input size.
    """
    input_symbolic_dict = {}
    pholder_symbolic_dict = {}
    input_index = 0
    pexpr = PythonPrinter().doprint
    for node in graph_module.graph.nodes:
        if node.op == "placeholder":
            tmeta_val = node.meta.get("val", node.meta.get("tensor_meta", None))
            if isinstance(tmeta_val, py_sym_types):
                val_str = pexpr(tmeta_val)
                input_symbolic_dict[val_str] = (input_index, sys.maxsize)
                pholder_symbolic_dict[val_str] = input_index
            elif type(tmeta_val) is torch._subclasses.FakeTensor:
                shape = node.meta["output_shapes"][0]
                for dim, sz in enumerate(shape):
                    sz_str = pexpr(sz)
                    if isinstance(sz, torch.SymInt) and sz_str not in input_symbolic_dict:
                        input_symbolic_dict[sz_str] = (input_index, dim)
            else:
                logger.debug(
                    "Graph input node type not inserted to input_symbolic_dict!!!:",
                    tmeta_val,
                )
            input_index += 1

    symbolic_meta = {}
    for md in outputs_metadata:
        out_shape_meta = md[0]
        for idx, sz_sympy in enumerate(out_shape_meta[0]):
            if isinstance(sz_sympy, sympy.Expr):
                sz_str = out_shape_meta[1][idx]
                if sz_str in input_symbolic_dict:
                    symbolic_meta[sz_str] = (
                        input_symbolic_dict[sz_str][0],
                        input_symbolic_dict[sz_str][1],
                        (),
                    )
                else:
                    assert sz_sympy.free_symbols is not None
                    symbolic_meta[sz_str] = (
                        sys.maxsize,
                        sys.maxsize,
                        sz_sympy.free_symbols,
                    )
                    for sym in sz_sympy.free_symbols:
                        sym_str = pexpr(sym)
                        assert sym_str in input_symbolic_dict
                        symbolic_meta[sym_str] = (
                            input_symbolic_dict[sym_str][0],
                            input_symbolic_dict[sym_str][1],
                            (),
                        )
    return symbolic_meta, pholder_symbolic_dict


def get_outputs_metadata(graph_module):
    """
    Returns a list of metadata of outputs from the graph, in the form of
    tuples(shape, dtype), in the order in which they appear in the graph.
    """
    outputs_metadata = []
    for node in graph_module.graph.nodes:
        if node.op == "output":
            for i in node.all_input_nodes:
                assert len(i.meta["output_shapes"]) == len(i.meta["output_dtypes"])
                for shape, dtype, strides in zip(
                    i.meta["output_shapes"],
                    i.meta["output_dtypes"],
                    (
                        [None]
                        if "output_strides_has_zero" not in i.meta or not i.meta["output_strides_has_zero"]
                        else i.meta["output_strides"]
                    ),
                ):
                    outputs_metadata.append((shape, dtype, strides))

    return outputs_metadata


def get_outputs_metadata_dynamic(graph_module):
    """
    Returns a list of metadata of outputs from the graph, in the form of
    tuples((sympy shape expr of each dims,
            string expr of each dims,
            token number of the expr of each dims,
            total number of dims), dtype),
    in the order in which they appear in the graph.
    """
    outputs_metadata = []
    sym_expr_list = []
    for node in graph_module.graph.nodes:
        if node.op == "output":
            for i in node.all_input_nodes:
                assert len(i.meta["output_shapes"]) == len(i.meta["output_dtypes"])
                for shape, dtype, strides in zip(
                    i.meta["output_shapes"],
                    i.meta["output_dtypes"],
                    (
                        [None]
                        if "output_strides_has_zero" not in i.meta or not i.meta["output_strides_has_zero"]
                        else i.meta["output_strides"]
                    ),
                ):
                    dynamic_shape_sympy = []
                    dynamic_shape_sym_expr_token = []
                    dynamic_shape_str = []
                    for sz in shape:
                        if isinstance(sz, int):
                            dynamic_shape_sympy.append(sz)
                            dynamic_shape_sym_expr_token.append(sys.maxsize)
                            dynamic_shape_str.append(sz)
                        elif isinstance(sz, torch.SymInt):
                            pexpr = PythonPrinter().doprint
                            sz_str = pexpr(sz)
                            sz_sympy = sympify(sz_str)
                            sz_sympy = substitute_sympyfn(sz_sympy)
                            dynamic_shape_sympy.append(sz_sympy)
                            dynamic_shape_str.append(sz_str)
                            if not sz_str in sym_expr_list:
                                sym_expr_list.append(sz_str)

                            sym_expr_token = sym_expr_list.index(sz_str)
                            dynamic_shape_sym_expr_token.append(sym_expr_token)
                        else:
                            logger.debug("Symbolic type not supported:", sz)
                            assert False

                    dim_size = len(dynamic_shape_sympy)
                    outputs_metadata.append(
                        (
                            (dynamic_shape_sympy, dynamic_shape_str, dynamic_shape_sym_expr_token, dim_size),
                            dtype,
                            strides,
                        )
                    )

    return outputs_metadata
