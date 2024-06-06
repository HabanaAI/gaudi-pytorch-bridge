###############################################################################
# Copyright (C) 2023-2024 Habana Labs, Ltd. an Intel Company
# All Rights Reserved.
#
# Unauthorized copying of this file or any element(s) within it, via any medium
# is strictly prohibited.
# This file contains Habana Labs, Ltd. proprietary and confidential information
# and is subject to the confidentiality and license agreements under which it
# was provided.
#
###############################################################################

import logging
import os
import sys

import habana_frameworks.torch.internal.bridge_config as bc
import sympy
import torch
from habana_frameworks.torch.dynamo.compile_backend import config as hpu_backend_config
from sympy import sympify
from torch.fx.experimental.proxy_tensor import py_sym_types

from .logger import dump_fx_graph, get_compile_backend_logger
from .random_utils import is_random_op
from .symbolic_execution import PythonPrinter, SymbolicShapeEvaluator

logger = get_compile_backend_logger()
enable_dynamic_output_preallocate = bc.get_pt_hpu_enable_dynamic_output_preallocate()


class HabanaGraphModule(torch.nn.Module):
    def __init__(
        self,
        jit_ir,
        graph_module,
        outputs_metadata,
        symbolic_metadata,
        pholder_symbolic_dict,
        is_training=False,
        dynamic=False,
    ):
        from ._recipe_compiler_C import EmptyBatchData

        logger.debug("Creating HabanaGraphModule")
        super().__init__()
        self._jit_ir = jit_ir
        self._fx_module = graph_module
        self._outputs_metadata = outputs_metadata
        self._pholder_symbolic_dict = pholder_symbolic_dict
        self._inference = not is_training
        self._recipe_id = None
        self._dynamic = dynamic
        self._symbol_evaluator = SymbolicShapeEvaluator(symbolic_metadata)
        self._has_randoms = False
        self._ds_output_prealloc = self._dynamic and enable_dynamic_output_preallocate
        self._outputs_batch_data = []
        if self._ds_output_prealloc:
            for md in self._outputs_metadata:
                self._outputs_batch_data.append(EmptyBatchData((), md[1], md[2]))
        else:
            for md in self._outputs_metadata:
                self._outputs_batch_data.append(EmptyBatchData(md[0], md[1], md[2]))

    def __call__(self, *args):
        outputs = []
        inputs = tuple(args)

        from ._recipe_compiler_C import batch_empty, graph_compile, graph_launch

        if self._ds_output_prealloc:
            self._symbol_evaluator.clear_symbolic_value_dict()
            for output, metadata in zip(self._outputs_batch_data, self._outputs_metadata):
                size = self._symbol_evaluator.calculate_shape(metadata[0], inputs)
                output.size = size

        outputs = batch_empty(self._outputs_batch_data)

        if self._recipe_id is None:
            self.check_for_random_ops()
            if self._has_randoms:
                inputs = (None, None) + inputs
            self._recipe_id = graph_compile(
                graph=self._jit_ir.graph,
                inputs=inputs,
                dynamic=self._dynamic,
                inference=self._inference,
                has_preallocated_outputs=bool(outputs),
                has_randoms=self._has_randoms,
                in_symbol_idx_map=self._pholder_symbolic_dict,
            )
            dump_fx_graph(self._fx_module, self._jit_ir.graph, self._recipe_id)
        elif self._has_randoms:
            inputs = (None, None) + inputs

        return graph_launch(
            recipe_id=self._recipe_id,
            inputs=inputs,
            outputs=outputs,
        )

    def check_for_random_ops(self):
        for n in self._fx_module.graph.nodes:
            if is_random_op(n):
                self._has_randoms = True
                return


def get_callable_recipe(jit_ir, graph_module: torch.fx.GraphModule, is_training=False, is_dynamic=False):
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
            outputs_metadata,
            symbolic_metadata,
            pholder_symbolic_dict,
            is_training=is_training,
            dynamic=is_dynamic,
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
