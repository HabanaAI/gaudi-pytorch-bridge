###############################################################################
# Copyright (C) 2023 Habana Labs, Ltd. an Intel Company
# All Rights Reserved.
#
# Unauthorized copying of this file or any element(s) within it, via any medium
# is strictly prohibited.
# This file contains Habana Labs, Ltd. proprietary and confidential information
# and is subject to the confidentiality and license agreements under which it
# was provided.
#
###############################################################################

import torch
import logging
import sys
import os
import habana_frameworks.torch.internal.bridge_config as bc

from .config import configuration_flags
from .logger import get_compile_backend_logger, dump_fx_graph

logger = get_compile_backend_logger()

from sympy.printing.printer import Printer
from sympy import sympify
from torch.fx.experimental.proxy_tensor import py_sym_types


enable_dynamic_output_preallocate = bc.get_pt_hpu_enable_dynamic_output_preallocate()

class CSEVariable:
    """A CSEVariable is just a name for an expression but it is useful to be able to annotate them on a backend dependent basis.
    The backends can inherit from this class and overload the "create_cse_var" Kernel to do that.
    The "update_on_args" method gives you a hook for annotations, see example of TritonCSEVariable in triton.py."""

    def __init__(self, name):
        self.name = name

    def __str__(self):
        return self.name

    def __hash__(self) -> int:
        return hash(self.name)

    def __eq__(self, other) -> bool:
        return type(other) == type(self) and other.name == self.name

    def update_on_args(self, name, args, kwargs):
        pass

class ExprPrinter(Printer):
    @staticmethod
    def paren(string):
        if (
            isinstance(string, CSEVariable)
            or re.match(r"^[a-z0-9_.]+$", string, re.I)
            or re.match(r"^\([^)]*\)$", string, re.I)
            or string == ""
        ):
            return string
        return f"({string})"

    def _print_Pow(self, expr):
        # Pow() confuses triton
        base, exp = expr.args
        base = self._print(base)
        assert exp.is_integer
        exp = int(exp)
        if exp > 0:
            return "*".join([self.paren(base)] * exp)
        elif exp < 0:
            return "1/" + self.paren("*".join([self.paren(base)] * abs(exp)))
        else:  # exp == 0
            return "1"

    def _print_Mul(self, expr):
        return "*".join(map(self.paren, map(self._print, expr.args)))

    def _print_Add(self, expr):
        return " + ".join(map(self.paren, map(self._print, expr.args)))

    def _print_Mod(self, expr):
        return " % ".join(map(self.paren, map(self._print, expr.args)))

    def _print_CleanDiv(self, expr):
        return self._print_FloorDiv(expr)

class PythonPrinter(ExprPrinter):
    def _print_ModularIndexing(self, expr):
        x, div, mod = expr.args
        x = self.paren(self.doprint(x))
        div = self.paren(self.doprint(div))
        mod = self.paren(self.doprint(mod))
        if div != "1":
            x = f"({x} // {div})"
        return f"{x} % {mod}"

    def _print_FloorDiv(self, expr):
        x, div = expr.args
        x = self.paren(self.doprint(x))
        div = self.paren(self.doprint(div))
        return f"({x} // {div})"

    def _print_floor(self, expr):
        assert len(expr.args) == 1
        return f"math.floor({self.paren(self._print(expr.args[0]))})"

class SymbolicShapeEvaluator():
    def __init__(self, symbolic_metadata):
        self._symbolic_value_dict = {}
        self._symbolic_metadata = symbolic_metadata

    def clear_symbolic_value_dict(self):
        self._symbolic_value_dict = {}

    def calculate_symbol_size(self, sym_expr, input_stack):
        pexpr = PythonPrinter().doprint
        def get_symbolic_value(sym_meta, inputs):
            input_idx = sym_meta[0]
            dim = sym_meta[1]
            input = inputs[input_idx]
            value = 0
            if isinstance(input, int):
                value = input
            elif isinstance(input, torch.Tensor):
                value = input.shape[dim]
            else:
                assert False, "Wrong input type to look for dimention value"
            return value

        sym_expr_str = pexpr(sym_expr)
        size = 0
        sym_meta = self._symbolic_metadata[sym_expr_str]
        if sym_expr_str in self._symbolic_value_dict:
            return self._symbolic_value_dict[sym_expr_str]
        elif sym_meta[0] is not sys.maxsize:
            size = get_symbolic_value(sym_meta, input_stack)
        else:
            sympi_expr = sympify(sym_expr_str)
            free_symbols = sym_meta[2]
            sym_value_pair = []
            for sub_sym in free_symbols:
                sub_sym_str = pexpr(sub_sym)
                sub_sym_meta = self._symbolic_metadata[sub_sym_str]
                value = get_symbolic_value(sub_sym_meta, input_stack)
                sym_value_pair.append((sub_sym, value))
            size = sympi_expr.subs(sym_value_pair)

        self._symbolic_value_dict[sym_expr_str] = size
        return size

    def calculate_shape(self, sym_shape, input_stack):
        """
        Return the concrete size after evaluating the symbolic expression.
        """
        concrete_size = []
        for sz in sym_shape:
            if isinstance(sz, int):
                concrete_size.append(sz)
            elif isinstance(sz, torch.SymInt):
                value = self.calculate_symbol_size(sz, input_stack)
                concrete_size.append(value)
            else:
                logger.debug("Symbolic type not supported:", sz)
                assert False
        return concrete_size

class HabanaGraphModule(torch.nn.Module):
    def __init__(self, jit_ir, graph_module, outputs_metadata, symbolic_metadata, is_training=False, dynamic=False):
        logger.debug("Creating HabanaGraphModule")
        super().__init__()
        self._jit_ir = jit_ir
        self._fx_module = graph_module
        self._outputs_metadata = outputs_metadata
        self._inference = not is_training
        self._recipe_id = None
        self._dynamic = dynamic
        self._symbol_evaluator = SymbolicShapeEvaluator(symbolic_metadata)

    def __call__(self, *args):
        outputs = []
        inputs = tuple(args)
        if self._dynamic and enable_dynamic_output_preallocate:
            self._symbol_evaluator.clear_symbolic_value_dict()

        for md in self._outputs_metadata:
            size = md[0]
            if self._dynamic and enable_dynamic_output_preallocate:
                size = self._symbol_evaluator.calculate_shape(md[0], inputs)
            outputs.append(torch.empty(size, dtype=md[1], device="hpu"))

        from ._recipe_compiler_C import graph_compile, graph_launch

        if self._recipe_id is None:
            self._recipe_id = graph_compile(graph=self._jit_ir.graph, inputs=inputs,
                                            dynamic=self._dynamic, inference=self._inference,
                                            has_preallocated_outputs=bool(outputs))
            dump_fx_graph(self._fx_module, self._recipe_id)
        return graph_launch(recipe_id=self._recipe_id, inputs=inputs, outputs=outputs)


def get_callable_recipe(jit_ir, graph_module: torch.fx.GraphModule, is_training=False, is_dynamic=False):
    """
    Calls backend to create compiled recipe or just returns unchanged module to
    run it eagerly depending on config.
    """
    outputs_metadata = []
    symbolic_metadata = {}
    if not is_dynamic and ((os.getenv("PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES", "").upper() not in [
            "ON", "1", "YES", "TRUE", "Y"]) or enable_dynamic_output_preallocate):
        outputs_metadata = get_outputs_metadata(graph_module)
    elif is_dynamic and enable_dynamic_output_preallocate:
        outputs_metadata = get_outputs_metadata(graph_module)
        symbolic_metadata = get_symbolic_metadata(graph_module, outputs_metadata)

    if configuration_flags["use_compiled_recipes"]:
        return HabanaGraphModule(jit_ir, graph_module, outputs_metadata, symbolic_metadata,
                                 is_training=is_training, dynamic=is_dynamic)
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
    input_index = 0
    pexpr = PythonPrinter().doprint
    for node in graph_module.graph.nodes:
        if node.op is "placeholder":
            tmeta_val = node.meta.get('val', node.meta.get('tensor_meta', None))
            if isinstance(tmeta_val, py_sym_types):
                val_str = pexpr(tmeta_val)
                input_symbolic_dict[val_str] = (input_index, sys.maxsize)
            elif type(tmeta_val) is torch._subclasses.FakeTensor:
                shape = node.meta["output_shapes"][0]
                for dim, sz in enumerate(shape):
                    sz_str = pexpr(sz)
                    if isinstance(sz, torch.SymInt) and sz_str not in input_symbolic_dict:
                        input_symbolic_dict[sz_str] = (input_index, dim)
            else:
                logger.debug("Graph input node type not inserted to input_symbolic_dict!!!:", tmeta_val)
            input_index += 1

    symbolic_meta = {}
    for md in outputs_metadata:
        for sz in md[0]:
            if isinstance(sz, torch.SymInt):
                sym_sz_str = pexpr(sz)
                if sym_sz_str in input_symbolic_dict:
                    symbolic_meta[sym_sz_str] = (input_symbolic_dict[sym_sz_str][0],
                                                 input_symbolic_dict[sym_sz_str][1], ())
                else:
                    sym_sz = sympify(sym_sz_str)
                    assert sym_sz.free_symbols is not None
                    symbolic_meta[sym_sz_str] = (sys.maxsize, sys.maxsize, sym_sz.free_symbols)
                    for sym in sym_sz.free_symbols:
                        sym_str = pexpr(sym)
                        assert sym_str in input_symbolic_dict
                        symbolic_meta[sym_str] = (input_symbolic_dict[sym_str][0],
                                                     input_symbolic_dict[sym_str][1], ())
    return symbolic_meta

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
                for shape, dtype in zip(i.meta["output_shapes"], i.meta["output_dtypes"]):
                    outputs_metadata.append((shape, dtype))

    return outputs_metadata
