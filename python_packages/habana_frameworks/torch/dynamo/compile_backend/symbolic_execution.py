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

import copy
import sys
import sympy
import torch

from sympy import sympify
from sympy.printing.printer import Printer
from .logger import get_compile_backend_logger

logger = get_compile_backend_logger()

class CSEVariable:
    """A CSEVariable is just a name for an expression but it is useful to be able to annotate them on a backend dependent basis
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

class SymExprNodeManager():
    node_name = "symexpr_py"
    def __init__(self, graph_module: torch.fx.GraphModule):
        self._graph_module = graph_module
        self._sym_expr_to_node_map = {}
        self._sym_placeholder_dict = {}
        self._insert_point_node = None

    def _create_symexpr_py_node(
        self,
        sympy_expr,
        sympy_symbols,
        py_node_args,
        node_type):

        def symexpr_python(*arguments, sympy_expr=copy.deepcopy(sympy_expr),
                            expr_symbols=copy.deepcopy(sympy_symbols)):
            sym_value_pair = []
            for idx, sub_sym in enumerate(expr_symbols):
                value = arguments[idx]
                sym_value_pair.append((sub_sym, value))
            size = sympy_expr.subs(sym_value_pair)
            return int(size)

        node_name = SymExprNodeManager.node_name
        with self._graph_module.graph.inserting_after(self._insert_point_node):
            new_kwargs = None
            new_node = self._graph_module.graph.create_node(
                "call_function",
                symexpr_python,
                tuple(py_node_args),
                new_kwargs,
                node_name,
                node_type
            )
            return new_node

    def add_sym_placeholder(self, meta_val, node):
        sym_str = PythonPrinter().doprint(meta_val)
        self._sym_placeholder_dict[sym_str] = node

    def set_insert_point(self, node):
        self._insert_point_node = node

    def get_match_sym_placeholder(self, sym_size_expr):
        pexpr = PythonPrinter().doprint
        sym_expr_str = pexpr(sym_size_expr)
        matched_node = None
        if sym_expr_str in self._sym_placeholder_dict:
            matched_node = self._sym_placeholder_dict[sym_expr_str]
        return matched_node

    def get_or_create(self, sym_size_expr, node_type):
        pexpr = PythonPrinter().doprint
        sym_expr_str = pexpr(sym_size_expr)
        if sym_expr_str in self._sym_expr_to_node_map:
            new_node = self._sym_expr_to_node_map[sym_expr_str]
        else:
            sympy_expr = sympify(sym_expr_str)
            sympy_expr_symbols = sympy_expr.free_symbols
            node_args = []
            for sym in sympy_expr_symbols:
                node_args.append(self._sym_placeholder_dict[pexpr(sym)])
            logger.debug("symexpr_python call_function creating for expr:", sym_expr_str)
            new_node = self._create_symexpr_py_node(sympy_expr, sympy_expr_symbols,
                                                    node_args, node_type)
            self._sym_expr_to_node_map[sym_expr_str] = new_node

        return new_node

class SymbolicShapeEvaluator:
    def __init__(self, symbolic_metadata):
        self._symbolic_value_dict = {}
        self._symbolic_metadata = symbolic_metadata

    def clear_symbolic_value_dict(self):
        self._symbolic_value_dict = {}

    def calculate_symbol_size(self, expr_sympy, expr_str, input_stack):
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

        size = 0
        sym_meta = self._symbolic_metadata[expr_str]
        if expr_str in self._symbolic_value_dict:
            return self._symbolic_value_dict[expr_str]
        elif sym_meta[0] is not sys.maxsize:
            size = get_symbolic_value(sym_meta, input_stack)
        else:
            pexpr = PythonPrinter().doprint
            free_symbols = sym_meta[2]
            sym_value_pair = []
            for sub_sym in free_symbols:
                sub_sym_str = pexpr(sub_sym)
                sub_sym_meta = self._symbolic_metadata[sub_sym_str]
                value = get_symbolic_value(sub_sym_meta, input_stack)
                sym_value_pair.append((sub_sym, value))
            size = expr_sympy.subs(sym_value_pair)

        self._symbolic_value_dict[expr_str] = size
        return size

    def calculate_shape(self, out_shape_meta, input_stack):
        """
        Return the concrete size after evaluating the symbolic expression.
        """
        concrete_size = []
        for idx, sz in enumerate(out_shape_meta[0]):
            if isinstance(sz, sympy.Expr):
                value = self.calculate_symbol_size(sz, out_shape_meta[1][idx], input_stack)
                concrete_size.append(value)
            else:
                concrete_size.append(sz)
        return concrete_size
