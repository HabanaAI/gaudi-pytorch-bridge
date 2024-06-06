###############################################################################
# Copyright (C) 2024 Habana Labs, Ltd. an Intel Company
# All Rights Reserved.
#
# Unauthorized copying of this file or any element(s) within it, via any medium
# is strictly prohibited.
# This file contains Habana Labs, Ltd. proprietary and confidential information
# and is subject to the confidentiality and license agreements under which it
# was provided.
#
###############################################################################
import pytest
import torch
from habana_frameworks.torch.utils.debug.dynamo_utils import FxGraphAnalyzer
from test_dynamo_utils import assert_helper
from test_utils import check_ops_executed_in_jit_ir, clear_t_compile_logs, format_tc


class TestBase:

    @staticmethod
    def run_hpu_compile(fn, *args):
        clear_t_compile_logs()

        compiled_fn = torch.compile(fn, backend="hpu_backend")
        return compiled_fn(*args)

    @staticmethod
    def check_ops_in_graph(ops_summary, fx_graph_ops, jit_ir_ops):
        for op, count_list in fx_graph_ops:
            assert_helper(ops_summary=ops_summary, op=op, count_list=count_list)
        check_ops_executed_in_jit_ir(jit_ir_ops)

    @staticmethod
    def run_test(fx_graph_ops, jit_ir_ops, fn, *args):
        with FxGraphAnalyzer(reset_dynamo=True) as fga:
            result = TestBase.run_hpu_compile(fn, *args)
        ops_summary = fga.get_ops_summary()
        TestBase.check_ops_in_graph(ops_summary, fx_graph_ops, jit_ir_ops)
        return result


class Test_single_value_P_for_bernoulli_pattern:

    class Test_bernoulli_default:

        @staticmethod
        def run_test(fn, *args):
            return TestBase.run_test(
                [("torch.ops.hpu.habana_bernoulli.Size", [(1, 0)])],
                {"habana_bernoulli_seed"},
                fn,
                *args,
            )

        @pytest.mark.parametrize("size", [[], [2, 3]], ids=format_tc)
        @pytest.mark.parametrize("p", [0.3], ids=format_tc)
        def test_full_bernoulli_method_case_optimized_out(self, size, p):

            def full_bernoulli_method(size, p):
                full = torch.full(size, p, device="hpu")
                return full.bernoulli()

            actual = self.run_test(
                full_bernoulli_method,
                size,
                p,
            ).to("cpu")

            assert list(actual.shape) == size

        @pytest.mark.parametrize("size", [[], [2, 3]], ids=format_tc)
        @pytest.mark.parametrize("p", [0.3], ids=format_tc)
        def test_fill_bernoulli_method_case_optimized_out(self, size, p):

            def fill_bernoulli_method(size, p):
                fill = torch.empty(size, device="hpu").fill_(p)
                return fill.bernoulli()

            actual = self.run_test(
                fill_bernoulli_method,
                size,
                p,
            ).to("cpu")

            assert list(actual.shape) == size

        @pytest.mark.parametrize("size", [[2], [2, 3, 4]], ids=format_tc)
        @pytest.mark.parametrize("p", [0.7], ids=format_tc)
        def test_full_bernoulli_function_case_optimized_out(self, size, p):

            def full_bernoulli_function(size, p):
                full = torch.full(size, p, device="hpu")
                return torch.bernoulli(full)

            actual = self.run_test(
                full_bernoulli_function,
                size,
                p,
            ).to("cpu")

            assert list(actual.shape) == size

        @pytest.mark.parametrize("size", [[2], [2, 3, 4]], ids=format_tc)
        @pytest.mark.parametrize("p", [0.7], ids=format_tc)
        def test_fill_bernoulli_function_case_optimized_out(self, size, p):

            def fill_bernoulli_function(size, p):
                fill = torch.empty(size, device="hpu").fill_(p)
                return torch.bernoulli(fill)

            actual = self.run_test(
                fill_bernoulli_function,
                size,
                p,
            ).to("cpu")

            assert list(actual.shape) == size

        @pytest.mark.parametrize("init_size, size", [([], [2, 4]), ([3, 4, 5], [2, 3, 4, 5])], ids=format_tc)
        @pytest.mark.parametrize("p", [0.3], ids=format_tc)
        def test_full_expand_bernoulli_method_case_optimized_out(self, init_size, p, size):

            def full_expand_bernoulli_method(init_size, p, size):
                full = torch.full(init_size, p, device="hpu")
                expand = full.expand(size)
                return expand.bernoulli()

            actual = self.run_test(
                full_expand_bernoulli_method,
                init_size,
                p,
                size,
            ).to("cpu")

            assert list(actual.shape) == size

        @pytest.mark.parametrize("init_size, size", [([], [2, 4]), ([3, 4, 5], [2, 3, 4, 5])], ids=format_tc)
        @pytest.mark.parametrize("p", [0.3], ids=format_tc)
        def test_fill_expand_bernoulli_method_case_optimized_out(self, init_size, p, size):

            def fill_expand_bernoulli_method(init_size, p, size):
                fill = torch.empty(init_size, device="hpu").fill_(p)
                expand = fill.expand(size)
                return expand.bernoulli()

            actual = self.run_test(
                fill_expand_bernoulli_method,
                init_size,
                p,
                size,
            ).to("cpu")

            assert list(actual.shape) == size

        @pytest.mark.parametrize(
            "init_size, size", [([3, 4], [2, 3, 4]), ([3, 4, 5, 6], [2, 3, 4, 5, 6])], ids=format_tc
        )
        @pytest.mark.parametrize("p", [0.7], ids=format_tc)
        def test_full_expand_bernoulli_function_case_optimized_out(self, init_size, p, size):

            def full_expand_bernoulli_function(init_size, p, size):
                full = torch.full(init_size, p, device="hpu")
                expand = full.expand(size)
                return torch.bernoulli(expand)

            actual = self.run_test(
                full_expand_bernoulli_function,
                init_size,
                p,
                size,
            ).to("cpu")

            assert list(actual.shape) == size

        @pytest.mark.parametrize(
            "init_size, size", [([3, 4], [2, 3, 4]), ([3, 4, 5, 6], [2, 3, 4, 5, 6])], ids=format_tc
        )
        @pytest.mark.parametrize("p", [0.7], ids=format_tc)
        def test_fill_expand_bernoulli_function_case_optimized_out(self, init_size, p, size):

            def fill_expand_bernoulli_function(init_size, p, size):
                fill = torch.empty(init_size, device="hpu").fill_(p)
                expand = fill.expand(size)
                return torch.bernoulli(expand)

            actual = self.run_test(
                fill_expand_bernoulli_function,
                init_size,
                p,
                size,
            ).to("cpu")

            assert list(actual.shape) == size

        @pytest.mark.parametrize("size", [[], [2, 3, 4]], ids=format_tc)
        @pytest.mark.parametrize("p", [0.4], ids=format_tc)
        def test_full_like_bernoulli_method_case_optimized_out(self, size, p):

            def full_like_bernoulli_method(size, p):
                tensor = torch.empty(size, device="hpu")
                full_like = torch.full_like(tensor, p)
                return full_like.bernoulli()

            actual = self.run_test(
                full_like_bernoulli_method,
                size,
                p,
            ).to("cpu")

            assert list(actual.shape) == size

        @pytest.mark.parametrize("size", [[3, 2], [5, 4, 3, 2]], ids=format_tc)
        @pytest.mark.parametrize("p", [0.6], ids=format_tc)
        def test_full_like_bernoulli_function_case_optimized_out(self, size, p):

            def full_like_bernoulli_function(size, p):
                tensor = torch.empty(size, device="hpu")
                full_like = torch.full_like(tensor, p)
                return torch.bernoulli(full_like)

            actual = self.run_test(
                full_like_bernoulli_function,
                size,
                p,
            ).to("cpu")

            assert list(actual.shape) == size

        @pytest.mark.parametrize("init_size, size", [([], [2, 4]), ([3, 4, 5], [2, 3, 4, 5])], ids=format_tc)
        @pytest.mark.parametrize("p", [0.3], ids=format_tc)
        def test_full_like_expand_bernoulli_method_case_optimized_out(self, init_size, p, size):

            def full_like_expand_bernoulli_method(init_size, p, size):
                tensor = torch.empty(init_size, device="hpu")
                full_like = torch.full_like(tensor, p)
                expand = full_like.expand(size)
                return expand.bernoulli()

            actual = self.run_test(
                full_like_expand_bernoulli_method,
                init_size,
                p,
                size,
            ).to("cpu")

            assert list(actual.shape) == size

        @pytest.mark.parametrize(
            "init_size, size", [([3, 4], [2, 3, 4]), ([3, 4, 5, 6], [2, 3, 4, 5, 6])], ids=format_tc
        )
        @pytest.mark.parametrize("p", [0.7], ids=format_tc)
        def test_full_like_expand_bernoulli_function_case_optimized_out(self, init_size, p, size):

            def full_like_expand_bernoulli_function(init_size, p, size):
                tensor = torch.empty(init_size, device="hpu")
                full_like = torch.full_like(tensor, p)
                expand = full_like.expand(size)
                return torch.bernoulli(expand)

            actual = self.run_test(
                full_like_expand_bernoulli_function,
                init_size,
                p,
                size,
            ).to("cpu")

            assert list(actual.shape) == size

    class Test_bernoulli_Tensor:

        @staticmethod
        def run_test(fn, *args):
            return TestBase.run_test(
                [("torch.ops.aten.bernoulli.p", [(1, 0)])],
                {"habana_bernoulli_seed"},
                fn,
                *args,
            )

        @pytest.mark.parametrize("size", [[], [2, 3, 4]], ids=format_tc)
        @pytest.mark.parametrize("p", [0.3], ids=format_tc)
        def test_full_bernoulli_inplace_case_optimized_out(self, size, p):

            def full(size, p):
                full = torch.full(size, p, device="hpu")
                bernoulli = torch.empty(size, device="hpu")
                return bernoulli.bernoulli_(full)

            actual = self.run_test(
                full,
                size,
                p,
            ).to("cpu")

            assert list(actual.shape) == size

        @pytest.mark.parametrize("size", [[], [2, 3, 4]], ids=format_tc)
        @pytest.mark.parametrize("p", [0.3], ids=format_tc)
        def test_fill_bernoulli_inplace_case_optimized_out(self, size, p):

            def fill(size, p):
                fill = torch.empty(size, device="hpu").fill_(p)
                bernoulli = torch.empty(size, device="hpu")
                return bernoulli.bernoulli_(fill)

            actual = self.run_test(
                fill,
                size,
                p,
            ).to("cpu")

            assert list(actual.shape) == size

        @pytest.mark.parametrize("init_size, size", [([], [2, 4]), ([3, 4, 5], [2, 3, 4, 5])], ids=format_tc)
        @pytest.mark.parametrize("p", [0.7], ids=format_tc)
        def test_full_expand_bernoulli_inplace_case_optimized_out(self, init_size, p, size):

            def full_expand(init_size, p, size):
                full = torch.full(init_size, p, device="hpu")
                expand = full.expand(size)
                bernoulli = torch.empty(size, device="hpu")
                return bernoulli.bernoulli_(expand)

            actual = self.run_test(
                full_expand,
                init_size,
                p,
                size,
            ).to("cpu")

            assert list(actual.shape) == size

        @pytest.mark.parametrize("init_size, size", [([], [2, 4]), ([3, 4, 5], [2, 3, 4, 5])], ids=format_tc)
        @pytest.mark.parametrize("p", [0.7], ids=format_tc)
        def test_fill_expand_bernoulli_inplace_case_optimized_out(self, init_size, p, size):

            def fill_expand(init_size, p, size):
                fill = torch.empty(init_size, device="hpu").fill_(p)
                expand = fill.expand(size)
                bernoulli = torch.empty(size, device="hpu")
                return bernoulli.bernoulli_(expand)

            actual = self.run_test(
                fill_expand,
                init_size,
                p,
                size,
            ).to("cpu")

            assert list(actual.shape) == size

    class Test_bernoulli_misc:

        @pytest.mark.parametrize("size", [(2, 3, 4, 5)], ids=format_tc)
        @pytest.mark.parametrize("p", [0.3], ids=format_tc)
        def test_full_has_two_users_no_matcher_replacement_takes_place(self, size, p):

            def full_used_twice(size, p):
                full = torch.full(size, p, device="hpu")
                bernoulli1 = torch.empty(size, device="hpu")
                bernoulli1.bernoulli_(full)
                bernoulli2 = torch.bernoulli(full)
                return bernoulli1, bernoulli2

            TestBase.run_test(
                [
                    ("torch.ops.aten.full.default", [(1, 0)]),
                    ("torch.ops.aten.bernoulli.default", [(1, 0)]),
                    ("torch.ops.aten.bernoulli.Tensor", [(1, 0)]),
                ],
                {
                    "habana_bernoulli_seed",
                },
                full_used_twice,
                size,
                p,
            )

        @pytest.mark.parametrize("init_size", [[3, 4]], ids=format_tc)
        @pytest.mark.parametrize("p", [0.4], ids=format_tc)
        @pytest.mark.parametrize("size", [(2, 3, 4)], ids=format_tc)
        def test_full_expand_has_two_users_no_matcher_replacement_takes_place(self, init_size, p, size):

            def full_expand_used_twice(init_size, p, size):
                full = torch.full(init_size, p, device="hpu")
                expand = full.expand(size)
                bernoulli1 = torch.empty(size, device="hpu")
                bernoulli1.bernoulli_(expand)
                bernoulli2 = torch.bernoulli(expand)
                return bernoulli1, bernoulli2

            TestBase.run_test(
                [
                    ("torch.ops.aten.full.default", [(1, 0)]),
                    ("torch.ops.aten.expand.default", [(1, 0)]),
                    ("torch.ops.aten.bernoulli.default", [(1, 0)]),
                    ("torch.ops.aten.bernoulli.Tensor", [(1, 0)]),
                ],
                {"habana_bernoulli_seed"},
                full_expand_used_twice,
                init_size,
                p,
                size,
            )

        @pytest.mark.parametrize("size", [(2, 3, 4)], ids=format_tc)
        @pytest.mark.parametrize("p", [0.3], ids=format_tc)
        def test_full_requires_grad_no_matcher_replacement_takes_place(self, size, p):

            def full_requires_grad(size, p):
                full = torch.full(size, p, device="hpu", requires_grad=True)
                bernoulli = torch.empty(size, device="hpu")
                bernoulli.bernoulli_(full)
                loss = bernoulli.sum()
                loss.backward()
                return full

            TestBase.run_test(
                [
                    ("torch.ops.aten.full.default", [None, (1, 0)]),
                    ("torch.ops.aten.bernoulli.Tensor", [(1, 0), None]),
                ],
                {"habana_bernoulli_seed"},
                full_requires_grad,
                size,
                p,
            )
