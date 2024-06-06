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
from test_utils import check_ops_executed_in_jit_ir, clear_t_compile_logs, format_tc, setup_teardown_env_fixture


class TestBase:

    @staticmethod
    def run_hpu_compile(fn, *args):
        clear_t_compile_logs()

        compiled_fn = torch.compile(fn, backend="hpu_backend")
        return compiled_fn(*args)

    @staticmethod
    def run_test(jit_ir_ops, fn, *args):
        actual = TestBase.run_hpu_compile(fn, *args)
        check_ops_executed_in_jit_ir(jit_ir_ops)
        return actual

    @staticmethod
    def is_binary(tensor):
        return torch.ne(tensor, 0).mul_(torch.ne(tensor, 1)).sum().item() == 0


class Test_bernoulli_function:

    @pytest.mark.parametrize("size", [[2, 3, 4]], ids=format_tc)
    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=format_tc)
    @pytest.mark.parametrize(
        "setup_teardown_env_fixture",
        [{"PT_HPU_USE_EAGER_FALLBACK": 0}],
        indirect=True,
    )
    def test_bernoulli_default(self, size, dtype, setup_teardown_env_fixture):

        def fn(size, dtype):
            p = torch.empty(size, dtype=dtype, device="hpu").uniform_(0, 1)
            return torch.bernoulli(p)

        actual = TestBase.run_test({"habana_bernoulli_seed"}, fn, size, dtype).to("cpu")
        assert TestBase.is_binary(actual)

    @pytest.mark.parametrize("size", [[2, 3, 4]], ids=format_tc)
    @pytest.mark.parametrize("p", [0.3], ids=format_tc)
    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=format_tc)
    @pytest.mark.parametrize(
        "setup_teardown_env_fixture",
        [{"PT_HPU_USE_EAGER_FALLBACK": 0}],
        indirect=True,
    )
    def test_bernoulli_with_scalar_p(self, size, p, dtype, setup_teardown_env_fixture):

        def fn(size, p, dtype):
            input = torch.empty(size, dtype=dtype, device="hpu").uniform_(0, 1)
            return torch.bernoulli(input, p)

        actual = TestBase.run_test({"habana_bernoulli_seed"}, fn, size, p, dtype).to("cpu")
        assert TestBase.is_binary(actual)


class Test_bernoulli_method:

    @pytest.mark.parametrize("size", [[2, 3, 4, 5]], ids=format_tc)
    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=format_tc)
    @pytest.mark.parametrize(
        "setup_teardown_env_fixture",
        [{"PT_HPU_USE_EAGER_FALLBACK": 0}],
        indirect=True,
    )
    def test_bernoulli_default(self, size, dtype, setup_teardown_env_fixture):

        def fn(size, dtype):
            p = torch.empty(size, dtype=dtype, device="hpu").uniform_(0, 1)
            return p.bernoulli()

        actual = TestBase.run_test({"habana_bernoulli_seed"}, fn, size, dtype).to("cpu")
        assert TestBase.is_binary(actual)

    @pytest.mark.parametrize("size", [[2, 3, 4, 5]], ids=format_tc)
    @pytest.mark.parametrize("p", [0.7], ids=format_tc)
    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=format_tc)
    @pytest.mark.parametrize(
        "setup_teardown_env_fixture",
        [{"PT_HPU_USE_EAGER_FALLBACK": 0}],
        indirect=True,
    )
    def test_bernoulli_with_scalar_p(self, size, p, dtype, setup_teardown_env_fixture):

        def fn(size, p, dtype):
            input = torch.empty(size, dtype=dtype, device="hpu").uniform_(0, 1)
            return input.bernoulli(p)

        actual = TestBase.run_test({"habana_bernoulli_seed"}, fn, size, p, dtype).to("cpu")
        assert TestBase.is_binary(actual)


class Test_bernoulli_inplace:

    @pytest.mark.parametrize("size", [[2, 3]], ids=format_tc)
    @pytest.mark.parametrize("p", [0.6], ids=format_tc)
    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=format_tc)
    @pytest.mark.parametrize(
        "setup_teardown_env_fixture",
        [{"PT_HPU_USE_EAGER_FALLBACK": 0}],
        indirect=True,
    )
    def test_bernoulli_with_scalar_p(self, size, p, dtype, setup_teardown_env_fixture):

        def fn(size, p, dtype):
            bernoulli = torch.empty(size, dtype=dtype, device="hpu")
            bernoulli.bernoulli_(p)
            return bernoulli

        actual = TestBase.run_test({"habana_bernoulli_seed"}, fn, size, p, dtype).to("cpu")
        assert TestBase.is_binary(actual)

    @pytest.mark.parametrize("size", [[2, 3]], ids=format_tc)
    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=format_tc)
    @pytest.mark.parametrize(
        "setup_teardown_env_fixture",
        [{"PT_HPU_USE_EAGER_FALLBACK": 0}],
        indirect=True,
    )
    def test_bernoulli_with_tensor_p(self, size, dtype, setup_teardown_env_fixture):

        def fn(size, dtype):
            p = torch.empty(size, dtype=dtype, device="hpu").uniform_(0, 1)
            bernoulli = torch.empty(size, device="hpu")
            bernoulli.bernoulli_(p)
            return bernoulli

        actual = TestBase.run_test({"habana_bernoulli_seed"}, fn, size, dtype).to("cpu")
        assert TestBase.is_binary(actual)


class Test_bernoulli_edge_cases:

    @pytest.mark.parametrize("size", [(2, 3, 4)], ids=format_tc)
    @pytest.mark.parametrize("dtype", [torch.float32], ids=format_tc)
    @pytest.mark.parametrize(
        "setup_teardown_env_fixture",
        [{"PT_HPU_USE_EAGER_FALLBACK": 0}],
        indirect=True,
    )
    def test_bernoulli_for_zeros_probabilities(self, size, dtype, setup_teardown_env_fixture):

        def fn(size, dtype):
            tensor = torch.zeros(size, dtype=dtype, device="hpu")
            return torch.bernoulli(tensor)

        actual = TestBase.run_test({"habana_bernoulli_seed"}, fn, size, dtype).to("cpu")
        num_of_ones = (actual == 1).sum()

        assert TestBase.is_binary(actual)
        assert num_of_ones == 0

    @pytest.mark.parametrize("size", [(2, 3, 4, 5)], ids=format_tc)
    @pytest.mark.parametrize("dtype", [torch.float32], ids=format_tc)
    @pytest.mark.parametrize(
        "setup_teardown_env_fixture",
        [{"PT_HPU_USE_EAGER_FALLBACK": 0}],
        indirect=True,
    )
    def test_bernoulli_for_ones_probabilities(self, size, dtype, setup_teardown_env_fixture):

        def fn(size, dtype):
            tensor = torch.ones(size, dtype=dtype, device="hpu")
            return torch.bernoulli(tensor)

        actual = TestBase.run_test({"habana_bernoulli_seed"}, fn, size, dtype).to("cpu")
        num_of_zeros = (actual == 0).sum()

        assert TestBase.is_binary(actual)
        assert num_of_zeros == 0
