#!/usr/bin/env python3
###############################################################################
#
#  Copyright (c) 2021-2025 Intel Corporation
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

import argparse
import os
import re
import shutil
import subprocess
import sys
import traceback

import run_test as rt
from test_infra_hpu import pytorch_version

ptVersion = pytorch_version + "/pytorch"

rt.__dict__["CUSTOM_HANDLERS"] = {}
# List of disabled tests
DISABLED_TESTS = [
    "test_cpp_api_parity",
    "test_cpp_extensions_jit",
    "test_cpp_extensions_aot_ninja",
    "test_cpp_extensions_aot_no_ninja",
    "test_jit",
    "test_jit_cuda_fuser",
    "test_jit_disabled",
    "test_jit_fuser_legacy",
    "test_jit_fuser_te",
    "test_jit_legacy",
    "test_jit_profiling",
    "benchmark_utils/test_benchmark_utils",
    "distributed/test_nccl",
    "distributions/test_constraints",
    "distributions/test_distributions",
] + rt.DISTRIBUTED_TESTS

ENABLED_TESTS = [
    "test_unary_ufuncs",
    "test_autograd",
    "test_binary_ufuncs",
    "test_complex",
    "test_foreach",
    "test_fx",
    "test_fx_experimental",
    "test_indexing",
    "test_linalg",
    "test_masked",
    "test_module_init",
    "test_modules",
    "test_nn",
    "test_numpy_interop",
    "test_reductions",
    "test_scatter_gather_ops",
    "test_segment_reductions",
    "test_serialization",
    "test_shape_ops",
    "test_sort_and_select",
    "test_ops",
    "test_spectral_ops",
    "test_tensor_creation_ops",
    "test_testing",
    "test_torch",
    "test_type_promotion",
    "test_view_ops",
    "test_decomp",
    "test_expanded_weights",
    "test_native_mha",
    "test_nestedtensor",
    "test_ops_gradients",
    "test_ops_jit",
    "test_prims",
    "nn/test_dropout",
    "nn/test_embeddings",
    "nn/test_pooling",
    "nn/test_multihead_attention",
    "nn/test_parametrization",
    "test_utils",
    "test_ops_fwd_gradients",
    "test_jit_llga_fuser",
    "test_legacy_vmap",
    "quantization/core/experimental/test_float8.py",
    "profiler_hpu/test_profiler.py",
    "profiler_hpu/test_record_function.py",
    "profiler_hpu/test_torch_tidy.py",
    "np_hpu/test_basic.py",
    "quantization_hpu/pt2e/test_quantize_pt2e.py",
    "dynamo_hpu/test_debug_utils.py",
    "dynamo_hpu/test_functions.py",
    "dynamo_hpu/test_logging.py",
    "dynamo_hpu/test_minifier.py",
    "dynamo_hpu/test_misc.py",
    "dynamo_hpu/test_repros.py",
    "dynamo_hpu/test_subclasses.py",
    "dynamo_hpu/test_unspec.py",
]
# Temporary fix to handle machine reap in nightly run
# EXTENDED_TESTS = ["test_unary_ufuncs" for x in range(8)]
HPU_CORE_TESTS = [
    "test_autograd",
    "test_masked",
    "test_module_init",
    "test_modules",
    "test_nn",
    "test_numpy_interop",
    "test_serialization",
    "test_sort_and_select",
    "test_sparse",
    "test_sparse_csr",
    "test_testing",
    "test_torch",
    "test_native_mha",
    "test_nestedtensor",
    "nn/test_dropout",
    "nn/test_embeddings",
    "nn/test_pooling",
    "nn/test_multihead_attention",
    "nn/test_parametrization",
    "test_utils",
    "test_jit_llga_fuser",
    "test_legacy_vmap",
    "test_transformers",
    "quantization/core/experimental/test_float8",
    "transformers_hpu/test_transformers",
    "profiler_hpu/test_profiler",
    "profiler_hpu/test_record_function",
    "np_hpu/test_basic",
    "quantization_hpu/pt2e/test_quantize_pt2e",
]
HPU_OPS1_TESTS = [
    "test_ops",
    "test_ops_jit",
    "test_decomp",
    "test_type_promotion",
    "test_view_ops",
    "test_expanded_weights",
    "test_reductions" "test_scatter_gather_ops",
    "test_segment_reductions",
    "test_shape_ops",
    "test_spectral_ops",
    "test_tensor_creation_ops",
]
HPU_OPS2_TESTS = [
    "test_prims",
    "test_indexing",
    "test_complex",
    "test_linalg",
    "test_foreach",
    "test_binary_ufuncs",
    "test_unary_ufuncs",
]
HPU_CI_TESTS = [
    # add only tests which always passes
    "test_autograd",
    "test_nn",
    "test_binary_ufuncs",
    "test_ops",
    "test_torch",
    "test_modules",
]
HPU_SLOW_TESTS = ["test_ops_gradients", "test_ops_fwd_gradients"]

HPU_DISTRIBUTED_FSDP_TESTS = [
    "distributed_hpu/fsdp/test_checkpoint_wrapper",
    "distributed_hpu/fsdp/test_distributed_checkpoint",
    "distributed_hpu/fsdp/test_fsdp_apply",
    "distributed_hpu/fsdp/test_fsdp_backward_prefetch",
    "distributed_hpu/fsdp/test_fsdp_checkpoint",
    "distributed_hpu/fsdp/test_fsdp_comm",
    "distributed_hpu/fsdp/test_fsdp_fine_tune",
    "distributed_hpu/fsdp/test_fsdp_flatten_params",
    "distributed_hpu/fsdp/test_fsdp_fx",
    "distributed_hpu/fsdp/test_fsdp_grad_acc",
    "distributed_hpu/fsdp/test_fsdp_clip_grad_norm",
    "distributed_hpu/fsdp/test_fsdp_ignored_modules",
    "distributed_hpu/fsdp/test_fsdp_input",
    "distributed_hpu/fsdp/test_fsdp_memory",
    "distributed_hpu/fsdp/test_fsdp_misc",
    "distributed_hpu/fsdp/test_fsdp_mixed_precision",
    "distributed_hpu/fsdp/test_fsdp_comm_hooks",
    "distributed_hpu/fsdp/test_fsdp_multiple_forward",
    "distributed_hpu/fsdp/test_fsdp_multiple_wrapping",
    "distributed_hpu/fsdp/test_fsdp_overlap",
    "distributed_hpu/fsdp/test_fsdp_pure_fp16",
    "distributed_hpu/fsdp/test_fsdp_tp_integration",
    "distributed_hpu/fsdp/test_fsdp_traversal",
    "distributed_hpu/fsdp/test_fsdp_uneven",
    "distributed_hpu/fsdp/test_fsdp_unshard_params",
    "distributed_hpu/fsdp/test_shard_utils",
    "distributed_hpu/fsdp/test_utils",
    "distributed_hpu/fsdp/test_wrap",
    "distributed_hpu/fsdp/test_fsdp_core",
    "distributed_hpu/fsdp/test_fsdp_exec_order",
    "distributed_hpu/fsdp/test_fsdp_freezing_weights",
    "distributed_hpu/fsdp/test_fsdp_hybrid_shard",
    "distributed_hpu/fsdp/test_fsdp_optim_state",
    "distributed_hpu/fsdp/test_fsdp_sharded_grad_scaler",
    "distributed_hpu/fsdp/test_fsdp_use_orig_params",
    "distributed_hpu/fsdp/test_fsdp_state_dict",
    "distributed_hpu/fsdp/test_fsdp_dtensor_state_dict",
    "distributed_hpu/fsdp/test_hsdp_dtensor_state_dict",
    "distributed_hpu/fsdp/test_fsdp_meta",
]

HPU_DISTRIBUTED_PIPELINE_TESTS = [
    "distributed_hpu/pipelining/test_backward",
    "distributed_hpu/pipelining/test_microbatch",
    "distributed_hpu/pipelining/test_pipe",
    "distributed_hpu/pipelining/test_schedule_multiproc.py",
    "distributed_hpu/pipelining/test_schedule",
    "distributed_hpu/pipelining/test_stage",
    "distributed_hpu/pipelining/test_transformer",
    "distributed_hpu/pipelining/test_unflatten",
]

HPU_DISTRIBUTED_FSDP2_TESTS = [
    "distributed_hpu/_composable/fsdp/test_fully_shard_autograd",
    "distributed_hpu/_composable/fsdp/test_fully_shard_clip_grad_norm_",
    "distributed_hpu/_composable/fsdp/test_fully_shard_comm",
    "distributed_hpu/_composable/fsdp/test_fully_shard_compile",
    "distributed_hpu/_composable/fsdp/test_fully_shard_extensions",
    "distributed_hpu/_composable/fsdp/test_fully_shard_frozen",
    "distributed_hpu/_composable/fsdp/test_fully_shard_grad_scaler",
    "distributed_hpu/_composable/fsdp/test_fully_shard_init",
    "distributed_hpu/_composable/fsdp/test_fully_shard_logging",
    "distributed_hpu/_composable/fsdp/test_fully_shard_memory",
    "distributed_hpu/_composable/fsdp/test_fully_shard_mixed_precision",
    "distributed_hpu/_composable/fsdp/test_fully_shard_overlap",
    "distributed_hpu/_composable/fsdp/test_fully_shard_state_dict",
    "distributed_hpu/_composable/fsdp/test_fully_shard_state",
    "distributed_hpu/_composable/fsdp/test_fully_shard_training",
]

HPU_DISTRIBUTED_MPI = [
    "distributed_hpu/test_nccl",
    "distributed_hpu/test_c10d_nccl",
    "distributed_hpu/test_c10d_spawn_nccl",
    "distributed_hpu/test_c10d_object_collectives",
    "distributed_hpu/test_data_parallel",
    "distributed_hpu/test_multi_threaded_pg",
    "distributed_hpu/test_store",
]

HPU_DISTRIBUTED_TESTS = [
    "distributed_hpu/test_functional_api",
    "distributed_hpu/test_inductor_collectives",
    "distributed_hpu/_composable/test_checkpoint",
    "distributed_hpu/_composable/test_contract",
    "distributed_hpu/_composable/test_replicate_with_compiler",
    "distributed_hpu/_composable/test_replicate",
    "distributed_hpu/_composable/test_compose",
    "distributed_hpu/launcher/run_test",
    "distributed_hpu/launcher/launch_test",
    "distributed_hpu/launcher/api_test",
    "distributed_hpu/elastic/agent/server/test/api_test",
    "distributed_hpu/elastic/events/lib_test",
    "distributed_hpu/elastic/metrics/api_test",
    "distributed_hpu/elastic/multiprocessing/tail_log_test",
    "distributed_hpu/elastic/multiprocessing/redirects_test",
    "distributed_hpu/elastic/multiprocessing/api_test",
    "distributed_hpu/elastic/multiprocessing/errors/error_handler_test",
    "distributed_hpu/elastic/multiprocessing/errors/api_test",
    "distributed_hpu/elastic/rendezvous/utils_test",
    "distributed_hpu/elastic/rendezvous/static_rendezvous_test",
    "distributed_hpu/elastic/rendezvous/etcd_server_test",
    "distributed_hpu/elastic/rendezvous/etcd_rendezvous_test",
    "distributed_hpu/elastic/rendezvous/etcd_rendezvous_backend_test",
    "distributed_hpu/elastic/rendezvous/dynamic_rendezvous_test",
    "distributed_hpu/elastic/rendezvous/c10d_rendezvous_backend_test",
    "distributed_hpu/elastic/rendezvous/api_test",
    "distributed_hpu/elastic/timer/api_test",
    "distributed_hpu/elastic/timer/file_based_local_timer_test",
    "distributed_hpu/elastic/timer/local_timer_example",
    "distributed_hpu/elastic/timer/local_timer_test",
    "distributed_hpu/elastic/utils/util_test",
    "distributed_hpu/elastic/utils/logging_test",
    "distributed_hpu/elastic/utils/data/cycling_iterator_test",
    "distributed_hpu/elastic/utils/distributed_test",
    "distributed_hpu/nn/jit/test_instantiator",
    "distributed_hpu/test_dynamo_distributed",
    "distributed_hpu/test_c10d_pypg",
    "distributed_hpu/test_c10d_common",
    "distributed_hpu/test_pg_wrapper",
    "distributed_hpu/test_c10d_functional_native",
    "distributed_hpu/_tensor/test_dtensor",
    "distributed_hpu/_tensor/test_experimental_ops",
    "distributed_hpu/_tensor/test_utils",
    "distributed_hpu/_tensor/test_redistribute",
    "distributed_hpu/_tensor/test_view_ops",
    "distributed_hpu/_tensor/test_optimizers",
    "distributed_hpu/_tensor/test_op_strategy",
    "distributed_hpu/_tensor/test_matrix_ops",
    "distributed_hpu/_tensor/test_dtensor_ops",
    "distributed_hpu/_tensor/test_convolution_ops",
    "distributed_hpu/_tensor/test_pointwise_ops",
    "distributed_hpu/_tensor/test_common_rules",
    "distributed_hpu/_tensor/test_init",
    "distributed_hpu/_tensor/test_tensor_ops",
    "distributed_hpu/_tensor/test_api",
    "distributed_hpu/_tensor/experimental/test_tp_transform",
    "distributed_hpu/_tensor/debug/test_op_coverage",
    "distributed_hpu/_tensor/debug/test_comm_mode",
    "distributed_hpu/algorithms/test_join",
    "distributed_hpu/algorithms/quantization/test_quantization",
    "distributed_hpu/algorithms/ddp_comm_hooks/test_ddp_hooks",
    "distributed_hpu/test_collective_utils",
    "distributed_hpu/test_device_mesh",
    "distributed_hpu/test_fake_pg",
    "distributed_hpu/checkpoint/test_dedup_tensors",
    "distributed_hpu/checkpoint/test_fsspec",
    "distributed_hpu/checkpoint/test_nested_dict",
    "distributed_hpu/checkpoint/test_planner",
    "distributed_hpu/checkpoint/test_state_dict_utils",
    "distributed_hpu/checkpoint/test_traverse",
    "distributed_hpu/checkpoint/test_utils",
    "distributed_hpu/checkpoint/test_state_dict",
    "distributed_hpu/test_c10d_spawn_ucc",
    "distributed_hpu/test_c10d_spawn_gloo",
    "distributed_hpu/test_c10d_logger",
    "distributed_hpu/test_compute_comm_reordering",
    "distributed_hpu/optim/test_apply_optimizer_in_backward",
    "distributed_hpu/optim/test_named_optimizer",
    "optim/test_zero_redundancy_optimizer",
]

HPU_DYNAMO_TESTS = [
    "dynamo_hpu/test_export",
    "dynamo_hpu/test_export_mutations",
    "dynamo_hpu/test_debug_utils",
    "dynamo_hpu/test_dynamic_shapes",
    "dynamo_hpu/test_comptime",
    "dynamo_hpu/test_aot_autograd",
    "dynamo_hpu/test_replay_record",
    "dynamo_hpu/test_recompile_ux",
    "dynamo_hpu/test_optimizers",
    "dynamo_hpu/test_modules",
    "dynamo_hpu/test_model_output",
    "dynamo_hpu/test_misc",
    "dynamo_hpu/test_minifier",
    "dynamo_hpu/test_global",
    "dynamo_hpu/test_functions",
    "dynamo_hpu/test_verify_correctness",
    "dynamo_hpu/test_unspec",
    "dynamo_hpu/test_subgraphs",
    "dynamo_hpu/test_backends",
    "dynamo_hpu/test_after_aot",
    "dynamo_hpu/test_autograd_function",
    "dynamo_hpu/test_compile",
    "dynamo_hpu/test_config",
    "dynamo_hpu/test_ctx_manager",
    "dynamo_hpu/test_decorators",
    "dynamo_hpu/test_exc",
    "dynamo_hpu/test_input_attr_tracking",
    "dynamo_hpu/test_logging",
    "dynamo_hpu/test_pre_dispatch",
    "dynamo_hpu/test_recompiles",
    "dynamo_hpu/test_repros",
    "dynamo_hpu/test_sources",
    "dynamo_hpu/test_subclasses",
    "dynamo_hpu/test_higher_order_ops",
    "dynamo_hpu/test_backward_higher_order_ops",
]
HPU_GPU_MIGRATION_TESTS = [
    "transformers_hpu/test_transformers",
]
NEW_EAGER_FRONTEND_TESTS = []
SKIP_RERUN = [
    "distributed_hpu/test_functional_api",
    "test_ops",
    "test_indexing",
    "test_decomp",
    "test_ops_jit",
    "test_sort_and_select",
    "test_expanded_weights",
]
# Tests to be executed with NEW_EAGER_FRONTEND
NEW_EAGER_FRONTEND_TESTS.extend(HPU_DYNAMO_TESTS)
NEW_EAGER_FRONTEND_TESTS.extend(HPU_GPU_MIGRATION_TESTS)
# Update the full suite test list.
ENABLED_TESTS.extend(HPU_DYNAMO_TESTS)
ENABLED_TESTS.extend(HPU_GPU_MIGRATION_TESTS)
# ENABLED_TESTS.extend(EXTENDED_TESTS)

# Extend distributed tests
HPU_DISTRIBUTED_TESTS.extend(HPU_DISTRIBUTED_MPI)

import torch.distributed as dist


def run_specific_test_hpu(test_module, test_directory, options, launcher_cmd):
    return_code = run_test(test_module, test_directory, options, launcher_cmd=launcher_cmd)
    assert isinstance(return_code, int) and not isinstance(return_code, bool), "Return code should be an integer"
    if return_code == 0:
        return None
    message = f"{test_module} failed!"
    if return_code < 0:
        # subprocess.Popen returns the child process' exit signal as
        # return code -N, where N is the signal number.
        signal_name = rt.SIGNALS_TO_NAMES_DICT[-return_code]
        message += f" Received signal: {signal_name}"
    return message


def get_num_cards():
    # Check ASIC
    p = subprocess.Popen(["lspci", "-d", "1da3:"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    num_cards = sum(1 for _ in p.stdout)
    # Check ASIC Simulators
    if num_cards == 0:
        num_cards = ((subprocess.Popen(["pgrep", "-c", "coral"], stdout=subprocess.PIPE)).communicate())[0].decode(
            "ascii"
        )
    return num_cards


def test_distributed_hpu(options, selected_tests, test_directory):
    has_failed = False
    failure_messages = []
    options_clone = rt.copy.deepcopy(options)
    options_clone.pytest = True
    world_size = 0
    rank = -1
    local_rank = -1
    try:
        print("Running HPU Distributed Test cases")
        # habana packages
        import habana_frameworks.torch.core as htcore
        import habana_frameworks.torch.distributed.hccl
        import habana_frameworks.torch.hpu as hpu
        from habana_frameworks.torch.distributed.hccl import initialize_distributed_hpu

        if dist.is_available():
            print("dist.is_available() : {}, Printing parameters ".format(dist.is_available()))
            world_size, rank, local_rank = initialize_distributed_hpu()
            print(f"INFO: world_size={world_size}, rank={rank}, local_rank={local_rank}")
        else:
            print("Distributed package not available , Aborting")
            print("Distributed package not available , Aborting", file=sys.stderr)
            raise RuntimeError("Distributed package not available")
            has_failed = True

        # TODO create DISTRIBUTED_TESTS_CONFIG file
        # SW-185004 : Since world_size is always set to 1, using the count of devices to set the rank
        # in mpiexec command.
        num_cards = get_num_cards()
        print("Number of card is {}".format(num_cards))
        if num_cards == 0:
            print("Exiting as Number of card is 0")
            sys.exit(1)

        mpi_available = subprocess.call("command -v mpiexec", shell=True) == 0 and sys.version_info > (3, 9)
        if not mpi_available:
            print("MPI not available -- MPI backend tests will be skipped", file=sys.stderr)
            raise RuntimeError("MPI backend not available")
        # mpi_cmd=['mpiexec','-n',str(world_size),'--allow-run-as-root']
        mpi_cmd = ["mpiexec", "-n", str(num_cards), "--allow-run-as-root"]
        for test_module in selected_tests:
            regexp = re.compile(r"fsdp")
            # mpi_cmd=['mpiexec','-n',str(world_size),'--allow-run-as-root']
            mpi_cmd = ["mpiexec", "-n", str(num_cards), "--allow-run-as-root"]
            if regexp.search(test_module) or test_module not in HPU_DISTRIBUTED_MPI:
                mpi_cmd = ""
            err_message = run_specific_test_hpu(test_module, test_directory, options, launcher_cmd=mpi_cmd)
            if err_message is None or options_clone.collect_only:
                continue
            has_failed = True
            failure_messages.append(err_message)
            if not options_clone.continue_through_error:
                raise RuntimeError(err_message)
        if not has_failed:
            print("The overall pytorch test suite status is PASS")
        else:
            print(f"The overall pytorch test suite status is FAIL. The failure_message: {failure_messages}")
            print("Execution logs can be found at {}".format(options.console_logs))
    finally:
        if options.coverage:
            from coverage import Coverage

            test_dir = os.path.dirname(os.path.abspath(__file__))
            with rt.set_cwd(test_dir):
                cov = Coverage()
                if rt.PYTORCH_COLLECT_COVERAGE:
                    cov.load()
                cov.combine(strict=False)
                cov.save()
                if not rt.PYTORCH_COLLECT_COVERAGE:
                    cov.html_report()
    if has_failed:
        print("Tests failed to execute", __file__, file=sys.stderr)
        sys.exit(1)


def test_dynamo_hpu(options, selected_tests, test_directory):
    has_failed = False
    failure_messages = []
    options_clone = rt.copy.deepcopy(options)
    options_clone.pytest = True
    try:
        print("Running HPU Dyanamo Test cases")
        for test_module in selected_tests:
            err_message = run_specific_test_hpu(test_module, test_directory, options, launcher_cmd=None)
            if err_message is None or options_clone.collect_only:
                continue
            has_failed = True
            failure_messages.append(err_message)
            if not options_clone.continue_through_error:
                raise RuntimeError(err_message)
        if not has_failed:
            print("The overall pytorch test suite status is PASS")
        else:
            print(f"The overall pytorch test suite status is FAIL. The failure_message: {failure_messages}")
            print("Execution logs can be found at {}".format(options.console_logs))
    finally:
        if options.coverage:
            from coverage import Coverage

            test_dir = os.path.dirname(os.path.abspath(__file__))
            with rt.set_cwd(test_dir):
                cov = Coverage()
                if rt.PYTORCH_COLLECT_COVERAGE:
                    cov.load()
                cov.combine(strict=False)
                cov.save()
                if not rt.PYTORCH_COLLECT_COVERAGE:
                    cov.html_report()
    if has_failed:
        print("Tests failed to execute", __file__, file=sys.stderr)
        sys.exit(1)


def test_gpu_migration_hpu(options, selected_tests, test_directory):
    has_failed = False
    failure_messages = []
    options_clone = rt.copy.deepcopy(options)
    options_clone.pytest = True
    try:
        print("Running HPU GPU Migration Test cases")
        for test_module in selected_tests:
            err_message = run_specific_test_hpu(test_module, test_directory, options, launcher_cmd=None)
            if err_message is None or options_clone.collect_only:
                continue
            has_failed = True
            failure_messages.append(err_message)
            if not options_clone.continue_through_error:
                raise RuntimeError(err_message)
        if not has_failed:
            print("The overall pytorch test suite status is PASS")
        else:
            print(f"The overall pytorch test suite status is FAIL. The failure_message: {failure_messages}")
            print("Execution logs can be found at {}".format(options.console_logs))
    finally:
        if options.coverage:
            from coverage import Coverage

            test_dir = os.path.dirname(os.path.abspath(__file__))
            with rt.set_cwd(test_dir):
                cov = Coverage()
                if rt.PYTORCH_COLLECT_COVERAGE:
                    cov.load()
                cov.combine(strict=False)
                cov.save()
                if not rt.PYTORCH_COLLECT_COVERAGE:
                    cov.html_report()
    if has_failed:
        print("Tests failed to execute", __file__, file=sys.stderr)
        sys.exit(1)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the PyTorch unit test suite",
        epilog="where TESTS is any of: {}".format(", ".join(ENABLED_TESTS)),
    )
    parser.add_argument(
        "-v", "--verbose", action="count", default=0, help="print verbose information and test-by-test results"
    )
    parser.add_argument("--jit", "--jit", action="store_true", help="run all jit tests")
    parser.add_argument(
        "--distributed-tests",
        "--distributed-tests",
        action="store_true",
        help="run all distributed tests",
    )
    parser.add_argument(
        "--distributed-fsdp-tests",
        "--distributed-fsdp-tests",
        action="store_true",
        help="run all distributed FSDP tests",
    )
    parser.add_argument(
        "--dynamo_hpu_tests",
        "--dynamo_hpu_tests",
        action="store_true",
        help="run all dynamo tests",
    )
    parser.add_argument(
        "--gpu_migration_hpu_tests",
        "--gpu_migration_hpu_tests",
        action="store_true",
        help="run all gpu migration tests",
    )
    parser.add_argument(
        "--functorch",
        "--functorch",
        action="store_true",
        help=(
            "If this flag is present, we will only run functorch tests. "
            "If this flag is not present, we will not run any functorch tests. "
            "This requires functorch to already be installed."
        ),
    )
    parser.add_argument(
        "-core",
        "--core",
        action="store_true",
        help="Only run core tests, or tests that validate PyTorch's ops, modules,"
        "and autograd. They are defined by CORE_TEST_LIST.",
    )
    parser.add_argument("--ops1", action="store_true", help="Only ops1 tests : ops,binary_ufuncs,ops_jit")
    parser.add_argument("--ops2", action="store_true", help="Only ops2 tests : unary,ops_jit")
    parser.add_argument("--ci_tests", action="store_true", help="Only ci_tests : autograd")
    parser.add_argument("--slow", action="store_true", help="Only tests in HPU_SLOW_TESTS")
    parser.add_argument(
        "--distributed_hpu",
        action="store_true",
        help="Run HPU customized distributed tests from list HPU_DISTRIBUTED_TESTS",
    )
    parser.add_argument(
        "--distributed_hpu_fsdp",
        action="store_true",
        help="Run HPU customized distributed FSDP tests from list HPU_DISTRIBUTED_FSDP_TESTS",
    )
    parser.add_argument(
        "--distributed_hpu_pipeline",
        action="store_true",
        help="Run HPU customized distributed pipelining tests from list HPU_DISTRIBUTED_PIPELINE_TESTS",
    )
    parser.add_argument(
        "--distributed_hpu_fsdp2",
        action="store_true",
        help="Run HPU customized distributed fsdp2 tests from list HPU_DISTRIBUTED_FSDP2_TESTS",
    )
    parser.add_argument(
        "--dynamo_hpu", action="store_true", help="Run HPU customized dyanmo tests from list HPU_DYNAMO_TESTS"
    )
    parser.add_argument(
        "--gpu_migration_hpu",
        action="store_true",
        help="Run HPU customized gpu migration tests from list HPU_GPU_MIGRATION_TESTS",
    )
    parser.add_argument(
        "-pt",
        "--pytest",
        default=True,
        action="store_false",
        help="If true, use `pytest` to execute the tests. E.g., this runs "
        "TestTorch with pytest in verbose and coverage mode: "
        "python rt.py -vci torch -pt",
    )
    parser.add_argument(
        "-c", "--coverage", action="store_true", help="enable coverage", default=rt.PYTORCH_COLLECT_COVERAGE
    )
    parser.add_argument(
        "-i",
        "--include",
        nargs="+",
        choices=rt.TestChoices(ENABLED_TESTS),
        default=ENABLED_TESTS,
        metavar="TESTS",
        help="select a set of tests to include (defaults to ALL tests)."
        " tests can be specified with module name, module.TestClass"
        " or module.TestClass.test_method",
    )
    parser.add_argument(
        "-x",
        "--exclude",
        nargs="+",
        choices=ENABLED_TESTS,
        metavar="TESTS",
        default=[],
        help="select a set of tests to exclude",
    )
    parser.add_argument(
        "-f",
        "--first",
        choices=ENABLED_TESTS,
        metavar="TESTS",
        help="select the test to start from (excludes previous tests)",
    )
    parser.add_argument(
        "-l",
        "--last",
        choices=ENABLED_TESTS,
        metavar="TESTS",
        help="select the last test to run (excludes following tests)",
    )
    parser.add_argument(
        "--bring-to-front",
        nargs="+",
        choices=rt.TestChoices(ENABLED_TESTS),
        default=[],
        metavar="TESTS",
        help="select a set of tests to run first. This can be used in situations"
        " where you want to run all tests, but care more about some set, "
        "e.g. after making a change to a specific component",
    )
    parser.add_argument("--ignore-win-blocklist", action="store_true", help="always run blocklisted windows tests")
    parser.add_argument("--determine-from", help="File of affected source filenames to determine which tests to run.")
    parser.add_argument(
        "--continue-through-error",
        default=True,
        action="store_false",
        help="Runs the full test suite despite one of the tests failing",
    )
    parser.add_argument(
        "additional_unittest_args",
        nargs="*",
        help="additional arguments passed through to unittest, e.g., "
        "python rt.py -i sparse -- TestSparse.test_factory_size_check",
    )
    parser.add_argument(
        "--shard",
        nargs=2,
        type=int,
        help="runs a shard of the tests (taking into account other selections), e.g., "
        "--shard 2 3 will break up the selected tests into 3 shards and run the tests "
        "in the 2nd shard (the first number should not exceed the second)",
    )
    parser.add_argument(
        "--exclude-jit-executor",
        default="True",
        action="store_true",
        help="exclude tests that are run for a specific jit config",
    )
    parser.add_argument(
        "--exclude-distributed-tests",
        default="True",
        action="store_true",
        help="exclude distributed tests",
    )
    parser.add_argument(
        "--exclude-dynamo-tests",
        default="True",
        action="store_true",
        help="exclude dynamo tests",
    )
    parser.add_argument(
        "--use-specified-test-cases-by",
        type=str,
        choices=["include", "bring-to-front"],
        default="include",
        help='used together with option "--run-specified-test-cases". When specified test case '
        "file is set, this option allows the user to control whether to only run the specified test "
        "modules or to simply bring the specified modules to front and also run the remaining "
        "modules. Note: regardless of this option, we will only run the specified test cases "
        " within a specified test module. For unspecified test modules with the bring-to-front "
        "option, all test cases will be run, as one may expect.",
    )
    parser.add_argument(
        "--fx2trt-tests",
        action="store_true",
        help="run all fx2trt tests",
    )
    parser.add_argument(
        "--exclude-fx2trt-tests",
        action="store_true",
        help="exclude fx2trt tests",
    )
    # Habana specific args
    parser.add_argument(
        "-xml", "--xml", default=True, action="store_false", help="Generate xml report file. Default = True"
    )
    parser.add_argument(
        "-html", "--html", default=True, action="store_false", help="Generate html report file. Default = True"
    )
    parser.add_argument(
        "-con", "--console_logs", default=True, action="store_false", help="Save console logs. Default = True"
    )
    parser.add_argument(
        "-o",
        "--outdir",
        default=os.path.join(os.path.dirname(os.path.realpath(__file__)), "test_results"),
        help="Path to save results. Default = test_results",
    )
    parser.add_argument("-ptargs", "--pytest_args", nargs="+", help="Additional args for pytest.")
    parser.add_argument(
        "-k",
        "--test_regex",
        default="_hpu",
        help="Pick tests to run as per pytest regex. Accuracy depends on pytest." 'Use either "-k" or "-tn".',
    )
    parser.add_argument(
        "-tn",
        "--test_name",
        default="",
        help="Pick test as per pytest test_module::test_class::test_method" 'Use either "-k" or "-tn".',
    )
    parser.add_argument("-s", "-s", default=False, action="store_true", help="shortcut for --capture=no in pytest")
    parser.add_argument(
        "-fp",
        "--file_parallel",
        help="Number of parallel files to execute. Forces one test file to schedule over one card only.",
    )
    parser.add_argument(
        "-fork", "--forked", default=False, action="store_true", help="Run pytest with --forked option. Default: False"
    )
    parser.add_argument("-n", "--num_threads", default=1, help="Number of threads. Only valid if forked==TRUE")
    # TODO: check if required?
    parser.add_argument(
        "-dir", "--test_dir", default=os.path.dirname(os.path.realpath(__file__)), help="Directory path to the tests"
    )
    parser.add_argument("--collect-only", default=False, action="store_true", help="List all the tests to be run")
    parser.add_argument("-m", "--mark", type=str, help="pytest marker to be applied")
    parser.add_argument("-ft", "--file_timeout", default=12000, help="Per file timeout value. Default = 12000s")
    parser.add_argument(
        "--mps",
        "--mps",
        action="store_true",
        help=("If this flag is present, we will only run test_mps and test_metal"),
    )
    return parser.parse_known_args()


# run non-device specific tests - run tests which do not have _hpu suffix
# cpu only tests are already skipped via the onlyCPU decorator
def bypass_test_filter(testmodule_str):
    if "dynamo" in testmodule_str:
        return True
    if "transformers_hpu" in testmodule_str:  # As the testname is cuda and it is migrated to HPU implicitly
        return True
    if "fx" in testmodule_str:
        return True
    if "distributed_hpu" in testmodule_str:
        return True
    return False


def get_executable_command(options, allow_pytest, disable_coverage=False):
    if options.coverage and not disable_coverage:
        executable = ["coverage", "run", "--parallel-mode", "--source=torch"]
    else:
        executable = [sys.executable, "-bb"]
    if options.pytest:
        if allow_pytest:
            executable += ["-m", "pytest"]
        else:
            rt.print_to_stderr("Pytest cannot be used for this test. Falling back to unittest.")
    return executable


def print_env(options):
    # Print env vars before running tests
    # Default width of 80 and 24
    fallback = (80, 24)
    col = shutil.get_terminal_size(fallback=fallback)[0]
    mark = " Start Printing env variables "
    print(
        ("=" * (int)((col - len(mark)) / 2)).ljust((int)((col - len(mark)) / 2))
        + mark.center(len(mark))
        + ("=" * (int)((col - len(mark)) / 2)).rjust((int)((col - len(mark)) / 2))
    )
    for k, v in os.environ.items():
        print(f"{k}={v}")
    mark = " Finish Printing env variables "
    print(
        ("=" * (int)((col - len(mark)) / 2)).ljust((int)((col - len(mark)) / 2))
        + mark.center(len(mark))
        + ("=" * (int)((col - len(mark)) / 2)).rjust((int)((col - len(mark)) / 2))
    )
    mark = " Start Printing test options "
    print(
        ("=" * (int)((col - len(mark)) / 2)).ljust((int)((col - len(mark)) / 2))
        + mark.center(len(mark))
        + ("=" * (int)((col - len(mark)) / 2)).rjust((int)((col - len(mark)) / 2))
    )
    print("options: ", options)
    mark = " Finish Printing test options "
    print(
        ("=" * (int)((col - len(mark)) / 2)).ljust((int)((col - len(mark)) / 2))
        + mark.center(len(mark))
        + ("=" * (int)((col - len(mark)) / 2)).rjust((int)((col - len(mark)) / 2))
    )


def get_num_threads(num_threads):
    # Caution: all the habana cards will be counted here.
    # In case machine has a mix of cards, change the relevant card id to'1da3:xxxx' in lspci
    p = subprocess.Popen(["lspci", "-d", "1da3:"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    num_cards = sum(1 for _ in p.stdout)
    # If running with Simulators
    if num_cards <= 0:
        p = subprocess.Popen(["pgrep", "coral"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        num_cards = sum(1 for _ in p.stdout)
    print("Found HPU devices {}".format(num_cards))
    return min(int(num_threads), int(num_cards))


def test_parallel(options, selected_tests, test_directory):
    has_failed = False
    failure_messages = []
    options_clone = rt.copy.deepcopy(options)
    options_clone.pytest = True
    if options.file_parallel:
        import multiprocessing
        from contextlib import contextmanager
        from functools import partial

        @contextmanager
        def poolcontext(*args, **kwargs):
            pool = multiprocessing.Pool(*args, **kwargs)
            yield pool
            pool.terminate()

        num_max_threads = get_num_threads(options.file_parallel)
        print(f"Testing {len(selected_tests)} files, running {num_max_threads} files in parallel")
        with poolcontext(processes=num_max_threads) as pool:
            results = pool.map(
                partial(run_test_module, test_directory=test_directory, options=options_clone),
                selected_tests,
                chunksize=1,
            )
            error_message = [result for result in results if result is not None]
            failure_messages.extend(error_message)
        if any(failure_messages):
            has_failed = True
    else:
        for test in selected_tests:
            err_message = run_test_module(test, test_directory, options_clone)
            if err_message is None or options_clone.collect_only:
                continue
            has_failed = True
            failure_messages.append(err_message)
            if not options_clone.continue_through_error:
                raise RuntimeError(err_message)
    return has_failed, failure_messages


def run_test_module(test: str, test_directory: str, options) -> rt.Optional[str]:
    test_module = rt.parse_test_module(test)
    # Printing the date here can help diagnose which tests are slow
    # rt.print_to_stderr(
    #    'Running {} ... [{}]'.format(test, rt.datetime.now()))
    handler = rt.CUSTOM_HANDLERS.get(test_module, run_test)
    return_code = handler(test_module, test_directory, options)
    assert isinstance(return_code, int) and not isinstance(return_code, bool), "Return code should be an integer"
    if return_code == 0:
        return None
    message = f"{test} failed!"
    if return_code < 0:
        # subprocess.Popen returns the child process' exit signal as
        # return code -N, where N is the signal number.
        signal_name = rt.SIGNALS_TO_NAMES_DICT[-return_code]
        message += f" Received signal: {signal_name}"
    return message


def get_module_name(test_module):
    splitpath = os.path.split(test_module)  # test module contains full path, check only one level
    test_folder = splitpath[len(splitpath) - 2]
    test_file = splitpath[len(splitpath) - 1]
    if test_folder != "":
        test_module_name = test_folder + "_" + test_file
    else:
        test_module_name = test_file
    test_module_name = test_module_name.replace("/", "_")
    return test_module_name


def run_test(test_module, test_directory, options, launcher_cmd=None, extra_unittest_args=None, leaf=True):
    unittest_args = options.additional_unittest_args.copy()
    if options.verbose:
        unittest_args.append("-" + "v" * options.verbose)  # in case of pytest
    if test_module in rt.RUN_PARALLEL_BLOCKLIST:
        unittest_args = [arg for arg in unittest_args if not arg.startswith("--run-parallel")]
    if extra_unittest_args:
        assert isinstance(extra_unittest_args, list)
        unittest_args.extend(extra_unittest_args)
    # test filters
    test_filter = ""
    if options.test_regex != "":
        test_filter = options.test_regex
    if (test_filter != "") and not bypass_test_filter(test_module):
        unittest_args.append("-k " + test_filter.strip())
    if options.test_name != "":
        test_filter = test_module + ".py" + "::" + options.test_name
    # xml report
    test_module_name = get_module_name(test_module)
    if options.xml:
        if options.forked:
            unittest_args.append("--junitxml={}".format(os.path.join(options.outdir, test_module_name + ".xml")))
        else:
            unittest_args.append("--junitxml={}".format(os.path.join(options.outdir, test_module_name + ".xml")))
        unittest_args.append("--override-ini=junit_suite_name={}".format(test_module_name))
    if options.html:
        unittest_args.append("--self-contained-html")
        if options.forked:
            unittest_args.append(
                "--html={}".format(os.path.join(options.outdir, "html_reports", "forked", test_module_name + ".html"))
            )
        else:
            unittest_args.append(
                "--html={}".format(os.path.join(options.outdir, "html_reports", "no_fork", test_module_name + ".html"))
            )
    # Forked options
    if options.forked:
        unittest_args.append("--forked")
    # Threaded workers
    # TODO: pytest-timeout doesn't gracefully exit with xdist. So disabling xdist until required.
    if not options.file_parallel and options.num_threads:
        num_threads = get_num_threads(options.num_threads)
        print("Running the test with {} threads".format(num_threads))
        if test_module in rt.RUN_PARALLEL_BLOCKLIST:
            unittest_args.append("-n 1")
        else:
            unittest_args.append("-n " + str(num_threads))
    # Catch the stdout of process
    # This is required in case of tests that crash, causing junitxml to not log them
    if options.console_logs:
        stdout_logs = os.path.join(options.outdir, "console_logs", "no_fork", test_module_name + "_stdout.log")
    unittest_args.append("--max-worker-restart=5000")
    # Additional flags
    if options.collect_only:
        unittest_args.append("--collect-only")
    if options.mark:
        unittest_args.append("-m=" + options.mark)
    if options.s:
        unittest_args.append("-s")
    # Additional pytest flags
    if options.pytest_args:
        pytest_args = ""
        for args in options.pytest_args:
            pytest_args = pytest_args + " " + args
        unittest_args.append(pytest_args)
    unittest_args.append("-rxXs")
    # If using pytest, replace -f with equivalent -x
    if options.pytest:
        unittest_args = [arg if arg != "-f" else "-x" for arg in unittest_args]
    # Can't call `python -m unittest test_*` here because it doesn't run code
    # in `if __name__ == '__main__': `. So call `python test_*.py` instead.
    if options.test_name != "":
        argv = [test_filter] + unittest_args
    else:
        argv = [test_module + ".py"] + unittest_args
    # Multiprocessing related tests cannot run with coverage.
    # Tracking issue: https://github.com/pytorch/pytorch/issues/50661
    disable_coverage = sys.platform == "win32" and test_module in rt.WINDOWS_COVERAGE_BLOCKLIST
    # Extra arguments are not supported with pytest
    executable = get_executable_command(
        options, allow_pytest=not extra_unittest_args, disable_coverage=disable_coverage
    )
    command = (launcher_cmd or []) + executable + argv
    rt.print_to_stderr("Executing {} ... [{}]".format(test_module, rt.datetime.now()))
    # handling the execution modes
    if test_module in NEW_EAGER_FRONTEND_TESTS:
        rt.print_to_stderr("Setting to New Eager mode for {}".format(test_module))
        print("Setting to New Eager mode for {}".format(test_module))
        os.environ["PT_HPU_LAZY_MODE"] = "0"
        print("Set PT_HPU_LAZY_MOD to {}".format(os.environ["PT_HPU_LAZY_MODE"]))
        rt.print_to_stderr("Set PT_HPU_LAZY_MOD to {}".format(os.environ["PT_HPU_LAZY_MODE"]))
    else:
        # rt.print_to_stderr("Setting default lazy mode for {}".format(test_module))
        # print("Setting to default lazy mode for {}".format(test_module))
        # os.environ["PT_HPU_LAZY_MODE"] = "1"
        # Execute with New Eager mode for PT2.1
        # os.environ["PT_HPU_LAZY_MODE"] = "0"
        print("Set PT_HPU_LAZY_MOD to {}".format(os.environ["PT_HPU_LAZY_MODE"]))
        rt.print_to_stderr("Set PT_HPU_LAZY_MOD to {}".format(os.environ["PT_HPU_LAZY_MODE"]))
    # Launch the thread
    from subprocess_tee import run

    # File specific timeout. currently set to 1hr per file.
    command = ["timeout", str(options.file_timeout)] + command
    result = run(command, echo=True)
    lines = result.stdout.splitlines()
    if result.returncode == 0:
        print("Execution result of {} is PASS".format(test_module))
    elif (
        not options.core and result.returncode == 255 and test_module not in SKIP_RERUN
    ):  # A segfaulted execution doesn't generate the xml.
        print("Execution result of {} is FAIL with return code {}".format(test_module, result.returncode))
        with open("../" + test_module + ".regressions", "a") as regfile:
            regfile.writelines(ptVersion + "/test/" + lines[-1] + "\n")
        print("Re-running the suite after marking the segfault regression")
        leaf = False
        run_test(test_module, test_directory, options, launcher_cmd, extra_unittest_args)
    elif (
        not options.core and result.returncode == 139 and test_module not in SKIP_RERUN
    ):  # A segfaulted execution doesn't generate the xml.
        print("Execution result of {} is FAIL with return code {}".format(test_module, result.returncode))
        with open("../" + test_module + ".regressions", "a") as regfile:
            regfile.writelines(ptVersion + "/test/" + lines[-1] + "\n")
        print("Re-running the suite after marking the segfault regression")
        leaf = False
        run_test(test_module, test_directory, options, launcher_cmd, extra_unittest_args)
    elif (
        not options.core and result.returncode == 134 and test_module not in SKIP_RERUN
    ):  # A timeout doesn't generate the xml.
        print("Execution result of {} is FAIL with return code {}".format(test_module, result.returncode))
        with open("../" + test_module + ".regressions", "a") as regfile:
            regfile.writelines(ptVersion + "/test/" + lines[-1] + "\n")
        print("Re-running the suite after marking the timeout regression")
        leaf = False
        run_test(test_module, test_directory, options, launcher_cmd, extra_unittest_args)
    elif (
        not options.core and result.returncode == 124 and test_module not in SKIP_RERUN
    ):  # A timeout doesn't generate the xml.
        print("Execution result of {} is FAIL with return code {}".format(test_module, result.returncode))
        with open("../" + test_module + ".regressions", "a") as regfile:
            regfile.writelines(ptVersion + "/test/" + lines[-1] + "\n")
        print("Re-running the suite after marking the timeout regression")
        leaf = False
        run_test(test_module, test_directory, options, launcher_cmd, extra_unittest_args)
    elif not options.core and result.returncode == 1 and test_module not in SKIP_RERUN:
        print("Execution result of {} is FAIL with return code {}".format(test_module, result.returncode))
        for line in lines:
            if " FAILED " in line and "::Test" in line and "::test_" in line:
                with open("../" + test_module + ".regressions", "a") as regfile:
                    regfile.writelines(ptVersion + "/test/" + line.split("FAILED", 1)[0].strip() + "\n")
                # break
        # This is required because some failures are causing false positive failures later in the execution
        print("In future Re-run the suite after marking the first failure.")
        # run_test(test_module, test_directory, options, launcher_cmd, extra_unittest_args)
    elif (
        not options.core and result.returncode == 137 and test_module not in SKIP_RERUN
    ):  # A out of memory error doesn't generate the xml.
        print("Execution result of {} is FAIL with return code {}".format(test_module, result.returncode))
        with open("../" + test_module + ".regressions", "a") as regfile:
            regfile.writelines(ptVersion + "/test/" + lines[-1] + "\n")
        print("Re-running the suite after marking the out of memory regression")
        leaf = False
        run_test(test_module, test_directory, options, launcher_cmd, extra_unittest_args)
    elif not options.core and result.returncode == 134 and test_module not in SKIP_RERUN:
        print("Execution result of {} is FAIL with return code {}".format(test_module, result.returncode))
        for line in lines:
            if " FAILED " in line and "::Test" in line and "::test_" in line:
                with open("../" + test_module + ".regressions", "a") as regfile:
                    regfile.writelines(ptVersion + "/test/" + line.split("FAILED", 1)[0].strip() + "\n")
                # break
        # This is required because some failures are causing false positive failures later in the execution
        print("In future Re-run the suite after marking the first failure.")
    else:
        print("Execution result of {} is FAIL with return code {}".format(test_module, result.returncode))
    # Print stdout only once, in the leaf recursion.
    if options.console_logs and leaf is True:
        with open(stdout_logs, "w") as log:
            log.write(result.stdout)
    return result.returncode


def get_selected_tests(options):
    selected_tests = options.include
    if options.core:
        print("Selecting  HPU_CORE_TESTS")
        selected_tests = HPU_CORE_TESTS
        selected_tests = list(filter(lambda test_name: test_name in HPU_CORE_TESTS, selected_tests))
    if options.ops1:
        print("Selecting only HPU_OPS1_TESTS")
        selected_tests = list(filter(lambda test_name: test_name in HPU_OPS1_TESTS, selected_tests))
    if options.ops2:
        print("Selecting HPU_OPS2_TESTS")
        selected_tests = list(filter(lambda test_name: test_name in HPU_OPS2_TESTS, selected_tests))
    if options.ci_tests:
        print("Selecting HPU_CI_TESTS")
        selected_tests = list(filter(lambda test_name: test_name in HPU_CI_TESTS, selected_tests))
    if options.slow:
        print("Selecting HPU_SLOW_TESTS")
        selected_tests = HPU_SLOW_TESTS
    if options.distributed_hpu:
        print("Selecting HPU_DISTRIBUTED_TESTS")
        selected_tests = HPU_DISTRIBUTED_TESTS
    if options.distributed_hpu_fsdp:
        print("Selecting HPU_DISTRIBUTED_FSDP_TESTS")
        selected_tests = HPU_DISTRIBUTED_FSDP_TESTS
    if options.distributed_hpu_pipeline:
        print("Selecting HPU_DISTRIBUTED_PIPELINE_TESTS")
        selected_tests = HPU_DISTRIBUTED_PIPELINE_TESTS
    if options.distributed_hpu_fsdp2:
        print("Selecting HPU_DISTRIBUTED_FSDP2_TESTS")
        selected_tests = HPU_DISTRIBUTED_FSDP2_TESTS
    if options.dynamo_hpu:
        print("Selecting HPU_DYNAMO_TESTS")
        selected_tests = HPU_DYNAMO_TESTS
    if options.gpu_migration_hpu:
        print("Selecting HPU_GPU_MIGRATION_TESTS")
        selected_tests = HPU_GPU_MIGRATION_TESTS
    print("Selected Tests : ", selected_tests)
    return selected_tests


def main():
    options, unknown_args = parse_args()
    test_directory = options.test_dir if options.test_dir else os.path.dirname(os.path.abspath(__file__))
    selected_tests = get_selected_tests(options)
    if options.test_regex != "" and options.test_name != "":
        print('ERROR: both "--test_regex" and "--test_name" provided. Will ignore test_regex.')
        options.test_regex = ""
    if options.coverage and not rt.PYTORCH_COLLECT_COVERAGE:
        rt.shell(["coverage", "erase"])
    if options.verbose:
        print_env(options)
        print(f"Selected tests:\n{selected_tests}")
    # Create dir for reports
    # XML reports
    try:
        xml_path = os.path.join(options.outdir, "xml_reports")
        if options.verbose:
            print("XML reports will be saved at: {}".format(xml_path))
        if not os.path.exists(xml_path):
            os.makedirs(xml_path)
    except:
        print("Cannot create directory for xml reports")
    # XML reports
    try:
        html_path = os.path.join(options.outdir, "html_reports")
        if options.verbose:
            print("HTML reports will be saved at: {}".format(html_path))
        if not os.path.exists(html_path):
            os.makedirs(html_path)
    except:
        print("Cannot create directory for html reports")
    # Console logs
    try:
        console_path = os.path.join(options.outdir, "console_logs")
        if options.verbose:
            print("Console logs will be saved at: {}".format(console_path))
        if not os.path.exists(console_path):
            os.makedirs(os.path.join(console_path, "no_fork"))
            os.makedirs(os.path.join(console_path, "fork"))
    except:
        print("Cannot create directory for console logs")
    # For distributed tests the command syntax is different
    if options.distributed_hpu or options.distributed_hpu_fsdp:
        test_distributed_hpu(options, selected_tests, test_directory)
        return
    # Run dyanmo tests with New Eager Frontend
    if options.dynamo_hpu:
        test_dynamo_hpu(options, selected_tests, test_directory)
        return
    # Run gpu migration tests with New Eager Frontend
    if options.gpu_migration_hpu:
        test_gpu_migration_hpu(options, selected_tests, test_directory)
        return
    # Execute the tests
    try:
        has_failed, failure_messages = test_parallel(options, selected_tests, test_directory)  # nopep8
        if not has_failed:
            print("The overall pytorch test suite status is PASS")
        else:
            print(f"The overall pytorch test suite status is FAIL. The failure_message: {failure_messages}")
            print("Execution logs can be found at {}".format(options.console_logs))
    finally:
        if options.coverage:
            from coverage import Coverage

            test_dir = os.path.dirname(os.path.abspath(__file__))
            with rt.set_cwd(test_dir):
                cov = Coverage()
                if rt.PYTORCH_COLLECT_COVERAGE:
                    cov.load()
                cov.combine(strict=False)
                cov.save()
                if not rt.PYTORCH_COLLECT_COVERAGE:
                    cov.html_report()
    if has_failed is True:
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        tb = traceback.format_exc()
        print(tb)
