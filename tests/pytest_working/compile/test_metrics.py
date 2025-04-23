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

import datetime
import json
import os
from contextlib import contextmanager

import habana_frameworks.torch.internal.bridge_config as bc
import habana_frameworks.torch.utils.debug as htdebug
import pytest
import torch
import torch.multiprocessing as pt_mp
from habana_frameworks.torch.hpu.metrics import (
    MetricNotFound,
    metric_debug_atexit,
    metric_debug_enable_saver,
    metric_debug_reload,
    metric_global,
    metric_localcontext,
    metrics_dump,
)
from habana_frameworks.torch.utils.event_dispatcher import EventDispatcher, EventId
from test_utils import compile_function_if_compile_mode


def compute_single_step(shape, device, sum_loops=1):
    dtype = torch.float32
    t1_cpu = torch.rand(shape, device="cpu", dtype=dtype)
    t2_cpu = torch.rand(shape, device="cpu", dtype=dtype)

    def fn(t1, t2):
        multiplied = t1 * t2
        summed = t1
        for _ in range(sum_loops):
            summed += t2
        out = summed * multiplied
        return out

    fn = compile_function_if_compile_mode(fn)
    t1 = t1_cpu.to(device=device)
    t2 = t2_cpu.to(device=device)
    out = fn(t1, t2)

    # move results to CPU, so compilation is enforced
    out = out.cpu()


class TestMetricsAPI:
    prev_val_cache_metrics = False

    @classmethod
    def setup_class(cls):
        cls.prev_val_cache_metrics = bc.get_pt_hpu_enable_cache_metrics()
        bc.set_pt_hpu_enable_cache_metrics(True)
        metric_debug_reload()

    @classmethod
    def teardown_class(cls):
        bc.set_pt_hpu_enable_cache_metrics(cls.prev_val_cache_metrics)
        metric_debug_reload()

    @pytest.fixture(scope="function")
    def gc_metric(self):
        m = metric_global("graph_compilation")
        m.reset()
        yield m

    @pytest.fixture(scope="function")
    def rc_metric(self):
        m = metric_global("recipe_cache")
        m.reset()
        yield m

    def test_graph_compilation_metric_different_shapes_in_loop(self, gc_metric, rc_metric):
        shapes = [[10, 20, x] for x in range(1, 11)]
        device = torch.device("hpu")
        torch.random.manual_seed(42)
        last_total_time = 0
        for curr_iter, shape in enumerate(shapes):
            compute_single_step(shape, device, curr_iter + 1)
            gc_metric_dict = dict(gc_metric.stats())
            assert gc_metric_dict["TotalNumber"] == (curr_iter + 1)
            assert gc_metric_dict["TotalTime"] > last_total_time
            last_total_time = gc_metric_dict["TotalTime"]

            rc_metric_dict = dict(rc_metric.stats())
            assert rc_metric_dict["TotalMiss"] == (curr_iter + 1)

            print(f"Current iteration {curr_iter}. GC metric: {gc_metric.stats()}")

    def test_graph_compilation_metric_same_shape_in_loop(self, gc_metric, rc_metric):
        device = torch.device("hpu")
        shape = [1, 2, 3, 4]
        torch.random.manual_seed(42)
        total_time_of_last_iter = -1
        total_test_cases = 10
        for curr_iter in range(total_test_cases):
            compute_single_step(shape, device)
            gc_metric_dict = dict(gc_metric.stats())
            assert gc_metric_dict["TotalNumber"] == 1
            assert gc_metric_dict["TotalTime"] == total_time_of_last_iter or total_time_of_last_iter == -1

            print(f"Current iteration {curr_iter}. GC metric: {gc_metric.stats()}")

        rc_metric_dict = dict(rc_metric.stats())
        assert rc_metric_dict["TotalMiss"] == 1
        assert rc_metric_dict["TotalHit"] == total_test_cases - 1

    @pytest.mark.parametrize(
        "metric_name", [("graph_compilation"), ("cpu_fallback"), ("memory_defragmentation"), ("recipe_cache")]
    )
    def test_metric_zero_at_beginning(self, metric_name):
        metric_debug_reload()
        metric = metric_global(metric_name)
        metric_dict = dict(metric.stats())

        if metric_name == "recipe_cache":
            assert metric_dict["TotalHit"] == 0
            assert metric_dict["TotalMiss"] == 0
            assert len(metric_dict["RecipeHit"].items()) == 0
            assert len(metric_dict["RecipeMiss"].items()) == 0
            assert len(metric_dict.items()) == 4
        if metric_name == "cpu_fallback":
            assert metric_dict["TotalNumber"] == 0
            assert len(metric_dict.items()) == 2
            assert len(metric_dict["FallbackOps"].items()) == 0
        if metric_name == "graph_compilation":
            assert metric_dict["TotalNumber"] == 0
            assert metric_dict["TotalTime"] == 0
            assert metric_dict["AvgTime"] == 0
        if metric_name == "memory_defragmentation":
            assert metric_dict["TotalNumber"] == 0
            assert metric_dict["MaxTime"] == 0
            assert metric_dict["TotalSuccessful"] == 0

    def test_graph_compilation_check_gc_global_metric_with_additional_event_handlers(self, gc_metric):
        device = torch.device("hpu")
        shape = [3, 2, 1]
        torch.random.manual_seed(42)

        ed = EventDispatcher.instance()

        h1 = ed.subscribe(EventId.GRAPH_COMPILATION)
        h2 = ed.subscribe(EventId.GRAPH_COMPILATION)

        compute_single_step(shape, device)

        gc_metric_dict = dict(gc_metric.stats())
        assert gc_metric_dict["TotalNumber"] == 1
        assert gc_metric_dict["TotalTime"] > 0
        assert gc_metric_dict["AvgTime"] > 0

        print(f"GC metric: {gc_metric.stats()}")

    def test_graph_compilation_check_gc_details(self, gc_metric, rc_metric):
        with env_var_in_scope({"PT_HPU_METRICS_GC_DETAILS": "1"}):
            shapes = [[10, 30, x] for x in range(1, 5)]
            device = torch.device("hpu")
            torch.random.manual_seed(42)
            last_recipe_number = 0
            for curr_iter, shape in enumerate(shapes):
                compute_single_step(shape, device, curr_iter + 1)
                gc_metric_dict = dict(gc_metric.stats())
                assert "recipe" in gc_metric_dict

                gc_metric_details_list = gc_metric_dict["recipe"]
                assert len(gc_metric_details_list) > last_recipe_number
                assert all(gc_metric_details_list[-1])
                last_recipe_number = len(gc_metric_details_list)

                print(f"Current iteration {curr_iter}. GC metric: {gc_metric.stats()}")

    def test_metric_context_manager(self, gc_metric):
        shapes = [[10, 30, x] for x in range(1, 11)]
        device = torch.device("hpu")

        torch.random.manual_seed(42)

        shapes = iter(shapes)

        with metric_localcontext("graph_compilation") as outer_gc_metric:
            with metric_localcontext("graph_compilation") as inner_gc_metric:
                [compute_single_step(next(shapes), device, i + 1) for i in range(1, 4)]
            assert dict(inner_gc_metric.stats())["TotalNumber"] == 3

            with metric_localcontext("graph_compilation") as inner_gc_metric:
                [compute_single_step(next(shapes), device, i + 1) for i in range(4, 6)]
            assert dict(inner_gc_metric.stats())["TotalNumber"] == 2

            with metric_localcontext("graph_compilation") as inner_gc_metric:
                [compute_single_step(next(shapes), device, i + 1) for i in range(6, 9)]
            assert dict(inner_gc_metric.stats())["TotalNumber"] == 3

            with metric_localcontext("graph_compilation") as inner_gc_metric:
                [compute_single_step(next(shapes), device, i + 1) for i in range(9, 11)]
            assert dict(inner_gc_metric.stats())["TotalNumber"] == 2

        assert dict(outer_gc_metric.stats())["TotalNumber"] == 10

        gc_metric_dict = dict(gc_metric.stats())
        assert gc_metric_dict["TotalNumber"] == 10

    def test_get_nonexisting_global_metric(self):
        metric = metric_global("non-existing metric")
        assert metric is None

    def test_get_nonexisting_local_metric(self):
        with pytest.raises(MetricNotFound), metric_localcontext("non-existing") as m:
            pass


def set_flag_in_env(name: str, value):
    if value is None:
        # Nothing to do here
        return
    elif isinstance(value, str):
        os.environ[name] = value
    elif isinstance(value, bool):
        os.environ[name] = str(int(value))
    elif isinstance(value, int):
        os.environ[name] = str(value)
    else:
        assert False, f"Value '{value}' invalid or not supported"


@contextmanager
def env_var_in_scope(vars={}):
    orig_vars = {}
    for key in vars.keys():
        orig_vars[key] = os.environ.get(key, None)
        set_flag_in_env(key, vars[key])
    try:
        yield
    finally:
        for key, orig_var in orig_vars.items():
            # restore environment variable
            if orig_var is not None:
                os.environ[key] = orig_var
            elif key in os.environ:
                del os.environ[key]


class TestMetricsDump:
    prev_val_cache_metrics = False

    @classmethod
    def setup_class(cls):
        cls.prev_val_cache_metrics = bc.get_pt_hpu_enable_cache_metrics()
        bc.set_pt_hpu_enable_cache_metrics(True)
        metric_debug_reload()

    @classmethod
    def teardown_class(cls):
        bc.set_pt_hpu_enable_cache_metrics(cls.prev_val_cache_metrics)
        metric_debug_reload()

    @pytest.fixture(scope="function")
    def runner(self):
        """Runner runs each function in separate process, so every time metrics
        are being initialized separately. Runner takes environment variables
        as parameter, so each run can be executed with separate set of
        environmental variables.
        """

        def runner_func(worker_function, *args, env={}, **kwargs):
            with env_var_in_scope(env):
                metric_debug_reload()
                metric_debug_enable_saver()
                worker_function(*args, **kwargs)
                htdebug.clear_dynamic_bucket_recipe_info()
                metric_debug_atexit()

        yield runner_func

    @staticmethod
    def _sample_worker_process():
        device = torch.device("hpu")
        torch.random.manual_seed(42)

        compute_single_step([3, 2, 1], device)
        compute_single_step([3, 2, 12], device)
        compute_single_step([3, 2, 123], device)

        m = metric_global("graph_compilation")
        print(f"name={m.name()}, stats={m.stats()}")

    @pytest.mark.parametrize(
        "base_name,multinode,expected_base_name",
        [
            ("metric_file", False, "metric_file"),
            ("metric_file.txt", False, "metric_file.txt"),
            ("metric_file", True, "metric_file-rank0"),
            ("metric_file.json", True, "metric_file-rank0.json"),
            ("metric_file.txt.json", True, "metric_file.txt-rank0.json"),
        ],
    )
    def test_metric_file_with_correct_name_is_created(self, runner, tmp_path, base_name, multinode, expected_base_name):
        metric_file_user_input = f"{tmp_path}/{base_name}"
        metric_file_target = f"{tmp_path}/{expected_base_name}"

        assert not os.path.exists(metric_file_target)
        env_vars = {"PT_HPU_METRICS_FILE": metric_file_user_input, "PT_HPU_METRICS_DUMP_TRIGGERS": "process_exit"}
        if multinode:
            env_vars["RANK"] = "0"

        runner(TestMetricsDump._sample_worker_process, env=env_vars)

        assert os.path.exists(metric_file_target)

    @staticmethod
    def _parse_text_obj(lines, curr_line, curr_root, curr_root_indent=0):
        OBJ_NAME_MAP = {
            "Metric name": "metric_name",
            "Generated on": "generated_on",
            "Triggered by": "triggered_by",
            "Statistics": "statistics",
        }

        while curr_line < len(lines):
            line = lines[curr_line]
            if line == "":
                # end of object
                break

            key, value = line.split(":", maxsplit=1)
            value = value.strip()
            indent = key.count("\t")
            assert curr_root_indent == indent
            key = key.strip()
            if key in OBJ_NAME_MAP:
                key = OBJ_NAME_MAP[key]

            if value == "":  # new sub-object
                curr_root[key] = {}
                curr_line = TestMetricsDump._parse_text_obj(lines, curr_line + 1, curr_root[key], curr_root_indent + 1)
            else:
                curr_root[key] = value
                curr_line += 1
        return curr_line

    @staticmethod
    def _parse_text(payload):
        lines = payload.split("\n")

        metrics = []
        num_processed_lines = 0

        while num_processed_lines < len(lines):
            root = {}
            num_processed_lines = TestMetricsDump._parse_text_obj(lines, num_processed_lines, root, 0)
            num_processed_lines += 1
            if root:
                metrics.append(root)

        assert num_processed_lines >= len(lines)
        return metrics

    @staticmethod
    def _parse_json(payload):
        return json.loads(payload)

    @staticmethod
    def _parse_dump(payload, format):
        if format == "text":
            return TestMetricsDump._parse_text(payload)
        if format == "json":
            return TestMetricsDump._parse_json(payload)
        return None

    @pytest.mark.parametrize("format", ["json", "text"])
    def test_metric_dump_on_process_exit(self, runner, tmp_path, format):
        metric_file = f"{tmp_path}/metric.{format}"
        env_vars = {
            "PT_HPU_METRICS_FILE": metric_file,
            "PT_HPU_METRICS_DUMP_TRIGGERS": "process_exit",
            "PT_HPU_METRICS_FILE_FORMAT": format,
        }

        runner(TestMetricsDump._sample_worker_process, env=env_vars)
        with open(metric_file) as f:
            payload = f.read()
        parsed = TestMetricsDump._parse_dump(payload, format)

        assert len(parsed) == 5
        metric = parsed[0]
        assert metric["metric_name"] == "graph_compilation"
        assert metric["triggered_by"] == "process_exit"
        assert int(metric["statistics"]["TotalNumber"]) == 3
        assert int(metric["statistics"]["TotalTime"]) > 0

        # check generated_on field if is in iso format
        assert datetime.datetime.fromisoformat(metric["generated_on"])

    @pytest.mark.parametrize("format", ["json", "text"])
    def test_metric_dump_on_process_exit(self, runner, tmp_path, format):
        metric_file = f"{tmp_path}/metric.{format}"
        env_vars = {
            "PT_HPU_METRICS_FILE": metric_file,
            "PT_HPU_METRICS_DUMP_TRIGGERS": "process_exit",
            "PT_HPU_METRICS_FILE_FORMAT": format,
        }

        runner(TestMetricsDump._sample_worker_process, env=env_vars)
        with open(metric_file) as f:
            payload = f.read()
        parsed = TestMetricsDump._parse_dump(payload, format)

        assert len(parsed) == 5
        gc_only = [p for p in parsed if p["metric_name"] == "graph_compilation"]

        metric_on_process_exit = gc_only[-1]
        assert metric_on_process_exit["metric_name"] == "graph_compilation"
        assert metric_on_process_exit["triggered_by"] == "process_exit"
        assert int(metric_on_process_exit["statistics"]["TotalNumber"]) == 3

    def test_metric_if_defaults_are_correct(self, runner, tmp_path):
        metric_file = f"{tmp_path}/metric.json"
        env_vars = {"PT_HPU_METRICS_FILE": metric_file}

        runner(TestMetricsDump._sample_worker_process, env=env_vars)
        with open(metric_file) as f:
            payload = f.read()
        parsed = TestMetricsDump._parse_dump(payload, "json")

        assert len(parsed) == 5
        metric = parsed[0]
        assert metric["metric_name"] == "graph_compilation"
        assert metric["triggered_by"] == "process_exit"
        assert int(metric["statistics"]["TotalNumber"]) == 3
        assert int(metric["statistics"]["TotalTime"]) > 0

        # check generated_on field if is in iso format
        assert datetime.datetime.fromisoformat(metric["generated_on"])

    @staticmethod
    def _sample_worker_process_that_does_nothing():
        pass

    def test_metric_no_dump_when_dev_not_acquired(self, runner, tmp_path):
        metric_file = f"{tmp_path}/metric.json"
        env_vars = {"PT_HPU_METRICS_FILE": metric_file}

        with env_var_in_scope(env_vars):
            metric_debug_reload()
            metric_debug_atexit()

        assert not os.path.exists(metric_file)

    @staticmethod
    def worker_process_for_mp(rank, world_size, call_initialize_dist_hpu):
        if call_initialize_dist_hpu:
            from habana_frameworks.torch.distributed import hccl

            hccl.initialize_distributed_hpu(world_size, rank, rank)

        device = torch.device("hpu")
        torch.random.manual_seed(42)
        compute_single_step([3, 2, 1], device)

    @staticmethod
    def _sample_worker_running_processes_via_torch_mp(world_size, call_initialize_dist_hpu):
        pt_mp.start_processes(
            TestMetricsDump.worker_process_for_mp,
            nprocs=world_size,
            args=(world_size, call_initialize_dist_hpu),
            daemon=False,
            start_method="spawn",
        )

    @pytest.mark.skip(reason="Test is not adjusted to single node")
    @pytest.mark.parametrize("call_init_dist_hpu", [True, False])
    def test_metric_run_processes_via_torch_mp(self, runner, tmp_path, call_init_dist_hpu):
        metric_file = f"{tmp_path}/metric.json"
        env_vars = {"PT_HPU_METRICS_FILE": metric_file}

        world_size = 2

        runner(
            TestMetricsDump._sample_worker_running_processes_via_torch_mp, world_size, call_init_dist_hpu, env=env_vars
        )

        for rank in range(world_size):
            if call_init_dist_hpu:
                core, ext = metric_file.rsplit(".")
                file_with_rank = f"{core}-rank{rank}.{ext}"
            else:
                file_with_rank = f"{metric_file}.{rank}" if rank > 0 else metric_file

            with open(file_with_rank) as f:
                payload = f.read()
            parsed = TestMetricsDump._parse_dump(payload, "json")

            assert len(parsed) == 5
            metric = parsed[0]
            assert metric["metric_name"] == "graph_compilation"
            assert metric["triggered_by"] == "process_exit"
            assert int(metric["statistics"]["TotalNumber"]) == 1
            assert int(metric["statistics"]["TotalTime"]) > 0

        if call_init_dist_hpu:
            assert not os.path.exists(metric_file), (
                "When 'initialize_distributed_hpu' is called then metrics should"
                " be stored in files with suffix 'rankX'"
            )

    @staticmethod
    def _sample_worker_process_with_manual_metric_dump(metric_file, metric_format):
        device = torch.device("hpu")
        torch.random.manual_seed(42)

        compute_single_step([3, 2, 1], device)
        compute_single_step([3, 2, 12], device)

        metrics_dump(metric_file, metric_format)

    @pytest.mark.parametrize("format", ["json", "text"])
    def test_manual_metric_dump(self, runner, tmp_path, format):
        metric_file = f"{tmp_path}/metric.{format}"
        runner(TestMetricsDump._sample_worker_process_with_manual_metric_dump, metric_file, format)

        with open(metric_file) as f:
            payload = f.read()

        parsed = TestMetricsDump._parse_dump(payload, format)
        assert len(parsed) == 5
        metric = parsed[0]
        assert metric["metric_name"] == "graph_compilation"
        assert metric["triggered_by"] == "user"
        assert int(metric["statistics"]["TotalNumber"]) == 2
        assert int(metric["statistics"]["TotalTime"]) > 0


def test_event_process():
    count = 0

    def increment(event_params):
        nonlocal count
        count += 1

    ed = EventDispatcher.instance()
    event_handle1 = ed.subscribe(EventId.CUSTOM_EVENT)
    event_handle2 = ed.subscribe(EventId.CUSTOM_EVENT)
    ed.publish(EventId.CUSTOM_EVENT, [])
    ed.publish(EventId.CUSTOM_EVENT, [])
    ed.process(event_handle1, increment)
    assert count == 2
    ed.publish(EventId.CUSTOM_EVENT, [])
    ed.publish(EventId.CUSTOM_EVENT, [])
    ed.publish(EventId.CUSTOM_EVENT, [])
    ed.process(event_handle1, increment)
    assert count == 5
    ed.process(event_handle2, increment)
    assert count == 10
    ed.process(event_handle1, increment)
    assert count == 10

    ed.unsubscribe(event_handle1)
    ed.unsubscribe(event_handle2)
