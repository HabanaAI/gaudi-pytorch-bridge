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

import datetime
import json
import multiprocessing
import os
from multiprocessing import Process, Queue

import pytest
import torch
import torch.multiprocessing as pt_mp
from habana_frameworks.torch.hpu.metrics import MetricNotFound, metric_global, metric_localcontext, metrics_dump
from habana_frameworks.torch.utils.event_dispatcher import EventDispatcher, EventId
from test_utils import env_var_in_scope, hpu

pytestmark = pytest.mark.skip(reason="Some of these tests don't free device and other tests are failing.")


@pytest.fixture(scope="module", autouse=True)
def set_multiprocess_start_method():
    # set spawn start method to not inherit already imported modules in child
    # processes used in metrics tests
    multiprocessing.set_start_method("spawn")


def compute_single_step(shape, device, sum_loops=1):
    dtype = torch.float32
    t1_cpu = torch.rand(shape, device="cpu", dtype=dtype)
    t2_cpu = torch.rand(shape, device="cpu", dtype=dtype)
    t1 = t1_cpu.to(device=device)
    t2 = t2_cpu.to(device=device)
    multiplied = t1 * t2
    summed = t1
    for _ in range(sum_loops):
        summed += t2
    out = summed * multiplied

    # move results to CPU, so compilation is enforced
    out = out.cpu()


class TestMetricsAPI:
    @pytest.fixture(scope="function")
    def gc_metric(self):
        m = metric_global("graph_compilation")
        m.reset()
        yield m

    def test_graph_compilation_metric_different_shapes_in_loop(self, gc_metric):
        shapes = [[10, 20, x] for x in range(1, 11)]
        device = hpu

        torch.random.manual_seed(42)

        last_total_time = 0
        for curr_iter, shape in enumerate(shapes):
            compute_single_step(shape, device, curr_iter + 1)
            gc_metric_dict = dict(gc_metric.stats())
            assert gc_metric_dict["TotalNumber"] == (curr_iter + 1)
            assert gc_metric_dict["TotalTime"] > last_total_time
            last_total_time = gc_metric_dict["TotalTime"]

            print(f"Current iteration {curr_iter}. GC metric: {gc_metric.stats()}")

    def test_graph_compilation_metric_same_shape_in_loop(self, gc_metric):
        device = hpu
        shape = [1, 2, 3]
        torch.random.manual_seed(42)

        total_time_of_last_iter = -1
        for curr_iter in range(10):
            compute_single_step(shape, device)
            gc_metric_dict = dict(gc_metric.stats())
            assert gc_metric_dict["TotalNumber"] == 1
            assert gc_metric_dict["TotalTime"] == total_time_of_last_iter or total_time_of_last_iter == -1

    @pytest.mark.skip(reason="Tests is chaning env variables")
    def test_graph_compilation_metric_different_shapes_in_loop(self, gc_metric, rc_metric):
        with env_var_in_scope({"PT_HPU_ENABLE_CACHE_METRICS": "1"}):
            shapes = [[10, 20, x] for x in range(1, 11)]
            device = torch.device("hpu")
            torch.random.manual_seed(42)

            last_total_time = 0
            for curr_iter, shape in enumerate(shapes):
                compute_single_step(shape, device, curr_iter + 1)
                gc_metric_dict = dict(gc_metric.stats())
                rc_metric_dict = dict(rc_metric.stats())
                assert gc_metric_dict["TotalNumber"] == (curr_iter + 1)
                assert rc_metric_dict["TotalMiss"] == (curr_iter + 1)
                assert gc_metric_dict["TotalTime"] > last_total_time
                last_total_time = gc_metric_dict["TotalTime"]

                print(f"Current iteration {curr_iter}. GC metric: {gc_metric.stats()}")

    @pytest.mark.skip(reason="Tests is chaning env variables")
    def test_graph_compilation_metric_same_shape_in_loop(self, gc_metric, rc_metric):
        with env_var_in_scope({"PT_HPU_ENABLE_CACHE_METRICS": "1"}):
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

    @staticmethod
    def _worker_metric_zero_at_beginning(q, metric_name):
        from habana_frameworks.torch.hpu import metric_global

        metric = metric_global(metric_name)
        metric_dict = dict(metric.stats())
        q.put(metric_dict)

    @pytest.mark.parametrize(
        "metric_name", [("graph_compilation"), ("cpu_fallback"), ("memory_defragmentation"), ("recipe_cache")]
    )
    def test_metric_zero_at_beginning(self, metric_name):
        """
        Spawns fresh process and verifies if metric are equal 0 at beginning.
        """
        q = Queue()
        p = Process(
            target=TestMetricsAPI._worker_metric_zero_at_beginning,
            args=(q, metric_name),
        )
        p.start()
        metric_dict = q.get(timeout=10)
        p.join()

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
        device = hpu
        shape = [3, 2, 1]
        torch.random.manual_seed(42)

        ed = EventDispatcher.instance()

        ed.subscribe(EventId.GRAPH_COMPILATION, lambda ts, p: print(f">>> lambda1 <<< {p}"))
        ed.subscribe(EventId.GRAPH_COMPILATION, lambda ts, p: print(f">>> lambda2 <<< {p}"))

        compute_single_step(shape, device)

        gc_metric_dict = dict(gc_metric.stats())
        assert gc_metric_dict["TotalNumber"] == 1
        assert gc_metric_dict["TotalTime"] > 0
        assert gc_metric_dict["AvgTime"] > 0

        print(f"GC metric: {gc_metric.stats()}")

    def test_metric_context_manager(self, gc_metric):
        shapes = [[10, 30, x] for x in range(1, 11)]
        device = hpu

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
        with pytest.raises(MetricNotFound):
            with metric_localcontext("non-existing") as m:  # noqa
                pass


class TestMetricsDump:
    @pytest.fixture(scope="function")
    def runner(self):
        """Runner runs each function in separate process, so every time metrics
        are being initialized separately. Runner takes environment variables
        as parameter, so each run can be executed with separate set of
        environmental variables.
        """

        def runner_func(worker_function, *args, env=None, **kwargs):
            with env_var_in_scope(env):
                p = Process(target=worker_function, args=args, kwargs=kwargs)
                p.start()
                p.join(timeout=30)

        yield runner_func

    @staticmethod
    def _sample_worker_process():
        device = hpu
        torch.random.manual_seed(42)

        compute_single_step([3, 2, 1], device)
        compute_single_step([3, 2, 12], device)
        compute_single_step([3, 2, 123], device)

        m = metric_global("graph_compilation")
        print(f"name={m.name()}, stats={m.stats()}")

    @pytest.mark.xfail(reason="File doesn't exists")
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
        env_vars = {
            "PT_HPU_METRICS_FILE": metric_file_user_input,
            "PT_HPU_METRICS_DUMP_TRIGGERS": "process_exit",
        }
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
        with open(metric_file, "r") as f:
            payload = f.read()
        parsed = TestMetricsDump._parse_dump(payload, format)

        assert len(parsed) == 4
        metric = parsed[0]
        assert metric["metric_name"] == "graph_compilation"
        assert metric["triggered_by"] == "process_exit"
        assert int(metric["statistics"]["TotalNumber"]) == 3
        assert int(metric["statistics"]["TotalTime"]) > 0

        # check generated_on field if is in iso format
        assert datetime.datetime.fromisoformat(metric["generated_on"])

    @pytest.mark.parametrize("format", ["json", "text"])
    def test_metric_dump_on_metric_change_and_process_exit(self, runner, tmp_path, format):
        metric_file = f"{tmp_path}/metric.{format}"
        env_vars = {
            "PT_HPU_METRICS_FILE": metric_file,
            "PT_HPU_METRICS_DUMP_TRIGGERS": "process_exit,metric_change",
            "PT_HPU_METRICS_FILE_FORMAT": format,
        }

        runner(TestMetricsDump._sample_worker_process, env=env_vars)
        with open(metric_file, "r") as f:
            payload = f.read()
        parsed = TestMetricsDump._parse_dump(payload, format)

        assert len(parsed) == 10  # 9 metrics chanages + process exit
        prev_total_time = 0
        prev_generated_on = None
        gc_only = [p for p in parsed if p["metric_name"] == "graph_compilation"]
        for idx, metric_on_metric_change in enumerate(gc_only[:3]):
            assert metric_on_metric_change["metric_name"] == "graph_compilation"
            assert metric_on_metric_change["triggered_by"] == "metric_change"
            assert int(metric_on_metric_change["statistics"]["TotalNumber"]) == (idx + 1)
            assert int(metric_on_metric_change["statistics"]["TotalTime"]) > prev_total_time
            prev_total_time = int(metric_on_metric_change["statistics"]["TotalTime"])

            curr_generated_on = datetime.datetime.fromisoformat(metric_on_metric_change["generated_on"])
            if prev_generated_on is not None:
                assert curr_generated_on > prev_generated_on
            prev_generated_on = curr_generated_on

        metric_on_process_exit = gc_only[-1]
        last_metric_on_metric_change = gc_only[-2]
        assert metric_on_process_exit["metric_name"] == "graph_compilation"
        assert metric_on_process_exit["triggered_by"] == "process_exit"
        assert int(metric_on_process_exit["statistics"]["TotalNumber"]) == 3
        assert int(metric_on_process_exit["statistics"]["TotalTime"]) == int(
            last_metric_on_metric_change["statistics"]["TotalTime"]
        )

    def test_metric_if_defaults_are_correct(self, runner, tmp_path):
        metric_file = f"{tmp_path}/metric.json"
        env_vars = {"PT_HPU_METRICS_FILE": metric_file}

        runner(TestMetricsDump._sample_worker_process, env=env_vars)
        with open(metric_file, "r") as f:
            payload = f.read()
        parsed = TestMetricsDump._parse_dump(payload, "json")

        assert len(parsed) == 4
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

        runner(TestMetricsDump._sample_worker_process_that_does_nothing, env=env_vars)
        assert not os.path.exists(metric_file)

    @staticmethod
    def worker_process_for_mp(rank, world_size, call_initialize_dist_hpu):
        if call_initialize_dist_hpu:
            import habana_frameworks.torch.distributed.hccl as hccl

            hccl.initialize_distributed_hpu(world_size, rank, rank)

        device = hpu
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

    @pytest.mark.parametrize("call_init_dist_hpu", [True, False])
    def test_metric_run_processes_via_torch_mp(self, runner, tmp_path, call_init_dist_hpu):
        metric_file = f"{tmp_path}/metric.json"
        env_vars = {"PT_HPU_METRICS_FILE": metric_file}

        world_size = 2

        runner(
            TestMetricsDump._sample_worker_running_processes_via_torch_mp,
            world_size,
            call_init_dist_hpu,
            env=env_vars,
        )

        for rank in range(world_size):
            if call_init_dist_hpu:
                core, ext = metric_file.rsplit(".")
                file_with_rank = f"{core}-rank{rank}.{ext}"
            else:
                file_with_rank = f"{metric_file}.{rank}" if rank > 0 else metric_file

            with open(file_with_rank, "r") as f:
                payload = f.read()
            parsed = TestMetricsDump._parse_dump(payload, "json")

            assert len(parsed) == 4
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
        device = hpu
        torch.random.manual_seed(42)

        compute_single_step([3, 2, 1], device)
        compute_single_step([3, 2, 12], device)

        metrics_dump(metric_file, metric_format)

    @pytest.mark.parametrize("format", ["json", "text"])
    def test_manual_metric_dump(self, runner, tmp_path, format):
        metric_file = f"{tmp_path}/metric.{format}"
        runner(
            TestMetricsDump._sample_worker_process_with_manual_metric_dump,
            metric_file,
            format,
        )

        with open(metric_file, "r") as f:
            payload = f.read()

        parsed = TestMetricsDump._parse_dump(payload, format)
        assert len(parsed) == 4
        metric = parsed[0]
        assert metric["metric_name"] == "graph_compilation"
        assert metric["triggered_by"] == "user"
        assert int(metric["statistics"]["TotalNumber"]) == 2
        assert int(metric["statistics"]["TotalTime"]) > 0
