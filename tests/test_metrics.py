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

import pytest
import torch
from habana_frameworks.torch.hpu.metrics import metric_global, metric_localcontext, MetricNotFound
from habana_frameworks.torch.utils.event_dispatcher import *
import multiprocessing as mp
from multiprocessing import Process, Queue


@pytest.fixture(scope="function")
def gc_metric():
    m = metric_global("gc")
    m.reset()
    yield m


def compute_single_step(shape, device):
    dtype = torch.float32
    t1_cpu = torch.rand(shape, device="cpu", dtype=dtype)
    t2_cpu = torch.rand(shape, device="cpu", dtype=dtype)
    t1 = t1_cpu.to(device=device)
    t2 = t2_cpu.to(device=device)
    multiplied = t1 * t2
    summed = t1 + t2
    out = summed * multiplied

    out = out.to(device="cpu")


def test_graph_compilation_metric_different_shapes_in_loop(gc_metric):
    shapes = [[10, 20, x] for x in range(1, 11)]
    device = torch.device('hpu')

    torch.random.manual_seed(42)

    last_total_time = 0
    for curr_iter, shape in enumerate(shapes):
        compute_single_step(shape, device)
        gc_metric_dict = dict(gc_metric.stats())
        assert gc_metric_dict["TotalNumber"] == (curr_iter + 1)
        assert gc_metric_dict["TotalTime"] > last_total_time
        last_total_time = gc_metric_dict["TotalTime"]

        print(f"Current iteration {curr_iter}. GC metric: {gc_metric.stats()}")


def test_graph_compilation_metric_same_shape_in_loop(gc_metric):
    device = torch.device('hpu')
    shape = [1, 2, 3]
    torch.random.manual_seed(42)

    total_time_of_last_iter = -1
    for curr_iter in range(10):
        compute_single_step(shape, device)
        gc_metric_dict = dict(gc_metric.stats())
        assert gc_metric_dict["TotalNumber"] == 1
        assert gc_metric_dict["TotalTime"] == total_time_of_last_iter or total_time_of_last_iter == -1

        print(f"Current iteration {curr_iter}. GC metric: {gc_metric.stats()}")


def _worker_graph_compilation_metric_zero_at_beginning(q):
    from habana_frameworks.torch.hpu import metric_global
    metric = metric_global("gc")
    metric_dict = dict(metric.stats())
    q.put(metric_dict)


def test_graph_compilation_metric_zero_at_beginning():
    """
    Spawns fresh process and verifies if metric are equal 0 at beginning.
    """
    mp.set_start_method('spawn')

    q = Queue()
    p = Process(target=_worker_graph_compilation_metric_zero_at_beginning, args=(q,))
    p.start()
    metric_dict = q.get(timeout=10)
    p.join()

    assert metric_dict["TotalNumber"] == 0
    assert metric_dict["TotalTime"] == 0
    assert metric_dict["AvgTime"] == 0


def test_graph_compilation_check_gc_global_metric_with_additional_event_handlers(gc_metric):
    device = torch.device('hpu')
    shape = [3, 2, 1]
    torch.random.manual_seed(42)

    ed = EventDispatcher.instance()

    h1 = ed.subscribe(EventId.GRAPH_COMPILATION, lambda p: print(f">>> lambda1 <<< {p}"))
    h2 = ed.subscribe(EventId.GRAPH_COMPILATION, lambda p: print(f">>> lambda2 <<< {p}"))

    compute_single_step(shape, device)

    gc_metric_dict = dict(gc_metric.stats())
    assert gc_metric_dict["TotalNumber"] == 1
    assert gc_metric_dict["TotalTime"] > 0
    assert gc_metric_dict["AvgTime"] > 0

    print(f"GC metric: {gc_metric.stats()}")


def test_metric_context_manager(gc_metric):
    shapes = [[10, 30, x] for x in range(1, 11)]
    device = torch.device('hpu')

    torch.random.manual_seed(42)

    shapes = iter(shapes)

    with metric_localcontext("gc") as outer_gc_metric:
        with metric_localcontext("gc") as inner_gc_metric:
            [compute_single_step(next(shapes), device) for i in range(3)]
        assert dict(inner_gc_metric.stats())["TotalNumber"] == 3

        with metric_localcontext("gc") as inner_gc_metric:
            [compute_single_step(next(shapes), device) for i in range(2)]
        assert dict(inner_gc_metric.stats())["TotalNumber"] == 2

        with metric_localcontext("gc") as inner_gc_metric:
            [compute_single_step(next(shapes), device) for i in range(3)]
        assert dict(inner_gc_metric.stats())["TotalNumber"] == 3

        with metric_localcontext("gc") as inner_gc_metric:
            [compute_single_step(next(shapes), device) for i in range(2)]
        assert dict(inner_gc_metric.stats())["TotalNumber"] == 2

    assert dict(outer_gc_metric.stats())["TotalNumber"] == 10

    gc_metric_dict = dict(gc_metric.stats())
    assert gc_metric_dict["TotalNumber"] == 10


def test_get_nonexisting_global_metric():
    metric = metric_global("non-existing metric")
    assert metric is None


def test_get_nonexisting_local_metric():
    with pytest.raises(MetricNotFound):
        with metric_localcontext("non-existing") as m:
            pass
