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
import abc
import atexit
import os
from contextlib import contextmanager
from statistics import mean
from typing import Sequence, Tuple

from habana_frameworks.torch.utils.event_dispatcher import EventDispatcher, EventId

from .exceptions import MetricNotFound
from .saver import MetricDumpFormat, MetricDumpTrigger, MetricSaver


def bool_helper(value):
    value = value.lower()
    if value in ("y", "yes", "t", "true", "on", "1"):
        return True
    else:
        return False


class MetricManager(object):
    def __init__(self) -> None:
        self._metrics_types = {}
        self._global_metrics = []
        self._metric_saver = MetricSaver()

        atexit.register(self._at_exit_callback)

        def mark_step_event_callback_fn(timestamp, event_params):
            self._metric_saver.process_trigger(MetricDumpTrigger.mark_step, self._global_metrics)

        def dev_acquired_event_callback_fn(timestamp, event_params):
            self._metric_saver.enable()

        ed = EventDispatcher.instance()
        self._mark_step_event_handle = ed.subscribe(EventId.MARK_STEP, mark_step_event_callback_fn)

        self._dev_acquired_event_handle = ed.subscribe(EventId.DEVICE_ACQUIRED, dev_acquired_event_callback_fn)

    def __del__(self):
        atexit.unregister(self._at_exit_callback)

        ed = EventDispatcher.instance()
        ed.unsubscribe(self._mark_step_event_handle)
        self._mark_step_event_handle = None

        ed.unsubscribe(self._dev_acquired_event_handle)
        self._dev_acquired_event_handle = None

        for global_metric in self._global_metrics:
            del global_metric
        self._global_metrics = []

        del self._metric_saver
        self._metric_saver = None

    def register(self, name, metric_class):
        assert name not in self._metrics_types, f"Metric with given name ({name}) is already registered"

        self._metrics_types[name] = metric_class
        self._global_metrics.append(metric_class())
        self._global_metrics[-1].on_metric_change(self._metric_saver.metric_change_callback)

    def get_global_metric(self, name: str):
        metrics = [m for m in self._global_metrics if m.name() == name]
        assert len(metrics) <= 1, "There are more than one metric with given name"

        return metrics[0] if len(metrics) == 1 else None

    def get_local_metric(self, name: str):
        if name in self._metrics_types:
            return self._metrics_types[name]()
        else:
            raise MetricNotFound(f"Metric with given name ({name}) doesn't exist.")

    def _at_exit_callback(self):
        self._metric_saver.process_trigger(MetricDumpTrigger.process_exit, self._global_metrics)
        self._metric_saver.close()

    def store_global_metrics(self, file_name, format):
        saver = MetricSaver(file_name, triggers=[MetricDumpTrigger.user], format=format)
        saver.enable()
        saver.process_trigger(MetricDumpTrigger.user, self._global_metrics)
        saver.close()


class Metric(metaclass=abc.ABCMeta):
    def __init__(self):
        self._metric_change_callback = None

    @abc.abstractmethod
    def name(self) -> str:
        """Returns metric name."""
        raise NotImplementedError

    @abc.abstractmethod
    def stats(self) -> Sequence[Tuple[str, int]]:
        """Returns list of tuples describing collected statistics.
        Each statistic is described as statistic name and its count.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def reset(self) -> None:
        """Resets collected statistics."""
        raise NotImplementedError

    @abc.abstractmethod
    def start(self) -> None:
        """Starts collecting statistics."""
        raise NotImplementedError

    @abc.abstractmethod
    def stop(self) -> None:
        """Stops collecting statistics."""
        raise NotImplementedError

    def on_metric_change(self, callback) -> None:
        """Registers callback called on every metric change."""
        self._metric_change_callback = callback

    def notify(self, timestamp, event_params):
        if self._metric_change_callback:
            self._metric_change_callback(timestamp, self.name(), self.stats())


class MemoryDefragmentationMetric(Metric):
    _TOTAL_DEFRAGMENTATION_TAG = "TotalNumber"
    _TOTAL_DEFRAGMENTATION_SUCCESSFUL_TAG = "TotalSuccessful"
    _DEFRAGMENTATION_SUCCESS_EVENT_NAME = "success"
    _DEFRAGMENTATION_MILLISECONDS_EVENT_NAME = "milliseconds"
    _TOTAL_DEFRAGMENTATION_MAX_TAG = "MaxTime"
    _TOTAL_DEFRAGMENTATION_MEAN_TAG = "AvgTime"

    def __init__(self):
        self._total_defragmentation_count = 0
        self._total_successful_defragmentation_count = 0
        self._defragmentation_time = []
        self._ed = EventDispatcher.instance()
        self._handle = None
        self.start()

    def _get_event_callback_fn(self):
        def callback(timestamp, event_params):
            event_params = dict(event_params)
            self._total_defragmentation_count += 1
            if bool_helper(event_params[self._DEFRAGMENTATION_SUCCESS_EVENT_NAME]):
                self._total_successful_defragmentation_count += 1
                self._defragmentation_time.append(int(event_params[self._DEFRAGMENTATION_MILLISECONDS_EVENT_NAME]))
            self.notify(timestamp, event_params)

        return callback

    def name(self):
        return "memory_defragmentation"

    def start(self):
        if not self._handle:
            self._handle = self._ed.subscribe(EventId.MEMORY_DEFRAGMENTATION, self._get_event_callback_fn())

    def stop(self):
        if self._handle:
            self._ed.unsubscribe(self._handle)
            self._handle = None

    def stats(self):
        return [
            (self._TOTAL_DEFRAGMENTATION_TAG, self._total_defragmentation_count),
            (self._TOTAL_DEFRAGMENTATION_SUCCESSFUL_TAG, self._total_successful_defragmentation_count),
            (
                self._TOTAL_DEFRAGMENTATION_MEAN_TAG,
                mean(self._defragmentation_time) if len(self._defragmentation_time) > 0 else 0,
            ),
            (
                self._TOTAL_DEFRAGMENTATION_MAX_TAG,
                max(self._defragmentation_time) if len(self._defragmentation_time) > 0 else 0,
            ),
        ]

    def reset(self):
        self._total_defragmentation_count = 0
        self._total_successful_defragmentation_count = 0
        self._defragmentation_time.clear()

    def __del__(self):
        self.stop()


class RecipeCacheMetric(Metric):
    _TOTAL_CACHE_HIT_TAG = "TotalHit"
    _RECIPE_CACHE_HIT_TAG = "RecipeHit"
    _TOTAL_CACHE_MISS_TAG = "TotalMiss"
    _RECIPE_CACHE_MISS_TAG = "RecipeMiss"
    _RECIPE_ID_EVENT_NAME = "recipe_id"

    def __init__(self):
        self.total_cache_hit = 0
        self.total_recipe_cache_hit = {}
        self.total_cache_miss = 0
        self.total_recipe_cache_miss = {}
        self._ed = EventDispatcher.instance()
        self._handle_hit = None
        self._handle_miss = None
        self.start()

    def _get_hit_event_callback_fn(self):
        def callback(timestamp, event_params):
            event_params = dict(event_params)
            self.total_cache_hit += 1
            recipe_id = event_params[self._RECIPE_ID_EVENT_NAME]
            self.total_recipe_cache_hit[recipe_id] = self.total_recipe_cache_hit.get(recipe_id, 0) + 1
            self.notify(timestamp, event_params)

        return callback

    def _get_miss_event_callback_fn(self):
        def callback(timestamp, event_params):
            event_params = dict(event_params)
            self.total_cache_miss += 1
            recipe_id = event_params[self._RECIPE_ID_EVENT_NAME]
            self.total_recipe_cache_miss[recipe_id] = self.total_recipe_cache_miss.get(recipe_id, 0) + 1
            self.notify(timestamp, event_params)

        return callback

    def name(self):
        return "recipe_cache"

    def start(self):
        if not self._handle_hit:
            self._handle_hit = self._ed.subscribe(EventId.CACHE_HIT, self._get_hit_event_callback_fn())
        if not self._handle_miss:
            self._handle_miss = self._ed.subscribe(EventId.CACHE_MISS, self._get_miss_event_callback_fn())

    def stop(self):
        if self._handle_hit:
            self._ed.unsubscribe(self._handle_hit)
            self._handle_hit = None
        if self._handle_miss:
            self._ed.unsubscribe(self._handle_miss)
            self._handle_miss = None

    def stats(self):
        return [
            (self._TOTAL_CACHE_HIT_TAG, self.total_cache_hit),
            (self._RECIPE_CACHE_HIT_TAG, self.total_recipe_cache_hit),
            (self._TOTAL_CACHE_MISS_TAG, self.total_cache_miss),
            (self._RECIPE_CACHE_MISS_TAG, self.total_recipe_cache_miss),
        ]

    def reset(self):
        self.total_cache_hit = 0
        self.total_recipe_cache_hit = {}
        self.total_cache_miss = 0
        self.total_recipe_cache_miss = {}

    def __del__(self):
        self.stop()


class CpuFallbackMetric(Metric):
    _TOTAL_FALLBACKS_TAG = "TotalNumber"
    _FALLBACK_OPS_TAG = "FallbackOps"
    _OP_NAME_EVENT_NAME = "op_name"

    def __init__(self):
        self._total_fallback_count = 0
        self._total_op_fallback_count = dict()
        self._ed = EventDispatcher.instance()
        self._handle = None
        self.start()

    def _get_event_callback_fn(self):
        def callback(timestamp, event_params):
            event_params = dict(event_params)
            self._total_fallback_count += 1
            op_name = event_params[self._OP_NAME_EVENT_NAME]
            self._total_op_fallback_count[op_name] = self._total_op_fallback_count.get(op_name, 0) + 1
            self.notify(timestamp, event_params)

        return callback

    def name(self):
        return "cpu_fallback"

    def start(self):
        if not self._handle:
            self._handle = self._ed.subscribe(EventId.CPU_FALLBACK, self._get_event_callback_fn())

    def stop(self):
        if self._handle:
            self._ed.unsubscribe(self._handle)
            self._handle = None

    def stats(self):
        return [
            (self._TOTAL_FALLBACKS_TAG, self._total_fallback_count),
            (self._FALLBACK_OPS_TAG, self._total_op_fallback_count),
        ]

    def reset(self):
        self._total_fallback_count = 0
        self._total_op_fallback_count = dict()

    def __del__(self):
        self.stop()


class GraphCompilationMetric(Metric):
    _TOTAL_NUMBER_TAG = "TotalNumber"
    _TOTAL_TIME_TAG = "TotalTime"
    _AVG_TIME_TAG = "AvgTime"
    _DURATION_EVENT_PARAM_NAME = "duration"
    _RECIPE_PARAM_NAME = "recipe"

    def __init__(self):
        super().__init__()
        self._total_num_of_compilation = 0
        self._total_time_of_compilation = 0
        self._ed = EventDispatcher.instance()
        self._handle = None
        self._recipe_names = []
        self._recipe_durations = []
        self.start()

    def _get_event_callback_fn(self):
        def callback_fn(timestamp, event_params):
            event_params = dict(event_params)
            self._total_num_of_compilation += 1
            self._total_time_of_compilation += int(event_params[self._DURATION_EVENT_PARAM_NAME])
            self._recipe_names.append(event_params[self._RECIPE_PARAM_NAME])
            self._recipe_durations.append(int(event_params[self._DURATION_EVENT_PARAM_NAME]))
            self.notify(timestamp, event_params)

        return callback_fn

    def name(self):
        return "graph_compilation"

    def start(self):
        if not self._handle:
            self._handle = self._ed.subscribe(EventId.GRAPH_COMPILATION, self._get_event_callback_fn())

    def stop(self):
        if self._handle:
            self._ed.unsubscribe(self._handle)
            self._handle = None

    def stats(self):
        result = {
            self._TOTAL_NUMBER_TAG: self._total_num_of_compilation,
            self._TOTAL_TIME_TAG: self._total_time_of_compilation,
            self._AVG_TIME_TAG: (
                float(self._total_time_of_compilation) / self._total_num_of_compilation
                if self._total_num_of_compilation > 0
                else 0
            ),
        }
        if "PT_HPU_METRICS_GC_DETAILS" in os.environ and bool_helper(os.getenv("PT_HPU_METRICS_GC_DETAILS")):
            result[self._RECIPE_PARAM_NAME] = list(
                map(lambda name, duration: (name, duration), self._recipe_names, self._recipe_durations)
            )
        return [(tag, value) for tag, value in result.items()]

    def reset(self):
        self._total_num_of_compilation = 0
        self._total_time_of_compilation = 0

    def __del__(self):
        self.stop()


_metric_mgr = None


def _init_metric_mgr():
    global _metric_mgr
    _metric_mgr = MetricManager()
    _metric_mgr.register("graph_compilation", GraphCompilationMetric)
    _metric_mgr.register("cpu_fallback", CpuFallbackMetric)
    _metric_mgr.register("memory_defragmentation", MemoryDefragmentationMetric)
    if "PT_HPU_ENABLE_CACHE_METRICS" in os.environ and bool_helper(os.getenv("PT_HPU_ENABLE_CACHE_METRICS")):
        _metric_mgr.register("recipe_cache", RecipeCacheMetric)


_init_metric_mgr()


def metric_global(name: str) -> Metric:
    """Returns global metric by name."""
    return _metric_mgr.get_global_metric(name)


@contextmanager
def metric_localcontext(name: str) -> Metric:
    """Context-manager metric API.

    Metric collection will start when entering the scope, and stop when exits
    the scope.

    Example usage:
    ```python
    with metric_localcontext("graph_compilation") as gc_local_metric:
      # do some work
      print(gc_local_metric)
    """
    metric = _metric_mgr.get_local_metric(name)

    try:
        yield metric
    finally:
        metric.stop()


def metrics_dump(file_name: str, format: MetricDumpFormat = MetricDumpFormat.json) -> None:
    """Stores global metrics in given file."""
    _metric_mgr.store_global_metrics(file_name, format=format)
