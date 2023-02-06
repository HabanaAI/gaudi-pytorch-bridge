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
from typing import Sequence, Tuple
import habana_frameworks.torch.utils.event_dispatcher as ed
from contextlib import contextmanager


class _MetricManager(object):
    def __init__(self) -> None:
        self._metrics_types = {}
        self._global_metrics = []

    def register(self, name, metric_class):
        assert name not in self._metrics_types, f"Metric with given name ({name}) is already registered"
        self._metrics_types[name] = metric_class
        self._global_metrics.append(metric_class())

    def get_global_metric(self, name: str):
        metrics = [m for m in self._global_metrics if m.name() == name]
        assert len(metrics) <= 1, "There are more than one metric with given name"

        return metrics[0] if len(metrics) == 1 else None

    def get_local_metric(self, name: str):
        if name in self._metrics_types:
            return self._metrics_types[name]()
        else:
            raise MetricNotFound(f"Metric with given name ({name}) doesn't exist.")


class MetricNotFound(Exception):
    pass


class Metric(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def name(self) -> str:
        """Returns metric name.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def stats(self) -> Sequence[Tuple[str, int]]:
        """Returns list of tuples describing collected statistics.
        Each statistic is described as statistic name and its count.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def reset(self) -> None:
        """Resets collected statistics.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def start(self) -> None:
        """Starts collecting statistics.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def stop(self) -> None:
        """Stops collecting statistics.
        """
        raise NotImplementedError


class GraphCompilationMetric(Metric):
    _TOTAL_NUMBER_TAG = "TotalNumber"
    _TOTAL_TIME_TAG = "TotalTime"
    _AVG_TIME_TAG = "AvgTime"
    _DURATION_EVENT_PARAM_NAME = "duration"

    def __init__(self):
        self._total_num_of_compilation = 0
        self._total_time_of_compilation = 0
        self._ed = ed.EventDispatcher.instance()
        self._handle = None
        self.start()

    def _get_callback_fn(self):
        def callback_fn(event_params):
            event_params = dict(event_params)
            self._total_num_of_compilation += 1
            self._total_time_of_compilation += event_params[self._DURATION_EVENT_PARAM_NAME]

        return callback_fn

    def name(self):
        return "gc"

    def start(self):
        if not self._handle:
            self._handle = self._ed.subscribe(ed.EventId.GRAPH_COMPILATION, self._get_callback_fn())

    def stop(self):
        if self._handle:
            self._ed.unsubscribe(self._handle)
            self._handle = None

    def stats(self):
        result = {
            self._TOTAL_NUMBER_TAG: self._total_num_of_compilation,
            self._TOTAL_TIME_TAG: self._total_time_of_compilation,
            self._AVG_TIME_TAG: float(self._total_time_of_compilation) /
            self._total_num_of_compilation if self._total_num_of_compilation > 0 else 0
        }
        return [(tag, value) for tag, value in result.items()]

    def reset(self):
        self._total_num_of_compilation = 0
        self._total_time_of_compilation = 0

    def __del__(self):
        self.stop()


_metric_mgr = _MetricManager()
_metric_mgr.register("gc", GraphCompilationMetric)


def metric_global(name: str) -> Metric:
    """Returns global metric by name.
    """
    return _metric_mgr.get_global_metric(name)


@contextmanager
def metric_localcontext(name: str) -> Metric:
    """Context-manager metric API.

      Metric collection will start when entering the scope, and stop when exits
      the scope.

      Example usage:
      ```python
      with metric_localcontext("gc") as gc_local_metric:
        # do some work
        print(gc_local_metric)
    """
    metric = _metric_mgr.get_local_metric(name)

    try:
        yield metric
    finally:
        metric.stop()
