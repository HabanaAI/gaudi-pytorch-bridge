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
from enum import Enum
import json
import os
from datetime import datetime

from .exceptions import InvalidMetricDumpTrigger, InvalidMetricDumpFileFormat


class MetricWriter(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def write(self, name, stats, dump_trigger):
        raise NotImplementedError

    @abc.abstractmethod
    def close(self):
        raise NotImplementedError

    def __del__(self):
        self.close()


class MetricJsonWriter(MetricWriter):
    """Writes Metric stats into JSON file.

    Metric statistics which are passed to writer as a list of tuples in format
    [("name", value), ("name2", value2), ...] are converted to dict and stored
    in the following JSON object:
    {
        name: <metric name>
        stats: {
            <name>: <value>,
            ...
        }
    }
    Subsequent metrics are stored in an array.
    """

    class MetricJsonEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, datetime):
                return obj.isoformat()
            # Let the base class default method raise the TypeError
            return super().default(obj)

    def __init__(self, name):
        self._handle = open(f"{name}", "w")
        self._handle.write("[")
        self._handle.flush()
        self._first_obj_written = False

    def write(self, name, stats, dump_trigger, timestamp):
        """Writes object to JSON file.
        """
        obj_to_write = {
            "metric_name": name,
            "triggered_by": dump_trigger,
            "generated_on": timestamp,
            "statistics": dict(stats)
        }

        if self._first_obj_written:
            # if there is already an object written to file, then append coma and
            # new line characters
            self._handle.write(",\n")

        self._handle.write(json.dumps(obj_to_write, cls=self.MetricJsonEncoder))
        self._handle.flush()
        self._first_obj_written = True

    def close(self):
        # close root array
        if self._handle:
            self._handle.write("]\n")
            self._handle.flush()
            self._handle.close()
            self._handle = None


class MetricTextWriter(MetricWriter):
    """Writes Metric stats into TEXT file.
    """

    def __init__(self, name):
        self._handle = open(f"{name}", "w")

    def write(self, name, stats, dump_trigger, timestamp):
        """Writes object to TEXT file.
        """
        lines_to_write = [
            f"Metric name: {name}\n",
            f"Triggered by: {dump_trigger}\n",
            f"Generated on: {timestamp}\n",
            f"Statistics:\n",
            *[f"\t{stat_name}: {stat_value}\n" for stat_name, stat_value in stats],
            "\n"
        ]

        self._handle.writelines(lines_to_write)
        self._handle.flush()

    def close(self):
        if self._handle:
            self._handle.close()
            self._handle = None


class MetricNullWriter(MetricWriter):
    def __init__(self, name=""):
        pass

    def write(self, name, stats, dump_trigger, timestamp):
        pass

    def close(self):
        pass


class MetricDumpFormat(str, Enum):
    json = "json"
    text = "text"


class MetricDumpTrigger(str, Enum):
    process_exit = "process_exit"
    mark_step = "mark_step"
    metric_change = "metric_change"
    user = "user"


class MetricSaver:
    METRIC_FILE_ENV_VAR = "HABANA_PT_METRICS_FILE"

    METRIC_FILE_FORMAT_ENV_VAR = "HABANA_PT_METRICS_FILE_FORMAT"
    METRIC_FILE_FORMAT_DEFAULT = MetricDumpFormat.json

    METRIC_DUMP_TRIGGER_ENV_VAR = "HABANA_PT_METRICS_DUMP_TRIGGERS"
    METRIC_DUMP_TRIGGER_DEFAULT = ",".join([MetricDumpTrigger.process_exit])

    FORMAT_TO_WRITER_MAP = {
        MetricDumpFormat.json: MetricJsonWriter,
        MetricDumpFormat.text: MetricTextWriter,
    }

    def _get_metric_dump_trigger_from_env(self):
        dump_trigger = os.environ.get(self.METRIC_DUMP_TRIGGER_ENV_VAR, self.METRIC_DUMP_TRIGGER_DEFAULT)
        dump_trigger = [MetricDumpTrigger[trigger] for trigger in dump_trigger.split(",")]
        return dump_trigger

    def _get_metric_dump_format_from_env(self):
        metric_file_format = os.environ.get(self.METRIC_FILE_FORMAT_ENV_VAR, self.METRIC_FILE_FORMAT_DEFAULT)
        metric_file_format = MetricDumpFormat[metric_file_format]
        return metric_file_format

    def _determine_file_name_for_multinode(self, name):
        try:
            base_name, ext = name.rsplit(".", maxsplit=1)
            ext = f".{ext}"
        except ValueError:
            base_name = name
            ext = ""

        rank_env_vars = ["RANK", "OMPI_COMM_WORLD_RANK"]
        for env_var in rank_env_vars:
            if env_var in os.environ:
                return f"{base_name}-rank{os.environ[env_var]}{ext}"
        return f"{base_name}{ext}"

    def __init__(self, file_name="", triggers=[MetricDumpTrigger.user], format=METRIC_FILE_FORMAT_DEFAULT):
        if file_name:
            use_env = False
            metric_file_name = file_name
        else:
            use_env = True
            metric_file_name = os.environ.get(self.METRIC_FILE_ENV_VAR, None)

        saver_enabled = True if metric_file_name else False

        self._metric_dump_triggers = []
        self._metric_writer = MetricNullWriter()

        if saver_enabled:
            self._metric_dump_triggers = self._get_metric_dump_trigger_from_env() if use_env else triggers
            metric_file_format = self._get_metric_dump_format_from_env() if use_env else format

            metric_file_name = self._determine_file_name_for_multinode(metric_file_name)
            self._metric_writer = MetricSaver.FORMAT_TO_WRITER_MAP[metric_file_format](metric_file_name)

    @property
    def metric_change_callback(self):
        def callback(timestamp, metric_name, stats):
            if MetricDumpTrigger.metric_change in self._metric_dump_triggers and self._metric_writer is not None:
                self._metric_writer.write(metric_name, stats, MetricDumpTrigger.metric_change, timestamp=timestamp)

        return callback

    def process_trigger(self, trigger, metrics=[]):
        if trigger in self._metric_dump_triggers:
            for metric in metrics:
                self._metric_writer.write(metric.name(), metric.stats(
                ), trigger, timestamp=datetime.now())

    def close(self):
        self._metric_writer.close()
        self._metric_writer = MetricNullWriter()
