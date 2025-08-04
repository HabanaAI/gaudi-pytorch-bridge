###############################################################################
# Copyright (c) 2025 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
###############################################################################

from dataclasses import dataclass

import habana_frameworks.torch.utils._utilization_metrics_C as utilization_metrics


@dataclass
class Usage:
    hl_smi: float
    esync: float


class HWUtilizationMetrics:
    """
    Provides a unified interface for tracking hardware utilization metrics.

    This class combines two sets of metrics:
      - HPU utilization (e.g., from HL-SMI), which indicates the average usage of the device.
      - Event synchronization metrics, which reflect the ratio of active event processing
        versus idle waiting times on compute stream.

    Example:
        metrics = HWUtilizationMetrics()
        metrics.start()
        # ... perform workload ...
        usage = metrics.get_usage()
        print(f"HL-SMI Usage: {usage.hl_smi}%")
        print(f"Event Synchronization Ratio: {usage.esync}%")
        metrics.stop()
    """

    def __init__(self) -> None:
        """
        Initialize the HWUtilizationMetrics.
        """
        self._started = False

    def start(self) -> None:
        """
        Start collecting utilization metrics. This method resets the counters and,
        if the profiler is already running, stops it first before restarting. This ensures
        that each call to start() begins with fresh counters.
        """
        utilization_metrics._start_hw_utilization_metrics()

    def resume(self) -> None:
        """
        Resume collecting utilization metrics. This method resumes metric collection if it was previously stopped.
        """
        utilization_metrics._resume_hw_utilization_metrics()

    def stop(self) -> None:
        """
        Stop collecting utilization metrics. If the profiler is not running, this method does nothing.
        """
        utilization_metrics._stop_hw_utilization_metrics()

    def _reset(self) -> None:
        """
        Reset all utilization counters for both HPU and stream utilization.
        This method is private and is automatically called when starting metric collection.
        """
        utilization_metrics._reset_hw_utilization_metrics()

    def get_usage(self) -> Usage:
        """
        Retrieve the current utilization metrics.

        Returns:
            Usage: A tuple instance containing the average HL-SMI usage
                    and the event synchronization ratio, both rounded to two decimal places.
        """
        esync_usage = utilization_metrics._get_hw_utilization_metrics()
        hl_smi = round(esync_usage.get("hl_smi", 0.0), 2)
        esync_ratio = round(esync_usage.get("esync", 0.0), 2)
        return Usage(hl_smi, esync_ratio)

    def display_usage(self) -> None:
        """
        Display the utilization metrics by printing them to the console.
        """
        usage = self.get_usage()
        print(f"Device usage based on hl-smi: {usage.hl_smi}%")
        print(f"Device usage based on event synchronization: {usage.esync}%")
