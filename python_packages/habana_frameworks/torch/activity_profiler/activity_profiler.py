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
import logging
from enum import Enum

import habana_frameworks.torch.utils._activity_profiler_C as hpu_profiler
from habana_frameworks.torch.utils.internal import is_lazy

import torch

logger = logging.getLogger(__name__)


class DebugActivity(Enum):
    SYNAPSE_FUNCTION_CALLS = 1
    BRIDGE_FUNCTION_CALLS = 2


# W/A for issue https://github.com/pytorch/pytorch/issues/146900
# Should be removed once the issue is fixed.
def clean_json(path):
    """
    Cleans a JSON file by removing control bytes (0x00-0x1F and 0x7F) from its contents.
    The cleaned content is then written back to the same file.
    """
    # Define control bytes: 0x00-0x1F and 0x7F
    control_bytes = set(range(0x00, 0x20))
    control_bytes.add(0x7F)

    with open(path, "rb") as f:
        file_bytes = f.read()

    cleaned_bytes = bytes(b for b in file_bytes if b not in control_bytes)
    decoded_string = cleaned_bytes.decode("utf-8", errors="replace")

    with open(path, "w", encoding="utf-8") as f:
        f.write(decoded_string)


def register_habana_activity_profiler():
    from collections.abc import Callable, Iterable
    from typing import Any

    original_activity = torch.profiler.ProfilerActivity

    class habana_autograd_profile_wrapper(torch.autograd.profiler.profile):
        def export_chrome_trace(self, path):
            super().export_chrome_trace(path)
            # W/A for issue https://github.com/pytorch/pytorch/issues/146900
            # Should be removed once the issue is fixed.
            if getattr(self, "with_stack", False):
                clean_json(path)
            hpu_profiler._export_logs(path)

    class habana_profile_wrapper(torch.profiler.profile):
        def __init__(
            self,
            *,
            activities: Iterable[torch.profiler.ProfilerActivity] | None = None,
            debug_activities: Iterable[DebugActivity] | None = None,
            schedule: Callable[[int], torch.profiler.ProfilerAction] | None = None,
            on_trace_ready: Callable[..., Any] | None = None,
            record_shapes: bool = False,
            profile_memory: bool = False,
            with_stack: bool = False,
            with_flops: bool = False,
            with_modules: bool = False,
            experimental_config: torch._C._profiler._ExperimentalConfig | None = None,
            use_cuda: bool | None = None
        ):
            activities = (
                (torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.HPU)
                if activities is None
                else activities
            )
            self.hpu_profiling_active = torch.profiler.ProfilerActivity.HPU in activities
            activities = [self._exchange_activity(activity) for activity in activities]
            bridge_profile = debug_activities is not None and DebugActivity.BRIDGE_FUNCTION_CALLS in debug_activities
            if debug_activities is not None and DebugActivity.SYNAPSE_FUNCTION_CALLS in debug_activities:
                logger.warning("DebugActivity.SYNAPSE_FUNCTION_CALLS is no longer supported in bridge")
            mandatory_events = self._get_mandatory_events()
            hpu_profiler._setup_activity_profiler_sources(bridge_profile, profile_memory, mandatory_events)

            super().__init__(
                activities=activities,
                schedule=schedule,
                on_trace_ready=on_trace_ready,
                record_shapes=record_shapes,
                profile_memory=profile_memory,
                with_stack=with_stack,
                with_flops=with_flops,
                with_modules=with_modules,
                experimental_config=experimental_config,
                use_cuda=use_cuda,
            )

        def _exchange_activity(self, activity):
            if activity == torch.profiler.ProfilerActivity.CPU:
                return original_activity.CPU
            if activity == torch.profiler.ProfilerActivity.CUDA:
                return original_activity.CUDA
            if activity == torch.profiler.ProfilerActivity.XPU:
                return original_activity.XPU
            if activity == torch.profiler.ProfilerActivity.MTIA:
                return original_activity.MTIA
            if activity == torch.profiler.ProfilerActivity.PrivateUse1:
                return original_activity.PrivateUse1

        def _get_mandatory_events(self):
            if is_lazy():
                mandatory_events = [
                    "SyncTensorsGraphInternal",
                    "ExecuteCachedGraph",
                    "LaunchSyncTensorsGraph",
                    "hpu_lazy",
                ]
            else:
                mandatory_events = ["LaunchRecipeTask", "add_new_recipe", "launch_recipe", "launch"]

            return mandatory_events

        def start_trace(self):
            if self.hpu_profiling_active:
                hpu_profiler._start_activity_profiler()
            super().start_trace()

        def stop_trace(self):
            super().stop_trace()
            if self.hpu_profiling_active:
                hpu_profiler._stop_activity_profiler()

    class HabanaProfilerActivity(Enum):
        CPU = 1
        CUDA = 2
        HPU = 3
        XPU = 4
        MTIA = 5
        PrivateUse1 = 6

    torch.profiler.profile = habana_profile_wrapper
    torch.profiler.ProfilerActivity = HabanaProfilerActivity
    torch.autograd.profiler.profile = habana_autograd_profile_wrapper


def register_habana_light_activity_profiler():
    from collections.abc import Callable, Iterable
    from typing import Any

    class habana_profile_light_wrapper(torch.profiler.profile):
        def __init__(
            self,
            *,
            activities: Iterable[torch.profiler.ProfilerActivity] | None = None,
            debug_activities: Iterable[DebugActivity] | None = None,
            schedule: Callable[[int], torch.profiler.ProfilerAction] | None = None,
            on_trace_ready: Callable[..., Any] | None = None,
            record_shapes: bool = False,
            profile_memory: bool = False,
            with_stack: bool = False,
            with_flops: bool = False,
            with_modules: bool = False,
            experimental_config: torch._C._profiler._ExperimentalConfig | None = None,
            use_cuda: bool | None = None
        ):
            bridge_profile = debug_activities is not None and DebugActivity.BRIDGE_FUNCTION_CALLS in debug_activities
            hpu_profiler._setup_habana_profiler_configs(bridge_profile, profile_memory)

            super().__init__(
                activities=activities,
                schedule=schedule,
                on_trace_ready=on_trace_ready,
                record_shapes=record_shapes,
                profile_memory=profile_memory,
                with_stack=with_stack,
                with_flops=with_flops,
                with_modules=with_modules,
                experimental_config=experimental_config,
                use_cuda=use_cuda,
            )

    torch.profiler.profile = habana_profile_light_wrapper


def register_habana_profiler():
    import habana_frameworks.torch.internal._bridge_config_C as bc

    if bc.get_pt_pytorch_profiler_use_kineto():
        register_habana_light_activity_profiler()
    else:
        register_habana_activity_profiler()


register_habana_profiler()
