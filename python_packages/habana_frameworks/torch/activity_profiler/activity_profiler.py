from enum import Enum

import habana_frameworks.torch.utils._activity_profiler_C as hpu_profiler
import torch


class DebugActivity(Enum):
    SYNAPSE_FUNCTION_CALLS = 1
    BRIDGE_FUNCTION_CALLS = 2


def register_habana_activity_profiler():
    from typing import Any, Callable, Iterable, Optional

    original_activity = torch.profiler.ProfilerActivity

    class habana_autograd_profile_wrapper(torch.autograd.profiler.profile):
        def export_chrome_trace(self, path):
            super().export_chrome_trace(path)
            hpu_profiler._export_logs(path)

    class habana_profile(torch.profiler.profile):
        def __init__(
            self,
            *,
            activities: Optional[Iterable[torch.profiler.ProfilerActivity]] = None,
            debug_activities: Optional[Iterable[DebugActivity]] = None,
            schedule: Optional[Callable[[int], torch.profiler.ProfilerAction]] = None,
            on_trace_ready: Optional[Callable[..., Any]] = None,
            record_shapes: bool = False,
            profile_memory: bool = False,
            with_stack: bool = False,
            with_flops: bool = False,
            with_modules: bool = False,
            experimental_config: Optional[torch._C._profiler._ExperimentalConfig] = None,
            use_cuda: Optional[bool] = None
        ):

            self.hpu_profiling_active = torch.profiler.ProfilerActivity.HPU in activities
            activities = [self._exchange_activity(activity) for activity in activities]
            synapse_logger = debug_activities is not None and DebugActivity.SYNAPSE_FUNCTION_CALLS in debug_activities
            bridge_profile = debug_activities is not None and DebugActivity.BRIDGE_FUNCTION_CALLS in debug_activities
            mandatory_events = self._get_mandatory_events()
            hpu_profiler._setup_activity_profiler_sources(
                synapse_logger, bridge_profile, profile_memory, mandatory_events
            )

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

        def _get_mandatory_events(self):
            return [
                "SyncTensorsGraphInternal",
                "ExecuteCachedGraph",
                "LaunchSyncTensorsGraph",
                "synEventRecord",
                "synEventSynchronize",
                "synLaunchWithExternalEvents",
                "hpu_lazy",
                "synMemCopyAsync",
            ]

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

    torch.profiler.profile = habana_profile
    torch.profiler.ProfilerActivity = HabanaProfilerActivity
    torch.autograd.profiler.profile = habana_autograd_profile_wrapper


register_habana_activity_profiler()
