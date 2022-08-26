import torch
import habana_frameworks.torch.utils._activity_profiler_C as hpu_profiler

def register_habana_activity_profiler():
    from enum import Enum
    from typing import Any, Callable, Iterable, Optional

    original_activity = torch.profiler.ProfilerActivity

    class habana_profile(torch.profiler.profile):
        def __init__(
                self,
                *,
                activities: Optional[Iterable[torch.profiler.ProfilerActivity]] = None,
                schedule: Optional[Callable[[int], torch.profiler.ProfilerAction]] = None,
                on_trace_ready: Optional[Callable[..., Any]] = None,
                record_shapes: bool = False,
                profile_memory: bool = False,
                with_stack: bool = False,
                with_flops: bool = False,
                with_modules: bool = False,
                use_cuda: Optional[bool] = None):

            self.hpu_profiling_active = torch.profiler.ProfilerActivity.HPU in activities
            activities = [self._exchange_activity(activity) for activity in activities]

            super().__init__(
                activities=activities,
                schedule=schedule,
                on_trace_ready=on_trace_ready,
                record_shapes=record_shapes,
                profile_memory=profile_memory,
                with_stack=with_stack,
                with_flops=with_flops,
                with_modules=with_modules,
                use_cuda=use_cuda
            )

        def _exchange_activity(self, activity):
            if activity == torch.profiler.ProfilerActivity.CPU:
                return original_activity.CPU
            if activity == torch.profiler.ProfilerActivity.CUDA:
                return original_activity.CUDA

        def start_trace(self):
            if self.hpu_profiling_active:
                hpu_profiler._start_activity_profiler()
            super().start_trace()

        def stop_trace(self):
            super().stop_trace()
            if self.hpu_profiling_active:
                hpu_profiler._stop_activity_profiler()

        def export_chrome_trace(self, path: str):
            super().export_chrome_trace(path)
            hpu_profiler._export_logs(path)

    class HabanaProfilerActivity(Enum):
        CPU = 1
        CUDA = 2
        HPU = 3

    torch.profiler.profile = habana_profile
    torch.profiler.ProfilerActivity = HabanaProfilerActivity

class habana_tracer:
    def __init__(self, tag: str):
        self.tag = tag
    def __enter__(self):
        self.id = hpu_profiler._add_custom_tag_begin(self.tag)
    def __exit__(self, type, value, traceback):
        hpu_profiler._add_custom_tag_end(self.id)

register_habana_activity_profiler()