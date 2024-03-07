import warnings

import habana_frameworks.torch.utils.profiler as htprofiler


def setup_profiler() -> None:
    warnings.warn(
        "habana_frameworks.torch.profiler.setup_profiler is deprecated. "
        "Please use habana_frameworks.torch.utils.profiler._setup_profiler"
    )
    htprofiler._setup_profiler()


def start_profiler() -> None:
    warnings.warn(
        "habana_frameworks.torch.profiler.start_profiler is deprecated. "
        "Please use habana_frameworks.torch.utils.profiler._start_profiler"
    )
    htprofiler._start_profiler()


def stop_profiler() -> None:
    warnings.warn(
        "habana_frameworks.torch.profiler.stop_profiler is deprecated. "
        "Please use habana_frameworks.torch.utils.profiler._stop_profiler"
    )
    htprofiler._stop_profiler()
