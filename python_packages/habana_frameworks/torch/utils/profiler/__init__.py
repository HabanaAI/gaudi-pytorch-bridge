from habana_frameworks.torch.utils.library_loader import load_habana_profiler

load_habana_profiler()

from habana_frameworks.torch.utils import _profiler_C


def _setup_profiler():
    _profiler_C.setup_profiler()


def _start_profiler():
    _profiler_C.start_profiler()


def _stop_profiler():
    _profiler_C.stop_profiler()
