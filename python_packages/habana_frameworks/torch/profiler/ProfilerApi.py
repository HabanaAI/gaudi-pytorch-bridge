def setup_profiler():
    from habana_frameworks.torch.utils.library_loader import load_habana_profiler
    load_habana_profiler()
    from habana_frameworks.torch import _profiler_C
    _profiler_C.setup_profiler()

def start_profiler():
    from habana_frameworks.torch import _profiler_C
    _profiler_C.start_profiler()

def stop_profiler():
    from habana_frameworks.torch import _profiler_C
    _profiler_C.stop_profiler()