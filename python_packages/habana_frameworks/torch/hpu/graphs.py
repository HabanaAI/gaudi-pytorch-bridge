import gc
import torch
import warnings
import habana_frameworks.torch as htorch
from habana_frameworks.torch import _hpu_C

class HPUGraph(object):
    r"""
    Wrapper around a HPU graph.

    .. warning::
        This API is in beta and may change in future releases.
    """
    def __init__(self):
        self.hpu_graph = _hpu_C.init_graph()

    def capture_begin(self):
        r"""
        Begins capturing HPU work on the current stream.
        """
        _hpu_C.capture_begin(self.hpu_graph)

    def capture_end(self):
        r"""
        Ends HPU graph capture on the current stream.
        After ``capture_end``, ``replay`` may be called on this instance.
        """
        _hpu_C.capture_end(self.hpu_graph)

    def replay(self):
        r"""
        Replays the HPU work captured by this graph.
        """
        _hpu_C.replay(self.hpu_graph)

class graph(object):
    r"""
    Context-manager that captures HPU work into a :class:`torch.hpu.HPUGraph`
    object for later replay.

    Arguments:
        hpu_graph (torch.hpu.HPUGraph): Graph object used for capture.
        stream (torch.hpu.Stream, optional): If supplied, will be set as the current stream in the context.
            If not supplied, ``graph`` sets its own internal side stream as the current stream in the context.

    .. warning::
        This API is in beta and may change in future releases.
    """
    default_capture_stream = None

    def __init__(self,
                 hpu_graph,
                 stream=None):
        # Lazy-init of default_capture_stream helps avoid circular-import errors.
        # Not thread safe, but graphs already have the general (explicitly documented)
        # restriction that only one capture may be underway at a time in the process.
        if self.__class__.default_capture_stream is None:
            self.__class__.default_capture_stream = htorch.hpu.Stream()

        self.capture_stream = stream if stream is not None else self.__class__.default_capture_stream
        assert self.capture_stream is not None
        self.stream_ctx = htorch.hpu.stream(self.capture_stream)
        self.hpu_graph = hpu_graph

    def __enter__(self):
        # Free as much memory as we can for the graph
        htorch.hpu.synchronize()
        gc.collect()

        self.stream_ctx.__enter__()

        self.hpu_graph.capture_begin()


    def __exit__(self, exc_type, exc_value, traceback):
        self.hpu_graph.capture_end()
        self.stream_ctx.__exit__(exc_type, exc_value, traceback)
        # returning None should propagate exceptions from either capture_end or stream_ctx.__exit__()
