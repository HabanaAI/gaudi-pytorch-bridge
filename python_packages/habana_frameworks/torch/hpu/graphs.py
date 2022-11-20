from typing import List
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
        self.hpu_graph = _hpu_C.HPUGraph()

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

    def replayV2(self, static_tlist: List[torch.Tensor], tlist: List[torch.Tensor]):
        r"""
        Replays the HPU work captured by this graph.

        Arguments:
            tlist: List of input tensors for the graph replay

        .. warning::
            This API is in beta and may change in future releases.
        """
        _hpu_C.replayV2(self.hpu_graph, static_tlist, tlist)

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

def make_graphed_callables(callables, sample_args, warmups=0):

    '''
    callables (torch.nn.Module or Python function, or tuple of these) – Callable or callables to graph.
        If you pass a tuple of callables, their order in the tuple must be the same order they’ll run in the live workload.

    sample_args (tuple of Tensors, or tuple of tuples of Tensors) – Samples args for each callable.
        If a single callable was passed, sample_args must be a single tuple of argument Tensors.
        If a tuple of callables was passed, sample_args must be tuple of tuples of argument Tensors.

    warmups (Int) -  number warmups run needed.
    '''

    just_one_callable = False

    if not isinstance(callables, tuple):
        just_one_callable = True
        callables = (callables,)
        sample_args = (sample_args,)

    for c, args in zip(callables, sample_args):
        if isinstance(c, torch.nn.Module):
            assert len(c._backward_hooks) == 0 and len(c._forward_hooks) == 0 and len(c._forward_pre_hooks) == 0, \
                "Modules must not have hooks registered at the time they are passed. However, registering hooks " + \
                 "on modules after passing them through make_graphed_callables is allowed."
            assert all(b.requires_grad is False for b in c.buffers()), "In any :class:`~torch.nn.Module` passed to " + \
                 ":func:`~make_graphed_callables`, only parameters may be trainable. All buffers must have " + \
                  "``requires_grad=False``."
        assert all(isinstance(arg, torch.Tensor) for arg in args), "In the beta API, sample_args " + \
            "for each callable must be a tuple of Tensors. Other types and keywordargs are not allowed."


    per_callable_len_user_args = [len(args) for args in sample_args]
    per_callable_module_params = [tuple(c.parameters()) if isinstance(c, torch.nn.Module) else ()
                                  for c in callables]
    per_callable_static_input_surfaces = [sample_args[i] + per_callable_module_params[i]
                                           for i in range(len(callables))]

    fwd_graphs = [htorch.hpu.HPUGraph() for _ in range(len(callables))]
    bwd_graphs = [htorch.hpu.HPUGraph() for _ in range(len(callables))]

    if warmups > 0:
        htorch.hpu.synchronize()
        with htorch.hpu.stream(htorch.hpu.Stream()):
            for func, args, static_input_surface in zip(callables,
                                                        sample_args,
                                                        per_callable_static_input_surfaces):
                for _ in range(warmups):
                    outputs = func(*args)
                    outputs = (outputs,) if isinstance(outputs, torch.Tensor) else outputs
                    grad_inputs = torch.autograd.grad(outputs=outputs,
                                                    inputs=tuple(i for i in static_input_surface if i.requires_grad),
                                                    grad_outputs=tuple(torch.empty_like(o) for o in outputs),
                                                    only_inputs=True,
                                                    allow_unused=False)
                del outputs, grad_inputs

    htorch.hpu.synchronize()
    # Capture forward graphs
    per_callable_static_outputs = []
    per_callable_output_was_tensor = []
    for func, args, fwd_graph in zip(callables,
                                     sample_args,
                                     fwd_graphs):
        with htorch.hpu.graph(fwd_graph):
            outputs = func(*args)
        if isinstance(outputs, torch.Tensor):
            per_callable_output_was_tensor.append(True)
            outputs = (outputs,)
        else:
            per_callable_output_was_tensor.append(False)
        per_callable_static_outputs.append(outputs)
    per_callable_static_grad_outputs = []
    per_callable_static_grad_inputs = []
    for static_input_surface, static_outputs, bwd_graph, module_params in \
            zip(reversed(per_callable_static_input_surfaces),
                reversed(per_callable_static_outputs),
                reversed(bwd_graphs),
                reversed(per_callable_module_params)):
        assert all(o.requires_grad for o in static_outputs), "Outputs of graphed callables must require grad."
        static_grad_outputs = tuple(torch.empty_like(o) for o in static_outputs)

        with htorch.hpu.graph(bwd_graph):
            grad_inputs = torch.autograd.grad(outputs=static_outputs,
                                              inputs=tuple(i for i in static_input_surface if i.requires_grad),
                                              grad_outputs=static_grad_outputs,
                                              only_inputs=True,
                                              allow_unused=False)

        static_grad_inputs = []
        grad_idx = 0
        for arg in static_input_surface:
            if arg.requires_grad:
                static_grad_inputs.append(grad_inputs[grad_idx])
                grad_idx += 1
            else:
                static_grad_inputs.append(None)
        static_grad_inputs = tuple(static_grad_inputs)

        per_callable_static_grad_outputs.append(static_grad_outputs)
        per_callable_static_grad_inputs.append(static_grad_inputs)
    per_callable_static_grad_outputs = list(reversed(per_callable_static_grad_outputs))
    per_callable_static_grad_inputs = list(reversed(per_callable_static_grad_inputs))

    def make_graphed_autograd_function(fwd_graph,
                                       bwd_graph,
                                       module_params,
                                       len_user_args,
                                       output_was_tensor,
                                       static_input_surface,
                                       static_outputs,
                                       static_grad_outputs,
                                       static_grad_inputs):
        class Graphed(torch.autograd.Function):
            @staticmethod
            def forward(ctx, *inputs):
                #for i in range(len_user_args):
                    # if static_input_surface[i].data_ptr() != inputs[i].data_ptr():
                    #     static_input_surface[i].copy_(inputs[i])
                #    static_input_surface[i].copy_(inputs[i])
                #fwd_graph.replay()
                fwd_graph.replayV2(static_input_surface, inputs)
                assert isinstance(static_outputs, tuple)
                return tuple(o.detach() for o in static_outputs)

            @staticmethod
            @torch.autograd.function.once_differentiable
            def backward(ctx, *grads):
                for g, grad in zip(static_grad_outputs, grads):
                    if g is None:
                        assert grad is None
                    #else:
                        # if g.data_ptr() != grad.data_ptr():
                        #     g.copy_(grad)
                        #g.copy_(grad)
                #bwd_graph.replay()
                bwd_graph.replayV2(static_grad_outputs, grads)

                # Input args that didn't require grad expect a None gradient.
                assert isinstance(static_grad_inputs, tuple)
                return tuple(b.detach() if b is not None else b for b in static_grad_inputs)

        def functionalized(*user_args):
            out = Graphed.apply(*(user_args + module_params))
            return out[0] if output_was_tensor else out

        return functionalized

    ret = []
    for i, func in enumerate(callables):
        graphed = make_graphed_autograd_function(fwd_graphs[i],
                                                 bwd_graphs[i],
                                                 per_callable_module_params[i],
                                                 per_callable_len_user_args[i],
                                                 per_callable_output_was_tensor[i],
                                                 per_callable_static_input_surfaces[i],
                                                 per_callable_static_outputs[i],
                                                 per_callable_static_grad_outputs[i],
                                                 per_callable_static_grad_inputs[i])

        if isinstance(func, torch.nn.Module):
            def make_graphed_forward(func, graph_training_state, graphed, orig_fwd):
                def new_fwd(*user_args):
                    if func.training == graph_training_state:
                        return graphed(*user_args)
                    else:
                        return orig_fwd(*user_args)
                    return new_fw
                return new_fwd
            func.forward = make_graphed_forward(func, func.training, graphed, func.forward)
            ret.append(func)
        else:
            ret.append(graphed)
    if just_one_callable:
        return ret[0]

    return tuple(ret)

class CachedParams:
    def __init__(self, graph_inputs, graph_outputs, graph):
        self.graph_inputs = graph_inputs
        self.graph_outputs = graph_outputs
        self.graph = graph


def input_hash(obj):
    if isinstance(obj, dict):
        return input_hash(tuple(obj.items()))
    elif isinstance(obj, list) or isinstance(obj, tuple):
        return hash(tuple(input_hash(el) for el in obj))
    elif torch.is_tensor(obj):
        return hash(obj.shape)
    else:
        return hash(obj)


def copy_to(dst, src):
    assert type(dst) == type(src)
    if isinstance(dst, dict):
        for (dk, dv), (sk, sv) in zip(dst.items(), src.items()):
            assert dk == sk
            copy_to(dv, sv)
    elif isinstance(dst, list) or isinstance(dst, tuple):
        for d, s in zip(dst, src):
            copy_to(d, s)
    elif torch.is_tensor(dst):
        dst.copy_(src, non_blocking=True)

def wrap_in_hpu_graph_func(func):
    import habana_frameworks.torch as ht
    stream = ht.hpu.Stream()
    cache = {}
    orig_fwd = func
    def forward(*args, **kwargs):
        inputs = (args, kwargs)
        h = input_hash(inputs)
        cached = cache.get(h)
        if cached is None:
            with ht.hpu.stream(stream):
                graph = ht.hpu.HPUGraph()
                graph.capture_begin()
                outputs = orig_fwd(*args, **kwargs)
                graph.capture_end()
                graph_inputs = inputs
                graph_outputs = outputs
                cache[h] = CachedParams(graph_inputs, graph_outputs, graph)
            return outputs
        copy_to(cached.graph_inputs, inputs)
        cached.graph.replay()
        return cached.graph_outputs
    return forward

def wrap_in_hpu_graph(module):
    import habana_frameworks.torch as ht
    stream = ht.hpu.Stream()
    cache = {}
    orig_fwd = module.forward
    def forward(*args, **kwargs):
        inputs = (args, kwargs)
        h = input_hash(inputs)
        cached = cache.get(h)
        if cached is None:
            with ht.hpu.stream(stream):
                graph = ht.hpu.HPUGraph()
                graph.capture_begin()
                outputs = orig_fwd(*args, **kwargs)
                graph.capture_end()
                graph_inputs = inputs
                graph_outputs = outputs
                cache[h] = CachedParams(graph_inputs, graph_outputs, graph)
            return outputs

        copy_to(cached.graph_inputs, inputs)
        cached.graph.replay()
        return cached.graph_outputs
    module.forward = forward
    return module