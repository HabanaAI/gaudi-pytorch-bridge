import collections
import copy
import gc
import inspect
import os
import warnings
from functools import wraps
from typing import List

import habana_frameworks.torch as htorch
import habana_frameworks.torch.core as htcore
import torch
from habana_frameworks.torch import _hpu_C
from habana_frameworks.torch.utils.debug import _hg_print as hpu_graph_print


def stringify(*args):
    string = ""
    for item in args:
        string = string + " " + str(item)
    return string


class HPUGraph(object):
    r"""
    Wrapper around a HPU graph.

    .. warning::
        This API is in beta and may change in future releases.
    """

    def __init__(self):
        self.hpu_graph = _hpu_C.HPUGraph()

    def capture_begin(self, dry_run=False):
        r"""
        Begins capturing HPU work on the current stream.
        """
        _hpu_C.capture_begin(self.hpu_graph, dry_run)

    def capture_end(self):
        r"""
        Ends HPU graph capture on the current stream.
        After ``capture_end``, ``replay`` may be called on this instance.
        """
        _hpu_C.capture_end(self.hpu_graph)

    def replay(self, asynchronous=False):
        r"""
        Replays the HPU work captured by this graph.
        """
        _hpu_C.replay(self.hpu_graph, asynchronous)

    def replayV2(self, static_tlist: List[torch.Tensor], tlist: List[torch.Tensor], asynchronous=False):
        r"""
        Replays the HPU work captured by this graph.

        Arguments:
            tlist: List of input tensors for the graph replay

        .. warning::
            This API is in beta and may change in future releases.
        """
        _hpu_C.replayV2(self.hpu_graph, static_tlist, tlist, asynchronous)

    def replayV3(self, tlistI: List[torch.Tensor], asynchronous=False):
        r"""
        Replays the HPU work captured by this graph.

        Arguments:
            tlistI: List of input tensors for the graph replay

        .. warning::
            This API is in beta and may change in future releases.
        """
        _hpu_C.replayV3(self.hpu_graph, tlistI, asynchronous)

    def clear_inputs(self):
        """
        Can be called by user on hpugraph
        Clears inputs to the graph held as outputs due to inplace operations
        in the HPUGraph
        """
        _hpu_C.clear_inputs(self.hpu_graph)

    def mark_user_outputs(self, static_tlist: List[torch.Tensor]):
        r"""
        Marks user needed output after graph capture

        Arguments:
            static_tlist: List of out tensors for the graph capture

        .. warning::
            This API is in beta and may change in future releases.
        """
        _hpu_C.mark_user_outputs(self.hpu_graph, static_tlist)

    def mark_user_inputs(self, static_tlist: List[torch.Tensor]):
        r"""
        Marks user provided input during graph capture

        Arguments:
            static_tlist: List of input tensors for the graph capture

        .. warning::
            This API is in beta and may change in future releases.
        """
        _hpu_C.mark_user_inputs(self.hpu_graph, static_tlist)

    def reset(self):
        r"""
        Destroys and free up memory of captured HPU graph.
        """
        if _hpu_C.destroy is not None:
            _hpu_C.destroy(self.hpu_graph)

    def __del__(self):
        self.reset()

    def get_user_input_match_indices(self):
        return _hpu_C.get_user_input_match_indices(self.hpu_graph)


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

    def __init__(self, hpu_graph, stream=None, dry_run=False):
        # Lazy-init of default_capture_stream helps avoid circular-import errors.
        # Not thread safe, but graphs already have the general (explicitly documented)
        # restriction that only one capture may be underway at a time in the process.
        if self.__class__.default_capture_stream is None:
            self.__class__.default_capture_stream = htorch.hpu.Stream()

        self.capture_stream = stream if stream is not None else self.__class__.default_capture_stream
        assert self.capture_stream is not None
        self.stream_ctx = htorch.hpu.stream(self.capture_stream)
        self.hpu_graph = hpu_graph
        self.dry_run = dry_run

    def __enter__(self):
        # Free as much memory as we can for the graph
        htorch.hpu.synchronize()
        gc.collect()

        self.stream_ctx.__enter__()
        self.capture_stream.is_capture = True
        self.hpu_graph.capture_begin(self.dry_run)

    def __exit__(self, exc_type, exc_value, traceback):
        self.hpu_graph.capture_end()
        self.capture_stream.is_capture = False
        self.stream_ctx.__exit__(exc_type, exc_value, traceback)
        # returning None should propagate exceptions from either capture_end or stream_ctx.__exit__()


def make_graphed_callables(
    callables,
    sample_args,
    warmups=0,
    allow_unused_input=False,
    asynchronous=False,
    disable_tensor_cache=False,
    dry_run=False,
):
    """
    callables (torch.nn.Module or Python function, or tuple of these) – Callable or callables to graph.
        If you pass a tuple of callables, their order in the tuple must be the same order they’ll run in the live workload.

    sample_args (tuple of Tensors, or tuple of tuples of Tensors) – Samples args for each callable.
        If a single callable was passed, sample_args must be a single tuple of argument Tensors.
        If a tuple of callables was passed, sample_args must be tuple of tuples of argument Tensors.

    warmups (Int) -  number warmups run needed.

    allow_unused_input (bool): If False, specifying inputs that were not used when computing outputs
    (and therefore their grad is always zero) is an error. Defaults to False.

    asynchronous (bool): If True, replay will be done asynchronously, main thread returns immediately after queing replay.
        Defaults to False.

    disable_tensor_cache (bool): If True, tensors won't be cached in hpu graph and memory can be saved.
        Defaults to False.

    dry_run (bool): If True, avoid actual launch of recipe.
        Defaults to False.

    """
    printvar = stringify(
        "make_graphed_callables",
        "Warmups",
        warmups,
        "allow_unused_input",
        allow_unused_input,
        "asynchronous",
        asynchronous,
        "disable_tensor_cache",
        disable_tensor_cache,
    )
    hpu_graph_print(printvar)
    just_one_callable = False

    if not isinstance(callables, tuple):
        just_one_callable = True
        callables = (callables,)
        sample_args = (sample_args,)

    for c, args in zip(callables, sample_args):
        if isinstance(c, torch.nn.Module):
            assert len(c._backward_hooks) == 0 and len(c._forward_hooks) == 0 and len(c._forward_pre_hooks) == 0, (
                "Modules must not have hooks registered at the time they are passed. However, registering hooks "
                + "on modules after passing them through make_graphed_callables is allowed."
            )
            assert all(b.requires_grad is False for b in c.buffers()), (
                "In any :class:`~torch.nn.Module` passed to "
                + ":func:`~make_graphed_callables`, only parameters may be trainable. All buffers must have "
                + "``requires_grad=False``."
            )
        assert all(isinstance(arg, torch.Tensor) for arg in args), (
            "In the beta API, sample_args "
            + "for each callable must be a tuple of Tensors. Other types and keywordargs are not allowed."
        )

    per_callable_len_user_args = [len(args) for args in sample_args]
    per_callable_module_params = [tuple(c.parameters()) if isinstance(c, torch.nn.Module) else () for c in callables]
    per_callable_static_input_surfaces = [sample_args[i] + per_callable_module_params[i] for i in range(len(callables))]

    fwd_graphs = [htorch.hpu.HPUGraph() for _ in range(len(callables))]
    bwd_graphs = [htorch.hpu.HPUGraph() for _ in range(len(callables))]

    if warmups > 0:
        htorch.hpu.synchronize()
        with htorch.hpu.stream(htorch.hpu.Stream()):
            for func, args, static_input_surface in zip(callables, sample_args, per_callable_static_input_surfaces):
                for _ in range(warmups):
                    outputs = func(*args)
                    outputs = (outputs,) if isinstance(outputs, torch.Tensor) else outputs
                    grad_inputs = torch.autograd.grad(
                        outputs=outputs,
                        inputs=tuple(i for i in static_input_surface if i.requires_grad),
                        grad_outputs=tuple(torch.empty_like(o) for o in outputs),
                        only_inputs=True,
                        allow_unused=allow_unused_input,
                    )
                del outputs, grad_inputs

    htorch.hpu.synchronize()
    # Capture forward graphs
    per_callable_static_outputs = []
    per_callable_output_was_tensor = []
    for func, args, fwd_graph in zip(callables, sample_args, fwd_graphs):
        with htorch.hpu.graph(fwd_graph, dry_run=True if disable_tensor_cache else dry_run):
            if disable_tensor_cache:
                fwd_graph.mark_user_inputs(args)
            outputs = func(*args)
        if isinstance(outputs, torch.Tensor):
            per_callable_output_was_tensor.append(True)
            outputs = (outputs,)
        else:
            per_callable_output_was_tensor.append(False)
        per_callable_static_outputs.append(outputs)

    per_callable_static_grad_outputs = []
    per_callable_static_grad_inputs = []
    bwd_mark_user_inputs_len = []
    for static_input_surface, args, static_outputs, bwd_graph, module_params in zip(
        reversed(per_callable_static_input_surfaces),
        reversed(sample_args),
        reversed(per_callable_static_outputs),
        reversed(bwd_graphs),
        reversed(per_callable_module_params),
    ):
        # assert all(o.requires_grad for o in static_outputs), "Outputs of graphed callables must require grad."
        static_grad_outputs = tuple(torch.empty_like(o) if o.requires_grad else None for o in static_outputs)

        with htorch.hpu.graph(bwd_graph, dry_run=True if disable_tensor_cache else dry_run):
            autograd_inputs = tuple(i for i in static_input_surface if i.requires_grad)
            if disable_tensor_cache:
                mark_user_inputs_list = get_user_input_tensor_list(
                    static_grad_outputs, ()
                ) + get_user_input_tensor_list(args, ())
                bwd_graph.mark_user_inputs(mark_user_inputs_list)
                bwd_mark_user_inputs_len.append(len(get_user_input_tensor_list(static_grad_outputs, ())))

            grad_inputs = torch.autograd.grad(
                outputs=tuple(o for o in static_outputs if o.requires_grad),
                inputs=autograd_inputs,
                grad_outputs=tuple(o for o in static_grad_outputs if o is not None),
                only_inputs=True,
                allow_unused=allow_unused_input,
                materialize_grads=allow_unused_input,
            )

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

    if disable_tensor_cache:
        per_callable_input_surfaces_optim = []
        for fwd_graph, static_input_surface, len_user_args in zip(
            fwd_graphs, per_callable_static_input_surfaces, per_callable_len_user_args
        ):
            len_module_params = len(per_callable_static_input_surfaces) - len_user_args
            matched_input_index = fwd_graph.get_user_input_match_indices()
            input_surface_list = list(static_input_surface)
            for i in range(len_user_args):
                if i in matched_input_index:
                    input_surface_list[i + len_module_params] = torch.empty(0)
            static_input_surface = tuple(input_surface_list)
            per_callable_input_surfaces_optim.append(static_input_surface)
        per_callable_static_input_surfaces = per_callable_input_surfaces_optim

        per_callable_grad_outputs_optim = []
        for bwd_graph, static_grad_outputs, bwd_uin_len in zip(
            bwd_graphs, reversed(per_callable_static_grad_outputs), reversed(bwd_mark_user_inputs_len)
        ):
            matched_input_index = bwd_graph.get_user_input_match_indices()
            grad_outputs_list = list(static_grad_outputs)
            for i in range(bwd_uin_len):
                if i in matched_input_index:
                    grad_outputs_list[i] = torch.empty(0)
            static_grad_outputs = tuple(grad_outputs_list)
            per_callable_grad_outputs_optim.append(static_grad_outputs)
        per_callable_static_grad_outputs = per_callable_grad_outputs_optim

    per_callable_static_grad_outputs = list(reversed(per_callable_static_grad_outputs))
    per_callable_static_grad_inputs = list(reversed(per_callable_static_grad_inputs))

    def make_graphed_autograd_function(
        fwd_graph,
        bwd_graph,
        module_params,
        len_user_args,
        output_was_tensor,
        static_input_surface,
        static_outputs,
        static_grad_outputs,
        static_grad_inputs,
        asynchronous,
        disable_tensor_cache,
    ):
        class Graphed(torch.autograd.Function):
            @staticmethod
            def forward(ctx, *inputs):
                if disable_tensor_cache:
                    marked_inputs = ()
                    matched_input_index = fwd_graph.get_user_input_match_indices()
                    for i in range(len_user_args):
                        if i not in matched_input_index:
                            static_input_surface[i].copy_(inputs[i])
                        marked_inputs = marked_inputs + (inputs[i],)
                    ctx.save_for_backward(*marked_inputs)
                    fwd_graph.replayV3(marked_inputs, asynchronous)
                    assert isinstance(static_outputs, tuple)
                    return tuple(o.detach() for o in static_outputs)
                else:
                    for i in range(len_user_args):
                        # if static_input_surface[i].data_ptr() != inputs[i].data_ptr():
                        #     static_input_surface[i].copy_(inputs[i])
                        static_input_surface[i].copy_(inputs[i])
                    fwd_graph.replay(asynchronous)
                    assert isinstance(static_outputs, tuple)
                    return tuple(o.detach() for o in static_outputs)

            @staticmethod
            @torch.autograd.function.once_differentiable
            def backward(ctx, *grads):
                if disable_tensor_cache:
                    marked_grads = ()
                    matched_input_index = bwd_graph.get_user_input_match_indices()
                    i = 0
                    for g, grad in zip(static_grad_outputs, grads):
                        if g is not None:
                            if i not in matched_input_index:
                                g.copy_(grad)
                            marked_grads = marked_grads + (grad,)
                            i = i + 1

                    bwd_graph.replayV3((marked_grads) + (ctx.saved_tensors), asynchronous)
                    return tuple(b.detach() if b is not None else b for b in static_grad_inputs)
                else:
                    for g, grad in zip(static_grad_outputs, grads):
                        if g is not None:
                            # if g.data_ptr() != grad.data_ptr():
                            #     g.copy_(grad)
                            g.copy_(grad)
                    bwd_graph.replay(asynchronous)
                    # Input args that didn't require grad expect a None gradient.
                    assert isinstance(static_grad_inputs, tuple)
                    return tuple(b.detach() if b is not None else b for b in static_grad_inputs)

        def functionalized(*user_args):
            out = Graphed.apply(*(user_args + module_params))
            return out[0] if output_was_tensor else out

        return functionalized

    ret = []
    for i, func in enumerate(callables):
        graphed = make_graphed_autograd_function(
            fwd_graphs[i],
            bwd_graphs[i],
            per_callable_module_params[i],
            per_callable_len_user_args[i],
            per_callable_output_was_tensor[i],
            per_callable_static_input_surfaces[i],
            per_callable_static_outputs[i],
            per_callable_static_grad_outputs[i],
            per_callable_static_grad_inputs[i],
            asynchronous,
            disable_tensor_cache,
        )

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
    def __init__(self, graph_inputs, graph_outputs, graph, out_tinfo_list=None, asynchronous=False):
        self.graph_inputs = graph_inputs
        self.graph_outputs = graph_outputs
        self.graph = graph
        self.out_tinfo_list = out_tinfo_list
        self.asynchronous = asynchronous


def input_hash(obj):
    if isinstance(obj, dict):
        return input_hash(tuple(obj.items()))
    elif isinstance(obj, list) or (isinstance(obj, tuple) and not isinstance(obj, torch.Size)):
        # torch.Size is specialization of tuple, so we don't want extra recursion.
        return hash((tuple(input_hash(el) for el in obj), torch.hpu.is_autocast_hpu_enabled()))
    elif torch.is_tensor(obj):
        return hash(tuple([obj.shape, _hpu_C.get_view_hash(obj), torch.hpu.is_autocast_hpu_enabled()]))
    else:
        return hash((obj, torch.hpu.is_autocast_hpu_enabled()))


def copy_to(dst, src):
    assert type(dst) == type(src)
    if isinstance(dst, dict):
        for (dk, dv), (sk, sv) in zip(dst.items(), src.items()):
            assert dk == sk
            copy_to(dv, sv)
    elif isinstance(dst, list) or (isinstance(dst, tuple) and not isinstance(dst, torch.Size)):
        for d, s in zip(dst, src):
            copy_to(d, s)
    elif torch.is_tensor(dst):
        dst.copy_(src, non_blocking=True)


def get_user_input_tensor_list(inputs, tlist):
    if isinstance(inputs, dict):
        for inp in inputs.items():
            tlist = get_user_input_tensor_list(inp, tlist)
    elif isinstance(inputs, list) or isinstance(inputs, tuple):
        for inp in inputs:
            tlist = get_user_input_tensor_list(inp, tlist)
    elif torch.is_tensor(inputs):
        tlist = tlist + (inputs,)
    return tlist


def is_seq_of_tensor(obj):
    if isinstance(obj, collections.abc.Sequence) and not isinstance(obj, str):
        for mem in obj:
            if not is_seq_of_tensor(mem):
                return False
        return True
    elif torch.is_tensor(obj):
        return True
    else:
        return False


def extract_tensors(data):
    """
    Returns a list of all tensors contained within a given data structure.
    """
    tensors = []
    if isinstance(data, torch.Tensor):
        tensors.append(data)
    elif isinstance(data, (list, tuple)):
        for item in data:
            tensors.extend(extract_tensors(item))
    elif isinstance(data, dict):
        for value in data.values():
            tensors.extend(extract_tensors(value))
    elif hasattr(data, "__dict__"):
        for value in data.__dict__.values():
            tensors.extend(extract_tensors(value))
    return tensors


def get_tensor_info(tensor):
    """
    Returns a dictionary with shape, dtype, and device information for a PyTorch tensor.
    """
    return {"shape": tensor.shape, "dtype": tensor.dtype, "device": tensor.device}


def wrapped_hpugraph_forward(
    cache, stream, orig_fwd, args, kwargs, disable_tensor_cache, asynchronous, dry_run, max_graphs
):
    """
    Wrapped forward method that captures and replays the HPU graph.

    Args:
        cache (dict): Cache to store captured graphs for reuse.
        stream (habana_frameworks.torch.hpu.Stream): HPU stream for graph capture and replay.
        orig_fwd (function): The original forward method of the module.
        args: Positional arguments passed to the forward method.
        kwargs: Keyword arguments passed to the forward method.
        disable_tensor_cache (bool): Specifies whether to use tensor cache during graph replay.
        asynchronous (bool): Specifies whether the graph replay should be asynchronous.
        dry_run (bool): Enable dry run, which helps to run model without allocating memory.
        max_graphs: maximum graphs which will be cached

    Returns:
        The output of the original forward method.

    Notes:
        - The graph is captured during the first call to forward and replayed for subsequent calls.
        - The replay can be synchronous or asynchronous based on the `asynchronous` argument.
        - The tensor cache is used during graph replay if `disable_tensor_cache` is True.
        - The tensors in the graph will be replaced with empty tensors after replaying to save memory
          if `disable_tensor_cache` is True.
        - if 'dry_run' do a trial run to find out the tensors which can be freed.
        - If `bypass_hpu_graphs=True` is present in kwargs the original fwd is called instead
        - If 'warmup_mode=True' is present in kwargs, we will skip replay for all iterations after the first one
    """
    warmup_mode = kwargs.pop("warmup_mode", False)

    if kwargs.pop("bypass_hpu_graphs", False):
        return orig_fwd(*args, **kwargs)
    inputs = (args, kwargs)

    h = input_hash(inputs)
    cached = cache.get(h)

    cached_tlist = extract_tensors(kwargs.pop("cache_tensors_list", []))
    # Read from env variable.
    env_tensor_cache = os.environ.get("PT_HPUGRAPH_DISABLE_TENSOR_CACHE")
    disable_tensor_cache = disable_tensor_cache if env_tensor_cache is None else env_tensor_cache == "1"

    # Enable dry run if tensor cache is disabled
    dry_run = True if disable_tensor_cache else dry_run
    if cached is None:
        if max_graphs is not None and len(cache) == max_graphs:
            return orig_fwd(*args, **kwargs)

        with htorch.hpu.stream(stream):
            graph = htorch.hpu.HPUGraph()
            graph.capture_begin(dry_run=dry_run)
            input_tensor_list = get_user_input_tensor_list(inputs, ())
            if disable_tensor_cache:
                graph.mark_user_inputs(input_tensor_list)
            outputs = orig_fwd(*args, **kwargs)
            graph.capture_end()
            graph_outputs = outputs
            matched_input_index = graph.get_user_input_match_indices()

            if not disable_tensor_cache:
                graph_inputs = inputs
                tinfo_list = None
                if dry_run:
                    graph.replay(asynchronous)
            else:
                tlist = extract_tensors(outputs)
                tinfo_list = [get_tensor_info(t) for t in tlist]
                tlist = cached_tlist + tlist
                graph.mark_user_outputs(tlist)
                saved_inputs = []
                for i in range(len(input_tensor_list)):
                    if i not in matched_input_index:
                        saved_inputs.append(input_tensor_list[i])
                graph_inputs = tuple(saved_inputs)
                if dry_run:
                    graph.replayV3(get_user_input_tensor_list(inputs, ()), asynchronous)

            cache[h] = CachedParams(graph_inputs, graph_outputs, graph, tinfo_list, asynchronous)

        return outputs

    if warmup_mode:
        hpu_graph_print("In warmup mode, skipping replay")
        htcore.mark_step()
        return cached.graph_outputs

    # use replayv1 here
    if not disable_tensor_cache:
        # Copy the user inputs
        copy_to(cached.graph_inputs, inputs)
        cached.graph.replay(cached.asynchronous)
    else:
        matched_input_index = cached.graph.get_user_input_match_indices()
        input_tensor_list = get_user_input_tensor_list(inputs, ())
        saved_inputs = []
        for i in range(len(input_tensor_list)):
            if i not in matched_input_index:
                saved_inputs.append(input_tensor_list[i])
        graph_inputs = tuple(saved_inputs)
        copy_to(cached.graph_inputs, graph_inputs)
        cached.graph.replayV3(input_tensor_list, cached.asynchronous)
    out = cached.graph_outputs
    # Enable this line to see the graph counts and memory stats
    # print("Graph count: ", len(cache), htorch.hpu.memory.memory_stats())
    return out


def wrap_in_hpu_graph_func(func, asynchronous=False, disable_tensor_cache=False, dry_run=False, max_graphs=None):
    """
    Wraps the forward method of a module in an HPU graph capture and replay mechanism.

    Args:
        module (torch.nn.Module): The module to be wrapped.
        asynchronous (bool, optional): Specifies whether the graph capture and replay should be asynchronous.
            Defaults to False.
        disable_tensor_cache (bool, optional): Specifies whether to use tensor cache during graph replay.
            Defaults to False.
        dry_run (bool): Enable dry run, which helps to run model without allocating memory.
        max_graphs: maximum graphs which will be cached

    Returns:
        torch.nn.Module: The module with the wrapped forward method.

    Raises:
        TypeError: If the input module is not an instance of torch.nn.Module.

    Notes:
        - This function modifies the input module by replacing its forward method.
        - The wrapped forward method captures the graph when it is first called, and replays the captured graph
          for subsequent calls.
        - If `disable_tensor_cache` is False, the graph replay uses a tensor cache for better performance.
        - If `disable_tensor_cache` is True, the graph replay replaces the tensors in the graph with empty tensors
          after replaying, saves memory.

    """
    stream = htorch.hpu.default_stream()
    cache = {}
    orig_fwd = func

    def forward(*args, **kwargs):
        return wrapped_hpugraph_forward(
            cache, stream, orig_fwd, args, kwargs, disable_tensor_cache, asynchronous, dry_run, max_graphs
        )

    return forward


def wrap_in_hpu_graph(module, asynchronous=False, disable_tensor_cache=False, dry_run=False, max_graphs=None):
    """
    Wraps the forward method of a module in an HPU graph capture and replay mechanism.

    Args:
        module (torch.nn.Module): The module to be wrapped.
        asynchronous (bool, optional): Specifies whether the graph capture and replay should be asynchronous.
            Defaults to False.
        disable_tensor_cache (bool, optional): Specifies whether to cache tensors during graph replay.
            Defaults to False.
        dry_run (bool): Enable dry run, which helps to run model without allocating memory.
        max_graphs: maximum graphs which will be cached

    Returns:
        torch.nn.Module: The module with the wrapped forward method.

    Raises:
        TypeError: If the input module is not an instance of torch.nn.Module.

    Notes:
        - This function modifies the input module by replacing its forward method.
        - The wrapped forward method captures the graph when it is first called, and replays the captured graph
          for subsequent calls.
        - If `disable_tensor_cache` is False, the graph replay uses a tensor cache for better performance.
        - If `disable_tensor_cache` is True, the graph replay replaces the tensors in the graph with empty tensors
          after replaying, saves memory.

    """
    stream = htorch.hpu.default_stream()
    cache = {}
    orig_fwd = module.forward

    @wraps(orig_fwd)
    def forward(*args, **kwargs):
        return wrapped_hpugraph_forward(
            cache, stream, orig_fwd, args, kwargs, disable_tensor_cache, asynchronous, dry_run, max_graphs
        )

    module.forward = forward

    # Can call model.clear_cache to release the HPU graph memory cached
    def clear_cache():
        cache.clear()

    def clear_inputs():
        for h in cache:
            cache[h].graph.clear_inputs()

    module.clear_inputs = clear_inputs
    module.clear_cache = clear_cache

    return module


class TensorPacker:
    def __init__(self, is_out_pack=False, verbose=False):
        self._is_out_pack = is_out_pack  # Whether the pack/unpack is for output of graph forward
        self._verbose = verbose

    class Index:
        def __init__(self, value):
            self.value = value

        def __repr__(self):
            return "#{0:d}".format(self.value)

    def pack(self, outs):
        tensor_list = []
        metadata = self._pack(outs, tensor_list)
        return tuple(tensor_list), metadata

    def _pack(self, outs, tensor_list):
        if torch.is_tensor(outs):
            metadata = self.Index(len(tensor_list))
            tensor_list.append(outs)

        elif isinstance(outs, tuple):
            metadata = list(copy.copy(outs))
            for idx, item in enumerate(outs):
                metadata[idx] = self._pack(item, tensor_list)
            metadata = tuple(metadata)

        elif isinstance(outs, dict):
            metadata = copy.copy(outs)
            for key in outs:
                metadata[key] = self._pack(outs[key], tensor_list)

        elif isinstance(outs, list):
            metadata = copy.copy(outs)
            for idx, item in enumerate(outs):
                metadata[idx] = self._pack(item, tensor_list)

        else:
            if self._verbose:
                print("[WARNING] Variable of type {0} will not be dynamic".format(type(outs)))
            return outs

        return metadata

    def unpack(self, tensors, metadata):
        output = self._unpack(tensors, metadata)
        return output

    def _unpack(self, tensors, metadata):
        if isinstance(metadata, self.Index):
            data = tensors[metadata.value]

        elif isinstance(metadata, tuple):
            data = list(copy.copy(metadata))
            for idx, item in enumerate(metadata):
                data[idx] = self._unpack(tensors, item)
            data = tuple(data)

        elif isinstance(metadata, dict):
            data = copy.copy(metadata)
            for key in metadata:
                data[key] = self._unpack(tensors, metadata[key])

        elif isinstance(metadata, list):
            data = copy.copy(metadata)
            for idx, item in enumerate(metadata):
                data[idx] = self._unpack(tensors, item)

        else:
            return metadata

        return data


class GraphModel(torch.nn.Module):
    def __init__(self, model, allow_unused_input=False, asynchronous=False, disable_tensor_cache=False, dry_run=False):
        super(GraphModel, self).__init__()
        self.model = model
        self.input_packer = TensorPacker()
        self.input_meta = None
        self.output_packer = TensorPacker(is_out_pack=True)
        self.output_meta = None
        self.assert_not_dataparallel()
        self.func_parameters = self.process_function_signature(self.model.forward)
        self.allow_unused_input = allow_unused_input
        self.asynchronous = asynchronous
        self.disable_tensor_cache = disable_tensor_cache
        self.dry_run = dry_run

    def forward(self, *args):
        full_args = self.input_packer.unpack(args, self.input_meta)
        outs = self.model(**full_args)
        out_tensors, self.output_meta = self.output_packer.pack(outs)
        return out_tensors

    def graph_forward(self, *args, **kwargs):
        full_args = GraphModel.get_full_args(self.func_parameters, *args, **kwargs)
        tensor_args, _ = self.input_packer.pack(full_args)
        out_tensors = self.hpu_graph(*tensor_args)
        return self.output_packer.unpack(out_tensors, self.output_meta)

    def init_hpu_graph(self, *args, **kwargs):
        full_args = GraphModel.get_full_args(self.func_parameters, *args, **kwargs)
        self.input_id = input_hash(full_args)
        tensor_args, self.input_meta = self.input_packer.pack(full_args)
        self.hpu_graph = make_graphed_callables(
            self,
            tensor_args,
            allow_unused_input=self.allow_unused_input,
            asynchronous=self.asynchronous,
            disable_tensor_cache=self.disable_tensor_cache,
            dry_run=self.dry_run,
        )

    def assert_not_dataparallel(self):
        assert not isinstance(self.model, torch.nn.parallel.DataParallel) and not isinstance(
            self.model, torch.nn.parallel.DistributedDataParallel
        ), "Use DataParallel/DistributedDataParallel only after wrapping with ModuleCacher"

    @staticmethod
    def process_function_signature(function):
        func_parameters = collections.OrderedDict(inspect.signature(function).parameters)

        UNSUPPORTED = [inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.VAR_POSITIONAL]
        for key in list(func_parameters):
            assert func_parameters[key].kind not in UNSUPPORTED, "Unsupported argument type : {0}".format(
                func_parameters[key].kind
            )
            if func_parameters[key].kind == inspect.Parameter.VAR_KEYWORD:
                print("[WARNING] Variable keyword arguments will not be supported.")
                del func_parameters[key]
                continue
            func_parameters[key] = func_parameters[key].default
        return func_parameters

    @staticmethod
    def get_full_args(forward_params, *args, **kwargs):
        args_full = copy.copy(forward_params)
        for idx, key in enumerate(args_full):
            if idx == len(args):
                break
            args_full[key] = args[idx]
        args_full.update(kwargs)
        return args_full

    @staticmethod
    def full_input_hash(forward_params, *args, **kwargs):
        return input_hash(GraphModel.get_full_args(forward_params, *args, **kwargs))


class ModuleCacher(torch.nn.Module):
    def __getstate__(self):
        return (
            self.max_graphs,
            self.dry_run,
            self.disable_tensor_cache,
            self.log_frequency,
            self.verbose,
            self.have_grad_accumulation,
            self.asynchronous,
            self.allow_unused_input,
            self.use_lfu,
            self.hpugraph_tracing,
            self.forward_params,
            self.inplace,
            self.orig_model,
        )

    def __setstate__(self, state):
        self.__init__(state[0])
        self.dry_run = state[1]
        self.disable_tensor_cache = state[2]
        self.log_frequency = state[3]
        self.verbose = state[4]
        self.have_grad_accumulation = state[5]
        self.asynchronous = state[6]
        self.allow_unused_input = state[7]
        self.use_lfu = state[8]
        self.hpugraph_tracing = state[9]
        self.forward_params = state[10]
        self.inplace = state[11]
        self.orig_model = state[12]
        model = copy.copy(self.orig_model)
        if not self.inplace:
            model = copy.copy(self.orig_model)
        self.model = model
        self.model.orig_forward = self.model.forward
        if self.use_lfu:
            self.model.forward = self.forward_lfs
        else:
            self.model.forward = self.forward
        self.model.set_iteration_count = self.set_iteration_count
        self.model.capture_start = self.capture_start
        self.model.capture_end = self.capture_end

    def __init__(self, max_graphs=10):
        super(ModuleCacher, self).__init__()
        self.max_graphs = max_graphs
        self.model_dict = {}
        self.input_count_dict = {}
        self.priority_keys = []
        self.is_capturing = False
        self.iteration_cnt = -1
        self.use_lazy_mode = os.environ.get("PT_HPU_LAZY_MODE", "1") == "1"
        self.hpugraph_tracing = False
        # Variables for statistics collection
        self.cached_hits_dict = {}
        self.uncached_train_hits = 0
        self.uncached_eval_hits = 0
        self.forward_cnt = 0
        self.set_iterations_call_cnt = 0
        self.disable_tensor_cache = False
        self.dry_run = False
        self.inplace = True

    def set_iteration_count(self, iter_num):
        self.forward_cnt = iter_num
        self.set_iterations_call_cnt += 1

    def is_hpugraph_tracing(self):
        return self.hpugraph_tracing

    def cache_replay(self, input_id, *args, **kwargs):
        self.cached_hits_dict[input_id] = self.cached_hits_dict.get(input_id, 0) + 1
        graph_model = self.model_dict[input_id]
        output = graph_model.graph_forward(*args, **kwargs)
        return output

    def cache_insert(self, input_id, *args, **kwargs):
        self.hpugraph_tracing = True
        graph_model = GraphModel(
            self.orig_model, self.allow_unused_input, self.asynchronous, self.disable_tensor_cache, self.dry_run
        )
        graph_model.init_hpu_graph(*args, **kwargs)
        self.model_dict[input_id] = graph_model
        ret = self.cache_replay(input_id, *args, **kwargs)
        self.hpugraph_tracing = False
        return ret

    def forward(self, *args, **kwargs):
        self.iteration_cnt += 1
        input_id = GraphModel.full_input_hash(self.forward_params, *args, **kwargs)
        use_cache = self.model.training and torch.is_grad_enabled() and self.use_lazy_mode

        if self.have_grad_accumulation and self.forward_cnt == 0:
            input_id = input_hash(
                (
                    input_id,
                    self.forward_cnt + 1,
                )
            )

        if self.verbose and self.iteration_cnt % self.log_frequency == 0:
            self.log_stats()

        if use_cache:
            if input_id in self.model_dict:
                return self.cache_replay(input_id, *args, **kwargs)

            if len(self.model_dict) < self.max_graphs:
                return self.cache_insert(input_id, *args, **kwargs)

        if self.model.training:
            self.uncached_train_hits += 1
        else:
            self.uncached_eval_hits += 1
        return self.model.orig_forward(*args, **kwargs)

    def capture_start(self):
        self.is_capturing = True

    def record(self, input_id):
        self.input_count_dict[input_id] = self.input_count_dict.get(input_id, 0) + 1

    def capture_end(self):
        self.priority_keys = sorted(self.input_count_dict.keys(), key=lambda x: self.input_count_dict[x], reverse=True)[
            : self.max_graphs
        ]
        self.is_capturing = False
        if self.verbose:
            self.log_stats()
        self.input_count_dict = {}

    def forward_lfs(self, *args, **kwargs):
        self.iteration_cnt += 1

        input_id = GraphModel.full_input_hash(self.forward_params, *args, **kwargs)
        use_cache = self.model.training and torch.is_grad_enabled() and self.use_lazy_mode and not self.is_capturing

        if self.have_grad_accumulation and self.forward_cnt == 0:
            input_id = input_hash(
                (
                    input_id,
                    self.forward_cnt + 1,
                )
            )

        if self.is_capturing:
            self.record(input_id)

        if self.verbose and self.iteration_cnt % self.log_frequency == 0:
            self.log_stats()

        if use_cache:
            if input_id in self.model_dict:
                return self.cache_replay(input_id, *args, **kwargs)

            if input_id in self.priority_keys:
                return self.cache_insert(input_id, *args, **kwargs)

        if self.model.training:
            self.uncached_train_hits += 1
        else:
            self.uncached_eval_hits += 1
        return self.model.orig_forward(*args, **kwargs)

    def __call__(
        self,
        model,
        use_lfu=False,
        inplace=True,
        allow_unused_input=False,
        asynchronous=False,
        have_grad_accumulation=False,
        log_frequency=100,
        verbose=False,
        disable_tensor_cache=False,
        dry_run=False,
    ):
        model.is_hpugraph_tracing = self.is_hpugraph_tracing
        self.inplace = inplace
        if not inplace:
            model = copy.copy(model)
        self.orig_model = copy.copy(model)
        self.forward_params = GraphModel.process_function_signature(self.orig_model.forward)
        self.model = model
        self.use_lfu = use_lfu
        self.model.orig_forward = self.model.forward
        if self.use_lfu:
            self.model.forward = self.forward_lfs
        else:
            self.model.forward = self.forward
        self.allow_unused_input = allow_unused_input
        self.asynchronous = asynchronous
        self.have_grad_accumulation = have_grad_accumulation
        self.model.set_iteration_count = self.set_iteration_count
        self.model.capture_start = self.capture_start
        self.model.capture_end = self.capture_end
        self.verbose = verbose
        self.log_frequency = log_frequency
        env_tensor_cache = os.environ.get("PT_HPUGRAPH_DISABLE_TENSOR_CACHE")
        self.disable_tensor_cache = disable_tensor_cache if env_tensor_cache is None else env_tensor_cache == "1"
        env_dry_run = os.environ.get("PT_HPUGRAPH_ENABLE_DRY_RUN")
        self.dry_run = dry_run if env_dry_run is None else env_dry_run == "1"
        return self.model

    def log_stats(self):
        print("HPU Graph Statistics")
        print("  Configs")
        print("    Max graphs                :-", self.max_graphs)
        print("    LFU Cache                 :-", self.use_lfu)
        print("    Async execution config    :-", self.asynchronous)
        print("    Grad accumulation config  :-", self.have_grad_accumulation)
        print("    Set iteration calls       :-", self.set_iterations_call_cnt)
        print("  Cache info ")
        print("    No. of HPUGraphs cached   :-", len(self.model_dict))
        print("    Priority keys             :-", self.priority_keys)
        print("    Input hash v. cached hits :-", self.cached_hits_dict)
        print("    Total cached hits         :-", sum(self.cached_hits_dict.values()))
        print("    Uncached train hits       :-", self.uncached_train_hits)
        print("    Uncached eval hits        :-", self.uncached_eval_hits)
        print(
            "    Total forwards executed   :-",
            sum(self.cached_hits_dict.values()) + self.uncached_train_hits + self.uncached_eval_hits,
        )
        if self.use_lfu and self.input_count_dict:
            print("    Input hash v. count         :", self.input_count_dict)

    def __del__(self):
        if self.verbose:
            self.log_stats()


def is_current_stream_capturing():
    current_stream = torch.hpu.current_stream()
    if hasattr(current_stream, "is_capture"):
        return current_stream.is_capture
    return False
