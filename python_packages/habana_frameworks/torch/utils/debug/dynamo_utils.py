import atexit
import itertools
from collections import defaultdict

import torch
from habana_frameworks.torch.dynamo.compile_backend.logger import get_compile_backend_logger

logger = get_compile_backend_logger()


class FxGraphAnalyzer:
    class OpCount:
        def __init__(self):
            self.graph_count = 0
            self.eager_count = 0

        def __str__(self):
            return f"{{graph_count = {self.graph_count}, eager_count = {self.eager_count}}}"

        def __repr__(self):
            return str(self)

    id_iter = itertools.count()
    registered_contexts: dict = dict()

    def __init__(self, reset_dynamo=False):
        self.reset_dynamo = reset_dynamo
        self.id = next(FxGraphAnalyzer.id_iter)
        self.graphs = list()
        atexit.register(self._at_exit_callback)

    def __del__(self):
        atexit.unregister(self._at_exit_callback)

    def __enter__(self):
        FxGraphAnalyzer.registered_contexts[self.id] = self
        if self.reset_dynamo:
            torch._dynamo.reset()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        FxGraphAnalyzer.registered_contexts.pop(self.id)

    def _at_exit_callback(self):
        def check_for_eager(graph_ops):
            return any(map(lambda elem: elem.eager_count, graph_ops.values()))

        fallback_ops = list(filter(check_for_eager, self.get_ops_summary()))
        if fallback_ops:
            logs = "ops in graph with eager fallbacks:\n"
            for graph_ops in fallback_ops:
                for k, v in graph_ops.items():
                    if v.eager_count > 0:
                        logs += str(k) + " " + str(v) + "\n"
            logger.critical(logs)

    def count_ops(self, nodes, ctx, in_submodule=False, ops_in_graph=None):
        if ops_in_graph is None:
            ops_in_graph = defaultdict(FxGraphAnalyzer.OpCount)
        for n in nodes:
            if n.op == "call_module":
                submodule = ctx.graph_module.get_submodule(n.target)
                self.count_ops(submodule.graph.nodes, ctx, True, ops_in_graph)
            elif n.op in {"call_function", "call_method"}:
                if (
                    "output_device" not in n.meta
                    or n.meta["output_device"] is None
                    or n.meta["output_device"].type != "hpu"
                ):
                    continue
                target_name = n._pretty_print_target(n.target)
                if in_submodule:
                    ops_in_graph[target_name].graph_count += 1
                else:
                    ops_in_graph[target_name].eager_count += 1

        if not in_submodule:
            self.graphs.append(dict(ops_in_graph))

    def get_ops_summary(self):
        return self.graphs
