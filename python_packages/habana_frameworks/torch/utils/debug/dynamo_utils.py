import itertools
from collections import defaultdict

import torch


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
    registered_contexts = dict()

    def __init__(self, reset_dynamo=False):
        self.reset_dynamo = reset_dynamo
        self.id = next(FxGraphAnalyzer.id_iter)
        self.graphs = list()

    def __enter__(self):
        FxGraphAnalyzer.registered_contexts[self.id] = self
        if self.reset_dynamo:
            torch._dynamo.reset()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        FxGraphAnalyzer.registered_contexts.pop(self.id)

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
