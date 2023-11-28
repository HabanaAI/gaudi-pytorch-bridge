import functools
import itertools
from collections import defaultdict


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

    def __init__(self):
        self.id = next(FxGraphAnalyzer.id_iter)
        self.ops = defaultdict(FxGraphAnalyzer.OpCount)

    def __enter__(self):
        FxGraphAnalyzer.registered_contexts[self.id] = self
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        FxGraphAnalyzer.registered_contexts.pop(self.id)

    def count_ops(self, nodes, ctx, in_submodule=False):
        for n in nodes:
            if 'output_device' not in n.meta or n.meta['output_device'] is None or n.meta['output_device'].type != 'hpu':
                continue

            if n.op == "call_module":
                submod = ctx.graph_module.get_submodule(n.target)
                self.count_ops(submod.graph.nodes, ctx, in_submodule=True)
            elif n.op == "call_function":
                if in_submodule:
                    self.ops[str(n.target)].graph_count += 1
                else:
                    self.ops[str(n.target)].eager_count += 1

    def get_ops_summary(self):
        return self.ops
