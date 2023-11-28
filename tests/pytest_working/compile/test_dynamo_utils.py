import torch
import habana_frameworks.torch
from habana_frameworks.torch.utils.debug.dynamo_utils import FxGraphAnalyzer


@torch.compile(backend='aot_hpu_training_backend')
def fn(x, y, device):
    res = x + y
    eager_fallback_res = torch.randint(high=100, size=[1], device=device, dtype=torch.int32)
    return res + eager_fallback_res


@torch.compile(backend='aot_hpu_training_backend')
def fn2(x, y):
    res = x + y
    res = res * x
    return res * res


def assert_helper(ops_summary, op, graph_count, eager_count):
    assert op in ops_summary
    assert ops_summary[op].graph_count == graph_count
    assert ops_summary[op].eager_count == eager_count


def test_simple():
    torch._dynamo.reset()
    with FxGraphAnalyzer() as fga:
        t1 = torch.tensor([6], device='hpu')
        t2 = torch.tensor([2], device='hpu')
        fn(t1, t2, 'hpu')

    # Should be:
    # 'aten.randint.default': {graph_count = 0, eager_count = 1},
    # 'aten.add.Tensor': {graph_count = 2, eager_count = 0}})
    ops_summary = fga.get_ops_summary()
    assert_helper(ops_summary, 'aten.randint.default', 0, 1)
    assert_helper(ops_summary, 'aten.add.Tensor', 2, 0)


def test_cpu():
    torch._dynamo.reset()
    with FxGraphAnalyzer() as fga:
        t1 = torch.tensor([6], device='cpu')
        t2 = torch.tensor([2], device='cpu')
        fn(t1, t2, 'cpu')

    assert not fga.get_ops_summary()


def test_multiple():
    torch._dynamo.reset()
    with FxGraphAnalyzer() as fga:
        t1 = torch.tensor([6], device='hpu')
        t2 = torch.tensor([2], device='hpu')
        with FxGraphAnalyzer() as fga2:
            fn2(t1, t2)
        with FxGraphAnalyzer() as fga3:
            fn(t1, t2, 'hpu')
        fn(t1.to('cpu'), t2.to('cpu'), 'cpu')

    # ops_summary should be:
    # 'aten.randint.default': {graph_count = 0, eager_count = 1},
    # 'aten.add.Tensor': {graph_count = 3, eager_count = 0},
    # 'aten.mul.Tensor': {graph_count = 2, eager_count = 0}}
    ops_summary = fga.get_ops_summary()
    assert_helper(ops_summary, 'aten.randint.default', 0, 1)
    assert_helper(ops_summary, 'aten.add.Tensor', 3, 0)
    assert_helper(ops_summary, 'aten.mul.Tensor', 2, 0)

    # ops_summary2 should be:
    # 'aten.add.Tensor': {graph_count = 1, eager_count = 0},
    # 'aten.mul.Tensor': {graph_count = 2, eager_count = 0}}
    ops_summary2 = fga2.get_ops_summary()
    assert_helper(ops_summary2, 'aten.add.Tensor', 1, 0)
    assert_helper(ops_summary2, 'aten.mul.Tensor', 2, 0)

    # ops_summary3 should be:
    # 'aten.randint.default': {graph_count = 0, eager_count = 1},
    # 'aten.add.Tensor': {graph_count = 2, eager_count = 0}})
    ops_summary3 = fga3.get_ops_summary()
    assert_helper(ops_summary3, 'aten.randint.default', 0, 1)
    assert_helper(ops_summary3, 'aten.add.Tensor', 2, 0)

