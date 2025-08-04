###############################################################################
# Copyright (c) 2025 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
###############################################################################

import habana_frameworks.torch.internal.bridge_config as bc
import torch
from habana_frameworks.torch.hpu.metrics import metric_debug_reload, metric_global


class TestJitCache:
    _prev_val_cache_metrics = False

    @classmethod
    def setup_class(cls) -> None:
        """Enable cache-metric collection."""
        cls._prev_val_cache_metrics = bc.get_pt_hpu_enable_cache_metrics()
        bc.set_pt_hpu_enable_cache_metrics(True)
        metric_debug_reload()

    @classmethod
    def teardown_class(cls) -> None:
        """Restore the original cache-metric setting."""
        bc.set_pt_hpu_enable_cache_metrics(cls._prev_val_cache_metrics)
        metric_debug_reload()

    def test_recipe_cache_miss_then_hit(self) -> None:
        """
        Compile two identical graphs and assert that the recipe-cache
        records exactly one miss and one hit.
        """
        metric = metric_global("recipe_cache")

        def graph1_func(a):
            b = a * 2
            return b

        def graph2_func(a):
            b = a * 2
            return b

        torch.manual_seed(0)

        input_tensor = torch.randn(2, 4, requires_grad=True).to("hpu")

        compiled1 = torch.compile(graph1_func, backend="hpu_backend", dynamic=False)
        compiled2 = torch.compile(graph2_func, backend="hpu_backend", dynamic=False)
        result1 = compiled1(input_tensor).to("cpu")
        result2 = compiled2(input_tensor).to("cpu")

        assert torch.allclose(result1, result2), "Compiled graphs returned different outputs"
        stats = dict(metric.stats())
        assert stats["TotalMiss"] == 1, f"Expected 1 cache miss, got {stats['TotalMiss']}"
        assert stats["TotalHit"] == 1, f"Expected 1 cache hit, got {stats['TotalHit']}"
