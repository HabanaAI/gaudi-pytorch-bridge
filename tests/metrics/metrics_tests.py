###############################################################################
# Copyright (C) 2023 Habana Labs, Ltd. an Intel Company
# All Rights Reserved.
#
# Unauthorized copying of this file or any element(s) within it, via any medium
# is strictly prohibited.
# This file contains Habana Labs, Ltd. proprietary and confidential information
# and is subject to the confidentiality and license agreements under which it
# was provided.
#
###############################################################################

import os

os.environ["PT_HPU_ENABLE_CACHE_METRICS"] = "1"
import pytest
import torch
from habana_frameworks.torch.hpu.metrics import metric_global
from habana_frameworks.torch.utils.library_loader import load_habana_module

load_habana_module()
torch.ops.load_library("build/libmetrics_tests.so")


class TestMemoryDefragmentationMetrics:
    @pytest.fixture(scope="function")
    def md_metric(self):
        m = metric_global("memory_defragmentation")
        m.reset()
        yield m

    def test_memory_metrics(self, md_metric):
        torch.ops.test_ops.trigger_test_metrics()
        md_stats = dict(md_metric.stats())
        print(md_stats)
        assert len(md_metric.stats()) == 4
        assert md_stats["TotalNumber"] == 2
        assert md_stats["TotalSuccessful"] == 1
        assert md_stats["MaxTime"] >= 100
        torch.ops.test_ops.trigger_test_metrics()
        md_stats = dict(md_metric.stats())
        print(md_stats)
        assert len(md_metric.stats()) == 4
        assert md_stats["TotalNumber"] == 4
        assert md_stats["TotalSuccessful"] == 2


class TestCacheMetrics:
    @pytest.fixture(scope="function")
    def rc_metric(self):
        m = metric_global("recipe_cache")
        m.reset()
        yield m

    def test_cache_metrics(self, rc_metric):
        assert len(dict(rc_metric.stats()).items()) == 4
        torch.ops.test_ops.trigger_test_metrics()
        rc_stats = dict(rc_metric.stats())
        print(rc_stats)
        assert len(rc_metric.stats()) == 4
        assert rc_stats["TotalHit"] == 3
        assert rc_stats["TotalMiss"] == 1
        assert rc_stats["RecipeHit"]["123"] == 1
        assert rc_stats["RecipeMiss"]["123"] == 1
        assert rc_stats["RecipeHit"]["456"] == 2


class TestCpuFallbackMetrics:
    @pytest.fixture(scope="function")
    def cf_metric(self):
        m = metric_global("cpu_fallback")
        m.reset()
        yield m

    def test_cpu_fallback_metrics(self, cf_metric):
        assert len(dict(cf_metric.stats()).items()) == 2
        torch.ops.test_ops.trigger_test_metrics()
        cf_stats = dict(cf_metric.stats())
        print(cf_stats)
        assert len(cf_metric.stats()) == 2
        assert cf_stats["TotalNumber"] == 2
        assert len(cf_stats["FallbackOps"].items()) == 2
        assert cf_stats["FallbackOps"]["metrics_trigger_fallback_op"] == 1
        assert cf_stats["FallbackOps"]["metrics_trigger_fallback_op_2"] == 1
        torch.ops.test_ops.trigger_test_metrics()
        cf_stats = dict(cf_metric.stats())
        print(cf_stats)
        assert len(cf_metric.stats()) == 2
        assert cf_stats["TotalNumber"] == 4
        assert len(cf_stats["FallbackOps"].items()) == 2
        assert cf_stats["FallbackOps"]["metrics_trigger_fallback_op"] == 2
        assert cf_stats["FallbackOps"]["metrics_trigger_fallback_op_2"] == 2
