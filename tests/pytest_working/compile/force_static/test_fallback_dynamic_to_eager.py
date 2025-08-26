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

import glob
import json
import shutil

import habana_frameworks.torch.utils.debug as htdebug
import pytest
import torch
from test_utils import compile_function_if_compile_mode


def func(x, y):
    # This is the dynamic part, we need run it in eager
    x = x + 1
    # Test _to_copy function call
    x = torch.ops.aten._to_copy.default(x, dtype=torch.bfloat16)
    x = x * 2

    # This is the static part, we need run it in compile
    y = y - 1
    y = y / 2

    # x.sum is dynamic part, we need run it in eager
    # y.mean() and x_sum - y_mean is static part and run in compile
    return x.sum() - y.mean()


def test_fallback_dynamic_to_eager():
    class Test_Ops:
        def __init__(self, fallback_dynamic_to_eager=True):
            self.fallback_dynamic_to_eager = fallback_dynamic_to_eager
            self.stats_path = pytest.stats_path
            self.run()
            self.check_compilation_type()
            htdebug._bridge_cleanup()

        def run(self):
            compiled_func = compile_function_if_compile_mode(
                func,
                dynamic=None,
                options={"use_eager_fallback": True, "fallback_dynamic_to_eager": self.fallback_dynamic_to_eager},
            )

            # CPU
            x = torch.randn([2, 5])
            y = torch.randn([3, 6])

            result_c = func(x, y)

            # HPU
            x_h = x.to("hpu")
            y_h = y.to("hpu")

            torch._dynamo.mark_dynamic(x_h, 0)
            torch._dynamo.mark_dynamic(x_h, 1)

            result_h = compiled_func(x_h, y_h)

            assert torch.allclose(result_h.to("cpu"), result_c, atol=0.001, rtol=0.001)

        def check_compilation_type(self):
            compile_types = []
            list_of_files = glob.glob(self.stats_path + "/*")
            assert len(list_of_files) > 0, "Compilation stat dumps not present"
            try:
                for file_ in list_of_files:
                    with open(file_) as f:
                        stats = json.loads(f.read() + "]")
                        for stat in stats:
                            for _, val in stat.items():
                                if "compilations" in val:
                                    compile_types.append(val["compilations"][0]["scope"])
            except:
                pass
            if self.fallback_dynamic_to_eager:
                assert "STATIC" in compile_types, "No static recipes with fallback_dynamic_to_eager=True"
                assert "DYNAMIC MIN + DYNAMIC MAX" not in compile_types, (
                    "Dynamic recipes with fallback_dynamic_to_eager=True"
                )
            else:
                assert "DYNAMIC MIN + DYNAMIC MAX" in compile_types, (
                    "No dynamic recipes with fallback_dynamic_to_eager=False"
                )

            shutil.rmtree(self.stats_path, ignore_errors=True)

    # Check with option enabled
    # 0 dnamic compilations should occur
    Test_Ops(fallback_dynamic_to_eager=True)

    # Check with option disabled
    # At least 1 dynamic compilations must occur
    Test_Ops(fallback_dynamic_to_eager=False)


def test_fallback_dynamic_to_eager_checked():
    compiled_func = compile_function_if_compile_mode(
        func, dynamic=None, options={"use_eager_fallback": False, "fallback_dynamic_to_eager": True}
    )

    x = torch.randn([2, 5])
    y = torch.randn([3, 6])

    # HPU
    x_h = x.to("hpu")
    y_h = y.to("hpu")

    torch._dynamo.mark_dynamic(x_h, 0)
    torch._dynamo.mark_dynamic(x_h, 1)

    try:
        result_h = compiled_func(x_h, y_h)
    except Exception as e:
        exception_str = str(e)

        # Check all the eager fallbacks caused by dynamic node
        assert "Eager fallback in nodes" in exception_str
        assert "torch.ops.aten.add" in exception_str
        assert "torch.ops.aten._to_copy" in exception_str
        assert "torch.ops.aten.mul" in exception_str
        assert "torch.ops.aten.sum" in exception_str

        assert "torch.ops.aten.sub" not in exception_str
        assert "torch.ops.aten.div" not in exception_str
        assert "torch.ops.aten.mean" not in exception_str
