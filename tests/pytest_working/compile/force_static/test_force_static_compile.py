###############################################################################
# Copyright (C) 2023-2024 Habana Labs, Ltd. an Intel Company
# All Rights Reserved.
#
# Unauthorized copying of this file or any element(s) within it, via any medium
# is strictly prohibited.
# This file contains Habana Labs, Ltd. proprietary and confidential information
# and is subject to the confidentiality and license agreements under which it
# was provided.
#
###############################################################################

import glob
import json
import shutil

import habana_frameworks.torch.utils.debug as htdebug
import pytest
import torch


def test_force_static_compile():

    class Test_Ops:

        def __init__(self, force_static_compile=True):
            self.force_static_compile = force_static_compile
            self.stats_path = pytest.stats_path
            self.run()
            self.check_compilation_type()
            htdebug._bridge_cleanup()

        def run(self):
            input_info = [
                [(96), (2, 16, 3), (48, 3, 1)],
                [(160), (2, 20, 4), (80, 4, 1)],
                [(250), (2, 25, 5), (125, 5, 1)],
            ]

            def raw_function(t1, shape, strides):
                t2 = t1 * 1
                t3 = torch.as_strided(t2, shape, strides)
                t4 = t3.add(0)
                t5 = t4.view(-1, 2)
                t6 = t5[:-1]
                result = t6 * 10
                return result

            torch._dynamo.reset()
            compiled_fn = torch.compile(
                raw_function,
                backend="hpu_backend",
                dynamic=True,
                options={"force_static_compile": self.force_static_compile},
            )

            for size, shape, strides in input_info:
                # CPU
                tensor = torch.randn(size)
                result = raw_function(tensor, shape, strides)

                # HPU
                tensor_h = tensor.to("hpu")
                result_h = compiled_fn(tensor_h, shape, strides)

                assert torch.allclose(result_h.to("cpu"), result, atol=0.001, rtol=0.001)

        def check_compilation_type(self):
            compile_types = []
            list_of_files = glob.glob(self.stats_path + "/*")
            assert len(list_of_files) > 0, "Compilation stat dumps not present"
            try:
                for file_ in list_of_files:
                    with open(file_, "r") as f:
                        stats = json.loads(f.read() + "]")
                        for stat in stats:
                            for _, val in stat.items():
                                if "compilations" in val:
                                    compile_types.append(val["compilations"][0]["scope"])
            except:
                pass
            if self.force_static_compile:
                assert "STATIC" in compile_types, "No static recipes with force_static_compile=True"
                assert (
                    "DYNAMIC MIN + DYNAMIC MAX" not in compile_types
                ), "Dynamic recipes with force_static_compile=True"
            else:
                assert (
                    "DYNAMIC MIN + DYNAMIC MAX" in compile_types
                ), "No dynamic recipes with force_static_compile=False"

            shutil.rmtree(self.stats_path, ignore_errors=True)

    # Check with Option Enbaled
    # "Zero" Dynamic compilations should occur
    Test_Ops(force_static_compile=True)

    # Check with Option Disabled
    # At least "One" Dynamic compilations Must occur
    Test_Ops(force_static_compile=False)
