###############################################################################
# Copyright (C) 2022 Habana Labs, Ltd. an Intel Company
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
import sys

import pytest

sys.path.append(f"{os.environ['PYTORCH_MODULES_ROOT_PATH']}/pytorch_helpers/synapse_logger/tools/")
import compute_time_stats as cts  # noqa


@pytest.mark.skip(reason="ValueError: min() arg is an empty sequence")
def test_compute_stats():
    input_functions = "empty_strided_hpu_lazy,slice_hpu_lazy,copy_hpu_lazy_"
    func_times = cts.generate_func_times(input_file="example_file.json", func_names=input_functions)
    assert len(func_times) == 3  # three functions

    for input_fn in input_functions.split(","):
        assert input_fn in func_times

    # variance of 1 element throws exception, so check
    # that it's returned 0 for single occurrence of copy_hpu_lazy_
    assert len(func_times["copy_hpu_lazy_"].samples) == 1 and func_times["copy_hpu_lazy_"].stddev == 0
