###############################################################################
# Copyright (C) 2024 Habana Labs, Ltd. an Intel Company
# All Rights Reserved.
#
# Unauthorized copying of this file or any element(s) within it, via any medium
# is strictly prohibited.
# This file contains Habana Labs, Ltd. proprietary and confidential information
# and is subject to the confidentiality and license agreements under which it
# was provided.
#
###############################################################################

import cProfile
import os
from functools import wraps


def cprofile(dirname):
    """
    Wraps function with cProfile context and dumps files to dirname.
    To convert them to dot format use gprof2dot python package and run
    `gprof2dot -f pstats <input_file> -i <output_file>.dot`
    """
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    elif not os.path.isdir(dirname):
        raise RuntimeError("Provided dirname exists but is not a directory")

    def _decorate(func):
        if not hasattr(func, "__cprofile_counter"):
            func.__cprofile_counter = 0

        @wraps(func)
        def wrapper(*args, **kwargs):
            fullpath = os.path.join(dirname, f"{func.__name__}_{os.getpid()}_{func.__cprofile_counter}")
            with cProfile.Profile() as pr:
                func(*args, **kwargs)
                pr.dump_stats(fullpath)
            func.__cprofile_counter += 1

        return wrapper

    return _decorate
