###############################################################################
#
#  Copyright (c) 2021-2024 Intel Corporation
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
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
