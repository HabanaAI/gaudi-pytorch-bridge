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
from contextlib import contextmanager

def set_flag_in_env(name: str, value):
    if value is None:
        # Nothing to do here
        return
    elif isinstance(value, str):
        os.environ[name] = value
    elif isinstance(value, bool):
        os.environ[name] = str(int(value))
    elif isinstance(value, int):
        os.environ[name] = str(value)
    else:
        assert False, f"Value '{value}' invalid or not supported"


@contextmanager
def env_var_in_scope(**kwargs):
    orig_vars = {}
    for key in kwargs.keys():
        orig_vars[key] = os.environ.get(key, None)
        set_flag_in_env(key, kwargs[key])
    try:
        yield
    finally:
        for key in orig_vars.keys():
            # restore environment variable
            if orig_vars[key] is not None:
                os.environ[key] = orig_vars[key]
            else:
                if key in os.environ:
                    del os.environ[key]

def disable_pgm_cache(func):
    def _disable_pgm_cache():
        with env_var_in_scope(PT_HPU_PGM_ENABLE_CACHE=0):
            func()
    return _disable_pgm_cache

def lazy_mode_zero(func):
    def _lazy_mode_zero():
        with env_var_in_scope(PT_HPU_LAZY_MODE=0):
            func()
    return _lazy_mode_zero