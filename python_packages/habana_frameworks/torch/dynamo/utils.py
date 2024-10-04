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


def auto_map(f):
    def wrapper(x, *args, **kwargs):
        if isinstance(x, (list, tuple)):
            return type(x)(map(lambda y: wrapper(y, *args, **kwargs), x))
        return f(x, *args, **kwargs)

    return wrapper


def str_join(inp):
    if isinstance(inp, (list, tuple)):
        out = ", ".join(str_join(x) for x in inp)
        if isinstance(inp, list):
            return f"[{out}]"
        else:
            return f"({out})"

    return str(inp)
