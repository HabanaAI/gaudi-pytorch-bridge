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
from habana_frameworks.torch import _hpu_C


# utils function to define base object proxy
def _proxy_module(name: str) -> type:
    def init_err(self):
        class_name = self.__class__.__name__

        raise RuntimeError("Tried to instantiate proxy base class {}.".format(class_name))

    return type(name, (object,), {"__init__": init_err})


def _register_proxy(module: str):
    if not hasattr(_hpu_C, module):
        _hpu_C.__dict__[module] = _proxy_module(module)


_register_proxy("_HpuStreamBase")
_register_proxy("_HpuEventBase")
