import os
from typing import Pattern

import habana_frameworks.torch as hftorch


def _check_hftorch_path():
    if not isinstance(hftorch.__path__, list) or len(hftorch.__path__) != 1:
        assert (False, "Bad __path__ of habana_frameworks.torch, expecting list with 1 element")


def get_include_dir():
    _check_hftorch_path()
    torch_path = hftorch.__path__[0]
    include_dir = os.path.join(torch_path, "include")
    return include_dir


def get_lib_dir():
    _check_hftorch_path()
    torch_path = hftorch.__path__[0]
    include_dir = os.path.join(torch_path, "lib")
    return include_dir
