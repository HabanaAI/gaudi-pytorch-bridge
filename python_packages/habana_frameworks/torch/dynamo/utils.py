###############################################################################
# Copyright (c) 2021-2025 Intel Corporation
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


def auto_map(f):
    def wrapper(x, *args, **kwargs):
        if isinstance(x, list | tuple):
            return type(x)(wrapper(y, *args, **kwargs) for y in x)
        return f(x, *args, **kwargs)

    return wrapper


def str_join(inp):
    if isinstance(inp, list | tuple):
        out = ", ".join(str_join(x) for x in inp)
        if isinstance(inp, list):
            return f"[{out}]"
        else:
            return f"({out})"

    return str(inp)


def str_to_bool(s):
    if isinstance(s, bool):
        return s
    elif isinstance(s, str):
        return s.lower() in ["true", "1", "yes", "y"]
    else:
        raise ValueError(f"Input must be a string or bool, got {type(s).__name__}: {s}")
