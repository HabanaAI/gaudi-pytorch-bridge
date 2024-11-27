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
