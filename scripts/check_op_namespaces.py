###############################################################################
#
#  Copyright (c) 2025 Intel Corporation
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

"""Op searcher

Usage:
    op_searcher <op_name>
"""

import torch
import torchvision
from docopt import docopt
from termcolor import colored


def test_namespace(namespace, op):
    if hasattr(namespace, op):
        print(namespace.__name__ + ": " + colored("OK", "green"))
    else:
        print(namespace.__name__ + ": " + colored("NO", "red"))


if __name__ == "__main__":
    arguments = docopt(__doc__)

    op_name = arguments["<op_name>"]

    test_namespace(torch.nn.functional, op_name)
    test_namespace(torch.linalg, op_name)
    test_namespace(torch, op_name)
    test_namespace(torch.ops.aten, op_name)
    test_namespace(torch.nn, op_name)
    test_namespace(torch.nn.utils, op_name)
    test_namespace(torch.Tensor, op_name)
    test_namespace(torch.special, op_name)
    test_namespace(torchvision.ops, op_name)
    test_namespace(torch.ops, op_name)
