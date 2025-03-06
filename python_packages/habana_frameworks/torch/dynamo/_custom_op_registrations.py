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

from typing import Optional

import torch
import torch.fx.node as fx_node
from torch.library import custom_op


def register_prepare_ops(fn, fake_fn, name: str, device_types: str = "cpu", schema: Optional[str] = None):
    op_name = f"hpu_prepare_ops::{name}"
    custom_fn = custom_op(op_name, fn, mutates_args=(), device_types=device_types, schema=schema)
    custom_fn.register_fake(fake_fn)
    fx_node._side_effectful_functions.add(eval(f"torch.ops.hpu_prepare_ops.{name}.default"))


def register_post_ops(fn, fake_fn, name: str, device_types: str = "cpu", schema: Optional[str] = None):
    op_name = f"hpu_post_ops::{name}"
    custom_fn = custom_op(op_name, fn, mutates_args=(), device_types=device_types, schema=schema)
    custom_fn.register_fake(fake_fn)
    fx_node._side_effectful_functions.add(eval(f"torch.ops.hpu_post_ops.{name}.default"))
