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

# mypy: ignore-errors

# noqa: F401, E501
# This is an auto-generated file. Please do not modify it by hand.
# To re-generate, run:
# cd ~/pytorch && python torchgen/fuse/gen_patterns.py

import operator

import torch
import torch._inductor
from torch._inductor.pattern_matcher import (
    CallFunction,
    Ignored,
    KeywordArg,
    MultiOutputPattern,
)

aten = torch.ops.aten
prims = torch.ops.prims

expand_default = CallFunction(aten.expand.default, KeywordArg("query"), Ignored())
view_default = CallFunction(aten.view.default, expand_default, Ignored(), _users=2)
transpose_int = CallFunction(aten.transpose.int, KeywordArg("key"), Ignored(), Ignored())
expand_default_1 = CallFunction(aten.expand.default, transpose_int, Ignored())
view_default_1 = CallFunction(aten.view.default, expand_default_1, Ignored(), _users=2)
bmm_default = CallFunction(aten.bmm.default, view_default, view_default_1)
view_default_2 = CallFunction(aten.view.default, bmm_default, Ignored())
div_Tensor = CallFunction(aten.div.Tensor, view_default_2, KeywordArg("inv_scale_factor"))
_softmax_default = CallFunction(aten._softmax.default, div_Tensor, Ignored(), False, _users=2)
native_dropout_default = CallFunction(
    aten.native_dropout.default, _softmax_default, KeywordArg("dropout_p"), True, _users=2
)
operator_getitem = CallFunction(operator.getitem, native_dropout_default, 0)
expand_default_2 = CallFunction(aten.expand.default, operator_getitem, Ignored())
view_default_3 = CallFunction(aten.view.default, expand_default_2, Ignored(), _users=2)
expand_default_3 = CallFunction(aten.expand.default, KeywordArg("value"), Ignored())
view_default_4 = CallFunction(aten.view.default, expand_default_3, Ignored(), _users=2)
bmm_default_1 = CallFunction(aten.bmm.default, view_default_3, view_default_4)
view_default_5 = CallFunction(aten.view.default, bmm_default_1, Ignored())
view_default_6 = CallFunction(aten.view.default, KeywordArg("tangents_1"), Ignored(), _users=2)
transpose_int_1 = CallFunction(aten.transpose.int, view_default_4, Ignored(), Ignored())
bmm_default_2 = CallFunction(aten.bmm.default, view_default_6, transpose_int_1)
view_default_7 = CallFunction(aten.view.default, bmm_default_2, Ignored())
operator_getitem_1 = CallFunction(operator.getitem, native_dropout_default, 1)
_to_copy_default = CallFunction(
    aten._to_copy.default, operator_getitem_1, dtype=Ignored(), layout=torch.strided, device=Ignored()
)
mul_Tensor = CallFunction(aten.mul.Tensor, _to_copy_default, Ignored())
mul_Tensor_1 = CallFunction(aten.mul.Tensor, view_default_7, mul_Tensor)
_softmax_backward_data_default = CallFunction(
    aten._softmax_backward_data.default, mul_Tensor_1, _softmax_default, Ignored(), Ignored()
)
div_Tensor_1 = CallFunction(aten.div.Tensor, _softmax_backward_data_default, KeywordArg("inv_scale_factor"))
view_default_8 = CallFunction(aten.view.default, div_Tensor_1, Ignored(), _users=2)
transpose_int_2 = CallFunction(aten.transpose.int, view_default_1, Ignored(), Ignored())
bmm_default_3 = CallFunction(aten.bmm.default, view_default_8, transpose_int_2)
view_default_9 = CallFunction(aten.view.default, bmm_default_3, Ignored())
transpose_int_3 = CallFunction(aten.transpose.int, view_default, Ignored(), Ignored())
bmm_default_4 = CallFunction(aten.bmm.default, transpose_int_3, view_default_8)
view_default_10 = CallFunction(aten.view.default, bmm_default_4, Ignored())
transpose_int_4 = CallFunction(aten.transpose.int, view_default_10, Ignored(), Ignored())
transpose_int_5 = CallFunction(aten.transpose.int, view_default_3, Ignored(), Ignored())
bmm_default_5 = CallFunction(aten.bmm.default, transpose_int_5, view_default_6)
view_default_11 = CallFunction(aten.view.default, bmm_default_5, Ignored())
_sfdp_pattern_3_training = MultiOutputPattern(
    [view_default_5, view_default_9, transpose_int_4, view_default_11, None, None]
)


expand_default = CallFunction(aten.expand.default, KeywordArg("query"), Ignored())
view_default = CallFunction(aten.view.default, expand_default, Ignored())
transpose_int = CallFunction(aten.transpose.int, KeywordArg("key"), Ignored(), Ignored())
expand_default_1 = CallFunction(aten.expand.default, transpose_int, Ignored())
view_default_1 = CallFunction(aten.view.default, expand_default_1, Ignored())
bmm_default = CallFunction(aten.bmm.default, view_default, view_default_1)
view_default_2 = CallFunction(aten.view.default, bmm_default, Ignored())
div_Tensor = CallFunction(aten.div.Tensor, view_default_2, KeywordArg("inv_scale_factor"))
_softmax_default = CallFunction(aten._softmax.default, div_Tensor, Ignored(), False)
expand_default_2 = CallFunction(aten.expand.default, _softmax_default, Ignored())
view_default_3 = CallFunction(aten.view.default, expand_default_2, Ignored())
expand_default_3 = CallFunction(aten.expand.default, KeywordArg("value"), Ignored())
view_default_4 = CallFunction(aten.view.default, expand_default_3, Ignored())
bmm_default_1 = CallFunction(aten.bmm.default, view_default_3, view_default_4)
_sfdp_pattern_3_inference = CallFunction(aten.view.default, bmm_default_1, Ignored(), _users=0)
