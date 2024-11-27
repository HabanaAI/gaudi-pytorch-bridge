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

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from test_utils import setup_teardown_env_fixture


class Fp32GroupNorm(nn.GroupNorm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, input):
        output = F.group_norm(
            input.float(),
            self.num_groups,
            self.weight.float() if self.weight is not None else None,
            self.bias.float() if self.bias is not None else None,
            self.eps,
        )
        return output.type_as(input)


class FeatureExtractor(nn.Module):
    def __init__(self, **cfg):
        super().__init__()

        n_in = cfg["n_in"]
        n_out = cfg["n_out"]
        k = cfg["k"]
        stride = cfg["stride"]
        is_group_norm = cfg["is_group_norm"]
        self.device = cfg["device"]

        self.conv = nn.Conv1d(n_in, n_out, k, stride=stride, bias=False)
        nn.init.kaiming_normal_(self.conv.weight)

        self.droupout = nn.Dropout(p=1e-20)
        self.group_norm = Fp32GroupNorm(n_out, n_out, affine=True)
        self.gelu = nn.GELU()

        self.is_group_norm = is_group_norm

    def forward(self, x):
        pre_shape = x.shape
        x = self.conv(x)
        print(f"pre shape:{pre_shape}  x shape:{x.shape}")

        x = self.droupout(x)
        x = self.group_norm(x)
        x = self.gelu(x)

        return x


@pytest.mark.skip(reason="Graph compile failed. synStatus 26")
@pytest.mark.skip(reason="Tests in this file are chaning env variables")
@pytest.mark.parametrize(
    "setup_teardown_env_fixture",
    [{"PT_HPU_LAZY_MODE": "1"}, {"PT_HPU_LAZY_MODE": "2"}],
    indirect=True,
)
def test_bn_3d(setup_teardown_env_fixture, device="hpu"):
    in_d = 1
    conv_feature_layers = [
        (16, 10, 5),
        (16, 3, 2),
        (16, 3, 2),
        (16, 3, 2),
        (16, 3, 2),
        (16, 2, 2),
        (16, 2, 2),
    ]
    conv_layers = nn.ModuleList()

    for dim, k, stride in conv_feature_layers:
        fe_blk = FeatureExtractor(
            n_in=in_d,
            n_out=dim,
            k=k,
            stride=stride,
            is_group_norm=(in_d == 1),
            device=device,
        ).to(device=device)
        conv_layers.append(fe_blk)
        in_d = dim

    x = torch.rand((8, 1, 1200)).float()
    x = x.to(device=device)

    for _, conv in enumerate(conv_layers):
        x = conv(x)

    loss = nn.CrossEntropyLoss()
    target = torch.rand(x.shape, device=device)
    output = loss(x, target)
    output.backward()

    print(f"x[0,0,0]: {x[0,0,0]}")
