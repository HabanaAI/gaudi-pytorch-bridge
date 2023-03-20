import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from habana_frameworks.torch.utils.library_loader import load_habana_module
import habana_frameworks.torch.core as htcore
# load_habana_module()


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


def test(device, use_lazy_mode):

    if device == "hpu":
        if use_lazy_mode:
            print(f"Lazy Mode ==========")
            os.environ["PT_HPU_LAZY_MODE"] = "1"
        else:
            print(f"Eager Mode ==========")
            os.environ["PT_HPU_LAZY_MODE"] = "2"

    in_d = 1
    # conv_feature_layers = [(512, 10, 5), (512, 3, 2), (512, 3, 2), (512, 3, 2), (512, 3, 2), (512, 2, 2), (512, 2, 2)]
    conv_feature_layers = [(16, 10, 5), (16, 3, 2), (16, 3, 2), (16, 3, 2), (16, 3, 2), (16, 2, 2), (16, 2, 2)]
    conv_layers = nn.ModuleList()

    for dim, k, stride in conv_feature_layers:
        fe_blk = FeatureExtractor(n_in=in_d, n_out=dim, k=k, stride=stride,
                                  is_group_norm=(in_d == 1), device=device).to(device=device)
        conv_layers.append(fe_blk)
        in_d = dim

    #  x = torch.rand((8, 1, 121920)).float()
    x = torch.rand((8, 1, 1200)).float()
    x = x.to(device=device)

    for idx, conv in enumerate(conv_layers):
        x = conv(x)

    loss = nn.CrossEntropyLoss()
    target = torch.rand(x.shape, device=device)
    output = loss(x, target)
    output.backward()

    print(f"x[0,0,0]: {x[0,0,0]}")


device = ["hpu"]


for d in device:
    print(f"Summary device={d.upper()}")
    print(f"===============================")

    test(device=d, use_lazy_mode=False)
    test(device=d, use_lazy_mode=True)
