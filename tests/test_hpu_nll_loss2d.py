import torch
import os
import sys

torch.ops.load_library(os.path.join(os.environ['BUILD_ROOT_LATEST'], "libhabana_pytorch_plugin.so"))
device = torch.device("hpu")

print("dev type", device.type)
N, C = 5, 4
loss = torch.nn.NLLLoss()
data = torch.randn(N, 16, 10, 10)
data_h = data.to(device)
target = torch.empty(N, 10, 10, dtype=torch.long).random_(0, C)
target_h = target.to(device)
output = loss(data, target)
print("cpu output ", output)
output_h = loss(data_h, target_h)
print("hpu output ", output_h.to("cpu"))
