import torch
import torch_hpu

print(torch_hpu.is_available())
print(torch_hpu.get_device_type())
print(torch_hpu.device_count())
