import torch
import torch_hpu

print(torch_hpu.is_available())
print(torch_hpu.get_device_type())
print(torch_hpu.device_count())
print(torch_hpu.get_device_name())
print(torch_hpu.get_device_name(0))
torch_hpu.get_device_name('hpu')
print(torch_hpu.get_device_name('hpu:0'))
d = torch.device('hpu')
print(torch_hpu.get_device_name(d))
