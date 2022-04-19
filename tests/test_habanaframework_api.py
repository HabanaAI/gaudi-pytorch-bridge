import torch
import habana_frameworks.torch.hpu as hpu
# Use torch_hpu APIs, equivalent to torch.cuda APIs
print("hpu available", hpu.is_available())
print("device count", hpu.device_count())
print("device name", hpu.get_device_name())
print("current device", hpu.current_device())
print("synchronize", hpu.synchronize())
import habana_frameworks.torch.core as htcore
htcore.mark_step()
import habana_frameworks.torch.utils.experimental as exp
print("device_type", exp._get_device_type())
print("compute_stream", exp._compute_stream())
x = torch.randn(10, device='hpu')
print("data_ptr", exp._data_ptr(x))
device_type = exp._get_device_type()
if (device_type == exp.synDeviceType.synDeviceGaudi):
    print("gaudi")
from habana_frameworks.torch.distributed.hccl import initialize_distributed_hpu
print("dist init", initialize_distributed_hpu())


