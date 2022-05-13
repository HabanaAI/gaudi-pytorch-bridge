import habana_frameworks.torch as htorch
# Use torch_hpu APIs, equivalent to torch.cuda APIs
print("hpu available", htorch.hpu.is_available())
print("hpu device count", htorch.hpu.device_count())
print("hpu device name", htorch.hpu.get_device_name())
print("hpu current device", htorch.hpu.current_device())
print("hpu synchronize", htorch.hpu.synchronize())

htorch.core.mark_step()

print("dist init", htorch.distributed.hccl.initialize_distributed_hpu())