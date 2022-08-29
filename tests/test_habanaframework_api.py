import torch
import habana_frameworks.torch as htorch

# Use torch_hpu APIs, equivalent to torch.cuda APIs
def test_basic_apis():
    print("hpu available", htorch.hpu.is_available())
    print("hpu device count", htorch.hpu.device_count())
    print("hpu device name", htorch.hpu.get_device_name())
    print("hpu current device", htorch.hpu.current_device())
    print("hpu synchronize", htorch.hpu.synchronize())

    htorch.core.mark_step()

    print("dist init", htorch.distributed.hccl.initialize_distributed_hpu())



def test_device_synchronize_api():
    tA_h = torch.zeros(10, 2).to('hpu')
    tB_h = torch.full((1000,), 1, device="hpu")
    ht.hpu.synchronize()  # Need verify with the log


if __name__ == "__main__":
    test_basic_apis()
    test_device_synchronize()
