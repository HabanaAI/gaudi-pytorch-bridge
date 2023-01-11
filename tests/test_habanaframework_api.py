import torch
import habana_frameworks.torch as htorch
import os

# Use torch_hpu APIs, equivalent to torch.cuda APIs
def test_basic_apis():
    print("hpu available", htorch.hpu.is_available())
    print("hpu device count", htorch.hpu.device_count())
    print("hpu device name", htorch.hpu.get_device_name())
    print("hpu current device", htorch.hpu.current_device())
    print("hpu synchronize", htorch.hpu.synchronize())
    print("hpu is_bf16_supported", htorch.hpu.is_bf16_supported())
    d = torch.device('hpu')
    print("hpu get_device_capability", htorch.hpu.get_device_capability(d))
    print("hpu get_device_properties", htorch.hpu.get_device_properties(d))
    print("hpu get_arch_list", htorch.hpu.get_arch_list())
    print("hpu get_gencode_flags", htorch.hpu.get_gencode_flags())
    if (htorch.hpu.device_count() >= 2):
        print("hpu can_device_access_peer", htorch.hpu.can_device_access_peer(0, 1))
        os.environ["HLS_MODULE_ID"] = "1"
        htorch.hpu.set_device(1)
        print (os.getenv("HLS_MODULE_ID"))
        os.environ["HLS_MODULE_ID"] = "0"
        with htorch.hpu.device(0):
            print (os.getenv("HLS_MODULE_ID"))

    htorch.core.mark_step()

    print("dist init", htorch.distributed.hccl.initialize_distributed_hpu())

def test_device_synchronize_api():
    tA_h = torch.zeros(10, 2).to('hpu')
    tB_h = torch.full((1000,), 1, device="hpu")
    ht.hpu.synchronize()  # Need verify with the log

if __name__ == "__main__":
    test_basic_apis()
    test_device_synchronize()
