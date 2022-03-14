import torch
import habana_frameworks.torch.core as htcore
from test_utils import compare_tensors

torch.manual_seed(0)

def test_hpu_host_to_device_copy():
  bucket_tensor = torch.randn((2, 2), dtype=torch.float32, device="hpu")
  output_tensor = torch.zeros((2, 2), dtype=torch.float32, device="hpu")
  bucket_tensor.copy_(output_tensor)

  chechpoint_params = torch.ones((2, 2), dtype=torch.float32)
  bucket_tensor.copy_(chechpoint_params)
  htcore.mark_step()

  compare_tensors(bucket_tensor, chechpoint_params, atol=0, rtol=0)

if __name__ == '__main__':
  test_hpu_host_to_device_copy()
