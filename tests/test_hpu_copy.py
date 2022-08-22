import torch
import habana_frameworks.torch.core as htcore
from test_utils import compare_tensors

torch.manual_seed(0)

def test_hpu_h2dcopy_1():
  bucket_tensor = torch.randn((2, 2), dtype=torch.float32, device='hpu')
  output_tensor = torch.zeros((2, 2), dtype=torch.float32, device='hpu')
  bucket_tensor.copy_(output_tensor)

  chechpoint_params = torch.ones((2, 2), dtype=torch.float32)
  bucket_tensor.copy_(chechpoint_params)

  htcore.mark_step()
  compare_tensors(bucket_tensor, chechpoint_params, atol=0, rtol=0)

def test_hpu_h2dcopy_2():
  hpu_tensor = torch.randn((2, 8), dtype=torch.float32)
  hpu_tensor = hpu_tensor.to(device='hpu')
  hpu_view_t = hpu_tensor.reshape(-1)
  cpu_tensor = torch.ones_like(hpu_view_t, dtype=torch.float32, device='cpu')

  with torch.no_grad():
    hpu_view_t.copy_(cpu_tensor)

  htcore.mark_step()
  compare_tensors(hpu_view_t, cpu_tensor, atol=0, rtol=0)

def test_hpu_h2dcopy_3():
  hpu_tensor = torch.randn((3, 3), dtype=torch.float32)
  hpu_tensor = hpu_tensor.to(device='hpu')
  cpu_tensor = torch.ones((2,  2), dtype=torch.float32)
  hpu_strided_t = hpu_tensor.as_strided((2, 2), (1, 2))

  with torch.no_grad():
    hpu_strided_t.copy_(cpu_tensor)

  htcore.mark_step()
  compare_tensors(hpu_strided_t, cpu_tensor, atol=0, rtol=0)

def test_hpu_h2dcopy_4():
  hpu_tensor = torch.randn((2, 8), dtype=torch.float32)
  hpu_tensor = hpu_tensor.to(device='hpu')
  hpu_view_t = hpu_tensor.reshape(-1)
  cpu_tensor = torch.ones_like(hpu_view_t, dtype=torch.float32, device='cpu')
  hpu_view_t.add_(2)

  with torch.no_grad():
    hpu_view_t.copy_(cpu_tensor)

  hpu_view_t = hpu_view_t.reshape(hpu_tensor.size())
  hpu_view_t.add_(2)

  cpu_tensor = cpu_tensor.reshape(hpu_tensor.size())
  cpu_tensor.add_(2)

  htcore.mark_step()
  compare_tensors(hpu_view_t, cpu_tensor, atol=0, rtol=0)

def test_hpu_h2dcopy_5():
  hpu_tensor = torch.randn((3, 2, 20, 20))
  hpu_tensor = hpu_tensor.to(device='hpu')
  hpu_tensor = hpu_tensor.to(memory_format=torch.channels_last)
  cpu_tensor = torch.rand_like(hpu_tensor, device='cpu', memory_format=torch.channels_last)

  hpu_tensor = hpu_tensor.view(hpu_tensor.size())

  with torch.no_grad():
    hpu_tensor.copy_(cpu_tensor)

  htcore.mark_step()
  compare_tensors(hpu_tensor, cpu_tensor, atol=0, rtol=0)

if __name__ == '__main__':
  test_hpu_h2dcopy_1()
  test_hpu_h2dcopy_2()
  test_hpu_h2dcopy_3()
  test_hpu_h2dcopy_4()
  test_hpu_h2dcopy_5()
