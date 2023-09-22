import torch
import habana_frameworks.torch.core as htcore
import habana_frameworks.torch.dynamo.compile_backend

def fn(val):
  return torch.full((2,2), val, dtype=torch.float, device="hpu")

compiled_fn = torch.compile(fn, backend="aot_hpu_training_backend")
compiled_fn(float('nan'))