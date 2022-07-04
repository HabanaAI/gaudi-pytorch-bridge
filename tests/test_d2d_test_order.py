import torch
import habana_frameworks.torch.core as htcore

device = 'hpu'
weight = torch.tensor([3.0], device=device)
t = torch.tensor([1.0], device=device)
temp = torch.tensor([4.0], device=device)

htcore.mark_step()

t1 = torch.add(weight, t) # t1 = weight + 1 = 3 + 1 = 4
weight.copy_(temp) #weight = 4

htcore.mark_step()
print(weight)
print(t1)
assert weight[0] == 4.0, f"Data mismatch in weight"
assert t1[0] == 4.0, f"Data mismatch in t1 (result of add)"
