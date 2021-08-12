import torch
from test_utils import compare_tensors
import habana_frameworks.torch.core as htcore

def graph():
 a = torch.randn(2,2).to('hpu')
 b = a.abs()
 c = b[None,None]
 y = c == 1
 return y, b

[a, b] = graph()
htcore.mark_step()
[c, d] = graph()
htcore.mark_step()