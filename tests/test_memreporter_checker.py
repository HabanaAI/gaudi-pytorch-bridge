from __future__ import print_function
import argparse
import os
import torch

def check_mem_reporter_file_created():
    import habana_frameworks.torch.core as htcore
    a = torch.ones([20,30,400,50]).to('hpu')
    import habana_frameworks.torch.utils.debug as htdebug
    htdebug._dump_memory_reporter()
    def run_iter():
      b = torch.transpose(a, 2,3)
      c = b.clone()
      x = c.to("cpu")
      x = torch.tensor([[1,2],[3,4]]).to('hpu')
      x1 = x.transpose(0,1)
      htcore.mark_step()

    for i in range(2):
      run_iter()
      htdebug._dump_memory_reporter()
    htcore.mark_step()
    htdebug._dump_memory_reporter()

    assert(os.path.exists("memory.reporter.json"))

if __name__ == '__main__':
    check_mem_reporter_file_created()
