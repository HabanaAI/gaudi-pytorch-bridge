from __future__ import print_function

import os

import habana_frameworks.torch.core as htcore
import habana_frameworks.torch.utils.debug as htdebug
import torch


def test_mem_stats_file_created():
    a = torch.ones([20, 30, 400, 50]).to("hpu")
    htdebug._memstat_devmem_start_collect("htcore.memstat_devmem_start_collect...", False)

    def run_iter():
        b = torch.transpose(a, 2, 3)
        c = b.clone()
        x = c.to("cpu")
        x = torch.tensor([[1, 2], [3, 4]]).to("hpu")
        x.transpose(0, 1)
        htcore.mark_step()

    for i in range(2):
        run_iter()
        htdebug._memstat_devmem_stop_collect("Transpose mem stat:" + str(i))
    htcore.mark_step()
    htdebug._memstat_devmem_stop_collect("Transpose mem stat END")


if __name__ == "__main__":
    assert os.path.exists("habana_log.livealloc.log_0")
    test_mem_stats_file_created()
