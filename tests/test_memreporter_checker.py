import os

import habana_frameworks.torch.core as htcore
import habana_frameworks.torch.utils.debug as htdebug
import pytest
import torch


@pytest.mark.xfail(reason="File doesn't exists")
def test_mem_reporter_file_created():
    a = torch.ones([2, 3, 40, 10]).to("hpu")
    htdebug._dump_memory_reporter()

    def run_iter():
        b = torch.transpose(a, 2, 3)
        c = b.clone()
        x = c.to("cpu")
        x = torch.tensor([[1, 2], [3, 4]]).to("hpu")
        x.transpose(0, 1)
        htcore.mark_step()

    for _ in range(2):
        run_iter()
        htdebug._dump_memory_reporter()
    htcore.mark_step()
    htdebug._dump_memory_reporter()

    file = "memory.reporter.json"
    if os.path.exists(file):
        os.remove(file)
    else:
        raise AssertionError(f"{file} doesn't exists")


if __name__ == "__main__":
    test_mem_reporter_file_created()
