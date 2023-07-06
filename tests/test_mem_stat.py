###############################################################################
# Copyright (C) 2023 Habana Labs, Ltd. an Intel Company
# All Rights Reserved.
#
# Unauthorized copying of this file or any element(s) within it, via any medium
# is strictly prohibited.
# This file contains Habana Labs, Ltd. proprietary and confidential information
# and is subject to the confidentiality and license agreements under which it
# was provided.
#
###############################################################################

# Memory Stats Apis description and test case
# 1. max_memory_allocated
# 2. reset_peak_memory_stats
# 3. memory_allocated
# 4. memory_summary
# 5. memory_stats
# 6. memory_reserved
# 7. max_memory_reserved
# 8. memory_cached
# 9. max_memory_cached
# 10. mem_get_info

# TORCH.HPU.MAX_MEMORY_ALLOCATED
# This API returns peak HPU memory allocated by tensors( in bytes). reset_peak_memory_stats() can be used to reset the starting point in tracing stats.

# TORCH.HPU.MEMORY_ALLOCATED
# Returns the current HPU memory occupied by tensors.

# TORCH.HPU.MEMORY_STATS
# Returns list of HPU memory statics. Below sample memory stats printout and details

# ('Limit', 3050939105) : amount of total memory on HPU device
# ('InUse', 20073088) : amount of allocated memory at any instance. ( starting point after reset_peak_memroy_stats() )
# ('MaxInUse', 20073088) : amount of total active memory allocated
# ('NumAllocs', 0) : number of allocations
# ('NumFrees', 0) : number of freed chunks
# ('ActiveAllocs', 0) : number of active allocations
# ('MaxAllocSize', 0) : maximum allocated size
# ('TotalSystemAllocs', 34) : total number of system allocations
# ('TotalSystemFrees', 2) : total number of system frees
# ('TotalActiveAllocs', 32)] : total number of active allocations

# TORCH.HPU.MEMORY_SUMMARY
# Returns human readable printout of current memory stats.

# TORCH.HPU.RESET_PEAK_MEMORY_STATS
# Resets starting point of memory occupied by tensors.

# TORCH.HPU.MEMORY_RESERVED
# Returns the current HPU memory managed by caching allocator in bytes for a given device.

# TORCH.HPU.MAX_MEMORY_RESERVED
# Returns the maximum HPU memory managed by caching allocator in bytes for a given device.

# TORCH.HPU.MEMORY_CACHED
# Deprecated same as memory_reserved.

# TORCH.HPU.MAX_MEMORY_CACHED
# Deprecated same as max_memory_reserved.

# TORCH.HPU.MEM_GET_INFO
# Returns the free and total memory occupied by a HPU device.

import habana_frameworks.torch.hpu as htcore
import torch
from test_utils import hpu


def test_mem_stat():
    input1 = torch.randn((64, 28, 28, 20), dtype=torch.float, requires_grad=True)
    input1_hpu = input1.contiguous(memory_format=torch.channels_last).to(hpu)  # noqa
    mem_summary1 = htcore.memory_summary()
    print("memory_summary1:")
    print(mem_summary1)
    htcore.reset_peak_memory_stats()
    input2 = torch.randn((64, 28, 28, 20), dtype=torch.float, requires_grad=True)
    input2_hpu = input2.contiguous(memory_format=torch.channels_last).to(hpu)  # noqa
    mem_summary2 = htcore.memory_summary()
    print("memory_summary2:")
    print(mem_summary2)
    mem_allocated = htcore.memory_allocated()
    print("memory_allocated: ", mem_allocated)
    mem_stats = htcore.memory_stats()
    print("memory_stats:")
    print(mem_stats)
    max_mem_allocated = htcore.max_memory_allocated()
    print("max_memory_allocated: ", max_mem_allocated)
    print("memory_reserved:", htcore.memory_reserved())
    print("max_memory_reserved:", htcore.max_memory_reserved())
    print("memory_cached:", htcore.memory_cached())
    print("max_memory_cached:", htcore.max_memory_cached())
    print("mem_get_info (free_mem, total_mem):", htcore.mem_get_info())
    assert htcore.memory_reserved() == htcore.memory_cached()
    assert htcore.memory_reserved() == htcore.max_memory_reserved()
    assert htcore.memory_cached() == htcore.max_memory_cached()
