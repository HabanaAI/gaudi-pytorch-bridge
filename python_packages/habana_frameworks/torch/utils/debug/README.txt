#How to check device memory statistics
###This APIs can be used to collect the memory allocation statistics.

#####1. _memstat_devmem_start_collect(msg, show_leaked_callstacks)
  This API will start collecting the statistscs and dump the values of statistics counters till this point. If show_leaked_callstacks is enabled, it will also dump the callstacks which are alive between start
and stop statistics collection process.

#####2. _memstat_devmem_stop_collect(msg)
  This API will dump the collected statistcs till this point and reset some of the counters. When next time this api is called, it will dump the difference in the statistics between this call and next call. '`msg`' can include the iteration counter or epoch counter number so that we can match the statistics values from file against the iterations/epochs.

The default file location is current folder and file name is '`habana_log.livealloc.log_0`' for rank 0 processes. In distributed environment, each rank will have a seperate file.
To change the filename you can set the environment variable `PT_HABANA_MEM_LOG_FILENAME`

The below is the sample of statistics data which are dumped in the file.

> Memory statistics collection started!!
Pool ID:                              0
Limit:                       3050939105 (2.84 GB)
InUse:                         50097280 (47.78 MB)
MaxInUse:                      50097280 (47.78 MB)
NumAllocs:                            1
NumFrees:                             0
ActiveAllocs:                         1
ScratchMem:                           0 (0.00 MB)
MaxAllocSize:                  48000000 (45.78 MB)
TotalSystemAllocs:                    2
TotalSystemFrees:                     0
TotActiveAllocs:                      2
Fragmentation:                        0
FragmentationMask:


Demo usage is available at `pytorch-integration/tests/test_memstats_checker.py`

#How to get overall device memory statistics reporter
###This APIs can be used to dump the memory statistics reporter.

##### _dump_memory_reporter()
  This APIs create memory reporter event and user request event captured under memory.reporter.json file when Environment variable PT_HABANA_MEM_LOG_LEVEL=7.

Demo usage is available at `pytorch-integration/tests/test_memreporter_checker.py`

we can also Memory logging level to get additional memory information. Below are the value that can be used:
Use Environment variable: PT_HABANA_MEM_LOG_LEVEL to get different level memory loggging.
  0 - Disable memory logging
  1 - logs full summary including backtrace
  2 - logs only allocation with backtrace
  3 - logs only free with backtrace
  4 - logs allocation and free, without backtrace
  5 - logs memory allocations and deallocation of workspace, tensors(virtual allocation/free, actaul device memory allocation/free and graph name with total memory allocated)
  6 - logs memory stats after every allocation/free of memory and record the memory allocation/dealloaction(ptr, size)
  7 - enable memory reporter to capture memory consumption, allocator stats, fragmentation stats, graph and tensors information under memory.reporter.json file at graph before launch, graph after launch, when oom appear, memory allocation fail event instance.

To get the Fragementation inforamtion with the Memory stats, Use Environment variable PT_HPU_POOL_LOG_FRAGMENTATION_INFO=1
