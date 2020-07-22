/******************************************************************************
 * Copyright (C) 2020 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */
#include <iostream>
#include <fstream>
#include <unordered_map>
#include <string>
#include <vector>
#include <algorithm>
#include <mutex>

class deviceMallocData final {
 private:
    using size_bt_pair_t = std::pair<size_t, std::vector<std::string>>;
    using ptr_bt_map_type_t = std::unordered_map< uint64_t, size_bt_pair_t >;
    ptr_bt_map_type_t ptr_bt_map;
    ptr_bt_map_type_t ptr_bt_map_last;

    size_t running_memory, iteration_high_watermark, overall_high_watermark;
    unsigned int iteration_number;

    const char *filename = "habana_log.livealloc.log";
    const char *fragment_csv_file = "habana_log.fragment.csv";
    bool take_bt, print_bt, print_free_bt, print_alloc_bt;
    size_t bt_depth;

    uint64_t dram_start_, dram_size_;
    std::ofstream out;

 public:
    static deviceMallocData& singleton();

    deviceMallocData(const deviceMallocData &) = delete;
    const deviceMallocData operator=(const deviceMallocData &) = delete;
    ~deviceMallocData();

    static bool sort_by_size(std::pair<uint64_t, size_bt_pair_t> a,
                             std::pair<uint64_t, size_bt_pair_t> b);
    static bool sort_by_ptr(std::pair<uint64_t, size_bt_pair_t> a,
                            std::pair<uint64_t, size_bt_pair_t> b);
    bool interesting_function(const std::string& name);
    void print_an_entry(const std::pair<uint64_t, size_bt_pair_t>& entry,
                        bool print_all_frames=false);
    void collect_backtrace(uint64_t ptr, bool alloc, size_t size=0, bool alloc_failure=false);
    void print_live_allocations(const char* msg = "");
    void report_fragmentation(bool from_free=false);
    void set_dram_start(uint64_t dram_start) {dram_start_ = dram_start;}
    void set_dram_size(uint64_t dram_size)   {dram_size_ = dram_size;}

    // TBD:: Make it private
    std::mutex m;
 private:
    deviceMallocData();
};

void log_synDeviceMalloc(uint64_t ptr, size_t size, bool failed=false);
void log_synDeviceFree(uint64_t ptr, bool failed=false);
void print_live_allocations(const char* msg = "");
void log_DRAM_start(uint64_t dram_start);
void log_DRAM_size(uint64_t dram_size);
