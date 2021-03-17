/******************************************************************************
 * Copyright (C) 2020 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */
#include <algorithm>
#include <fstream>
#include <iostream>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace synapse_helpers {
enum mem_log_level {
  MEM_LOG_DISABLE = 0,
  MEM_LOG_ALL, /* logs full summary including bt */
  MEM_LOG_ALLOC, /* logs only alloc */
  MEM_LOG_FREE, /* logs only free  */
  MEM_LOG_ALLOC_FREE_NOBT, /*logs alloc and free, no backtrace */
  MEM_LOG_GRAPH_LAUNCH, /*logs memory before graph launch, no backtrace */
};

class deviceMallocData final {
 private:
  using size_bt_pair_t = std::pair<size_t, std::vector<std::string>>;
  using ptr_bt_map_type_t = std::unordered_map<uint64_t, size_bt_pair_t>;
  ptr_bt_map_type_t ptr_bt_map;
  ptr_bt_map_type_t ptr_bt_map_last;
  ptr_bt_map_type_t duplicate_ptr_bt_map;

  size_t running_memory, iteration_high_watermark, overall_high_watermark;
  unsigned int iteration_number;

  const char* filename = "habana_log.livealloc.log";
  const char* fragment_csv_file = "habana_log.fragment.csv";
  bool take_bt, print_free_bt, print_alloc_bt;
  size_t bt_depth;
  bool logging_enabled_;

  uint64_t dram_start_, dram_size_;
  std::ofstream out;

 public:
  static deviceMallocData& singleton();

  deviceMallocData(const deviceMallocData&) = delete;
  const deviceMallocData operator=(const deviceMallocData&) = delete;
  ~deviceMallocData();

  static bool sort_by_size(
      std::pair<uint64_t, size_bt_pair_t>& a,
      std::pair<uint64_t, size_bt_pair_t>& b);
  static bool sort_by_ptr(
      std::pair<uint64_t, size_bt_pair_t>& a,
      std::pair<uint64_t, size_bt_pair_t>& b);
  bool interesting_function(const std::string& name);
  void print_an_entry(
      const std::pair<uint64_t, size_bt_pair_t>& entry,
      bool print_all_frames = false);
  void collect_backtrace(
      uint64_t ptr,
      bool alloc,
      size_t size = 0,
      bool alloc_failure = false);
  void print_live_allocations(const char* msg = "");
  void report_fragmentation(bool from_free = false);
  void set_dram_start(uint64_t dram_start) {
    dram_start_ = dram_start;
  }
  void set_dram_size(uint64_t dram_size) {
    dram_size_ = dram_size;
  }

  bool is_logging_enabled() {
    return logging_enabled_;
  }

  // TBD:: Make it private
  std::mutex m;

 private:
  deviceMallocData();
};

void log_synDeviceMalloc(uint64_t ptr, size_t size, bool failed = false);
void log_synDeviceFree(uint64_t ptr, bool failed = false);
void print_live_allocations(const char* msg = "");
void log_DRAM_start(uint64_t dram_start);
void log_DRAM_size(uint64_t dram_size);
} // namespace synapse_helpers
