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
#include "device.h"

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

  std::string filename;
  const char* fragment_csv_file = "habana_log.fragment.csv";
  bool take_bt, print_free_bt, print_alloc_bt, mem_statuscheck_running;
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
  std::string get_formatted_func_name(
      std::string string,
      bool print_all_frames,
      bool* dot_marker_placed);
  void print_an_entry(
      const std::pair<uint64_t, size_bt_pair_t>& entry,
      bool print_all_frames = false);
  void collect_backtrace(
      uint64_t ptr,
      bool alloc,
      size_t size = 0,
      bool alloc_failure = false);
  void print_to_file(const char* msg);
  void print_live_allocations(const char* msg = "");
  void dump_collected_data(const char* msg = "");
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

  void set_back_trace(bool enable) {
    take_bt = enable;
    logging_enabled_ = (take_bt || print_free_bt || print_alloc_bt);
  }

  void set_memstats_check_flag(bool flag) {
    mem_statuscheck_running = flag;
  }

  bool get_memstats_check_flag() {
    return mem_statuscheck_running;
  }

  // TBD:: Make it private
  std::mutex m;

 private:
  deviceMallocData();
};

void log_synDeviceMalloc(uint64_t ptr, size_t size, bool failed = false);
void log_synDeviceFree(uint64_t ptr, bool failed = false);
void print_to_file(const char* msg);
void print_live_allocations(const char* msg = "");
void log_DRAM_start(uint64_t dram_start);
void log_DRAM_size(uint64_t dram_size);
void set_back_trace(bool enable);
void set_memstats_check_flag(bool flag);
void memstats_dump(synapse_helpers::device& device, const char* msg);
} // namespace synapse_helpers
