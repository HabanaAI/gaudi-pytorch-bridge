/*******************************************************************************
 * Copyright (C) 2020-2023 Habana Labs, Ltd. an Intel Company
 * All Rights Reserved.
 *
 * Unauthorized copying of this file or any element(s) within it, via any medium
 * is strictly prohibited.
 * This file contains Habana Labs, Ltd. proprietary and confidential information
 * and is subject to the confidentiality and license agreements under which it
 * was provided.
 *
 *******************************************************************************
 */

#pragma once

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
  MEM_LOG_MEMORY_STATS, /*logs memory allocation/free and the mem status*/
  MEM_LOG_RECORD, /* logs memory allocations and deallocation */
  MEM_REPORTER, /* memory reporter */
};

enum mem_reporter_type {
  MEM_REPORTER_GRAPH_BEFORE_LAUNCH =
      0, /* report event when before graph launch */
  MEM_REPORTER_GRAPH_AFTER_LAUNCH, /* report event when after graph launch */
  MEM_REPORTER_ALLOC_FAILS, /* report event when alloc fails */
  MEM_REPORTER_OOM, /* report event when oom */
  MEM_REPORTER_USER_CALL, /* report event when user request */
  MEM_DEFRAGMENT_START, /* repoort event when defragmentation start */
  MEM_DEFRAGMENT_SUCCESS, /* repoort event when defragmentation success */
  MEM_DEFRAGMENT_FAIL, /* repoort event when defragmentation fail */
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
  std::string memory_reporter_name;
  const char* fragment_csv_file = "habana_log.fragment.csv";
  bool take_bt = false;
  bool print_free_bt = false;
  bool print_alloc_bt = false;
  bool mem_statuscheck_running = false;
  bool enable_recording = false;
  bool print_memory_stats = false;
  size_t bt_depth;
  bool logging_enabled_;
  bool mem_reporter_enable_{false};
  bool fragment_json_enabled_{false};

  uint64_t dram_start_, dram_size_;
  std::ofstream out;
  std::mutex out_mut;

  void print_to_file(const char* msg);

  class Lockedfstream {
   public:
    Lockedfstream(std::ofstream& fs, std::mutex& mut) : fs_(fs), gl_(mut){};
    ~Lockedfstream() {
      fs_.flush();
    };

    Lockedfstream(const Lockedfstream&) = delete;
    Lockedfstream& operator=(const Lockedfstream&) = delete;

    template <typename T>
    Lockedfstream& operator<<(const T& data) {
      fs_ << data;
      return *this;
    }

   private:
    std::ofstream& fs_;
    std::lock_guard<std::mutex> gl_;
  };
  Lockedfstream get_out_stream() {
    return Lockedfstream(out, out_mut);
  }
  Lockedfstream get_memory_reporter_out_stream() {
    return Lockedfstream(memory_reporter_out, memory_reporter_out_mut);
  }
  Lockedfstream get_memory_json_out_stream() {
    return Lockedfstream(memory_json_out, memory_json_out_mut);
  }

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
      Lockedfstream& out_stream,
      const std::pair<uint64_t, size_bt_pair_t>& entry,
      bool print_all_frames = false);
  void collect_backtrace(
      uint64_t ptr,
      bool alloc,
      size_t size = 0,
      bool alloc_failure = false);

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

  bool get_memstats_check_flag() const {
    return mem_statuscheck_running;
  }

  bool is_recording_enabled() const {
    return enable_recording;
  }

  bool is_mem_stats_log_enabled() const {
    return print_memory_stats;
  }

  bool is_mem_reporter_enabled() const {
    return mem_reporter_enable_;
  }

  void create_memory_reporter_event(
      synapse_helpers::device& device,
      std::string& event_name);

  bool is_fragment_json_enabled() const {
    return fragment_json_enabled_;
  }

  template <typename T>
  static void print(std::stringstream& ss, const T& arg) {
    ss << " " << std::hex << arg;
  }

  template <typename T>
  static void print(std::stringstream& ss, const absl::Span<T>& arg) {
    for (auto a : arg) {
      print(ss, a);
    }
  }

  template <typename... ArgT>
  void record(const char* operation, const ArgT&... args) {
    std::stringstream ss;
    ss << "TRACE " << operation;
    (print(ss, args), ...);
    print_to_file(ss.str().c_str());
  }

  void record_graph_tensor_info(
      const std::string& name,
      const bool is_graph_input,
      const bool is_graph_output,
      const uint64_t index,
      uint64_t size);

  void update_graph_tensor_info(const uint64_t index, uint64_t start);

  void record_tensor_info(
      const std::string& name,
      const bool is_param,
      const bool is_grad,
      const bool is_optim_state,
      const bool is_graph_input,
      const bool is_graph_output,
      const uint64_t t_start,
      const uint64_t t_end);

  void update_workspace_record(uint64_t start, uint64_t end) {
    workspace_start = start;
    workspace_end = end;
  }

  void create_fragment_json_entry(
      synapse_helpers::device& device,
      std::string& graph_name);

 private:
  std::ofstream memory_reporter_out;
  std::mutex memory_reporter_out_mut;

  deviceMallocData();
  std::ofstream memory_json_out;
  std::mutex memory_json_out_mut;
  uint64_t workspace_start{0}, workspace_end{0};
  std::vector<std::tuple<uint64_t, uint64_t, std::string>> params;
  std::vector<std::tuple<uint64_t, uint64_t, std::string>> grads;
  std::vector<std::tuple<uint64_t, uint64_t, std::string>> optim_states;
  std::vector<std::tuple<uint64_t, uint64_t, std::string>> graph_input;
  std::vector<std::tuple<uint64_t, uint64_t, std::string>> graph_output;
  std::unordered_map<uint64_t, size_t> graph_input_indices{},
      graph_output_indices{};
};

void log_synDeviceMalloc(uint64_t ptr, size_t size, bool failed = false);
void log_synDeviceFree(uint64_t ptr, bool failed = false);
void log_synDevicePoolCreate(
    uint64_t free_mem,
    uint64_t mem_acquire_perc,
    uint64_t base_mem_ptr);
void log_synDeviceWorkspace(uint64_t ptr, size_t size);
void log_synDeviceAlloc(uint64_t ptr, size_t size);
void log_synDeviceDeallocate(uint64_t ptr);
void log_synDeviceLockMemory(
    absl::Span<const synapse_helpers::device_ptr> ptrs);
void log_synDeviceAllocFail(
    synapse_helpers::device& device,
    bool is_workspace,
    size_t size);
void log_synDeviceMemStats(device_memory& dev_mem);

void log_graph_info(
    synapse_helpers::device& device,
    std::string graph_name,
    size_t size,
    size_t wsize,
    size_t actual_size);

void log_synDeviceRecordGraphTensorInfo(
    const std::string& name,
    const bool is_graph_input,
    const bool is_graph_output,
    const uint64_t index,
    uint64_t size);
void log_synDeviceUpdateGraphTensorInfo(const uint64_t index, uint64_t start);
void log_synDeviceRecordTensorInfo(
    const std::string& name,
    const bool is_param,
    const bool is_grad,
    const bool is_optim_state,
    const bool is_graph_input,
    const bool is_graph_output,
    uint64_t start,
    uint64_t end);
void log_tensor_info(
    std::string tensor_name,
    uint64_t index,
    uint64_t v_addr,
    uint64_t d_addr);
void print_live_allocations(const char* msg = "");
void log_DRAM_start(uint64_t dram_start);
void log_DRAM_size(uint64_t dram_size);
void set_back_trace(bool enable);
void set_memstats_check_flag(bool flag);
void memstats_dump(synapse_helpers::device& device, const char* msg);
bool memory_reporter_enable();
void memory_reporter_event_create(
    synapse_helpers::device& device,
    synapse_helpers::mem_reporter_type event_type);
} // namespace synapse_helpers
