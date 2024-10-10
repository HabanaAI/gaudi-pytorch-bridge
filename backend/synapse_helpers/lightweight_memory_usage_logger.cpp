/*******************************************************************************
 * Copyright (C) 2022-2023 Habana Labs, Ltd. an Intel Company
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

#ifndef DISABLE_MEMORY_MONITORING
#include "lightweight_memory_usage_logger.h"
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include "backend/synapse_helpers/env_flags.h"
#include "habana_helpers/logging.h"
#include "pool_allocator/CoalescedStringentPoolAllocator.h"

using namespace std;
namespace {

const string output_file_name = string("memory_usage_");
const string output_file_ext = string(".csv");
const char* delimiter = ",";
constexpr chrono::duration<int64_t> write_period = std::chrono::seconds(1);

class MemMonitorBase {
 public:
  virtual ~MemMonitorBase() = default;
};

std::string apply_rank_prefix(const std::string& original_path) {
  const char* prefix = std::getenv("RANK");
  if (prefix != nullptr) {
    return std::string(prefix) + original_path;
  }
  return original_path;
}

template <typename... DataSource>
class MemMonitor : public MemMonitorBase {
 public:
  MemMonitor(DataSource&&... data_sources)
      : data_sources_(std::forward<DataSource>(data_sources)...) {
    ofs_.open(apply_rank_prefix(output_file_name_), ofstream::out);
    if (!ofs_.is_open()) {
      PT_BRIDGE_WARN(
          "Memory monitoring: Can't open file ",
          apply_rank_prefix(output_file_name_));
      return;
    }
    thread_ = thread(&MemMonitor::thread_writer, this);
  }

  ~MemMonitor() override {
    if (!thread_.joinable())
      return;
    stop_thread_ = true;
    cv_.notify_one();
    thread_.join();
  }

  void thread_writer();
  static std::string GetTimeStamp() {
    std::time_t current_time = std::time(nullptr);
    std::tm current_time_tm{};
    localtime_r(&current_time, &current_time_tm);
    std::array<char, 25> strbuf;
    strftime(
        strbuf.data(), strbuf.size(), "%Y-%m-%dT%H_%M_%S", &current_time_tm);
    return string(strbuf.data());
  }

  template <typename... Tp>
  void print_header(std::tuple<Tp...>& t) {
    std::apply([this](auto&... args) { (args.GetHeader(ofs_), ...); }, t);
    ofs_ << "\n";
  }

  template <typename... Tp>
  void print_data(std::tuple<Tp...>& t) {
    std::apply([this](auto&... args) { (args.GetData(ofs_), ...); }, t);
    ofs_ << "\n";
  }

 private:
  const string output_file_name_ =
      output_file_name + GetTimeStamp() + output_file_ext;
  static const std::chrono::duration<int64_t> write_period_;
  static const uint64_t flush_interval_ = 60;
  atomic<bool> stop_thread_{false};
  ofstream ofs_;
  std::tuple<DataSource...> data_sources_;
  std::mutex mtx_;
  std::condition_variable cv_;
  thread thread_{};
};

template <typename... DataSource>
const std::chrono::duration<int64_t> MemMonitor<DataSource...>::write_period_ =
    write_period;
template <typename... DataSource>
void MemMonitor<DataSource...>::thread_writer() {
  uint64_t next_flush = flush_interval_;
  print_header(data_sources_);
  std::unique_lock<std::mutex> lock(mtx_);
  while (!stop_thread_) {
    cv_.wait_for(lock, write_period_, [this] { return stop_thread_.load(); });
    if (stop_thread_)
      return;
    print_data(data_sources_);
    if (--next_flush == 0) {
      ofs_.flush();
      next_flush = flush_interval_;
    }
  }
}

class DeviceStatistics {
 public:
  void GetHeader(std::ostream& os) const {
    os << "HPU memory in use (KB)" << delimiter << "Peak HPU memory in use (KB)"
       << delimiter;
  }
  void GetData(std::ostream& os) const {
    std::lock_guard<std::mutex> lock(mtx_);
    if (hpu_allocator_ == nullptr) {
      os << delimiter;
    } else {
      synapse_helpers::MemoryStats hpu_allocator_stats;
      hpu_allocator_->get_stats(&hpu_allocator_stats);
      os << hpu_allocator_stats.bytes_in_use / 1000 << delimiter
         << hpu_allocator_stats.peak_bytes_in_use / 1000 << delimiter;
    }
  }
  void setDevice(
      const synapse_helpers::pool_allocator::CoalescedStringentPooling*
          hpu_allocator) {
    std::lock_guard<std::mutex> lock(mtx_);
    hpu_allocator_ = hpu_allocator;
  }

 private:
  mutable std::mutex mtx_;
  const synapse_helpers::pool_allocator::CoalescedStringentPooling*
      hpu_allocator_ = nullptr;
};

class RamStatistics {
 public:
  RamStatistics() : ifs("/proc/self/status"){};
  static uint64_t extract_num(const string& str) {
    auto it_start =
        find_if(str.begin(), str.end(), [](char a) { return isdigit(a); });
    auto it_end =
        find_if_not(it_start, str.end(), [](char a) { return isdigit(a); });
    auto num = string(it_start, it_end);
    return std::atol(num.c_str());
  }
  void GetHeader(std::ostream& os) const {
    for (auto& el : record_list_name_)
      os << el << delimiter;
  }
  void GetData(std::ostream& os) {
    ifs.clear();
    ifs.seekg(0);
    fill(data_.begin(), data_.end(), 0);
    string mem_data;
    while (getline(ifs, mem_data)) {
      for (size_t i = 0; i < record_list_.size(); ++i) {
        if (mem_data.rfind(record_list_[i]) != std::string::npos)
          data_[i] = extract_num(mem_data);
      }
    }
    for (auto el : data_)
      os << el << delimiter;
  }

 private:
  ifstream ifs;
  static constexpr uint64_t record_nums_ = 4;
  static const array<string, record_nums_> record_list_;
  static const array<string, record_nums_> record_list_name_;
  array<uint64_t, record_nums_> data_;
};
const array<string, RamStatistics::record_nums_> RamStatistics::record_list_ = {
    "VmRSS",
    "VmHWM",
    "VmSize",
    "VmSwap"};
const array<string, RamStatistics::record_nums_>
    RamStatistics::record_list_name_ = {
        "CPU Resident memory - VmRSS (KB)",
        "CPU Peak resident memory - VmHWM (KB)",
        "CPU Virtual memory - VmSize (KB)",
        "CPU Swapped memory - VmSwap (KB)"};

class TimeSource {
 public:
  void GetHeader(std::ostream& os) const {
    os << "TimeStamp (ms)" << delimiter;
  }
  void GetData(std::ostream& os) const {
    auto duration = chrono::high_resolution_clock::now() - init_timestamp_;
    auto time = chrono::duration_cast<chrono::milliseconds>(duration).count();
    os << time << delimiter;
  }

 private:
  chrono::time_point<std::chrono::high_resolution_clock> init_timestamp_ =
      chrono::high_resolution_clock::now();
};

class InitMemMonitor {
 public:
  InitMemMonitor() {
    if (GET_ENV_FLAG_NEW(PT_ENABLE_LIGHTWEIGHT_MEMORY_USAGE_LOGGING)) {
      devstats_ = std::make_unique<DeviceStatistics>();
      mem_monitor_ =
          CreateMemMonitor(TimeSource{}, RamStatistics{}, *devstats_);
    }
  }
  template <typename... DataSource>
  static unique_ptr<MemMonitor<DataSource...>> CreateMemMonitor(
      DataSource&&... data_sources) {
    return std::make_unique<MemMonitor<DataSource...>>(
        std::forward<DataSource>(data_sources)...);
  }
  void setDevice(
      const synapse_helpers::pool_allocator::CoalescedStringentPooling*
          allocator_ptr) {
    if (devstats_)
      devstats_->setDevice(allocator_ptr);
  }

 private:
  unique_ptr<DeviceStatistics> devstats_{};
  unique_ptr<MemMonitorBase> mem_monitor_{};
};

InitMemMonitor& get_mem_monitor() {
  static InitMemMonitor mem_monitor;
  return mem_monitor;
}
} // namespace

namespace synapse_helpers::LightweightMemoryMonitor {
void setDevice(const synapse_helpers::pool_allocator::CoalescedStringentPooling*
                   allocator_ptr) {
  if (GET_ENV_FLAG_NEW(PT_ENABLE_LIGHTWEIGHT_MEMORY_USAGE_LOGGING))
    get_mem_monitor().setDevice(allocator_ptr);
}
void resetDevice() {
  if (GET_ENV_FLAG_NEW(PT_ENABLE_LIGHTWEIGHT_MEMORY_USAGE_LOGGING))
    get_mem_monitor().setDevice(nullptr);
}
} // namespace synapse_helpers::LightweightMemoryMonitor

#endif
