/**
* Copyright (c) 2021-2024 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*/
#include "synapse_logger.h"

#include <dlfcn.h>
#include <link.h>
#include <sys/time.h>
#include <syscall.h>
#include <unistd.h>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <mutex>
#include <string>
#include <type_traits>

#include <absl/strings/str_format.h>
#include <absl/strings/string_view.h>

#include "object_dump.h"
#include "synapse_api.h"

uint64_t NowNanos() {
  return static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::system_clock::now().time_since_epoch())
          .count());
}

uint64_t NowMicros() {
  return static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::microseconds>(
          std::chrono::system_clock::now().time_since_epoch())
          .count());
}

namespace lib_synapse {
void LoadSymbols(void* lib_handle);
}
#ifndef BINARY_NAME
#define BINARY_NAME "pytorch_synapse_logger.so"
#endif
const char* base_file_name = ".local.synapse_log";

namespace {
void checked_dlclose(void* lib_handle) {
  if (lib_handle)
    dlclose(lib_handle);
}
} // namespace

namespace synapse_logger {

std::atomic_int enabled_slog_level(S_ERROR);

int get_slog_level_config() {
  return enabled_slog_level.load();
}

std::unique_ptr<void, void (&)(void*)> dlopen_or_die(
    const char* name,
    int flag) {
  std::unique_ptr<void, void (&)(void*)> handle(
      dlopen(name, flag), checked_dlclose);
  CHECK_NULL(handle.get());
  return handle;
}

SynapseLogger::SynapseLogger()
    : log_start_time_{},
      log_file_name_(
          absl::StrFormat("%s.json", absl::string_view(base_file_name))),
      data_file_name_(
          absl::StrFormat("%s.data", absl::string_view(base_file_name))),
      logger_lib_handle_(dlopen_or_die(
          "${ORIGIN}/" BINARY_NAME,
          RTLD_GLOBAL | RTLD_NOLOAD | RTLD_NOW)),
      synapse_lib_handle_(
          dlopen_or_die("libSynapse.so", RTLD_GLOBAL | RTLD_NOW)),
      dev_attr_recorded(false) {
  SLOG(S_TRACE) << __FUNCTION__ << "\n";
  static_cast<void>(dev_attr_recorded);
  std::signal(SIGUSR1, SynapseLogger::command_signal_handler);
  lib_synapse::LoadSymbols(synapse_lib_handle_.get());
  link_map* l_map = nullptr;
  CHECK_TRUE(dlinfo(synapse_lib_handle_.get(), RTLD_DI_LINKMAP, &l_map) == 0);
  synapse_lib_path_ = l_map->l_name;

  const char* c_commands = std::getenv("HBN_SYNAPSE_LOGGER_COMMANDS");

  if (c_commands != nullptr) {
    absl::string_view sv{c_commands};
    absl::string_view separator{":"};

    size_t e = sv.find(separator);
    while (e != absl::string_view::npos) {
      command(sv.substr(0, e));
      sv = sv.substr(e + separator.size());
      e = sv.find(separator);
    }
    command(sv);
  }
  if (!lazy_open_) {
    command("restart");
  }
  if (c_commands == nullptr) {
    command("disable");
  }
}

void SynapseLogger::command_signal_handler(int) {
  const char* command_file_name = "synapse_logger_command";
  std::ifstream f(command_file_name);
  if (!f.good()) {
    SLOG(S_ERROR) << "Got command signal " << SIGUSR1 << " but command file "
                  << command_file_name << " cannot be read.\n";
    return;
  }
  std::string command_str(std::istreambuf_iterator<char>{f}, {});
  logger.command(command_str);

  /*std::stringstream command_stream;
  command_stream << f.rdbuf();
  command(command_stream.str());
  */
}

void SynapseLogger::lazy_open() {
  if (lazy_open_ && !fout_.is_open()) {
    SLOG(S_INFO) << "lazy open\n";
    restart();
  }
}

static std::once_flag lazy_init_flag{};

void SynapseLogger::dump_trace_info() {
  if (optimize_trace_) {
    std::call_once(lazy_init_flag, &SynapseLogger::lazy_open, logger);
    std::lock_guard<std::mutex> lock{log_lock_};

    for (long unsigned int i = 0; i < trace_info.payload.size(); i++) {
      fout_ << R"({"tid":)" << trace_info.tid[i] << R"(, "pid":)"
            << trace_info.pid << R"(, "ts":)" << trace_info.dtime[i] << ", "
            << trace_info.payload[i] << "},\n";
    }
  }
  fout_ << std::flush;
}

thread_local pid_t SynapseLogger::threadId = syscall(__NR_gettid);

void SynapseLogger::log(absl::string_view payload) {
  std::call_once(lazy_init_flag, &SynapseLogger::lazy_open, logger);
  std::lock_guard<std::mutex> lock(log_lock_);

  if (optimize_trace_) {
    trace_info.tid.push_back(SynapseLogger::threadId);
    trace_info.dtime.push_back(NowMicros());
    trace_info.payload.push_back(payload.data());
  } else {
    pid_t tid = syscall(__NR_gettid);
    pid_t pid = getpid();

    int64_t dtime = NowMicros();

    fout_ << R"({"tid":)" << tid << R"(, "pid":)" << pid << R"(, "ts":)"
          << dtime << ", " << payload << "},\n";
  }

  if (eager_flush_) {
    fout_ << std::flush;
  }
}

void SynapseLogger::on_log(
    std::string_view name,
    std::string_view args,
    bool begin) {
  std::lock_guard<std::mutex> lock(on_log_lock_);
  if (observer_) {
    pid_t tid = syscall(__NR_gettid);
    pid_t pid = getpid();
    int64_t dtime = NowNanos();
    observer_->on_log(name, args, pid, tid, dtime, begin);
  }
}

void SynapseLogger::dump_host_data(
    const void* ptr,
    int byte_size,
    data_dump_category data_category) {
  if (is_enabled(data_category)) {
    auto offset = dump_data(ptr, byte_size);
    ostr_t out{get_ostr()};
    out << R"("name":"object", "ph":"i", "args":{"type":"uint8_t*", "at":")"
        << ptr << "\"";
    out << ", \"data_offset\":" << offset << ", \"byte_size\":" << byte_size
        << "}";
    log(out.str());
  }
}

size_t SynapseLogger::dump_data(const void* ptr, int byte_size) {
  std::call_once(lazy_init_flag, &SynapseLogger::lazy_open, logger);
  size_t offset = data_fout_.tellp();
  std::lock_guard<std::mutex> lock(log_lock_);
  data_fout_.write((char*)ptr, byte_size);
  data_fout_.flush();
  return offset;
}

void SynapseLogger::disable() {
  source_cat_mask_ = 0;
  {
    std::lock_guard<std::mutex> lock(log_lock_);
    if (data_fout_) {
      data_fout_.close();
    }
    if (fout_.is_open()) {
      fout_.close();
    }
  }
}
void SynapseLogger::disable_mask() {
  source_cat_mask_ = 0;
}

void SynapseLogger::register_event_observer(SynapseLoggerObserver* observer) {
  std::lock_guard<std::mutex> lock(on_log_lock_);
  observer_ = observer;
}

void SynapseLogger::restart() {
  {
    std::lock_guard<std::mutex> tlock(transfer_lock_);
    transfers_.clear();
  }
  std::lock_guard<std::mutex> lock(log_lock_);
  if (0 == source_cat_mask_) {
    source_cat_mask_ =
        static_cast<uint64_t>(data_dump_category::SYNAPSE_API_CALL) |
        static_cast<uint64_t>(
            data_dump_category::CUSTOM_RUNTIME_TRACE_PROVIDER);
  }
  if (use_null_backend_) {
    // Disable api call and tensor data logging for null backend.
    source_cat_mask_ &=
        ~(static_cast<uint64_t>(data_dump_category::SYNAPSE_API_CALL) |
          static_cast<uint64_t>(
              data_dump_category::CUSTOM_RUNTIME_TRACE_PROVIDER) |
          static_cast<uint64_t>(data_dump_category::VAR_TENSOR_DATA) |
          static_cast<uint64_t>(data_dump_category::CONST_TENSOR_DATA));
  }
  if (fout_.is_open()) {
    fout_.close();
  }
  if (data_fout_.is_open()) {
    data_fout_.close();
  }
  // log_start_time_ = std::chrono::high_resolution_clock::now();
  clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &log_start_time_);
  fout_.open(log_file_name_, std::ios::out);
  data_fout_.open(data_file_name_, std::ios::out | std::ios::binary);
  fout_ << "[\n" << std::fixed << std::setw(11) << std::setprecision(6);
}

void SynapseLogger::command(absl::string_view cmd) {
  absl::string_view separator{"="};
  absl::string_view cmd_name;
  absl::string_view cmd_params;

  size_t e = cmd.find(separator);
  if (e != absl::string_view::npos) {
    cmd_name = cmd.substr(0, e);
    cmd_params = cmd.substr(e + separator.size());
  } else {
    cmd_name = cmd;
  }
  if (cmd_name == "start_data_capture") {
    source_cat_mask_ |=
        (static_cast<uint64_t>(data_dump_category::VAR_TENSOR_DATA) |
         static_cast<uint64_t>(data_dump_category::CONST_TENSOR_DATA));
  } else if (cmd_name == "stop_data_capture") {
    source_cat_mask_ &=
        ~(static_cast<uint64_t>(data_dump_category::VAR_TENSOR_DATA) |
          static_cast<uint64_t>(data_dump_category::CONST_TENSOR_DATA));
    std::lock_guard<std::mutex> tlock(transfer_lock_);
    transfers_.clear();
  } else if (cmd_name == "stop_vtensor_capture") {
    source_cat_mask_ &=
        ~static_cast<uint64_t>(data_dump_category::VAR_TENSOR_DATA);
    std::lock_guard<std::mutex> tlock(transfer_lock_);
    transfers_.clear();
  } else if (cmd_name == "stop_ctensor_capture") {
    source_cat_mask_ &=
        ~static_cast<uint64_t>(data_dump_category::CONST_TENSOR_DATA);
    std::lock_guard<std::mutex> tlock(transfer_lock_);
    transfers_.clear();
  } else if (cmd_name == "start_vtensor_capture") {
    source_cat_mask_ |=
        static_cast<uint64_t>(data_dump_category::VAR_TENSOR_DATA);
    std::lock_guard<std::mutex> tlock(transfer_lock_);
    transfers_.clear();
  } else if (cmd_name == "start_ctensor_capture") {
    source_cat_mask_ |=
        static_cast<uint64_t>(data_dump_category::CONST_TENSOR_DATA);
    std::lock_guard<std::mutex> tlock(transfer_lock_);
    transfers_.clear();
  } else if (cmd_name == "eager_flush") {
    eager_flush_ = true;
  } else if (cmd_name == "no_eager_flush") {
    eager_flush_ = false;
  } else if (cmd_name == "use_pid_suffix") {
    // make sure you send this before 'restart'
    log_file_name_ = absl::StrFormat("%s.%d.json", base_file_name, getpid());
    data_file_name_ = absl::StrFormat("%s.%d.data", base_file_name, getpid());
  } else if (cmd_name == "optimize_trace") {
    optimize_trace_ = true;
    char* reserve_count_str = std::getenv("TRACE_RESERVE_COUNT");
    trace_info.trace_reserve_count = reserve_count_str
        ? atoi(reserve_count_str)
        : trace_info.trace_reserve_count;
  } else if (cmd_name == "file_name") {
    std::lock_guard<std::mutex> lock(log_lock_);
    std::ostringstream log_file_name_ss, data_file_name_ss;
    log_file_name_ = std::string(cmd_params);
    log_file_name_.append(".json");
    data_file_name_ = std::string(cmd_params);
    data_file_name_.append(".data");
    SLOG(S_INFO) << "Output log file name set to " << log_file_name_
                 << std::endl;
    SLOG(S_INFO) << "Output data file name set to " << data_file_name_
                 << std::endl;
  } else if (cmd_name == "restart" || cmd_name == "enable") {
    restart();
  } else if (cmd_name == "category_mask") {
    unsigned mask =
        strtoll(static_cast<std::string>(cmd_params).c_str(), nullptr, 0);
    source_cat_mask_ = mask;
    if (mask == 0) {
      SLOG(S_INFO) << "Category mask for logger set to zero  (\"" << cmd_params
                   << "\" requested)";
    }
  } else if (cmd_name == "disable") {
    disable();
  } else if (cmd_name == "lazy_open") {
    disable();
    lazy_open_ = true;
  } else if (cmd_name == "use_null_backend") {
    // Disable api call and tensor data logging for null backend.
    use_null_backend_ = true;
    source_cat_mask_ &=
        ~(static_cast<uint64_t>(data_dump_category::SYNAPSE_API_CALL) |
          static_cast<uint64_t>(
              data_dump_category::CUSTOM_RUNTIME_TRACE_PROVIDER) |
          static_cast<uint64_t>(data_dump_category::VAR_TENSOR_DATA) |
          static_cast<uint64_t>(data_dump_category::CONST_TENSOR_DATA));
    std::lock_guard<std::mutex> tlock(transfer_lock_);
    transfers_.clear();
  } else if (cmd_name == "disable_mask") {
    disable_mask();
  } else if (cmd_name == "disable_log") {
    synapse_logger::enabled_slog_level.store(S_ERROR);

  } else if (cmd_name == "enable_log") {
    synapse_logger::enabled_slog_level.store(S_INFO);
  } else {
    SLOG(S_ERROR) << "Unknown command " << cmd_name << ".\n";
    return;
  }
  SLOG(S_INFO) << "Done command: " << cmd_name << "\n";
} // namespace synapse_logger

void SynapseLogger::dump_reference(
    const std::string& ref,
    const std::string& ref_type,
    float* vec,
    int n) {
  ostr_t out{get_ostr()};
  out << R"("name":"reference", "args":{"to":")" << ref << R"(", "length":)"
      << n;
  unsigned num_elements = n * sizeof(float);
  out << ", \"data_offset\":" << data_fout_.tellp()
      << ", \"byte_size\":" << num_elements << R"(, "data_cast":")" << ref_type
      << "\"}";
  log(out.str());
  std::lock_guard<std::mutex> lock(log_lock_);
  data_fout_.write((char*)vec, num_elements);
  data_fout_.flush();
}

SynapseLogger logger;

void start_hw_profile() {
  if (logger.last_acquired_id() != SynapseLogger::SYN_DEVICE_ID_UNASSIGNED) {
    synStatus status{synDeviceSynchronize(logger.last_acquired_id())};
    if (status != synSuccess) {
      SLOG(S_ERROR) << "synDeviceSynchronize failed: " << status;
    }
    status = synProfilerStart(synTraceDevice, logger.last_acquired_id());
    if (status != synSuccess) {
      SLOG(S_ERROR) << "synProfilerStart failed: " << status;
    }
  }
}

void stop_hw_profile() {
  if (logger.last_acquired_id() != SynapseLogger::SYN_DEVICE_ID_UNASSIGNED) {
    synStatus status{synDeviceSynchronize(logger.last_acquired_id())};
    if (status != synSuccess) {
      SLOG(S_ERROR) << "synDeviceSynchronize failed: " << status;
    }
    status = synProfilerStop(synTraceDevice, logger.last_acquired_id());
    if (status != synSuccess) {
      SLOG(S_ERROR) << "synProfilerStop failed: " << status;
    }
    synProfilerGetTrace(
        synTraceDevice,
        logger.last_acquired_id(),
        synTraceFormatTEF,
        nullptr,
        nullptr,
        nullptr);
    if (status != synSuccess) {
      SLOG(S_ERROR) << "synProfilerGetTrace failed: " << status;
    }
  }
}

void put_log(const std::string& what) {
  logger.log(what);
}

void dump_reference(
    const std::string& ref,
    const std::string& ref_type,
    float* vec,
    int n) {
  SLOG(S_TRACE) << __PRETTY_FUNCTION__ << " called for " << ref << " vec "
                << vec << " n " << n << "\n";
  logger.dump_reference(ref, ref_type, vec, n);
}
void command(const std::string& x) {
  logger.command(x);
}

bool logger_is_enabled(data_dump_category cat) {
  return logger.is_enabled(cat);
}

bool log_observer_is_enabled(std::string_view name) {
  return logger.is_event_logger_enabled(name);
}

void log(absl::string_view payload) {
  logger.log(payload);
}

void on_log(std::string_view name, std::string_view args, bool begin) {
  logger.on_log(name, args, begin);
}

std::string getSynapseLibPath() {
  return logger.getSynapseLibPath();
}

extern "C" void register_synapse_logger_oberver(
    synapse_logger::SynapseLoggerObserver* observer) {
  logger.disable();
  logger.register_event_observer(observer);
}
} // namespace synapse_logger
