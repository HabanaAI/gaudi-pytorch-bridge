/******************************************************************************
 * Copyright (C) 2020 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */
#include "synapse_logger.h"

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
#include <sys/time.h>
#include <syscall.h>
#include <type_traits>
#include <unistd.h>
#include <dlfcn.h>

#include "absl/strings/string_view.h"

#include "object_dump.h"
#include "synapse_api.h"

uint64_t NowMicros() {
  return static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch())
          .count());
}

namespace lib_synapse {
void LoadSymbols(void* lib_handle);
}
namespace lib_hcl {
void LoadSymbols(void* lib_handle_);
}
#ifndef BINARY_NAME
#define BINARY_NAME "synapse_logger.so"
#endif

namespace {
void checked_dlclose(void* lib_handle) {
  if (lib_handle) dlclose(lib_handle);
}
}  // namespace

namespace synapse_logger {

std::unique_ptr<void, void (&)(void*)> dlopen_or_die(const char* name, int flag) {
  std::unique_ptr<void, void (&)(void*)> handle(dlopen(name, flag), checked_dlclose);
  CHECK_NULL(handle.get());
  return handle;
}

SynapseLogger::SynapseLogger()
    : log_start_time_{},
      log_file_name_(".local.synapse_log.json"),
      data_file_name_(".local.synapse_log.data"),
      logger_lib_handle_(dlopen_or_die("${ORIGIN}/" BINARY_NAME, RTLD_GLOBAL | RTLD_NOLOAD | RTLD_NOW)),
      synapse_lib_handle_(dlopen_or_die("libSynapse.so", RTLD_GLOBAL | RTLD_NOW)) {
  SLOG(S_TRACE) << __FUNCTION__ << "\n";
  std::signal(SIGUSR1, SynapseLogger::command_signal_handler);
  lib_synapse::LoadSymbols(synapse_lib_handle_.get());
  lib_hcl::LoadSymbols(synapse_lib_handle_.get());
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
}

void SynapseLogger::command_signal_handler(int) {
  const char* command_file_name = "synapse_logger_command";
  std::ifstream f(command_file_name);
  if (!f.good()) {
    SLOG(S_ERROR) << "Got command signal " << SIGUSR1 << " but command file " << command_file_name
                  << " cannot be read.\n";
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

void SynapseLogger::log(absl::string_view payload) {
  std::call_once(lazy_init_flag, &SynapseLogger::lazy_open, logger);
  std::lock_guard<std::mutex> lock(log_lock_);
  pid_t tid = syscall(__NR_gettid);
  pid_t pid = getpid();

  int64_t dtime = NowMicros();

  fout_ << R"({"tid":)" << tid << R"(, "pid":)" << pid << R"(, "ts":)" << dtime << ", " << payload << "},\n";
  if (eager_flush_) {
    fout_ << std::flush;
  }
}

void SynapseLogger::dump_host_data(const void* ptr, int byte_size, data_dump_category data_category) {
  if (is_enabled(data_category)) {
    auto offset = dump_data(ptr, byte_size);
    ostr_t out{get_ostr()};
    out << R"("name":"object", "ph":"i", "args":{"type":"host_data", "at":")" << ptr << "\"";
    out << ", \"data_offset\":" << offset << ", \"byte_size\":" << byte_size << "}";
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
void SynapseLogger::restart() {
  {
    std::lock_guard<std::mutex> tlock(transfer_lock_);
    transfers_.clear();
  }
  std::lock_guard<std::mutex> lock(log_lock_);
  if (0 == source_cat_mask_) {
    source_cat_mask_ = static_cast<uint64_t>(data_dump_category::SYNAPSE_API_CALL);
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
    source_cat_mask_ |= (static_cast<uint64_t>(data_dump_category::VAR_TENSOR_DATA) |
                         static_cast<uint64_t>(data_dump_category::CONST_TENSOR_DATA));
  } else if (cmd_name == "stop_data_capture") {
    source_cat_mask_ &= ~(static_cast<uint64_t>(data_dump_category::VAR_TENSOR_DATA) |
                          static_cast<uint64_t>(data_dump_category::CONST_TENSOR_DATA));
    std::lock_guard<std::mutex> tlock(transfer_lock_);
    transfers_.clear();
  } else if (cmd_name == "stop_vtensor_capture") {
    source_cat_mask_ &= ~static_cast<uint64_t>(data_dump_category::VAR_TENSOR_DATA);
    std::lock_guard<std::mutex> tlock(transfer_lock_);
    transfers_.clear();
  } else if (cmd_name == "stop_ctensor_capture") {
    source_cat_mask_ &= ~static_cast<uint64_t>(data_dump_category::CONST_TENSOR_DATA);
    std::lock_guard<std::mutex> tlock(transfer_lock_);
    transfers_.clear();
  } else if (cmd_name == "start_vtensor_capture") {
    source_cat_mask_ |= static_cast<uint64_t>(data_dump_category::VAR_TENSOR_DATA);
    std::lock_guard<std::mutex> tlock(transfer_lock_);
    transfers_.clear();
  } else if (cmd_name == "start_ctensor_capture") {
    source_cat_mask_ |= static_cast<uint64_t>(data_dump_category::CONST_TENSOR_DATA);
    std::lock_guard<std::mutex> tlock(transfer_lock_);
    transfers_.clear();
  } else if (cmd_name == "eager_flush") {
    eager_flush_ = true;
  } else if (cmd_name == "no_eager_flush") {
    eager_flush_ = false;
  } else if (cmd_name == "file_name") {
    std::lock_guard<std::mutex> lock(log_lock_);
    std::ostringstream log_file_name_ss, data_file_name_ss;
    log_file_name_ = std::string(cmd_params);
    log_file_name_.append(".json");
    data_file_name_ = std::string(cmd_params);
    data_file_name_.append(".data");
    SLOG(S_INFO) << "Output log file name set to " << log_file_name_ << std::endl;
    SLOG(S_INFO) << "Output data file name set to " << data_file_name_ << std::endl;
  } else if (cmd_name == "restart" || cmd_name == "enable") {
    restart();
  } else if (cmd_name == "category_mask") {
    unsigned mask = strtoll(static_cast<std::string>(cmd_params).c_str(), nullptr, 0);
    source_cat_mask_ = mask;
    if (mask == 0) {
      SLOG(S_INFO) << "Category mask for logger set to zero  (\"" << cmd_params << "\" requested)";
    }
  } else if (cmd_name == "disable") {
    disable();
  } else if (cmd_name == "lazy_open") {
    disable();
    lazy_open_ = true;
  } else {
    SLOG(S_ERROR) << "Unknown command " << cmd_name << ".\n";
    return;
  }
  SLOG(S_INFO) << "Done command: " << cmd_name << "\n";
}  // namespace synapse_logger

void SynapseLogger::dump_reference(const std::string& ref, const std::string& ref_type, float* vec, int n) {
  ostr_t out{get_ostr()};
  out << R"("name":"reference", "args":{"to":")" << ref << R"(", "length":)" << n;
  unsigned num_elements = n * sizeof(float);
  out << ", \"data_offset\":" << data_fout_.tellp() << ", \"byte_size\":" << num_elements << R"(, "data_cast":")"
      << ref_type << "\"}";
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
      SLOG(S_ERROR) << "synDeviceSynchronize failed: " << status;
    }
    synProfilerGetTrace(synTraceDevice, logger.last_acquired_id(), synTraceFormatTEF, nullptr, nullptr);
    if (status != synSuccess) {
      SLOG(S_ERROR) << "synDeviceSynchronize failed: " << status;
    }
  }
}

void put_log(const std::string& what) { logger.log(what); }

void dump_reference(const std::string& ref, const std::string& ref_type, float* vec, int n) {
  SLOG(S_TRACE) << __PRETTY_FUNCTION__ << " called for " << ref << " vec " << vec << " n " << n << "\n";
  logger.dump_reference(ref, ref_type, vec, n);
}
void command(const std::string& x) { logger.command(x); }

bool logger_is_enabled(data_dump_category cat) { return logger.is_enabled(cat); }

void log(absl::string_view payload) { logger.log(payload); }
}  // namespace synapse_logger
