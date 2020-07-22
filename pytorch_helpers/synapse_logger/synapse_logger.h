/******************************************************************************
 * Copyright (C) 2020 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */
#pragma once
#include <absl/strings/string_view.h>
#include <absl/types/variant.h>
#include <dlfcn.h>
#include <hcl_api_types.h>
#include <unistd.h>
#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <ctime>
#include <deque>
#include <exception>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>

#include "object_dump.h"
#include "synapse_api_types.h"
#include "synapse_common_types.h"

enum ErrorLevel { S_ERROR = 0, S_INFO = 1, S_TRACE = 2 };

namespace synapse_logger {

constexpr std::array<const char*, 3> slog_levels = {{"ERROR", "INFO", "TRACE"}};
const int enabled_slog_level = S_INFO;
constexpr const char* get_slog_level(ErrorLevel level) { return synapse_logger::slog_levels[level]; }

inline bool is_status_success(HCLStatus status) { return (eHCLSuccess == status); }

inline bool is_status_success(synStatus status) { return (synSuccess == status); }

#define SLOG(level)                                                                                        \
  if (synapse_logger::enabled_slog_level >= level)                                                         \
  (level <= S_ERROR ? std::cerr : std::clog) << "synapse_logger " << synapse_logger::get_slog_level(level) \
                                             << ". pid=" << getpid() << " at " << __FILE__ << ":" << __LINE__ << " "

#define CHECK_NULL(x)                              \
  do {                                             \
    if (nullptr == (x)) {                          \
      SLOG(S_ERROR) << " (" << dlerror() << ")\n"; \
      std::terminate();                            \
    }                                              \
  } while (0)

inline unsigned count_prod(const unsigned* dims, unsigned len) {
  unsigned result = 1;
  for (unsigned p = 0; p < len; ++p) {
    result *= dims[p];
  }
  return result;
}

inline uint32_t size_of_syn_data_type(synDataType dataType) {
  switch (dataType) {
    case syn_type_int8:   // alias to syn_type_fixed
    case syn_type_uint8:  // 8-bit unsigned integer
      return 1;
    case syn_type_bf16:   // 16-bit float- 8 bits exponent, 7 bits mantisa, 1 bit sign
    case syn_type_int16:  // 16-bit integer
      return 2;
    case syn_type_float:  // alias to syn_type_single
    case syn_type_int32:  // 32-bit integer
      return 4;
    default:
      return -1;  // invalid
  }
}

class SynapseLogger {
 public:
  static const uint32_t SYN_DEVICE_ID_UNASSIGNED{0xFFFFFFFF};

  SynapseLogger();
  ~SynapseLogger() {  // NOLINT
    SLOG(S_TRACE) << "###SYN_LOG_DESTROY\n";
    flush();
  }

  void flush() {
    std::lock_guard<std::mutex> lock{log_lock_};
    fout_ << std::flush;
  }
  void log(absl::string_view payload);

  void dump_host_data(const void* ptr, int byte_size,
                      data_dump_category data_category = data_dump_category::VAR_TENSOR_DATA);
  size_t dump_data(const void* ptr, int byte_size);
  void command(absl::string_view x);
  void restart();
  void disable();
  void lazy_open();

  void dump_reference(const std::string& ref, const std::string& ref_type, float* vec, int n);

  void dump_device_alloc_data(const uint64_t ptr, size_t num_bytes, const synDeviceId deviceId, synStatus status,
                              data_dump_category data_category = data_dump_category::DEVICE_ALLOC_TRACKING);
  void dump_device_free_data(const uint64_t ptr, synStatus status,
                             data_dump_category data_category = data_dump_category::DEVICE_ALLOC_TRACKING);
  void dump_device_attr(const synDeviceAttribute* deviceAttr, uint64_t* val, const unsigned querySize,
                        data_dump_category data_category = data_dump_category::DEVICE_ALLOC_TRACKING);

  struct recorded_event {
    synStreamHandle stream_handle;
    synEventHandle event_handle;
  };

  struct host_transfer {
    uint64_t source;
    void* destination;
    uint64_t size;
  };

  using host_xfer_or_event = absl::variant<host_transfer, recorded_event>;
  using stream_deque = std::deque<host_xfer_or_event>;

  void stream_synchronized(synStreamHandle streamHandle) {
    if (!is_enabled(data_dump_category::VAR_TENSOR_DATA)) {
      return;
    }
    stream_deque transfers;
    {
      std::lock_guard<std::mutex> lock(transfer_lock_);
      std::swap(transfers_[streamHandle], transfers);
    }
    for (auto& xfer_v : transfers) {
      if (absl::holds_alternative<host_transfer>(xfer_v)) {
        auto& xfer = absl::get<host_transfer>(xfer_v);
        dump_host_data(xfer.destination, xfer.size);
      }
    }
  }

  void event_recorded(synStreamHandle stream_handle, synEventHandle event_handle) {
    if (!is_enabled(data_dump_category::VAR_TENSOR_DATA)) {
      return;
    }
    std::lock_guard<std::mutex> lock{transfer_lock_};
    transfers_[stream_handle].emplace_back(recorded_event{stream_handle, event_handle});
  }

  void event_synchronized(synEventHandle event_handle) {
    if (!is_enabled(data_dump_category::VAR_TENSOR_DATA)) {
      return;
    }
    for (auto& xfer_queue_iter : transfers_) {
      stream_deque transfers;
      {
        std::lock_guard<std::mutex> lock(transfer_lock_);
        stream_deque& xfer_queue{xfer_queue_iter.second};
        // Find if event is present on stream
        auto event_pos =
            std::find_if(xfer_queue.begin(), xfer_queue.end(), [event_handle](const host_xfer_or_event& maybe_event) {
              if (absl::holds_alternative<recorded_event>(maybe_event)) {
                auto& event = absl::get<recorded_event>(maybe_event);
                return (event.event_handle == event_handle);
              }
              return false;
            });
        if (event_pos == xfer_queue.end()) {
          // Event not on this stream, moving on to next one
          continue;
        }
        // Remove everything before event from stream_deque, storing local copy
        std::move(xfer_queue.begin(), event_pos, std::back_inserter(transfers));
        xfer_queue.erase(xfer_queue.begin(), event_pos);
      }

      // Dump any transfer stored on local copy
      for (auto& maybe_xfer : transfers) {
        if (absl::holds_alternative<host_transfer>(maybe_xfer)) {
          auto& xfer = absl::get<host_transfer>(maybe_xfer);
          dump_host_data(xfer.destination, xfer.size);
        }
      }
    }
  }

  void store_transfer_to_host(const synStreamHandle streamHandle, const uint64_t src, const uint64_t size,
                              const uint64_t dst) {
    if (!is_enabled(data_dump_category::VAR_TENSOR_DATA)) {
      return;
    }
    std::lock_guard<std::mutex> lock(transfer_lock_);
    transfers_[streamHandle].emplace_back(
        host_transfer{.source = src, .destination = reinterpret_cast<void*>(dst), .size = size});
  }

  synDeviceId last_acquired_id() { return last_acquired_id_; }
  void last_acquired_id(synDeviceId id) { last_acquired_id_ = id; }

  void* last_dst{};
  unsigned last_size{};
  bool is_enabled(data_dump_category cat) { return (0 != (source_cat_mask_ & static_cast<uint64_t>(cat))); }

 private:
  // std::chrono::time_point<std::chrono::high_resolution_clock> log_start_time_;
  std::atomic_uint64_t source_cat_mask_{std::numeric_limits<uint64_t>::max() & ~(static_cast<uint64_t>(data_dump_category::DEVICE_ALLOC_TRACKING))};
  struct timespec log_start_time_;
  std::string log_file_name_;
  std::string data_file_name_;
  std::ofstream fout_;
  std::ofstream data_fout_;
  std::mutex log_lock_{};
  std::mutex transfer_lock_{};
  std::mutex ostr_lock_{};
  std::unique_ptr<void, void (&)(void*)> logger_lib_handle_;
  std::unique_ptr<void, void (&)(void*)> synapse_lib_handle_;
  std::unordered_map<synStreamHandle, stream_deque> transfers_;
  synDeviceId last_acquired_id_{SYN_DEVICE_ID_UNASSIGNED};
  std::atomic_bool eager_flush_{true};
  std::atomic_bool lazy_open_{false};
  static void command_signal_handler(int);
  bool dev_attr_recorded;
};

extern SynapseLogger logger;

inline std::ostream& operator<<(std::ostream& stream, const synQuantizationParams& qp) {
  stream << "{ \"m_zp\":" << qp.m_zp << ", \"m_scale\":" << qp.m_scale << ", \"m_qDataType\":" << qp.m_qDataType
         << "}";
  return stream;
}

inline void log_synTensorDescriptor(std::ostream& out, const synTensorDescriptor* obj) {
  out << R"("name":"object", "ph":"i", "args":{"at":")" << (void*)obj
      << R"(", "type":"synTensorDescriptor", "fields":{ "m_batchPos":)" << obj->m_batchPos
      << ", \"m_dataType\":" << obj->m_dataType << ", \"m_dims\":" << obj->m_dims << ", \"m_sizes\":[";
  for (unsigned i = 0; i < SYN_MAX_TENSOR_DIM - 1; ++i) {
    out << obj->m_sizes[i] << ", ";
  }
  out << obj->m_sizes[SYN_MAX_TENSOR_DIM - 1];
  out << "], \"m_quantizationParams\":[";
  for (unsigned i = 0; i < SYN_NUM_DATA_TYPES - 1; ++i) {
    out << obj->m_quantizationParams[i] << ", ";
  }
  out << obj->m_quantizationParams[SYN_NUM_DATA_TYPES - 1];

  out << R"(], "m_ptr":")" << obj->m_ptr << ", \"m_isWeights\":" << obj->m_isWeights
      << ", \"m_isQuantized\":" << obj->m_isQuantized << R"(, "m_name":")" << (obj->m_name ? obj->m_name : "nullptr")
      << "\"}}";
}

inline std::ostream& operator<<(std::ostream& out, const synTensorDescriptor* obj) {
  log_synTensorDescriptor(out, obj);
  return out;
}

void dump_reference(const std::string& ref, const std::string& ref_type, float* vec, int n);
void command(const std::string& x);

}  // namespace synapse_logger
