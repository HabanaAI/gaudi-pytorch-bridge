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
#include "backend/synapse_helpers/device.h"

#include <absl/types/variant.h>
#include <stdlib.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <ostream>
#include <sstream>
#include <string>

#include <thread>
#include <vector>

#include <synapse_api.h>

#include "backend/helpers/dynamic_shape_info.h"
#include "backend/helpers/event_dispatcher.h"
#include "backend/kernel/refinement_engine.h"

#include "habana_helpers/logging.h"
#include "habana_kernels/fallback_helper.h"
#include "pytorch_helpers/habana_helpers/logging.h"
#include "pytorch_helpers/habana_helpers/python_utils.h"

#include "backend/synapse_helpers/devmem_logger.h"
#include "backend/synapse_helpers/env_flags.h"
#include "backend/synapse_helpers/session.h"
#include "backend/synapse_helpers/tcmalloc_helper.h"
#include "backend/synapse_helpers/util.h"

#define PRINT_ENV_FLAG_DEFAULT(name) \
  std::clog << " " << #name << " = " << GET_ENV_FLAG_NEW(name) << "\n";

namespace synapse_helpers {
/**
 * END: These will be removed when all lazy kernels start using shape
 * functions.
 */

std::string get_mem_str(uint64_t nbytes) {
  std::ostringstream oss;
  oss << std::setfill('0') << std::setw(12) << nbytes << " bytes <";
  uint64_t gb{0x40000000};
  if (nbytes > gb) {
    oss << std::setfill('0') << std::setw(4) << nbytes / gb << " GB ";
    nbytes %= gb;
  }
  uint64_t mb{0x100000};
  if (nbytes > mb) {
    oss << std::setfill('0') << std::setw(4) << nbytes / mb << " MB ";
    nbytes %= mb;
  }
  uint64_t kb{0x400};
  if (nbytes > kb) {
    oss << std::setfill('0') << std::setw(4) << nbytes / kb << " KB ";
    nbytes %= kb;
  }
  oss << std::setfill('0') << std::setw(4) << nbytes << " B>";

  return oss.str();
}

// Since computation on stream is asynchronous, in order to share workspace
// buffer, it has to be fixed in size otherwise, there need to be implemented
// mechanism to adjust its size at runtime, but that would require an explcit
// barrier on the computation stream and reallocation of this buffer. For now
// it's fixed to 10GB, since for BERT SQUAD, batch12 on fp32, the largest
// recipe requires WS of size ~9.7GB
// TODO: as a WA for memory issue, modified it to 5GB for the resnet run of
// BS=64,
//       may require changes or revert in future
constexpr std::size_t GLOBAL_WORKSPACE_SIZE = 5e9;

std::weak_ptr<device> device::device_in_use;
std::mutex device::device_mtx;

void active_recipe_counter::increase() {
  std::unique_lock<std::mutex> cond_lock(counter_mutex_);
  ++counter_state_;
  ++total_submitted_;
}

void active_recipe_counter::decrease_and_notify() {
  {
    std::unique_lock<std::mutex> cond_lock(counter_mutex_);
    if (counter_state_ == 0)
      PT_SYNHELPER_FATAL("Counter state is invalid")
    --counter_state_;
    ++total_freed_;
  }
  cv_.notify_all();
}

bool active_recipe_counter::is_zero() {
  std::unique_lock<std::mutex> cond_lock(counter_mutex_);
  bool zflag = (0 == counter_state_ ? true : false);
  return zflag;
}

uint32_t active_recipe_counter::wait_for_next_decrease_call() {
  std::unique_lock<std::mutex> cond_lock(counter_mutex_);
  if (counter_state_ > 0) {
    if ((total_submitted_ - total_freed_) != counter_state_) {
      PT_SYNHELPER_DEBUG(
          "wait_for_next_decrease_call counter::",
          counter_state_,
          " total_submitted::",
          total_submitted_,
          " total_freed::",
          total_freed_);
    }
    // NOTE: This is a common blocking flow for recipe execution.
    // Due to its blocking nature, and in many cases like mark_step(),
    // there is a possibility that in background it has been already
    // holding GIL lock.
    // Hence it is essential here that we release the GIL lock
    // before entering to wait state, so that other threads can
    // acquire GIL lock and proceed.
    AutoNoGIL gil_release;
    cv_.wait_for(cond_lock, std::chrono::milliseconds(100));
  }
  return counter_state_;
}

uint32_t active_recipe_counter::get_count() {
  std::unique_lock<std::mutex> cond_lock(counter_mutex_);
  return counter_state_;
}

void CheckDynamicMinMaxPolicyOrder() {
  const std::string MANDATE_POLICY_ORDER = "3,1";

  std::string min_policy_seq =
      GET_ENV_FLAG_NEW(PT_HPU_DYNAMIC_MIN_POLICY_ORDER);
  std::string max_policy_seq =
      GET_ENV_FLAG_NEW(PT_HPU_DYNAMIC_MAX_POLICY_ORDER);

  bool min_check = min_policy_seq.size() >= MANDATE_POLICY_ORDER.size() &&
      0 ==
          min_policy_seq.compare(
              min_policy_seq.size() - MANDATE_POLICY_ORDER.size(),
              MANDATE_POLICY_ORDER.size(),
              MANDATE_POLICY_ORDER);
  bool max_check = max_policy_seq.size() >= MANDATE_POLICY_ORDER.size() &&
      0 ==
          max_policy_seq.compare(
              max_policy_seq.size() - MANDATE_POLICY_ORDER.size(),
              MANDATE_POLICY_ORDER.size(),
              MANDATE_POLICY_ORDER);

  if (!min_check) {
    PT_DYNAMIC_SHAPE_FATAL(
        "Incorrect PT_HPU_DYNAMIC_MIN_POLICY_ORDER specified. Policy should end with \"3,1\".");
  }
  if (!max_check) {
    PT_DYNAMIC_SHAPE_FATAL(
        "Incorrect PT_HPU_DYNAMIC_MAX_POLICY_ORDER specified. Policy should end with \"3,1\".");
  }
}

void DisableDynamicShapeGaudi3() {
  constexpr uint32_t maxStringLength{1024};
  char deviceName[maxStringLength];
  auto status = synDeviceGetName(deviceName, maxStringLength, 0);
  if (status != synSuccess) {
    PT_SYNHELPER_DEBUG(
        Logger::formatStatusMsg(status), "Failed to get device name.");
  }

  if ((strcmp(deviceName, "GAUDI3") == 0) ||
      (strcmp(deviceName, "GRECO") == 0)) {
    habana_helpers::DisableRefineDynamicShape();
  }
}

int GetSystemRamInKB(void) {
  FILE* meminfo = fopen("/proc/meminfo", "r");
  if (meminfo != NULL) {
    char line[256];
    while (fgets(line, sizeof(line), meminfo)) {
      int ram;
      if (sscanf(line, "MemTotal: %d kB", &ram) == 1) {
        fclose(meminfo);
        return ram;
      }
    }
    fclose(meminfo);
  }
  return 0;
}

void dumpEnvSettings() {
  int node_id = 0;
  char* ptr1;
  char* ptr2;
  ptr1 = std::getenv("RANK");
  ptr2 = std::getenv("OMPI_COMM_WORLD_RANK");
  if (ptr1 != nullptr) {
    node_id = std::stoul(ptr1, nullptr, 16);
  } else if (ptr2 != nullptr) {
    node_id = std::stoul(ptr2, nullptr, 16);
  } else {
    node_id = 0;
  }

  // print only from main process
  if (!node_id) {
    if (const char* env_p = std::getenv("HB_BUILD_VER")) {
      std::clog
          << "=============================HABANA SW VERSION======================================= \n";
      std::clog << " HB_BUILD_VER = " << env_p << '\n';
    }
    std::clog
        << "=============================HABANA PT BRIDGE CONFIGURATION =========================== \n";

    // Below should be logged only flags documented in
    // https://docs.habana.ai/en/latest/PyTorch/Runtime_Flags.html
    // Make sure to update the user docs, if new flag is added.
    // NOTE: Only flags represented in env_flags.h are logged.
    PRINT_ENV_FLAG_DEFAULT(PT_HPU_LAZY_MODE)
    PRINT_ENV_FLAG_DEFAULT(PT_RECIPE_CACHE_PATH)
    PRINT_ENV_FLAG_DEFAULT(PT_CACHE_FOLDER_DELETE)
    PRINT_ENV_FLAG_DEFAULT(PT_HPU_MAX_COMPOUND_OP_SIZE)
    PRINT_ENV_FLAG_DEFAULT(PT_HPU_LAZY_ACC_PAR_MODE)

    if (GET_ENV_FLAG_NEW(PT_ENABLE_FP8_CAST_STOCHASTIC_ROUNDING)) {
      PT_BRIDGE_WARN(
          "PT_ENABLE_FP8_CAST_STOCHASTIC_ROUNDING is enabled. Casts to torch.fp8 will be executed "
          "in stochastic rounding mode.");
    }
    auto keep_input_mutations = std::getenv("PT_HPU_KEEP_INPUT_MUTATIONS");
    if (keep_input_mutations) {
      std::clog << " PT_HPU_KEEP_INPUT_MUTATIONS = " << keep_input_mutations
                << "\n";
    }

    std::clog
        << "---------------------------: System Configuration :---------------------------\n";
    std::clog << "Num CPU Cores : " << std::thread::hardware_concurrency()
              << "\n";
    std::clog << "CPU RAM       : " << GetSystemRamInKB() << " KB\n";
    std::clog
        << "------------------------------------------------------------------------------\n";
  }
}
device::device(
    std::shared_ptr<session> synapse_session,
    synDeviceId device_id,
    synDeviceType device_type,
    const create_allocator_fnc& create_allocator)
    : synapse_session_(std::move(synapse_session)),
      type_{device_type},
      id_{device_id},
      event_handle_cache_{*this, 0},
      time_event_handle_cache_{*this, EVENT_COLLECT_TIME},
      memory_mapper_{*this},
      recipe_handle_cache_{*this},
      host_memory_{*this},
      device_memory_{*this} {
  // create default stream
  create_default_stream();
  HABANA_ASSERT(create_allocator != nullptr);
  allocator_ = create_allocator(id_);
  ReleaseFreeMemory();
  DisableDynamicShapeGaudi3();
  dumpEnvSettings();
  is_hcl_same_addr_enabled_ =
      GET_ENV_FLAG_NEW(PT_ENABLE_HCL_SAME_ADDRESS_RESOLUTION) &&
      GET_ENV_FLAG_NEW(PT_ENABLE_HCL_STREAM);

  // use the first allocated buffer always for same_address functionality
  // if each rank uses the same address for the recv/intermediate addresses;
  // then we can use the same address and it will save the address resolution
  // (since the address is known)
  if (is_hcl_same_addr_enabled_ && (std::getenv("HLS_MODULE_ID") != nullptr)) {
    size_t prealloc_size = 2ULL * 1024 * 1024 * 1024; // 2GByte
    void* v_ptr{nullptr};
    device_memory_.malloc(&v_ptr, prealloc_size);
    device_ptr prealloc_addr = reinterpret_cast<device_ptr>(v_ptr);
    device_memory_.fix_address(reinterpret_cast<void*>(prealloc_addr));
    HABANA_ASSERT(prealloc_addr != device_nullptr);
    preallocated_reduction_buffer_ = absl::make_optional<owned_device_ptr>(
        prealloc_addr, prealloc_size, *this);
  }

  // GLOBAL_WORKSPACE_SIZE is set based on PT_HPU_INITIAL_WORKSPACE_SIZE in GB
  const size_t init_size =
      GET_ENV_FLAG_NEW(PT_HPU_INITIAL_WORKSPACE_SIZE) * 1024 * 1024 * 1024;
  if (init_size > 0) {
    workspace_buffer_ = get_workspace_buffer(init_size);

    PT_SYNHELPER_DEBUG(
        "Allocating static workspace at ",
        (void*)workspace_buffer_,
        " size ",
        synapse_helpers::get_mem_str(workspace_size_));
  }

  is_caching_enabled_ = GET_ENV_FLAG_NEW(PT_ENABLE_HABANA_CACHING);
  is_stream_async_enabled_ = GET_ENV_FLAG_NEW(PT_ENABLE_HABANA_STREAMASYNC);
  host_memory_cache_enabled_ = GET_ENV_FLAG_NEW(PT_ENABLE_HOST_MEMORY_CACHE);
  max_dma_copy_retry_count_ =
      GET_ENV_FLAG_NEW(PT_HABANA_MAX_DMA_COPY_RETRY_COUNT);
  dma_copy_retry_delay_ = std::chrono::milliseconds(
      GET_ENV_FLAG_NEW(PT_HABANA_DMA_COPY_RETRY_DELAY));
  max_recipe_limit_in_queue_ =
      GET_ENV_FLAG_NEW(PT_HPU_MAX_RECIPE_SUBMISSION_LIMIT);
  enable_memory_defragmentation_ =
      GET_ENV_FLAG_NEW(PT_ENABLE_MEMORY_DEFRAGMENTATION);
  enable_memory_defrag_info_ = GET_ENV_FLAG_NEW(PT_ENABLE_DEFRAGMENTATION_INFO);

  habana_helpers::SetRefineDynamicShape(
      GET_ENV_FLAG_NEW(PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES));

  CheckDynamicMinMaxPolicyOrder();

  // Create the refinement thread
  habana::RefinementEngine::GetEngine().Initialize();
}

synapse_error_v<std::shared_ptr<device>> device::get_or_create(
    const std::set<synDeviceType>& allowed_device_types,
    const create_allocator_fnc& allocator) {
  std::lock_guard<std::mutex> lock(device_mtx);
  std::shared_ptr<device> device_ptr = device_in_use.lock();
  if (device_ptr != nullptr) {
    if (!allowed_device_types.count(device_ptr->type())) {
      return synapse_error{
          "Process already acquired device of different type.",
          synDeviceTypeMismatch};
    }
    return device_ptr;
  }

  return device::create(allowed_device_types, allocator);
}

synapse_error_v<std::shared_ptr<device>> device::get_by_id(
    synDeviceId requested_id) {
  std::lock_guard<std::mutex> lock(device_mtx);
  std::shared_ptr<device> device_ptr = device_in_use.lock();
  if (device_ptr != nullptr) {
    if (requested_id == device_ptr->id()) {
      return device_ptr;
    }
  }
  return synapse_error{
      "Device with given id is not open by anyone!", synObjectNotInitialized};
}

synapse_error_v<std::shared_ptr<device>> device::create(
    const std::set<synDeviceType>& allowed_device_types,
    const create_allocator_fnc& create_allocator) {
  PT_SYNHELPER_DEBUG("synHPU Init");
  uint32_t new_device_id;
  synStatus status{synStatus::synSuccess};

  if (create_allocator == nullptr) {
    return synapse_helpers::synapse_error{
        "You should pass non null create_allocator_fnc for device creation.",
        synInvalidArgument};
  }

  auto synapse_session_create_result{synapse_helpers::session::get_or_create()};
  if (absl::holds_alternative<synapse_helpers::synapse_error>(
          synapse_session_create_result)) {
    auto error = absl::get<synapse_helpers::synapse_error>(
        synapse_session_create_result);
    return error;
  }

  auto synapse_session =
      synapse_helpers::get_value(std::move(synapse_session_create_result));

  synDeviceType acquired_device_type = synDeviceGaudi;
  if (std::getenv("HLS_MODULE_ID") != nullptr) {
    // Required for  multi chip configuration
    status = synDeviceAcquireByModuleId(
        &new_device_id, std::stoll(std::getenv("HLS_MODULE_ID")));
    if (status == synSuccess) {
      synDeviceInfo dinfo;
      auto status_info = synDeviceGetInfo(new_device_id, &dinfo);
      if (status_info != synSuccess) {
        return synapse_error{"Device get info failed.", status_info};
      }
      acquired_device_type = dinfo.deviceType;
    }
  } else {
    for (auto const& device_type : allowed_device_types) {
      status = synDeviceAcquireByDeviceType(&new_device_id, device_type);
      if (status == synSuccess) {
        PT_SYNHELPER_DEBUG(
            Logger::formatStatusMsg(status),
            "Device acquire successful for device_type: ",
            device_type);
        acquired_device_type = device_type;
        break;
      } else {
        PT_SYNHELPER_DEBUG(
            Logger::formatStatusMsg(status),
            "Device acquire failed for device_type: ",
            device_type);
      }
    }
  }

  if (status != synSuccess) {
    return synapse_error{"Device acquire failed.", status};
  } else {
    habana_helpers::EmitEvent(
        habana_helpers::EventDispatcher::Topic::DEVICE_ACQUIRED);
  }

  std::shared_ptr<device> device_ptr{new device(
      synapse_session, new_device_id, acquired_device_type, create_allocator)};

  uint64_t free_mem, total_mem;
  status = synDeviceGetMemoryInfo(device_ptr->id(), &free_mem, &total_mem);
  if (synStatus::synSuccess != status) {
    PT_SYNHELPER_FATAL(
        Logger::formatStatusMsg(status), "Cannot obtain device memory size.");
  }
  PT_SYNHELPER_DEBUG(
      "Device memory size: total=", total_mem, " free=", free_mem);

  // assign weak_ptr for future gets.
  device_in_use = device_ptr;
  return device_ptr;
}

int device::get_count_by_current_type() {
  int count = 0;
  synStatus status{synStatus::synSuccess};
  status = synDeviceGetCountByDeviceType((uint32_t*)&count, type_);
  if (status != synSuccess) {
    PT_SYNHELPER_DEBUG(
        Logger::formatStatusMsg(status), "Fail to get device count.");
  }

  return count;
}

std::shared_ptr<session> device::get_or_create_session() {
  auto synapse_session_create_result{synapse_helpers::session::get_or_create()};
  if (absl::holds_alternative<synapse_helpers::synapse_error>(
          synapse_session_create_result)) {
    auto error = absl::get<synapse_helpers::synapse_error>(
        synapse_session_create_result);
    PT_SYNHELPER_WARN("Fail to create session. error: ", error.error);
    return nullptr;
  }
  auto synapse_session =
      synapse_helpers::get_value(std::move(synapse_session_create_result));
  return synapse_session;
}

int device::get_total_device_count() {
  int count = -1;

  // This call can be made prior to device acquire so
  // creating a synapse session, so SynApi will become available
  auto sessionPtr = get_or_create_session();

  synStatus status{synStatus::synSuccess};
  status = synDeviceGetCount((uint32_t*)&count);
  if (status != synSuccess) {
    PT_SYNHELPER_WARN(
        Logger::formatStatusMsg(status), "Fail to get device count.");
  }

  return count;
}

void device::cleanup() {
  if (cleanup_done_) {
    return;
  }
  cleanup_done_ = true;

  global_context_.Clear();

  // Refinement thread cleanup is the first call since
  // it might be in the process of compiling a new recipe.
  // The compilation is allowed to complete for graceful termination.
  habana::RefinementEngine::GetEngine().Shutdown();

  // Wait for all futures to finish.
  // NOTE: If GIL is acquired by any other thread, there is a good chance that
  // we will hang here. Make sure the call for cleanup is coming from the main
  // python thread which has the GIL
  if (GET_ENV_FLAG_NEW(PT_WAIT_FOR_ALL_FUTURES_IN_CLEANUP)) {
    sem_.wait_for_all_futures();
  }

  flush_stream_events();

  if (is_hcl_same_addr_enabled_ && (std::getenv("HLS_MODULE_ID") != nullptr)) {
    device_ptr prealloc_addr = preallocated_reduction_buffer_->get();
    allocator_->free((void*)prealloc_addr);
  }
  // free workspace buffer
  {
    std::unique_lock<std::mutex> lock(ws_mutex_);
    allocator_->free(reinterpret_cast<void*>(workspace_buffer_));
  }

  framework_specific_cleanup_();

  // We should unmap all buffers BEFORE device is released.
  auto status = memory_mapper_.drop_cache();
  if (synStatus::synSuccess != status) {
    PT_SYNHELPER_FATAL(
        Logger::formatStatusMsg(status), "memory_mapper::drop_cache() failed.");
  }
  synapse_helpers::memstats_dump(*this, "Stats after cleanup.");
  streams_.clear();
  user_event_flag_map_.clear();
}

device::~device() {
  PT_SYNHELPER_DEBUG("Device dectructor entry");
  cleanup();

  habana::HpuFallbackHelper::get()->print_fallback_freq();
}

// only used when generic stream is not used
uint64_t device::get_compute_stream_count() {
  if (!GET_ENV_FLAG_NEW(PT_HPU_ENABLE_GENERIC_STREAM)) {
    if (type_ == synDeviceGaudi)
      return 2;
    if (type_ == synDeviceGaudi2)
      return 4;
    if (type_ == synDeviceGaudi3)
      return 4;
    return 1;
  }
  PT_SYNHELPER_FATAL("get_compute_stream_count not supported");
  return 0;
}

// Each stream is associate to 1 GC thread, creating unlimited
// streams can cause OS resource unavailable. in practical
// situations only use few user streams. CUDA has a limiation
// of 32 and later stream are assigned in round robin fashion.
//
#define GENERIC_STREAM_LIMIT 32
void device::create_stream(hpuStream_t& hpu_stream, bool high_priority) {
  std::unique_lock<std::mutex> lock(stream_mutex_);
  if (GET_ENV_FLAG_NEW(PT_HPU_ENABLE_GENERIC_STREAM)) {
    hpu_stream = ++stream_index_;
    if (stream_index_ <= GENERIC_STREAM_LIMIT) {
      streams_[hpu_stream] = absl::make_unique<stream>(*this);
    }
    PT_SYNHELPER_DEBUG(
        "STREAM:: New device stream created with index", stream_index_);
  } else {
    // special case handling only for Network stream in case of non generic
    // stream. Priority is never used otherwise
    if (high_priority &&
        default_streams_.find(NETWORK) == default_streams_.end()) {
      default_streams_[NETWORK] = absl::make_unique<stream>(*this);
      PT_SYNHELPER_DEBUG(
          "STREAM:: network stream handle", *default_streams_[NETWORK]);
    } else {
      auto compute_stream_count = get_compute_stream_count();
      hpu_stream = ++stream_index_;
      if (stream_index_ < compute_stream_count) {
        streams_[hpu_stream] = absl::make_unique<stream>(*this);
      }
      PT_SYNHELPER_DEBUG(
          "STREAM:: New device stream created with index", stream_index_);
    }
  }
}

// In Case of Generic stream, NETWORK type is not part of the default.
// as the collectives always uses a different stream.
void device::create_default_stream_event(synEventHandle keyHandle, bool flags) {
  std::unique_lock<std::mutex> lock(usr_event_mutex_);
  if (GET_ENV_FLAG_NEW(PT_HPU_ENABLE_GENERIC_STREAM)) {
    std::array<synEventHandle, _END_TYPE> event_array;
    for (int i = 0; i < (int)default_streams_.size(); i++) {
      if (flags) {
        event_array[i] = get_time_event_handle_cache().get_free_handle();
      } else {
        event_array[i] = get_event_handle_cache().get_free_handle();
      }
    }
    user_event_handle_map_[keyHandle] = event_array;
    PT_SYNHELPER_DEBUG("Create_default_stream_event handle::", keyHandle);
  }
}

void device::record_event_default_stream(synEventHandle keyHandle) {
  std::unique_lock<std::mutex> lock(usr_event_mutex_);
  if (GET_ENV_FLAG_NEW(PT_HPU_ENABLE_GENERIC_STREAM)) {
    PT_SYNHELPER_DEBUG("record_event_default_stream with handle::", keyHandle);
    auto it = user_event_handle_map_.find(keyHandle);
    HABANA_ASSERT(it != user_event_handle_map_.end());

    std::array<synEventHandle, _END_TYPE> event_array = it->second;
    for (int i = 0; i < (int)default_streams_.size(); i++) {
      auto type = (default_stream_type)i;
      auto status = synEventRecord(event_array[i], *default_streams_[type]);
      if (synStatus::synSuccess != status) {
        PT_DEVICE_FATAL(
            "synStreamWaitEvent failed: ",
            status,
            " for stream::",
            *default_streams_[type]);
      }
    }
  }
}

void device::wait_event_default_stream(synEventHandle handle) {
  std::unique_lock<std::mutex> lock(usr_event_mutex_);
  PT_SYNHELPER_DEBUG("wait_event_default_stream with handle::", handle);
  if (GET_ENV_FLAG_NEW(PT_HPU_ENABLE_GENERIC_STREAM)) {
    for (int i = 0; i < (int)default_streams_.size(); i++) {
      auto type = (default_stream_type)i;
      auto status = synStreamWaitEvent(*default_streams_[type], handle, 0);
      if (synStatus::synSuccess != status) {
        PT_DEVICE_FATAL(
            "synStreamWaitEvent failed: ",
            status,
            " for stream::",
            *default_streams_[type]);
      }
    }
  }
}

void device::synchronize_event_default_stream(synEventHandle keyHandle) {
  std::unique_lock<std::mutex> lock(usr_event_mutex_);
  if (GET_ENV_FLAG_NEW(PT_HPU_ENABLE_GENERIC_STREAM)) {
    auto it = user_event_handle_map_.find(keyHandle);
    HABANA_ASSERT(it != user_event_handle_map_.end());

    std::array<synEventHandle, _END_TYPE> event_array = it->second;
    for (int i = 0; i < (int)default_streams_.size(); i++) {
      auto status = synEventSynchronize(event_array[i]);
      if (synStatus::synSuccess != status) {
        PT_DEVICE_FATAL(
            "synEventSynchronize failed: ",
            status,
            " for Event::",
            event_array[i]);
      }
    }
  }
}

bool device::query_event_default_stream(synEventHandle keyHandle) {
  std::unique_lock<std::mutex> lock(usr_event_mutex_);
  bool result = false;
  if (GET_ENV_FLAG_NEW(PT_HPU_ENABLE_GENERIC_STREAM)) {
    auto it = user_event_handle_map_.find(keyHandle);
    HABANA_ASSERT(it != user_event_handle_map_.end());

    std::array<synEventHandle, _END_TYPE> event_array = it->second;
    for (int i = 0; i < (int)default_streams_.size(); i++) {
      auto status = synEventQuery(event_array[i]);
      if (synStatus::synSuccess == status) {
        result = true;
      } else {
        PT_DEVICE_DEBUG(
            "synEventQuery failed: ", status, " for Event::", event_array[i]);
        result = false;
        break;
      }
    }
  }
  return result;
}

uint64_t device::eplased_time_default_stream(
    synEventHandle keyHandle1,
    synEventHandle keyHandle2) {
  std::unique_lock<std::mutex> lock(usr_event_mutex_);
  uint64_t max_time_ms = 0;
  if (GET_ENV_FLAG_NEW(PT_HPU_ENABLE_GENERIC_STREAM)) {
    auto it = user_event_handle_map_.find(keyHandle1);
    HABANA_ASSERT(it != user_event_handle_map_.end());

    std::array<synEventHandle, _END_TYPE> event_array1 = it->second;

    it = user_event_handle_map_.find(keyHandle2);
    HABANA_ASSERT(it != user_event_handle_map_.end());

    std::array<synEventHandle, _END_TYPE> event_array2 = it->second;
    for (int i = 0; i < (int)default_streams_.size(); i++) {
      uint64_t time_ms = 0;
      auto status =
          synEventElapsedTime(&time_ms, event_array1[i], event_array2[i]);
      if (synStatus::synSuccess != status) {
        PT_DEVICE_FATAL("synEventElapsedTime failed: ", status);
      }
      max_time_ms = std::max(time_ms, max_time_ms);
    }
  }
  return max_time_ms;
}

void device::delete_event_default_stream(synEventHandle keyHandle, bool flags) {
  std::unique_lock<std::mutex> lock(usr_event_mutex_);
  if (GET_ENV_FLAG_NEW(PT_HPU_ENABLE_GENERIC_STREAM)) {
    auto it = user_event_handle_map_.find(keyHandle);
    HABANA_ASSERT(it != user_event_handle_map_.end());

    std::array<synEventHandle, _END_TYPE> event_array = it->second;
    for (int i = 0; i < (int)default_streams_.size(); i++) {
      if (flags) {
        get_time_event_handle_cache().release_handle(event_array[i]);
      } else {
        get_event_handle_cache().release_handle(event_array[i]);
      }
    }
    if (flags) {
      get_time_event_handle_cache().release_handle(keyHandle);
    } else {
      get_event_handle_cache().release_handle(keyHandle);
    }
    user_event_handle_map_.erase(keyHandle);
  }
}

// Default stream ==> 4 synapse stream, so if the query or
// synchronize is done on default stream, Need to do it on
// all 4 streams.
bool device::query_default_stream() {
  std::unique_lock<std::mutex> lock(stream_mutex_);
  if (GET_ENV_FLAG_NEW(PT_HPU_ENABLE_GENERIC_STREAM)) {
    for (auto& s : default_streams_) {
      auto& stream = *s.second;
      auto status = stream.query();
      if (status != synSuccess) {
        PT_SYNHELPER_DEBUG(
            Logger::formatStatusMsg(status), "STREAM:: synStreamQuery");
        return false;
      }
    }
    return true;
  } else {
    auto& stream = *default_streams_[COMPUTE];
    auto status = stream.query();
    if (status != synSuccess) {
      PT_SYNHELPER_DEBUG(
          Logger::formatStatusMsg(status), "STREAM:: synStreamQuery failed");
      return false;
    }
    return true;
  }
  return true;
}

void device::synchronize_default_stream() {
  std::unique_lock<std::mutex> lock(stream_mutex_);
  if (GET_ENV_FLAG_NEW(PT_HPU_ENABLE_GENERIC_STREAM)) {
    for (auto& s : default_streams_) {
      auto& stream = *s.second;
      stream.synchronize();
    }
  } else {
    auto& stream = *default_streams_[COMPUTE];
    stream.synchronize();
  }
}

void device::create_default_stream() {
  std::unique_lock<std::mutex> lock(stream_mutex_);
  default_streams_[COMPUTE] = absl::make_unique<stream>(*this);

  PT_SYNHELPER_DEBUG(
      "STREAM:: compute stream handle", *default_streams_[COMPUTE]);
  default_streams_[DMA_D2D] = absl::make_unique<stream>(*this);
  PT_SYNHELPER_DEBUG(
      "STREAM:: DMA_D2D stream handle", *default_streams_[DMA_D2D]);
  default_streams_[DMA_H2D] = absl::make_unique<stream>(*this);
  PT_SYNHELPER_DEBUG(
      "STREAM:: DMA_H2D stream handle", *default_streams_[DMA_H2D]);
  default_streams_[DMA_D2H] = absl::make_unique<stream>(*this);
  PT_SYNHELPER_DEBUG(
      "STREAM:: DMA_D2H stream handle", *default_streams_[DMA_D2H]);
}

stream& device::get_stream(hpuStream_t id, default_stream_type stream_type) {
  std::unique_lock<std::mutex> lock(stream_mutex_);
  if (GET_ENV_FLAG_NEW(PT_HPU_FORCE_USE_DEFAULT_STREAM)) {
    auto& stream = *default_streams_[stream_type];
    return stream;
  }

  if (GET_ENV_FLAG_NEW(PT_HPU_ENABLE_GENERIC_STREAM)) {
    if (id == 0) { // default stream any type stream
      auto& stream = *default_streams_[stream_type];
      PT_SYNHELPER_DEBUG(
          "STREAM:: get stream handle ", stream, " for id::", id);
      return stream;
    } else {
      auto index = id;
      if (id >= GENERIC_STREAM_LIMIT) {
        index = (id % GENERIC_STREAM_LIMIT);
        if (index == 0)
          index = 1; // start round robin from the 1 as 0 is default.
        else if (index < GENERIC_STREAM_LIMIT)
          index += 1;
      }
      auto it = streams_.find(index);
      HABANA_ASSERT(it != streams_.end());

      auto& stream = *it->second;
      return stream;
    }
  } else {
    if (id == 0 || stream_type != COMPUTE) { // any type stream
      auto& stream = *default_streams_[stream_type];
      PT_SYNHELPER_DEBUG(
          "STREAM:: get stream handle ", stream, " for id::", id);
      return stream;
    } else {
      auto compute_stream_count = get_compute_stream_count();
      auto index = id;
      // if not using generic stream, compute stream is assigned
      // in round robin fashion to user_stream in case if
      // it exceed actaul stream count.
      if (id >= compute_stream_count) {
        index = id % compute_stream_count;
      }

      if (index == 0) { // coumpute stream
        auto& stream = *default_streams_[stream_type];
        PT_SYNHELPER_DEBUG(
            "STREAM:: get stream handle ", stream, " for id::", id);
        return stream;
      }

      auto it = streams_.find(index);
      HABANA_ASSERT(
          it != streams_.end(), "Invalid Compute stream streamId::", index);

      auto& stream = *it->second;
      PT_SYNHELPER_DEBUG(
          "STREAM:: get stream handle",
          stream,
          " for id::",
          id,
          " index::",
          index);
      return stream;
    }
  }
}

void device::delete_stream(hpuStream_t id) {
  std::unique_lock<std::mutex> lock(stream_mutex_);
  auto it = streams_.find(id);
  HABANA_ASSERT(it != streams_.end());
  streams_.erase(id);
}

void device::flush_stream_events() {
  {
    std::unique_lock<std::mutex> lock(stream_mutex_);
    for (auto& s : default_streams_) {
      auto& stream = *s.second;
      stream.flush();
    }

    for (auto& s : streams_) {
      auto& stream = *s.second;
      stream.flush();
    }
  }
  auto start = std::chrono::steady_clock::now();
  while (true) {
    if (sem_.is_flushed())
      break;
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  PT_SYNHELPER_DEBUG(
      "stream flush completed in ",
      std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::steady_clock::now() - start)
          .count());
}

std::ostream& operator<<(std::ostream& stream, const device& syn_device) {
  stream << "synDevice at " << &syn_device;
  switch (syn_device.type()) {
    case synDeviceGaudi:
      stream << " Gaudi ";
      break;
    case synDeviceGaudi2:
      stream << " Gaudi2 ";
      break;
    case synDeviceGaudi3:
      stream << " Gaudi3 ";
      break;
    default:
      stream << " UNKNOWN ";
  }
  auto flag_guard = synapse_helpers::ostream_flag_guard::create(stream);
  stream << std::hex << syn_device.id() << std::dec;
  return stream;
}

device_ptr device::malloc(size_t size) {
  return reinterpret_cast<device_ptr>(allocator_->alloc(size));
}

void device::free(device_ptr ptr) {
  return allocator_->free(reinterpret_cast<void*>(ptr));
}

inline bool device::copy_data_to_device_(
    void* cpu_data,
    device_ptr destination,
    device_ptr event_addr,
    size_t total_bytes,
    const event_done_callback& done_cb,
    bool is_pinned,
    synapse_helpers::hpuStream_t hpu_stream) {
  PT_SYNHELPER_DEBUG(
      "Copy CPU Tensor to Device ",
      cpu_data,
      " to ",
      (void*)destination,
      ", total_bytes=",
      total_bytes);
  synStatus status;

  void* mapped_cpu_data = cpu_data;
  synapse_helpers::stream& stream_handle = get_stream(hpu_stream, DMA_H2D);
  uint8_t* dst_ptr;
  if (!is_pinned) {
    status = host_memory_.malloc((void**)&dst_ptr, total_bytes);
    if (status != synStatus::synSuccess) {
      PT_SYNHELPER_FATAL(Logger::formatStatusMsg(status), "Host malloc failed");
      return false;
    }
    std::copy(
        reinterpret_cast<uint8_t*>(cpu_data),
        reinterpret_cast<uint8_t*>(cpu_data) + total_bytes,
        dst_ptr);
    mapped_cpu_data = dst_ptr;
  } else {
    PT_SYNHELPER_DEBUG("copy_data_to_device uses Pinned memory");
  }

  PT_SYNHELPER_DEBUG("Used stream handle: ", stream_handle);

  unsigned attempt = 0;
  std::shared_ptr<device_ptr_lock> locked;
  do {
    locked = std::make_shared<device_ptr_lock>(lock_addresses(destination));
    status = synMemCopyAsync(
        stream_handle,
        reinterpret_cast<uint64_t>(mapped_cpu_data),
        total_bytes,
        locked->at(0),
        synDmaDir::HOST_TO_DRAM);

    if (status == synStatus::synSuccess) {
      if (attempt != 0) {
        PT_SYNHELPER_WARN(
            "DMA to HPU start succeeded on ", attempt + 1, " attempt.");
      }
      break;
    } else if (attempt < max_dma_copy_retry_count_ - 1) {
      PT_SYNHELPER_WARN(
          Logger::formatStatusMsg(status),
          "DMA to HPU start failed. Attempt ",
          attempt + 1,
          "/",
          max_dma_copy_retry_count_,
          ".");
      std::this_thread::sleep_for(dma_copy_retry_delay_);
    } else {
      if (!is_pinned) {
        host_memory_.free((void*)dst_ptr);
      }
      PT_SYNHELPER_FATAL(
          Logger::formatStatusMsg(status), "DMA to HPU start failed");
      return false;
    }
  } while (++attempt < max_dma_copy_retry_count_);

  sem_.add_producer(
      {event_addr},
      stream_handle,
      [this, dst_ptr, is_pinned, done_cb, locked]() mutable {
        if (!is_pinned)
          host_memory_.free((void*)dst_ptr);
        done_cb();
        locked = nullptr;
      });
  return true;
}

synapse_error device::copy_data_to_device(
    void* cpu_data,
    device_ptr destination,
    device_ptr event_addr,
    size_t total_bytes,
    const event_done_callback& done_cb,
    bool non_blocking,
    bool is_pinned,
    synapse_helpers::hpuStream_t hpu_stream) {
  /* in case of write, we can invoke a fill (compute)
   * stream or via DMA. if we have a fill and a copy
   * Need to wait for the fill compute stream to complete
   * before copy, so wait */
  synapse_helpers::stream& stream_handle = get_stream(hpu_stream, DMA_H2D);
  sem_.enqueue_wait_event(event_addr, stream_handle);

  /*
   * If non-blocking copy and non pinned memory and tensor size >= 1 MB
   *  - Schedule copy data function to a async thread and add future
   * Else
   *  - Continue copy data function in the same main thread
   */
  if (true == non_blocking && false == is_pinned &&
      GET_ENV_FLAG_NEW(PT_HPU_ENABLE_H2D_COPY_ASYNC_THREAD) &&
      total_bytes >= GET_ENV_FLAG_NEW(PT_HPU_H2D_COPY_MIN_TENSOR_SIZE)) {
    std::shared_future<bool> copy_future = std::async(
        std::launch::async | std::launch::deferred,
        &device::copy_data_to_device_,
        this,
        cpu_data,
        destination,
        event_addr,
        total_bytes,
        done_cb,
        is_pinned,
        hpu_stream);
    submit_future(destination, std::move(copy_future));
  } else { // Continue in the same main thread
    (void)device::copy_data_to_device_(
        cpu_data,
        destination,
        event_addr,
        total_bytes,
        done_cb,
        is_pinned,
        hpu_stream);
  }
  return {};
}

synapse_error device::copy_data_to_device(
    transfer_manifest const& transfers,
    event_done_callback unref_cb,
    synapse_helpers::hpuStream_t hpu_stream) {
  synStatus status;

  synapse_helpers::stream& stream_handle = get_stream(hpu_stream, DMA_H2D);
  for (std::size_t i = 0; i < transfers.size(); ++i) {
    sem_.enqueue_wait_event(transfers[i].dst_event_addr, stream_handle);
  }

  std::vector<std::uint64_t> mapped_srcs(transfers.size());
  std::vector<std::uint64_t> lens(transfers.size());
  std::vector<std::uint64_t> dsts(transfers.size());
  std::vector<std::uint64_t> dsts_event_addr(transfers.size());

  // Allocate host memory for all cpu tensors
  uint8_t* host_mem_ptr;
  auto total_bytes = std::accumulate(
      transfers.begin(),
      transfers.end(),
      0,
      [&](size_t total, const transfer_desc& curr) {
        return total + curr.bytes_to_transfer;
      });
  status = host_memory_.malloc((void**)&host_mem_ptr, total_bytes);
  if (status != synStatus::synSuccess) {
    PT_SYNHELPER_FATAL(Logger::formatStatusMsg(status), "Host malloc failed");
    return {};
  }

  // Copy cpu tensors data to host memory
  uint8_t* mem_ptr = host_mem_ptr;
  for (std::size_t i = 0; i < transfers.size(); ++i) {
    auto src = transfers[i].src;
    auto len = transfers[i].bytes_to_transfer;
    std::copy(
        reinterpret_cast<uint8_t*>(src),
        reinterpret_cast<uint8_t*>(src) + len,
        mem_ptr);
    mapped_srcs[i] = reinterpret_cast<uint64_t>(mem_ptr);
    mem_ptr += len;
    lens[i] = len;
    dsts[i] = transfers[i].dst;
    dsts_event_addr[i] = transfers[i].dst_event_addr;
  }

  unsigned attempt = 0;
  auto locked = std::make_shared<device_ptr_lock>(lock_addresses(dsts));
  HABANA_ASSERT(
      transfers.size() ==
      std::size_t(std::distance(locked->begin(), locked->end())));
  absl::Span<const device_ptr> locked_dsts{locked->begin(), transfers.size()};
  do {
    status = synMemCopyAsyncMultiple(
        stream_handle,
        mapped_srcs.data(),
        lens.data(),
        locked_dsts.data(),
        synDmaDir::HOST_TO_DRAM,
        transfers.size());

    if (status == synStatus::synSuccess) {
      if (attempt != 0) {
        PT_SYNHELPER_WARN(
            Logger::formatStatusMsg(status),
            "DMA to HPU start succeeded on ",
            attempt + 1,
            " attempt.");
      }
      break;
    } else if (attempt < max_dma_copy_retry_count_ - 1) {
      PT_SYNHELPER_WARN(
          Logger::formatStatusMsg(status),
          "DMA to HPU start failed. Attempt ",
          attempt + 1,
          "/",
          max_dma_copy_retry_count_,
          ".");
      std::this_thread::sleep_for(dma_copy_retry_delay_);
    } else {
      host_memory_.free((void*)host_mem_ptr);
      PT_SYNHELPER_FATAL(
          Logger::formatStatusMsg(status), "DMA to HPU start failed");
      return {};
    }
  } while (++attempt < max_dma_copy_retry_count_);

  sem_.add_producer(
      std::move(dsts_event_addr),
      stream_handle,
      [this, host_mem_ptr, unref_cb, locked]() mutable {
        host_memory_.free((void*)host_mem_ptr);
        unref_cb();
        locked = nullptr;
      });
  return {};
}

synapse_error device::copy_data_to_host(
    device_ptr device_data,
    void* destination,
    device_ptr event_addr,
    size_t total_bytes,
    const event_done_callback& done_cb,
    bool is_pinned,
    synapse_helpers::hpuStream_t hpu_stream) {
  PT_SYNHELPER_DEBUG(
      "Copy Device Tensor to CPU ",
      (void*)device_data,
      " ",
      destination,
      " total_bytes=",
      total_bytes);

  synStatus status;
  synapse_helpers::stream& stream_handle = get_stream(hpu_stream, DMA_D2H);
  PT_SYNHELPER_DEBUG("Used stream handle: ", stream_handle);
  sem_.enqueue_wait_event(event_addr, stream_handle);

  void* mapped_destination = destination;
  uint8_t* dst_ptr;
  if (!is_pinned) {
    status = host_memory_.malloc((void**)&dst_ptr, total_bytes);
    if (status != synStatus::synSuccess) {
      PT_SYNHELPER_WARN(Logger::formatStatusMsg(status), "Host malloc failed");
      return synapse_error{"Host Malloc failed with status.", status};
    }
    mapped_destination = dst_ptr;
  } else {
    PT_SYNHELPER_DEBUG("copy_data_to_host uses Pinned memory");
  }

  unsigned attempt = 0;
  std::shared_ptr<device_ptr_lock> locked;
  do {
    locked = std::make_shared<device_ptr_lock>(lock_addresses(device_data));
    status = synMemCopyAsync(
        stream_handle,
        locked->at(0),
        total_bytes,
        reinterpret_cast<uint64_t>(mapped_destination),
        synDmaDir::DRAM_TO_HOST);
    if (status == synStatus::synSuccess) {
      if (attempt != 0) {
        PT_SYNHELPER_WARN(
            "DMA from HPU start succeeded on ", attempt + 1, " attempt.");
      }
      break;
    } else if (attempt < max_dma_copy_retry_count_ - 1) {
      PT_SYNHELPER_WARN(
          Logger::formatStatusMsg(status),
          "DMA from HPU start failed. Attempt ",
          attempt + 1,
          "/",
          max_dma_copy_retry_count_,
          ".");
      std::this_thread::sleep_for(dma_copy_retry_delay_);
    } else {
      if (!is_pinned) {
        host_memory_.free((void*)dst_ptr);
      }
      return synapse_error{"DMA from HPU start failed.", status};
    }
  } while (++attempt < max_dma_copy_retry_count_);

  sem_.add_producer(
      {},
      stream_handle,
      [this,
       done_cb,
       dst_ptr,
       total_bytes,
       destination,
       is_pinned,
       locked]() mutable {
        if (!is_pinned) {
          std::copy(
              dst_ptr,
              dst_ptr + total_bytes,
              reinterpret_cast<uint8_t*>(destination));
          host_memory_.free((void*)dst_ptr);
        }
        done_cb();
        locked = nullptr;
      });

  return {};
}

synapse_error device::copy_data_within_device(
    device_ptr source,
    device_ptr destination,
    device_ptr src_event_addr,
    device_ptr dst_event_addr,
    size_t total_bytes,
    event_done_callback unref_cb,
    synapse_helpers::hpuStream_t hpu_stream) {
  synStatus status;

  synapse_helpers::stream& stream_handle = get_stream(hpu_stream, DMA_D2D);
  sem_.enqueue_wait_event(src_event_addr, stream_handle);
  auto locked =
      std::make_shared<device_ptr_lock>(lock_addresses(source, destination));
  status = synMemCopyAsync(
      stream_handle,
      locked->at(0),
      total_bytes,
      locked->at(1),
      synDmaDir::DRAM_TO_DRAM);
  if (synStatus::synSuccess != status) {
    return synapse_error{"DMA inside HPU start failed.", status};
  }
  auto done_cb = [unref_cb, locked]() mutable {
    unref_cb();
    locked = nullptr;
  };
  sem_.add_producer({dst_event_addr}, stream_handle, std::move(done_cb));

  return {};
}

synapse_error device::copy_data_within_device(
    transfer_manifest const& transfers,
    event_done_callback unref_cb,
    stream* const next_operation_stream,
    synapse_helpers::hpuStream_t hpu_stream) {
  synStatus status;

  std::vector<std::uint64_t> all_addresses(2 * transfers.size());
  std::vector<std::uint64_t> dsts(transfers.size());
  std::vector<std::uint64_t> lens(transfers.size());
  std::vector<std::uint64_t> dsts_event_addr(transfers.size());

  synapse_helpers::stream& stream_handle = get_stream(hpu_stream, DMA_D2D);
  for (std::size_t i = 0; i < transfers.size(); ++i) {
    sem_.enqueue_wait_event(transfers[i].src_event_addr, stream_handle);
    all_addresses[i] = transfers[i].src;
    dsts[i] = all_addresses[i + transfers.size()] = transfers[i].dst;
    lens[i] = transfers[i].bytes_to_transfer;
    dsts_event_addr[i] = transfers[i].dst_event_addr;
  }

  auto locked =
      std::make_shared<device_ptr_lock>(lock_addresses(all_addresses));
  HABANA_ASSERT(
      transfers.size() * 2 ==
      std::size_t(std::distance(locked->begin(), locked->end())));
  absl::Span<const device_ptr> locked_srcs{locked->begin(), transfers.size()};
  absl::Span<const device_ptr> locked_dsts{
      locked->begin() + transfers.size(), transfers.size()};

  status = synMemCopyAsyncMultiple(
      stream_handle,
      locked_srcs.data(),
      lens.data(),
      locked_dsts.data(),
      synDmaDir::DRAM_TO_DRAM,
      transfers.size());
  if (synStatus::synSuccess != status) {
    return synapse_error{"dma inside hpu start failed.", status};
  }
  auto done_cb = [unref_cb, locked]() mutable {
    unref_cb();
    locked = nullptr;
  };

  if (nullptr == next_operation_stream) {
    sem_.add_producer(
        std::move(dsts_event_addr), stream_handle, std::move(done_cb));
  } else {
    // If next operation stream is known then user wants us to put event on
    // this stream immediately and not pass it into the SEM.
    record_and_wait_for_event(
        stream_handle, *next_operation_stream, std::move(done_cb));
  }
  return {};
} // namespace synapse_helpers

device_ptr device::get_workspace_buffer(size_t size) {
  std::unique_lock<std::mutex> lock(ws_mutex_);
  if (size == 0) {
    PT_SYNHELPER_DEBUG("workspace Allocation of size is zero");
    return 0;
  }
  void* buffer = device_memory_.workspace_alloc(
      (void*)workspace_buffer_, workspace_size_, size);
  workspace_buffer_ = reinterpret_cast<device_ptr>(buffer);
  if (buffer == nullptr) {
    PT_SYNHELPER_FATAL("workspace Allocation of size ::", size, " failed!");
  }
  uint32_t usage_cnt = 0;
  if (workspace_usage_.find(size) != workspace_usage_.end()) {
    usage_cnt = workspace_usage_.at(size);
  }
  workspace_usage_[size] = ++usage_cnt;
  PT_SYNHELPER_DEBUG(
      "Allocated workspace buffer at",
      (void*)workspace_buffer_,
      " size ",
      synapse_helpers::get_mem_str(workspace_size_));

  real_workspace_size_ = size;

  return workspace_buffer_;
}

size_t device::get_least_workspace_size(
    size_t persistent_size,
    size_t req_workspace_size) {
  size_t least_workspace_size = 0;
  // 1. based on the rank select the workspace size which is >=
  // req_workspace_size and < current_ws_size.
  // 2. if the slected least_workspace_size is > than the req_workspace_size,
  // check with the new ws memory is available.
  // 3. if available, use the least_workspace_size else use the
  // req_workspace_size
  size_t current_ws_size = workspace_size_;
  if (workspace_usage_.size() > 1) {
    uint32_t usage_rank = 0;
    auto itr = workspace_usage_.begin();
    for (; itr != workspace_usage_.end(); ++itr) {
      if (itr->first >= req_workspace_size && itr->first < current_ws_size &&
          itr->second > usage_rank) {
        least_workspace_size = itr->first;
        usage_rank = itr->second;
      }
    }
  }

  if ((least_workspace_size > req_workspace_size) &&
      (device_memory_.is_memory_available(
          persistent_size, current_ws_size, least_workspace_size))) {
    return least_workspace_size;
  }

  return req_workspace_size;
}

void device::cleanup_workspace_buffer() {
  std::unique_lock<std::mutex> lock(ws_mutex_);
  if (workspace_size_ == 0)
    return;
  auto& recipe_counter = get_active_recipe_counter();
  while (recipe_counter.get_count() > 1) {
    recipe_counter.wait_for_next_decrease_call();
  }
  allocator_->free(reinterpret_cast<void*>(workspace_buffer_));
  workspace_buffer_ = 0;
  workspace_size_ = 0;
}

void device::add_wait_events_on_stream(
    const std::vector<device_ptr>& input_tensors,
    stream& stream) {
  for (const auto& input_addr : input_tensors) {
    PT_SYNHELPER_DEBUG(
        "Wait event address ", reinterpret_cast<void*>(input_addr))
    sem_.enqueue_wait_event(input_addr, stream);
  }
}

void device::add_wait_event_on_stream(
    const std::string& event_id,
    stream& stream) {
  sem_.enqueue_wait_event(event_id, stream);
}

void device::register_producer_on_stream(
    std::vector<device_ptr>&& bound_addresses,
    stream& stream,
    event_done_callback done_cb,
    synEventHandle event_handle) {
  sem_.add_producer(
      std::move(bound_addresses), stream, std::move(done_cb), event_handle);
}

void device::submit_future(
    device_ptr device_addr,
    std::shared_future<bool> fut) {
  sem_.add_future(device_addr, std::move(fut));
}

void device::register_producer_on_stream(
    std::vector<device_ptr>&& bound_addresses,
    const std::string& event_id,
    stream& stream,
    event_done_callback done_cb) {
  sem_.add_producer(
      std::move(bound_addresses), event_id, stream, std::move(done_cb));
}

void device::register_producer_on_stream(stream& stream, shared_event event) {
  if (!event->is_partial()) {
    PT_SYNHELPER_FATAL(
        "Only mapped partial events can be registered without address");
  }
  sem_.add_producer(stream, event);
}

void device::add_event_id(
    const std::string& event_id,
    const std::string& new_id) {
  sem_.add_event_id(event_id, new_id);
}

void device::wait_until_event_ready(const std::string& event_id) {
  sem_.wait_until_done(event_id);
}

void device::wait_for_future(device_ptr address) {
  sem_.wait_for_future(address);
}

void device::wait_until_address_ready(device_ptr address) {
  sem_.wait_until_done(address);
}

void device::wait_for_event(shared_event& event) {
  sem_.wait_until_done(event);
}

shared_event device::map_event_to_tensor(
    stream& stream,
    const synRecipeHandle recipe_handle,
    synLaunchTensorInfoExt* tensor_info,
    event_done_callback done_cb) {
  return sem_.map_event_to_tensor(
      stream, recipe_handle, tensor_info, std::move(done_cb));
}

void device::record_and_wait_for_event(
    stream& record_stream,
    stream& other_stream,
    event_done_callback done_callback) {
  auto event_ref = std::make_shared<event>(
      get_event_handle_cache(),
      record_stream,
      std::vector<device_ptr>{},
      "",
      std::move(done_callback));
  record_stream.register_pending_event(event_ref);
  event_ref->stream_wait_event(other_stream);
}

std::set<synDeviceType> device::get_supported_devices() {
  return {
      synDeviceType::synDeviceGaudi,
      synDeviceType::synDeviceGaudi2,
      synDeviceType::synDeviceGaudi3};
}

void device::synchronize() {
  auto status = synDeviceSynchronize(id_);
  if (status != synSuccess) {
    PT_SYNHELPER_FATAL(
        Logger::formatStatusMsg(status), "synDeviceSynchronize failed.");
  }
}

void device::release() {
  auto status = synDeviceRelease(id_);
  if (status != synSuccess) {
    PT_SYNHELPER_FATAL(
        Logger::formatStatusMsg(status), "synDeviceRelease failed.");
  }
}

std::string device::get_device_capability() {
  char pDriverVersion[256];
  auto status = synDriverGetVersion(pDriverVersion, 256);
  if (status != synSuccess) {
    PT_SYNHELPER_FATAL(
        Logger::formatStatusMsg(status), "synDriverGetVersion failed.");
  }
  return std::string(pDriverVersion);
}

std::string device::get_device_properties(int id) {
  synDeviceInfo device_info;
  auto status = synDeviceGetInfo(id, &device_info);
  if (status != synSuccess) {
    PT_SYNHELPER_FATAL(
        Logger::formatStatusMsg(status), "synDeviceGetInfo failed.");
  }

  std::string properties = "";
  properties = properties +
      "(sramBaseAddress=" + std::to_string(device_info.sramBaseAddress) +
      ", dramBaseAddress=" + std::to_string(device_info.dramBaseAddress) +
      ", sramSize=" + std::to_string(device_info.sramSize) +
      ", dramSize=" + std::to_string(device_info.dramSize) +
      ", tpcEnabledMask=" + std::to_string(device_info.tpcEnabledMask) +
      ", dramEnabled=" + std::to_string(device_info.dramEnabled) +
      ", fd=" + std::to_string(device_info.fd) +
      ", device_type=" + std::to_string(device_info.deviceType) + ")";

  return properties;
}

habana::backend::GlobalContext& device::get_global_context() {
  return global_context_;
}

void owned_device_ptr::device_ptr_deleter::operator()(device_ptr* ptr) {
  if (ptr) {
    PT_SYNHELPER_DEBUG(
        "Free buffer ptr ",
        reinterpret_cast<void*>(reinterpret_cast<device_ptr>(ptr)));
    device_->free(reinterpret_cast<device_ptr>(ptr));
  }
}

device_id::~device_id() {
  if (id_ != device::INVALID_ID) {
    auto status = synDeviceRelease(id_);
    if (status != synSuccess) {
      PT_SYNHELPER_FATAL(
          Logger::formatStatusMsg(status), "synDeviceRelease failed.");
    }
  }
}

} // namespace synapse_helpers
