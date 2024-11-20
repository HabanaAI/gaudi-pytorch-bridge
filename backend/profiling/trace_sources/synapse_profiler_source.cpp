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

#include "backend/profiling/trace_sources/synapse_profiler_source.h"
#include <vector>
#include "backend/habana_device/HPUGuardImpl.h"

namespace habana {
namespace profile {

uint64_t NowNanos() {
  return static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::system_clock::now().time_since_epoch())
          .count());
}

std::string get_device_name() {
  constexpr uint32_t maxStringLength{1024};
  char deviceName[maxStringLength];
  auto status = synDeviceGetName(deviceName, maxStringLength, 0);
  if (status != synSuccess) {
    PT_SYNHELPER_DEBUG(
        Logger::formatStatusMsg(status), "Failed to get device name.");
    return "";
  }
  return deviceName;
}

uint64_t get_memory_size() {
  uint64_t free_mem{}, total_mem{};
  auto status = synDeviceGetMemoryInfo(0, &free_mem, &total_mem);
  if (status != synSuccess) {
    PT_SYNHELPER_DEBUG(
        Logger::formatStatusMsg(status), "Failed to get device name.");
    return 0;
  }
  return total_mem;
}

SynapseProfilerSource::SynapseProfilerSource() {
  auto env = std::getenv("HABANA_PROFILE");
  bool hpu_profiling_available =
      (env != nullptr) && (absl::string_view{env} != "0");

  if (!hpu_profiling_available) {
    throw std::runtime_error(
        "Tensorboard callback for HPU hardware profiling disabled. To enable set \"HABANA_PROFILE\"");
  }
  parser_ = std::make_unique<HpuTraceParser>(offset_);
}

void SynapseProfilerSource::start(TraceSink&) {
  // Necessary to initialize the device to use synapse api calls
  HABANAGuardImpl h;
  h.getDevice();
  uint64_t hpu_start_time_ns{};
  synProfilerGetCurrentTimeNS(&hpu_start_time_ns);
  long double hpu_start_time = hpu_start_time_ns;
  long double wall_start_time = NowNanos();
  parser_->update(hpu_start_time, wall_start_time);

  uint32_t bytes_req = 0;
  synStatus status = synProfilerQueryRequiredMemory(0, &bytes_req);
  if (status != synSuccess) {
    std::cerr << "synProfilerQueryRequiredMemory failed" << std::endl;
  }

  if (bytes_req > 0) {
    void* data_ptr{nullptr};
    auto& device = habana::HPUDeviceContext::get_device(0);
    device.get_device_memory().malloc(&data_ptr, bytes_req);
    auto user_buff =
        reinterpret_cast<void*>(device.get_fixed_address(data_ptr));
    status = synProfilerSetUserBuffer(0, user_buff);
    if (status != synSuccess) {
      std::cerr << "synProfilerSetUserBuffer failed" << std::endl;
    }
  }

  status = synProfilerStart(synTraceAll, 0);
  if (status != synSuccess) {
    std::cerr << "synProfilerStart failed" << std::endl;
  }
}

void SynapseProfilerSource::stop() {
  wall_stop_time_ = NowNanos();
  synStatus status = synProfilerStop(synTraceAll, 0);
  if (status != synSuccess) {
    std::cerr << "synProfilerStop failed" << std::endl;
  }
}

void SynapseProfilerSource::extract(TraceSink& output) {
  initHpuDetails(output);
  convertLogs(output);
}

TraceSourceVariant SynapseProfilerSource::get_variant() {
  return TraceSourceVariant::SYNAPSE_PROFILER;
}

void SynapseProfilerSource::set_offset(unsigned offset) {
  offset_ = offset;
}

void SynapseProfilerSource::convertLogs(TraceSink& output) {
  size_t size{}, count{};
  getLogsSize(size, count);
  if (count == 0) {
    std::cerr << "No profiler entries" << std::endl;
    return;
  }
  auto events = std::make_unique<unsigned char[]>(size);
  if (!getEntries(size, count, events.get())) {
    return;
  }
  parser_->Export(
      reinterpret_cast<synTraceEvent*>(events.get()),
      count - 1,
      wall_stop_time_,
      output);
}

void SynapseProfilerSource::getLogsSize(size_t& size, size_t& count) {
  auto status = synProfilerGetTrace(
      synTraceAll, 0, synTraceFormatTEF, nullptr, &size, &count);
  if (status != synSuccess) {
    std::cerr << "synProfilerGetTrace failed" << std::endl;
  }
}

bool SynapseProfilerSource::getEntries(size_t& size, size_t& count, void* out) {
  auto status = synProfilerGetTrace(
      synTraceAll, 0, synTraceFormatTEF, out, &size, &count);
  if (status != synSuccess) {
    std::cerr << "synProfilerGetTrace failed" << std::endl;
    return false;
  }
  return true;
}

void SynapseProfilerSource::initHpuDetails(TraceSink& output) {
  auto name = get_device_name();
  auto memory = get_memory_size();
  output.addDeviceDetails({{"name", name}});
  output.addDeviceDetails({{"totalGlobalMem", memory}});
}
} // namespace profile
} // namespace habana