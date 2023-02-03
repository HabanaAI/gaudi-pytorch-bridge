#include "pytorch_helpers/habana_helpers/profiling/trace_sources/synapse_profiler_source.h"
#include <vector>
#include "pytorch_helpers/habana_device/HPUGuardImpl.h"

namespace habana {

static int64_t getTimeUs() {
  return std::chrono::duration_cast<std::chrono::microseconds>(
             std::chrono::system_clock::now().time_since_epoch())
      .count();
}

std::string get_device_name() {
  constexpr uint32_t maxStringLength{1024};
  char deviceName[maxStringLength];
  auto status = synDeviceGetName(deviceName, maxStringLength, 0);
  if (status != synSuccess) {
    PT_SYNHELPER_DEBUG("Failed to get device name. Status: ", status);
    return "";
  }
  return deviceName;
}

uint64_t get_memory_size() {
  uint64_t free_mem{}, total_mem{};
  auto status = synDeviceGetMemoryInfo(0, &free_mem, &total_mem);
  if (status != synSuccess) {
    PT_SYNHELPER_DEBUG("Failed to get device name. Status: ", status);
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
}

void SynapseProfilerSource::start() {
  // Necessary to initialize the device to use synapse api calls
  HABANAGuardImpl h;
  h.getDevice();
  uint64_t hpu_start_time_ns{};
  synProfilerGetCurrentTimeNS(&hpu_start_time_ns);
  long double hpu_start_time = hpu_start_time_ns / 1000.0L;
  long double wall_start_time = getTimeUs();
  parser_ = std::make_unique<HpuTraceParser>(hpu_start_time, wall_start_time);

  synStatus status = synProfilerStart(synTraceAll, 0);
  if (status != synSuccess) {
    std::cerr << "synProfilerStart failed" << std::endl;
  }
}

void SynapseProfilerSource::stop() {
  uint64_t wall_stop_time_ns{};
  synProfilerGetCurrentTimeNS(&wall_stop_time_ns);
  wall_stop_time_ = wall_stop_time_ns / 1000;
  synStatus status = synProfilerStop(synTraceAll, 0);
  if (status != synSuccess) {
    std::cerr << "synProfilerStop failed" << std::endl;
  }
}

void SynapseProfilerSource::extract(TraceSink& output) {
  initHpuDetails(output);
  convertLogs(output);
}

void SynapseProfilerSource::convertLogs(TraceSink& output) {
  size_t size{}, count{};
  getLogsSize(size, count);
  if (count == 0) {
    std::cerr << "No profiler entries" << std::endl;
    return;
  }
  std::vector<synTraceEvent2> events;
  // size is not divisible by sizeof(synTraceEvent2)
  // investigate and remove it (SW-102567)
  events.resize(size / sizeof(synTraceEvent2) + 1);
  if (!getEntries(size, count, events.data())) {
    return;
  }
  parser_->Export(events.data(), count - 1, wall_stop_time_, output);
}

void SynapseProfilerSource::getLogsSize(size_t& size, size_t& count) {
  auto status = synProfilerGetTrace2(
      synTraceAll, 0, synTraceFormatTEF, nullptr, &size, &count);
  if (status != synSuccess) {
    std::cerr << "synProfilerGetTrace2 failed" << std::endl;
  }
}

bool SynapseProfilerSource::getEntries(size_t& size, size_t& count, void* out) {
  auto status = synProfilerGetTrace2(
      synTraceAll, 0, synTraceFormatTEF, out, &size, &count);
  if (status != synSuccess) {
    std::cerr << "synProfilerGetTrace2 failed" << std::endl;
    return false;
  }
  return true;
}

void SynapseProfilerSource::initHpuDetails(TraceSink& output) {
  auto name = get_device_name();
  auto memory = get_memory_size();
  output.addDeviceDetails(
      {{"name", name}, {"totalGlobalMem", std::to_string(memory)}});
}
} // namespace habana