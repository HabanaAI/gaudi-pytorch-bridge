#include "synapse_profiler.h"
#include <vector>
#include "pytorch_helpers/habana_device/HPUGuardImpl.h"

namespace habana {

uint64_t nowNanos(clockid_t clock) {
  constexpr uint64_t kSecondsToNanos = 1000ULL * 1000ULL * 1000ULL;
  struct timespec ts;
  clock_gettime(clock, &ts);
  return (
      static_cast<uint64_t>(ts.tv_sec) * kSecondsToNanos +
      static_cast<uint64_t>(ts.tv_nsec));
}

SynapseProfiler::SynapseProfiler() {
  auto env = std::getenv("HABANA_PROFILE");
  bool hpu_profiling_available =
      (env != nullptr) && (absl::string_view{env} != "0");

  if (!hpu_profiling_available) {
    throw std::runtime_error(
        "Tensorboard callback for HPU hardware profiling disabled. To enable set \"HABANA_PROFILE\"");
  }
}

void SynapseProfiler::start() {
  // Necessary to initialize the device to use synapse api calls
  HABANAGuardImpl h;
  h.getDevice();
  auto hpu_start_time = nowNanos(CLOCK_MONOTONIC_RAW) / 1000;
  auto wall_start_time = nowNanos(CLOCK_REALTIME) / 1000;
  parser_ =
      std::make_unique<HpuTraceParser>(*this, hpu_start_time, wall_start_time);

  synStatus status = synProfilerStart(synTraceAll, 0);
  if (status != synSuccess) {
    std::cerr << "synProfilerStart failed" << std::endl;
  }
}

void SynapseProfiler::stop() {
  synStatus status = synProfilerStop(synTraceAll, 0);
  if (status != synSuccess) {
    std::cerr << "synProfilerStop failed" << std::endl;
  }
  convertLogs();
}

void SynapseProfiler::convertLogs() {
  auto wall_stop_time = nowNanos(CLOCK_REALTIME) / 1000;

  size_t size{}, count{};
  getLogsSize(size, count);
  if (count == 0) {
    std::cerr << "No profiler entries" << std::endl;
    return;
  }
  std::vector<synTraceEvent> events;
  // size is not divisible by sizeof(synTraceEvent)
  // investigate and remove it (SW-102567)
  events.resize(size / sizeof(synTraceEvent) + 1);
  if (!getEntries(size, count, events.data())) {
    return;
  }
  parser_->Export(events.data(), count - 1, wall_stop_time);
}

void SynapseProfiler::getLogsSize(size_t& size, size_t& count) {
  auto status = synProfilerGetTrace(
      synTraceAll, 0, synTraceFormatTEF, nullptr, &size, &count);
  if (status != synSuccess) {
    std::cerr << "synProfilerGetTrace failed" << std::endl;
  }
}

bool SynapseProfiler::getEntries(size_t& size, size_t& count, void* out) {
  auto status = synProfilerGetTrace(
      synTraceAll, 0, synTraceFormatTEF, out, &size, &count);
  if (status != synSuccess) {
    std::cerr << "synProfilerGetTrace failed" << std::endl;
    return false;
  }
  return true;
}

uint64_t SynapseProfiler::startCustomMeasurement(const std::string& tag) {
  static uint64_t id{0};
  uint64_t time;
  synProfilerGetCurrentTimeNS(&time);
  custom_measurements_[++id] = std::make_pair(time, tag);
  return id;
}

void SynapseProfiler::stopCustomMeasurement(uint64_t id) {
  auto custom_measurement_it = custom_measurements_.find(id);
  if (custom_measurement_it != custom_measurements_.end()) {
    synProfilerAddCustomMeasurement(
        custom_measurement_it->second.second.c_str(),
        custom_measurement_it->second.first);
  } else {
    std::cerr << "custom measurement " << id << " not found" << std::endl;
  }
}
} // namespace habana