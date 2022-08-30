#include <iostream>
#include "absl/strings/string_view.h"
#include "json_parser.h"
#include "synapse_profiler.h"
#define FMT_HEADER_ONLY
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsign-compare"
#include "spdlog/common.h"
#include "spdlog/fmt/bundled/format.h"
#pragma GCC diagnostic pop
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#include "Config.h"
#include "libkineto.h"
#pragma GCC diagnostic pop

namespace habana {

using namespace libkineto;
using namespace std::chrono;

class KinetoActivityProfiler : public SynapseProfiler {
 public:
  KinetoActivityProfiler(std::deque<GenericTraceActivity>& activities)
      : SynapseProfiler(json_parser_), activities_{activities} {}

  void addActivity(
      const std::string& name,
      bool isKernel,
      int64_t device,
      int64_t resource,
      uint64_t start,
      uint64_t end) {
    GenericTraceActivity ev{
        defaultTraceSpan(),
        isKernel ? ActivityType::CONCURRENT_KERNEL : ActivityType::HPU_OP,
        name};
    ev.startTime = start;
    ev.endTime = end;
    ev.device = device;
    ev.resource = resource;
    if (isKernel) {
      ev.addMetadata("device", ev.device);
    }
    activities_.push_back(ev);
  }

  void addDevice(const std::string& name, int64_t device) {
    GenericTraceActivity name_meta{
        defaultTraceSpan(), ActivityType::HPU_META_OP, ""};
    name_meta.startTime = 0;
    name_meta.endTime = 0;
    name_meta.activityName = "process_name";
    name_meta.device = device;
    name_meta.resource = 0;
    name_meta.addMetadata("name", std::string("\"") + name + "\"");
    activities_.push_back(name_meta);

    GenericTraceActivity sort_meta{
        defaultTraceSpan(), ActivityType::HPU_META_OP, ""};
    sort_meta.startTime = 0;
    sort_meta.endTime = 0;
    sort_meta.device = device;
    sort_meta.resource = 0;
    sort_meta.activityName = "process_sort_index";
    sort_meta.addMetadata(
        "sort_index",
        std::to_string(device < 8 ? device + 0x1000000ll : device));
    activities_.push_back(sort_meta);
  }

  virtual void addResource(
      const std::string& name,
      int64_t device,
      int64_t resource,
      int64_t sort_index = -1) {
    GenericTraceActivity name_meta{
        defaultTraceSpan(), ActivityType::HPU_META_OP, ""};
    name_meta.startTime = 0;
    name_meta.endTime = 0;
    name_meta.activityName = "thread_name";
    name_meta.device = device;
    name_meta.resource = resource;
    name_meta.addMetadata("name", std::string("\"") + name + "\"");
    activities_.push_back(name_meta);

    GenericTraceActivity sort_meta{
        defaultTraceSpan(), ActivityType::HPU_META_OP, ""};
    sort_meta.startTime = 0;
    sort_meta.endTime = 0;
    sort_meta.device = device;
    sort_meta.resource = resource;
    sort_meta.activityName = "thread_sort_index";
    sort_meta.addMetadata("sort_index", std::to_string(sort_index));
    activities_.push_back(sort_meta);
  }

 private:
  const TraceSpan& defaultTraceSpan() {
    static TraceSpan span(0, 0, "PyTorch Profiler", "");
    return span;
  }

  std::deque<GenericTraceActivity>& activities_;
  Parser json_parser_;
};

class ProfilerSession : public libkineto::IActivityProfilerSession {
 public:
  explicit ProfilerSession(int64_t, int64_t) {
    status_ = TraceStatus::READY;
    profiler_ = std::make_unique<KinetoActivityProfiler>(activities_);
  }

  void start() override {
    profiler_->start();
    status_ = TraceStatus::RECORDING;
  }

  void stop() override {
    profiler_->stop();
    status_ = TraceStatus::READY;
  }

  std::vector<std::string> errors() override {
    return {};
  }

  std::unique_ptr<libkineto::CpuTraceBuffer> getTraceBuffer() override {
    auto buf = std::make_unique<libkineto::CpuTraceBuffer>();
    buf->activities.swap(activities_);
    return buf;
  }

  void processTrace(ActivityLogger& logger) override {
    for (const auto& activity : activities_) {
      activity.log(logger);
    }
  }

 private:
  std::deque<GenericTraceActivity> activities_;
  std::unique_ptr<KinetoActivityProfiler> profiler_;
};

class ActivityProfiler : public libkineto::IActivityProfiler {
 public:
  ActivityProfiler() {}
  virtual ~ActivityProfiler() override {}

  virtual const std::string& name() const override {
    return device_name;
  }

  virtual const std::set<ActivityType>& availableActivities() const override {
    return supported_activities;
  }

  virtual std::unique_ptr<IActivityProfilerSession> configure(
      const std::set<ActivityType>& activity_types,
      const KINETO_NAMESPACE::Config& config) override {
    auto start_time_ms =
        duration_cast<milliseconds>(system_clock::now().time_since_epoch())
            .count();
    return configure(start_time_ms, 0, activity_types, config);
  }

  virtual std::unique_ptr<IActivityProfilerSession> configure(
      int64_t start_time_ms,
      int64_t duration_ms,
      const std::set<ActivityType>& activity_types,
      const KINETO_NAMESPACE::Config&) override {
    auto env = std::getenv("HABANA_PROFILE");
    bool hpu_profiling_available =
        (env != nullptr) && (absl::string_view{env} != "0");
    bool hpu_profiling_requested =
        activity_types.find(ActivityType::HPU_OP) != activity_types.end() ||
        activity_types.find(ActivityType::HPU_META_OP) != activity_types.end();

    if (hpu_profiling_requested) {
      if (hpu_profiling_available) {
        auto session =
            std::make_unique<ProfilerSession>(start_time_ms, duration_ms);
        return session;
      } else {
        std::cerr
            << "Tensorboard callback for HPU hardware profiling disabled. To enable set \"HABANA_PROFILE\""
            << std::endl;
      }
    }
    return nullptr;
  }

 private:
  const std::set<ActivityType> supported_activities{
      ActivityType::HPU_OP,
      ActivityType::CONCURRENT_KERNEL,
      ActivityType::HPU_META_OP};
  std::string device_name{"HPU"};
};

std::unique_ptr<IActivityProfiler> register_activity_profiler() {
  return std::make_unique<ActivityProfiler>();
}

auto register_activity_profiler_factory = [] {
  libkineto::api().registerProfilerFactory(register_activity_profiler);
  return 0;
};
}; // namespace habana
#undef FMT_HEADER_ONLY