/**
 * Copyright (c) 2025 Intel Corporation
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

#include "utilization_metrics.h"
#include <synapse_api.h>
#include "backend/habana_device/HPUDevice.h"
#include "pytorch_helpers/habana_helpers/logging.h"

namespace synapse_helpers {
namespace {

using hlml_return_t = int;

struct hlml_utilization_t {
  unsigned int aip; // Device (AIP) utilization percentage.
  unsigned int memory; // Memory utilization percentage.
};

int ResolveDeviceIndex(int device_index) {
  synDeviceInfoV2 device_info;
  auto status = ::synDeviceGetInfoV2(device_index, &device_info);
  if (status == synFail) {
    PT_SYNHELPER_FATAL("synDeviceGetInfo failed.");
  }
  return device_info.deviceIndex;
}

} // namespace

LibLoader::LibLoader(const std::string& path)
    : libPath_(path), handle(nullptr) {
  loadLibrary();
}

LibLoader::~LibLoader() {
  unloadLibrary();
}

void LibLoader::loadLibrary() {
  std::lock_guard<std::mutex> guard(libMutex);
  handle = dlopen(libPath_.c_str(), RTLD_LOCAL | RTLD_NOW);
  if (!handle) {
    PT_SYNHELPER_FATAL("synDeviceGetInfo failed '", libPath_, "': ", dlerror());
  }
}

void LibLoader::unloadLibrary() {
  std::lock_guard<std::mutex> guard(libMutex);
  if (handle) {
    dlclose(handle);
    handle = nullptr;
  }
  funcPtrCache.clear();
}

HlmlPowerProvider::HlmlPowerProvider(const std::string& libPath)
    : device_(nullptr), libPath_(libPath) {}

void HlmlPowerProvider::init() {
  using hlml_init_func_t = hlml_return_t (*)();
  if (!lib_) {
    lib_ = std::make_unique<LibLoader>(libPath_);
  }
  auto initFunc = lib_->getFunction<hlml_init_func_t>("hlml_init");
  hlml_return_t ret = initFunc();
  if (ret != 0) {
    PT_SYNHELPER_FATAL("hlml_init failed with ret:", ret);
  }
  int device_id = habana::HPUDeviceContext::get_device().id();
  device_id = ResolveDeviceIndex(device_id);
  device_ = getDevice(device_id);
}

hlml_device_t HlmlPowerProvider::getDevice(unsigned int index) {
  using get_device_func_t = hlml_return_t (*)(unsigned int, hlml_device_t*);
  hlml_device_t device;
  auto getDeviceFunc =
      lib_->getFunction<get_device_func_t>("hlml_device_get_handle_by_index");
  hlml_return_t ret = getDeviceFunc(index, &device);
  if (ret != 0 || device == nullptr) {
    PT_SYNHELPER_FATAL(
        "hlml_device_get_handle_by_index failed for index:",
        index,
        " with ret:",
        ret);
  }
  PT_SYNHELPER_DEBUG("Retrieved device handle for index ", index);
  return device;
}

double HlmlPowerProvider::getUtilization() {
  using get_util_func_t = hlml_return_t (*)(hlml_device_t, hlml_utilization_t*);
  auto getUtilFunc =
      lib_->getFunction<get_util_func_t>("hlml_device_get_utilization_rates");
  hlml_utilization_t util = {0, 0};
  hlml_return_t ret = getUtilFunc(device_, &util);
  if (ret != 0) {
    PT_SYNHELPER_FATAL(
        "hlml_device_get_utilization_rates failed with ret:", ret);
  }
  PT_SYNHELPER_DEBUG(
      "Device Utilization:",
      "AIP:",
      util.aip,
      "%",
      "Memory:",
      util.memory,
      "%");
  return static_cast<double>(util.aip);
}

void HlmlPowerProvider::hlmlShutdown() {
  using hlm_shutdown_func_t = hlml_return_t (*)();
  auto getShutdownFunc =
      lib_->getFunction<hlm_shutdown_func_t>("hlml_shutdown");
  hlml_return_t ret = getShutdownFunc();
  if (ret != 0) {
    PT_SYNHELPER_FATAL("hlml_shutdown failed with ret:", ret);
  }
}

HPUUtilizationPoller::HPUUtilizationPoller(int interval)
    : interval_(interval),
      started_(false),
      totalUtil_(0.0),
      sampleCount_(0),
      usage_(0.0),
      provider_() {}

HPUUtilizationPoller::~HPUUtilizationPoller() {
  if (started_.load()) {
    stop();
  }
}

void HPUUtilizationPoller::start() {
  if (started_.load()) {
    PT_SYNHELPER_WARN(
        "HPUUtilizationPoller start called while already started");
    return;
  }
  reset();
  started_.store(true);
  pollThread_ = std::thread(&HPUUtilizationPoller::pollLoop, this);
}

void HPUUtilizationPoller::stop() {
  if (!started_.load()) {
    PT_SYNHELPER_WARN("HPUUtilizationPoller stop called when not started");
    return;
  }
  {
    std::lock_guard<std::mutex> lock(cvMutex_);
    started_.store(false);
  }
  cv_.notify_all();
  if (pollThread_.joinable()) {
    pollThread_.join();
  }
  {
    std::lock_guard<std::mutex> lock(mutex_);
    usage_ = (sampleCount_ == 0) ? usage_ : (totalUtil_ / sampleCount_);
  }
}

void HPUUtilizationPoller::reset() {
  std::lock_guard<std::mutex> lock(mutex_);
  totalUtil_ = 0.0;
  sampleCount_ = 0;
  usage_ = 0.0;
}

void HPUUtilizationPoller::resume() {
  if (started_.load()) {
    PT_SYNHELPER_WARN(
        "HPUUtilizationPoller::resume called when already running")
    return;
  }
  started_.store(true);
  pollThread_ = std::thread(&HPUUtilizationPoller::pollLoop, this);
}

double HPUUtilizationPoller::getUtilization() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return (sampleCount_ == 0) ? usage_ : (totalUtil_ / sampleCount_);
}

void HPUUtilizationPoller::pollLoop() {
  try {
    provider_.init();
  } catch (const std::exception& ex) {
    PT_SYNHELPER_WARN(
        "Failed to initialize HlmlPowerProvider in pollLoop: ", ex.what());
    return;
  }

  while (started_.load()) {
    try {
      double util = provider_.getUtilization();
      {
        std::lock_guard<std::mutex> lock(mutex_);
        if (util > 0) {
          totalUtil_ += util;
        }
        ++sampleCount_;
      }
    } catch (const std::exception& ex) {
      PT_SYNHELPER_WARN("Error polling utilization: ", ex.what());
      break;
    }

    std::unique_lock<std::mutex> lock(cvMutex_);
    cv_.wait_for(lock, std::chrono::seconds(interval_), [this] {
      return !started_.load();
    });
  }
  try {
    provider_.hlmlShutdown();
  } catch (const std::exception& ex) {
    PT_SYNHELPER_WARN("Error hlml shutdown: ", ex.what());
  }
}

Timer::Timer() {
  start_ = std::chrono::high_resolution_clock::now();
}

void Timer::start() {
  start_ = std::chrono::high_resolution_clock::now();
}

Interval Timer::getInterval() {
  auto now = std::chrono::high_resolution_clock::now();
  Interval ret = {start_, now};
  start_ = now;
  return ret;
}

StreamUtilizationMetric::StreamUtilizationMetric()
    : started_(false), totalTime_(0), idleTime_(0) {}

void StreamUtilizationMetric::start() {
  std::lock_guard<std::mutex> lock(mutex_);
  if (!started_) {
    totalTime_ = 0;
    idleTime_ = 0;
    startTime_ = std::chrono::high_resolution_clock::now();
    started_ = true;
  } else {
    PT_SYNHELPER_WARN(
        "StreamUtilizationMetric::start called when already started");
  }
}

void StreamUtilizationMetric::stop() {
  std::lock_guard<std::mutex> lock(mutex_);
  if (started_) {
    auto endTime = std::chrono::high_resolution_clock::now();
    totalTime_ += std::chrono::duration_cast<std::chrono::milliseconds>(
                      endTime - startTime_)
                      .count();
    startTime_ = endTime;
    started_ = false;
  } else {
    PT_SYNHELPER_WARN("StreamUtilizationMetric::stop called when not started");
  }
}

void StreamUtilizationMetric::reset() {
  std::lock_guard<std::mutex> lock(mutex_);
  totalTime_ = 0;
  idleTime_ = 0;
  startTime_ = std::chrono::high_resolution_clock::now();
}

void StreamUtilizationMetric::resume() {
  std::lock_guard<std::mutex> lock(mutex_);
  if (!started_) {
    startTime_ = std::chrono::high_resolution_clock::now();
    started_ = true;
  } else {
    PT_SYNHELPER_WARN(
        "StreamUtilizationMetric::resume called when already started");
  }
}

double StreamUtilizationMetric::getUtilization() const {
  std::lock_guard<std::mutex> lock(mutex_);
  if (totalTime_ == 0)
    return 0.0;
  return static_cast<double>(totalTime_ - idleTime_) /
      static_cast<double>(totalTime_) * 100.0;
}

void StreamUtilizationMetric::update(Interval idle) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (started_) {
    if (idle.first > startTime_) {
      idleTime_ += std::chrono::duration_cast<std::chrono::milliseconds>(
                       idle.second - idle.first)
                       .count();
    } else if (idle.second > startTime_) {
      idleTime_ += std::chrono::duration_cast<std::chrono::milliseconds>(
                       idle.second - startTime_)
                       .count();
    }
  }
}

UtilizationMetrics& UtilizationMetrics::getInstance() {
  static UtilizationMetrics instance;
  return instance;
}

void UtilizationMetrics::start() {
  poolerUtilization.start();
  streamUtilization.start();
}

void UtilizationMetrics::stop() {
  poolerUtilization.stop();
  streamUtilization.stop();
}

void UtilizationMetrics::reset() {
  poolerUtilization.reset();
  streamUtilization.reset();
}

void UtilizationMetrics::resume() {
  poolerUtilization.resume();
  streamUtilization.resume();
}

std::pair<double, double> UtilizationMetrics::getUtilization() {
  return {
      poolerUtilization.getUtilization(), streamUtilization.getUtilization()};
}

void UtilizationMetrics::update(Interval idle) {
  streamUtilization.update(idle);
}

} // namespace synapse_helpers
