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
#pragma once

#include <dlfcn.h>
#include <hlml.h>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>

namespace synapse_helpers {
typedef void* hlml_device_t;

class LibLoader {
 public:
  explicit LibLoader(const std::string& path);
  ~LibLoader();

  LibLoader(const LibLoader&) = delete;
  LibLoader& operator=(const LibLoader&) = delete;

  // Template method to retrieve a function pointer of the given type.
  // Example usage:
  //    using hlml_init_func_t = int (*)();
  //    hlml_init_func_t hlml_init =
  //    lib.getFunction<hlml_init_func_t>("hlml_init");
  template <typename Func>
  Func getFunction(const std::string& funcName) {
    std::lock_guard<std::mutex> guard(libMutex);
    auto it = funcPtrCache.find(funcName);
    if (it != funcPtrCache.end()) {
      return reinterpret_cast<Func>(it->second);
    }
    dlerror();
    void* ptr = dlsym(handle, funcName.c_str());
    const char* error = dlerror();
    if (error != nullptr) {
      throw std::runtime_error("dlsym error for '" + funcName + "': " + error);
    }
    funcPtrCache[funcName] = ptr;
    return reinterpret_cast<Func>(ptr);
  }

 private:
  std::string libPath_;
  void* handle;
  std::unordered_map<std::string, void*> funcPtrCache;
  std::mutex libMutex;

  void loadLibrary();
  void unloadLibrary();
};

class HlmlPowerProvider {
 public:
  explicit HlmlPowerProvider(const std::string& libPath = "libhlml.so");

  void init();

  double getUtilization();
  void loadLib(const std::string& libPath);
  void hlmlShutdown();

 private:
  hlml_device_t getDevice(unsigned int index);

  std::unique_ptr<LibLoader> lib_;
  hlml_device_t device_;
  std::string libPath_;
};

class HPUUtilizationPoller {
 public:
  explicit HPUUtilizationPoller(int interval = 1);
  ~HPUUtilizationPoller();

  void start();
  void stop();
  void reset();
  void resume();
  double getUtilization() const;

 private:
  void pollLoop();

  int interval_;
  std::atomic<bool> started_;
  double totalUtil_;
  size_t sampleCount_;
  double usage_;
  HlmlPowerProvider provider_;

  std::thread pollThread_;

  mutable std::mutex mutex_;

  std::condition_variable cv_;
  std::mutex cvMutex_;
};

using Interval = std::pair<
    std::chrono::time_point<std::chrono::high_resolution_clock>,
    std::chrono::time_point<std::chrono::high_resolution_clock>>;

class Timer {
 public:
  Timer();
  void start();
  Interval getInterval();

 private:
  std::chrono::time_point<std::chrono::high_resolution_clock> start_;
};

class StreamUtilizationMetric {
 public:
  StreamUtilizationMetric();

  void start();
  void stop();
  void reset();
  void resume();
  double getUtilization() const;

  void update(Interval idle);

 private:
  StreamUtilizationMetric(const StreamUtilizationMetric&) = delete;
  StreamUtilizationMetric& operator=(const StreamUtilizationMetric&) = delete;

  mutable std::mutex mutex_;
  bool started_;
  std::chrono::time_point<std::chrono::high_resolution_clock> startTime_;

  int64_t totalTime_ = 0;
  int64_t idleTime_ = 0;
};

class UtilizationMetrics {
 public:
  static UtilizationMetrics& getInstance();

  void start();
  void stop();
  void reset();
  void resume();
  std::pair<double, double> getUtilization();

  void update(Interval idle);

 private:
  UtilizationMetrics() = default;
  UtilizationMetrics(const UtilizationMetrics&) = delete;
  UtilizationMetrics& operator=(const UtilizationMetrics&) = delete;

  StreamUtilizationMetric streamUtilization;
  HPUUtilizationPoller poolerUtilization;
};

} // namespace synapse_helpers
