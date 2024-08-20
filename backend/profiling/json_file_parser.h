/******************************************************************************
 * Copyright (C) 2023 Habana Labs, Ltd. an Intel Company
 * All Rights Reserved.
 *
 * Unauthorized copying of this file or any element(s) within it, via any medium
 * is strictly prohibited.
 * This file contains Habana Labs, Ltd. proprietary and confidential information
 * and is subject to the confidentiality and license agreements under which it
 * was provided.
 *
 ******************************************************************************
 */

#pragma once
#include <chrono>
#include <fstream>
#include <sstream>
#include <string>
#include <string_view>
#include "backend/profiling/profiling.h"
#include "backend/synapse_helpers/env_flags.h"
#include "nlohmann/json.hpp"

namespace habana {
namespace profile {

template <class ClockT>
inline int64_t timeSinceEpoch(const std::chrono::time_point<ClockT>& t) {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             t.time_since_epoch())
      .count();
}

// std::chrono header start
#ifdef _GLIBCXX_USE_C99_STDINT_TR1
#define _KINETO_GLIBCXX_CHRONO_INT64_T int64_t
#elif defined __INT64_TYPE__
#define _KINETO_GLIBCXX_CHRONO_INT64_T __INT64_TYPE__
#else
#define _KINETO_GLIBCXX_CHRONO_INT64_T long long
#endif
// std::chrono header end

// There are tools like Chrome Trace Viewer that uses double to represent
// each element in the timeline. Double has a 53 bit mantissa to support
// up to 2^53 significant digits (up to 9007199254740992). This holds at the
// nanosecond level, about 3 months and 12 days. So, let's round base time to
// 3 months intervals, so we can still collect traces across ranks relative
// to each other.
// A month is 2629746, so 3 months is 7889238.
using _trimonths =
    std::chrono::duration<_KINETO_GLIBCXX_CHRONO_INT64_T, std::ratio<7889238>>;
#undef _GLIBCXX_CHRONO_INT64_T

class ChromeTraceBaseTime {
 public:
  ChromeTraceBaseTime() = default;
  static ChromeTraceBaseTime& singleton();
  void init() {
    get();
  }
  int64_t get() {
    // Make all timestamps relative to 3 month intervals.
    static int64_t base_time =
        timeSinceEpoch(std::chrono::time_point<std::chrono::system_clock>(
            std::chrono::floor<_trimonths>(std::chrono::system_clock::now())));
    return base_time;
  }
};

class JsonFileParser : public TraceSink {
 public:
  JsonFileParser() = default;
  ~JsonFileParser() override = default;

  void addActivity(
      const Activity& activity,
      const std::optional<RecipeInfo>& recipeInfo,
      uint64_t time,
      bool begin) override;

  void addCompleteActivity(
      const Activity& activity,
      const std::optional<RecipeInfo>& recipeInfo,
      uint64_t start,
      uint64_t end) override;

  void addFlowEvent(
      std::string_view name,
      std::string_view cat,
      const Flow& start,
      const Flow& finish) override;

  void addMemoryEvent(
      int64_t device,
      int64_t resource,
      int64_t time,
      uint64_t addr,
      int64_t bytes,
      int64_t device_id,
      int64_t device_type,
      uint64_t total_allocated,
      uint64_t total_reserved) override;

  void addDevice(std::string_view name, int64_t id) override;

  void addResource(
      std::string_view name,
      int64_t deviceId,
      int64_t id,
      int64_t sortIndex) override;

  virtual void addDeviceDetails(
      const std::unordered_map<std::string, std::string>& device_details)
      override;

  virtual void addDeviceDetails(
      const std::unordered_map<std::string, int64_t>& device_details) override;

  nlohmannV340::json& getCreateArray(
      nlohmannV340::json& json_file,
      const std::string_view& name);

  void merge(const std::string_view& path);

  /**
   * @brief Clean-up data in the object.
   *
   * Clean-up and reset data containers and variables in the object.
   */
  virtual void clear();

  virtual int64_t transToRelativeTime(int64_t time) override;

 private:
  void addToEvents(const nlohmannV340::json& obj);

  std::string toHex(uint64_t handle);

  std::string splitNs(uint64_t value);

  nlohmannV340::json constructEvent(
      const Activity& activity,
      const std::optional<RecipeInfo>& recipeInfo,
      int64_t ts);

  nlohmannV340::json constructFlow(
      std::string_view name,
      std::string_view cat,
      int64_t pid,
      int64_t tid,
      int64_t ts,
      bool start);

  nlohmannV340::json constructMemoryEvent(
      int64_t pid,
      int64_t tid,
      int64_t ts,
      uint64_t addr,
      int64_t bytes,
      int64_t device_id,
      int64_t device_type,
      uint64_t total_allocated,
      uint64_t total_reserved);

  std::string mapActivityTypeToString(ActivityType type);

  uint64_t flow_id_counter_ = 0;
  uint64_t profiler_event_index_ = 0;
  nlohmannV340::json traceEvents_;
  nlohmannV340::json deviceProperties_;
};
}; // namespace profile
}; // namespace habana
