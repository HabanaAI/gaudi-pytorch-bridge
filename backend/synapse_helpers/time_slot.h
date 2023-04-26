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
#pragma once

#include <absl/types/optional.h>
#include <synapse_api.h>
#include <synapse_api_types.h>
#include <synapse_common_types.h>
#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include "backend/synapse_helpers/device_types.h"
#include "backend/synapse_helpers/event_handle_cache.h"
#include "habana_helpers/logging.h"

#include <c10/util/Backtrace.h>
#include <iostream>

namespace synapse_helpers {

class TimeSlotBase {
 public:
  virtual absl::optional<uint64_t> getTime() const = 0;
  virtual ~TimeSlotBase() = default;
};

class TimeSlot : public TimeSlotBase {
 public:
  TimeSlot(
      CachedEventHandle&& event_start,
      CachedEventHandle&& event_end,
      synStreamHandle handle)
      : stream_handle_(handle),
        event_start_(std::move(event_start)),
        event_end_(std::move(event_end)) {}
  ~TimeSlot() = default;

  TimeSlot(const TimeSlot&) = delete;
  TimeSlot& operator=(const TimeSlot&) = delete;

  void Start() {
    auto status = synEventRecord(event_start_.get(), stream_handle_);
    if (status != synSuccess)
      PT_SYNHELPER_FATAL(
          Logger::formatStatusMsg(status), "Failed to record start event");
  };
  void Stop() {
    auto status = synEventRecord(event_end_.get(), stream_handle_);
    if (status != synSuccess)
      PT_SYNHELPER_FATAL(
          Logger::formatStatusMsg(status), "Failed to record end event");
  };

  absl::optional<uint64_t> getTime() const override {
    uint64_t elapseTime{};
    // Time is reported in nanoseconds
    auto status =
        synEventElapsedTime(&elapseTime, event_start_.get(), event_end_.get());
    return (status == synSuccess) ? elapseTime : absl::optional<uint64_t>{};
  }

 private:
  synStreamHandle stream_handle_;
  CachedEventHandle event_start_;
  CachedEventHandle event_end_;
};

class TimeScope {
 public:
  TimeScope(const std::shared_ptr<TimeSlot>& ts) {
    if (ts) {
      ts_ = ts;
      ts_->Start();
    }
  };

  TimeScope(std::shared_ptr<TimeSlot>&& ts) {
    if (ts) {
      ts_ = std::move(ts);
      ts_->Start();
    }
  };

  virtual ~TimeScope() {
    if (ts_)
      ts_->Stop();
  };

  TimeScope(const TimeSlot&) = delete;
  TimeScope& operator=(const TimeScope&) = delete;
  TimeScope(TimeScope&&) = default;
  TimeScope& operator=(TimeScope&&) = default;

 private:
  std::shared_ptr<TimeSlot> ts_{};
};

} // namespace synapse_helpers
