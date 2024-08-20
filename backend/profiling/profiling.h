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

#include <strings.h>
#include <list>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace habana {
namespace profile {

enum class ActivityType { KERNEL, RUNTIME, MEMCPY, MEMSET, CPU_INSTANT_EVENT };
enum class TraceSourceVariant : unsigned {
  SYNAPSE_PROFILER = 0,
  SYNAPSE_LOGGER = 10000,
  BRIDGE_LOGS = 20000,
  MEMORY_LOGS = 30000
};

class TraceSource;

struct Activity {
  std::string_view name;
  std::unordered_map<std::string, std::string> args;
  ActivityType type;
  int64_t device;
  int64_t resource;
};

struct RecipeInfo {
  uint16_t recipeId;
  std::string_view recipeName;
  uint64_t streamHandle;
  uint64_t eventHandle;
};

struct Flow {
  int64_t device;
  int64_t resource;
  int64_t time;
};

class TraceSink {
 public:
  virtual ~TraceSink(){};
  virtual void addActivity(
      const Activity& activity,
      const std::optional<RecipeInfo>& recipeInfo,
      uint64_t time,
      bool begin) = 0;

  virtual void addCompleteActivity(
      const Activity& activity,
      const std::optional<RecipeInfo>& recipeInfo,
      uint64_t start,
      uint64_t end) = 0;

  virtual void addFlowEvent(
      std::string_view name,
      std::string_view cat,
      const Flow& start,
      const Flow& finish) = 0;

  virtual void addMemoryEvent(
      int64_t device,
      int64_t resource,
      int64_t time,
      uint64_t addr,
      int64_t bytes,
      int64_t device_id,
      int64_t device_type,
      uint64_t total_allocated,
      uint64_t total_reserved) = 0;

  virtual void addDevice(std::string_view name, int64_t device) = 0;

  virtual void addResource(
      std::string_view name,
      int64_t device,
      int64_t resource,
      int64_t sort_index = -1) = 0;

  virtual void addDeviceDetails(
      const std::unordered_map<std::string, std::string>& device_details) = 0;

  virtual void addDeviceDetails(
      const std::unordered_map<std::string, int64_t>& device_details) = 0;

  /**
   * @brief Clean-up data in the object.
   *
   * Clean-up and reset data containers and variables in the object.
   */
  virtual void clear() = 0;

  virtual int64_t transToRelativeTime(int64_t time) = 0;
};

class TraceSource {
 public:
  virtual ~TraceSource(){};
  virtual void start(TraceSink& output) = 0;
  virtual void stop() = 0;
  virtual void extract(TraceSink& output) = 0;
  virtual TraceSourceVariant get_variant() = 0;
  virtual void set_offset(unsigned offset) = 0;
};

class Profiler {
 public:
  Profiler(TraceSink& sink);
  void init_sources(
      bool synapse_logger,
      bool bridge,
      bool memory,
      const std::vector<std::string>& mandatory_events);
  void start();
  void stop();

 private:
  TraceSink& trace_sink_;
  std::list<std::unique_ptr<TraceSource>> trace_sources_;
};

namespace bridge {
void trace_start(std::string_view id);
void trace_end(std::string_view id);
bool is_enabled(std::string_view id);
}; // namespace bridge

}; // namespace profile
}; // namespace habana