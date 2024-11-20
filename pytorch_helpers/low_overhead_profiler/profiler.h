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
#include <stdint.h>
#include <array>
#include <atomic>
#include <climits>
#include <list>
#include <mutex>
#include <string_view>
#include <thread>
#include <vector>

#define LOP_TRACE_NAMED(x, l) LOP::ScopedProfiler tracer(x, l);
#define LOP_TRACE_NAMED_FAST(x) LOP::FastScopedProfiler tracer(x);

#define LOP_TRACE_FUNC(l) LOP_TRACE_NAMED(__PRETTY_FUNCTION__, l);
#define LOP_TRACE_FUNC_FAST() LOP_TRACE_NAMED_FAST(__PRETTY_FUNCTION__);

#define NUM_OF_PIPELINE_STAGES 5

namespace LOP {

enum class PipelineStageID {
  PIPELIE_STAGE_MAIN_ID = 0,
  PIPELIE_STAGE_LOWERING_ID = 1,
  PIPELIE_STAGE_COMPILE_ID = 2,
  PIPELIE_STAGE_EXECUTE_ID = 3,
  PIPELIE_STAGE_BACKGROUND_ID = 4,
  PIPELIE_STAGE_DEFAULT_ID = -1
};

struct Event {
  uint64_t timestamp;
  uint64_t jit_cache_key;
  const char* name;
  std::string op_name;
  uint32_t thread_id;
  uint32_t cpu_id;
  uint32_t user_event_id;
  int32_t pipeline_stage_id;
  uint64_t pipeline_queue_length;
  uint64_t jit_cache_hit_count;
  bool is_begin;
  uint64_t device_queue_length;
  uint64_t stage_time;
};

struct ProfilerEngine {
  ProfilerEngine();
  ~ProfilerEngine();

  static ProfilerEngine& get_inst();

  inline bool is_enabled() {
    return this->enabled;
  }
  inline bool is_loglevel(uint32_t log_level) {
    return this->env_log_level <= log_level;
  }
  void enable() {
    this->enabled = true;
  }
  void disable() {
    this->enabled = false;
  }
  void flush();

  std::atomic<bool> enabled;
  std::atomic<bool> flushed;

  double ticks_per_ns_ratio;
  std::array<std::atomic<uint64_t>, NUM_OF_PIPELINE_STAGES> events_counter;
  std::vector<std::vector<Event>> events_table;
  uint32_t env_log_level;
};

void emit_event_fast(
    bool is_begin,
    const char* name,
    std::string_view op_name,
    int32_t pipe_stage_id = -1,
    uint64_t queue_length = 0,
    uint64_t jit_key = 0,
    uint64_t jit_cache_hit_count = 0,
    uint64_t device_queue_length = 0);

class ScopeEventImpl {
 public:
  ScopeEventImpl(
      const char* event_name,
      const std::string& op_name,
      int32_t pipeline_stage_id,
      uint64_t jit_key,
      uint64_t jit_cache_hit_count,
      uint64_t queue_length,
      uint64_t device_queue_length)
      : event_name_(event_name),
        op_name_(op_name),
        pipeline_stage_id_(pipeline_stage_id),
        jit_key_(jit_key),
        jit_cache_hit_count_(jit_cache_hit_count),
        queue_length_(queue_length),
        device_queue_length_(device_queue_length) {
    // Emit the start event
    LOP::emit_event_fast(
        true, event_name_, op_name_, pipeline_stage_id_, queue_length_);
  }

  ~ScopeEventImpl() {
    // Emit the end event
    LOP::emit_event_fast(
        false,
        event_name_,
        op_name_,
        pipeline_stage_id_,
        queue_length_,
        jit_key_,
        jit_cache_hit_count_,
        device_queue_length_);
  }

 private:
  const char* event_name_;
  const std::string op_name_;
  int32_t pipeline_stage_id_;
  uint64_t jit_key_;
  uint64_t jit_cache_hit_count_;
  uint64_t queue_length_;
  uint64_t device_queue_length_;
};

class ScopeEvent {
 public:
  ScopeEvent(
      const char* event_name,
      const std::string& op_name,
      int32_t pipeline_stage_id,
      uint64_t jit_key,
      uint64_t jit_cache_hit_count,
      uint64_t queue_length,
      uint64_t device_queue_length) {
    bool enable_lop_collection =
        GET_ENV_FLAG_NEW(PT_HPU_ENABLE_LOP_METRICS_COLLECTION) ||
        GET_ENV_FLAG_NEW(PT_HPU_ENABLE_LOP_TRACES_COLLECTION);
    if (enable_lop_collection)
      scope_event_ = std::make_unique<ScopeEventImpl>(
          event_name,
          op_name,
          pipeline_stage_id,
          jit_key,
          jit_cache_hit_count,
          queue_length,
          device_queue_length);
  }

 private:
  std::unique_ptr<ScopeEventImpl> scope_event_;
};

} // namespace LOP
