/*******************************************************************************
 * Copyright (C) 2023-2024 Habana Labs, Ltd. an Intel Company
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
#include <stdint.h>
#include <array>
#include <atomic>
#include <climits>
#include <list>
#include <mutex>
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
  uint32_t thread_id;
  uint32_t cpu_id;
  uint32_t user_event_id;
  int32_t pipeline_stage_id;
  uint64_t pipeline_queue_length;
  uint64_t jit_cache_hit_count;
  bool is_begin;
  uint64_t device_queue_length;
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
    int32_t pipe_stage_id = -1,
    uint64_t queue_length = 0,
    uint64_t jit_key = 0,
    uint64_t jit_cache_hit_count = 0,
    uint64_t device_queue_length = 0);

} // namespace LOP
