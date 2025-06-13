/**
 * Copyright (c) 2021-2025 Intel Corporation
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
#include <immintrin.h>
#include <stdint.h>
#include <sys/stat.h>
#include <unistd.h>
#include <x86intrin.h>
#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <list>
#include <map>
#include <mutex>
#include <queue>
#include <stack>
#include <string>
#include <string_view>
#include <thread>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "backend/synapse_helpers/env_flags.h"
#include "profiler.h"

// RING_SIZE HAVE TO BE POWER OF 2 - due to algorithm used later.
#define EVENT_TABLE_SIZE 10000000
#define STAGE_INITIALIZER \
  { 0, 0, 0, 0, 0 }
#define NUM_EXPONENTIAL_BUCKETS 3
#define NUM_EQUIDISTANT_BUCKETS 5
#define NUM_TOP_OPS 5
#define NUM_TIMING_DATA_CNTS 100

namespace LOP {

inline uint64_t barriered_rdtsc() {
  uint64_t tsc;
  __asm__ __volatile__(
      "rdtscp\n\t"
      "shl $32, %%rdx\n\t"
      "or %%rdx, %%rax\n\t"
      "lfence\n\t"
      : "=a"(tsc)
      :
      : "%rcx", "%rdx", "memory");

  return tsc;
}

ProfilerEngine::ProfilerEngine()
    : enabled(true),
      flushed(false),
      enable_traces(false),
      events_mutex{},
      events_counter{},
      events_table(
          NUM_OF_PIPELINE_STAGES,
          std::vector<Event>(GET_ENV_FLAG_NEW(PT_HPU_EVENT_TABLE_SIZE))) {
  char* disable_string = std::getenv("LOP_DISABLE");
  if (!disable_string || !static_cast<uint32_t>(std::stoi(disable_string))) {
    std::chrono::nanoseconds time_1_ns(1);
    std::chrono::nanoseconds time_1000_ns(1000);
    auto pre_test1 = barriered_rdtsc();
    for (int i = 0; i < 1000; ++i)
      std::this_thread::sleep_for(time_1_ns);
    auto post_test1 = barriered_rdtsc();
    auto fragmented_test_result = post_test1 - pre_test1;

    auto pre_test2 = barriered_rdtsc();
    std::this_thread::sleep_for(time_1000_ns);
    auto post_test2 = barriered_rdtsc();
    auto consolidated_test_result = post_test2 - pre_test2;

    auto overhead = (fragmented_test_result - consolidated_test_result) / 999;

    std::chrono::milliseconds time_100_ms(100);
    auto pre = barriered_rdtsc();
    std::this_thread::sleep_for(time_100_ms);
    auto post = barriered_rdtsc();
    uint64_t tsc_frequency_per_second = (post - pre - overhead) * 10;
    this->ticks_per_ns_ratio = static_cast<double>(tsc_frequency_per_second) /
        1000.0 / 1000.0 / 1000.0;

    printf("TSC freq: %luHz\n", tsc_frequency_per_second);
    printf("          %f ticks per nanosecond\n", this->ticks_per_ns_ratio);

    char* log_level_string = std::getenv("LOP_LOG_LEVEL");
    if (log_level_string) {
      this->env_log_level = static_cast<uint32_t>(std::stoi(log_level_string));
    } else {
      this->env_log_level = 3;
    }
  }
}

struct OpGroup {
  std::string name;
  uint64_t total_time;
  uint32_t count;
  double avg_time;

  OpGroup(const std::string& n, uint64_t total, uint32_t cnt)
      : name(n),
        total_time(total),
        count(cnt),
        avg_time(static_cast<double>(total) / cnt) {}
};

// Calculate adaptive cutoff (90th percentile by default)
uint64_t calculate_adaptive_cutoff(
    std::vector<uint64_t>& event_times,
    uint64_t max_time,
    double percentile = 0.9) {
  if (event_times.empty()) {
    return max_time;
  }
  std::sort(event_times.begin(), event_times.end());
  uint64_t cutoff_index =
      static_cast<uint64_t>(event_times.size() * percentile);
  return event_times[cutoff_index];
}

// create buckets: initialization and filling
void create_buckets(
    const std::vector<Event>& events,
    const std::string& target_event_name,
    uint64_t min_time,
    uint64_t max_time,
    uint64_t adaptive_cutoff,
    uint64_t jit_cache_hit_count_threshold,
    std::vector<uint64_t>& buckets,
    uint64_t& equidistant_bucket_size,
    uint64_t& exponential_range,
    uint64_t& exponential_base) {
  // Initialize equidistant and exponential buckets
  uint64_t equidistant_range = adaptive_cutoff - min_time;
  equidistant_bucket_size =
      std::max(equidistant_range / NUM_EQUIDISTANT_BUCKETS, (uint64_t)1);

  exponential_range = max_time - adaptive_cutoff;
  exponential_base =
      (exponential_range > 0) ? std::pow(2, NUM_EXPONENTIAL_BUCKETS) : 1;

  for (const auto& event : events) {
      if (event.jit_cache_hit_count > jit_cache_hit_count_threshold &&
          !event.is_begin && event.name == target_event_name) {
      uint64_t event_time = event.stage_time;
      if (event_time <= adaptive_cutoff) { // Equidistant bucket
        uint64_t offset = (event_time > min_time) ? (event_time - min_time) : 0;
        uint64_t bucket_index = std::min(
            offset / equidistant_bucket_size,
            (uint64_t)(NUM_EQUIDISTANT_BUCKETS - 1));
        buckets[bucket_index]++;
      } else { // Exponential bucket
        uint64_t offset = event_time - adaptive_cutoff;
        uint64_t bucket_index =
            NUM_EQUIDISTANT_BUCKETS; // Start after equidistant buckets
        while (offset > 0 &&
               bucket_index <
                   (NUM_EQUIDISTANT_BUCKETS + NUM_EXPONENTIAL_BUCKETS - 1)) {
          offset /= 2; // Exponential decay
          bucket_index++;
        }
        buckets[bucket_index]++;
      }
      }
  }
}

void print_histogram(
    uint64_t min_time,
    uint64_t max_time,
    const std::vector<Event>& events,
    const std::string& target_event_name,
    FILE* metrics_file) {
  std::vector<uint64_t> event_times;
  uint64_t jit_cache_hit_count_threshold =
      GET_ENV_FLAG_NEW(PT_HPU_LOP_JIT_WARM_UP_STEPS);
  for (const auto& event : events) {
      if (event.jit_cache_hit_count > jit_cache_hit_count_threshold &&
          event.name == target_event_name) {
      if (event.is_begin)
        continue;
      event_times.push_back(event.stage_time);
      }
  }
  uint64_t adaptive_cutoff = calculate_adaptive_cutoff(event_times, max_time);

  std::vector<uint64_t> buckets(
      NUM_EQUIDISTANT_BUCKETS + NUM_EXPONENTIAL_BUCKETS, 0);
  uint64_t equidistant_bucket_size, exponential_range, exponential_base;
  create_buckets(
      events,
      target_event_name,
      min_time,
      max_time,
      adaptive_cutoff,
      jit_cache_hit_count_threshold,
      buckets,
      equidistant_bucket_size,
      exponential_range,
      exponential_base);

  fprintf(metrics_file, "        ------------------------------------------\n");
  fprintf(metrics_file, "        Time Range (ns)\tFrequency\n");
  fprintf(metrics_file, "        ------------------------------------------\n");

  for (int i = 0; i < NUM_EQUIDISTANT_BUCKETS;
       ++i) { // print equidistant buckets
      uint64_t bucket_min = min_time + i * equidistant_bucket_size;
      uint64_t bucket_max = (i == NUM_EQUIDISTANT_BUCKETS - 1)
          ? adaptive_cutoff
          : (bucket_min + equidistant_bucket_size - 1);
      fprintf(
          metrics_file,
          "        [%lu, %lu]\t%lu\n",
          bucket_min,
          bucket_max,
          buckets[i]);
  }
  uint64_t prev_max = adaptive_cutoff;
  for (int i = 0; i < NUM_EXPONENTIAL_BUCKETS; ++i) {
    uint64_t bucket_min = prev_max + 1;
    uint64_t bucket_max = (i == NUM_EXPONENTIAL_BUCKETS - 1)
      ? max_time : (bucket_min + (1 << i) * exponential_range / exponential_base - 1);
    fprintf(
        metrics_file,
        "        [%lu, %lu]\t%lu\n",
        bucket_min,
        bucket_max,
        buckets[NUM_EQUIDISTANT_BUCKETS + i]);
    prev_max = bucket_max;
  }

  fprintf(metrics_file, "        ------------------------------------------\n");
}

void print_device_queue_histogram(
    uint64_t min_device_queue_len,
    uint64_t max_device_queue_len,
    const std::vector<Event>& events,
    FILE* metrics_file) {
  uint64_t device_queue_range = max_device_queue_len - min_device_queue_len + 1;
  std::vector<uint64_t> device_queue_len(device_queue_range, 0);
  uint64_t jit_cache_hit_count_threshold =
      GET_ENV_FLAG_NEW(PT_HPU_LOP_JIT_WARM_UP_STEPS);
  uint64_t total_events = 0;
  for (const auto& event : events) {
    if (event.jit_cache_hit_count > jit_cache_hit_count_threshold) {
      if (event.is_begin)
        continue;

      device_queue_len[event.device_queue_length]++;
      total_events++;
    }
  }

  fprintf(metrics_file, "   ------------------------------------------\n");
  fprintf(metrics_file, "   Device Queue Depth Percentage");
  fprintf(metrics_file, "   ------------------------------------------\n");

  for (uint64_t i = max_device_queue_len; i >= min_device_queue_len; i--) {
    float device_queue_percent =
        (static_cast<float>(device_queue_len[i]) / total_events) * 100;
    fprintf(metrics_file, "   %lu\t %f\n", i, device_queue_percent);
    if (i == 0) {
      break;
    }
  }
  fprintf(metrics_file, "   ------------------------------------------\n");
}

bool all_stages_empty(
    const std::unordered_map<int, std::unordered_map<std::string, uint64_t>>&
        stage_counter) {
  for (const auto& [pipeline_stage, events] : stage_counter) {
    if (!events.empty()) {
      return false; // Found a non-empty stage
    }
  }
  return true; // All stages are empty
}

void ProfilerEngine::flush() {
  printf("ProfilerEngine::flush at PID:%u\n", getpid());
  fflush(stdout);

  if (this->flushed) {
    printf("Tried to flush already flushed LOP. Doing nothing.");
    return;
  }

  this->enabled = false;
  std::unordered_map<int, std::unordered_map<std::string, uint64_t>>
      stage_total_time;
  std::unordered_map<int, std::unordered_map<std::string, uint64_t>>
      stage_counter;
  std::unordered_map<int, std::unordered_map<std::string, uint64_t>>
      pipeline_queue_length;
  uint64_t max_default_value = std::numeric_limits<uint64_t>::max();
  uint64_t min_default_value = std::numeric_limits<uint64_t>::min();
  std::unordered_map<int, std::unordered_map<std::string, uint64_t>> min_time;
  std::unordered_map<int, std::unordered_map<std::string, uint64_t>> max_time;
  std::unordered_map<int, std::unordered_map<std::string, uint64_t>>
      min_queue_len;
  std::unordered_map<int, std::unordered_map<std::string, uint64_t>>
      max_queue_len;
  std::unordered_map<int, std::unordered_map<std::string, uint64_t>>
      stage_mean_time;
  std::unordered_map<int, std::unordered_map<std::string, uint64_t>>
      mean_queue_length;
  std::unordered_map<int, std::unordered_map<std::string, uint64_t>>
      stage_time_variance;
  std::unordered_map<int, std::unordered_map<std::string, uint64_t>>
      stage_time_std;
  std::unordered_map<int, std::unordered_map<std::string, int64_t>>
      stage_queue_length_variance;
  std::unordered_map<int, std::unordered_map<std::string, int64_t>>
      stage_queue_length_std;
  int64_t current_index[NUM_OF_PIPELINE_STAGES] = STAGE_INITIALIZER;
  uint64_t device_total_queue_length = 0;
  uint64_t min_device_queue_len = 0;
  uint64_t max_device_queue_len = 0;
  uint64_t mean_device_queue_length = 0;
  int64_t device_queue_length_variance = 0;
  int64_t device_queue_length_std = 0;
  uint64_t total_events_execute_stage = 0;
  uint64_t wait_time = 0;
  std::vector<std::stack<Event>> event_stack(
      NUM_OF_PIPELINE_STAGES); // stack to push begin events & pop it when end
                               // events are met (nested events)
  uint64_t jit_cache_hit_count_threshold =
      GET_ENV_FLAG_NEW(PT_HPU_LOP_JIT_WARM_UP_STEPS);
  const char* base_dir_path = std::getenv("HABANA_LOGS");
  std::string dir_path = "/";
  const char* rank = std::getenv("RANK");
  if (rank != nullptr && rank[0] != '\0') {
    dir_path = "/" + std::string(rank) + "/";
  }
  int rc = mkdir((std::string(base_dir_path) + dir_path).c_str(), S_IRWXU);
  if (rc && errno != EEXIST) {
    // fail to create device/rank folder under HABANA_LOGS directory,
    // so keep under lop files under HABANA_LOGS directory.
    dir_path = "/";
  }

  std::string event_file_path = std::string(base_dir_path) + dir_path +
      "events_pid" + std::to_string(getpid()) + ".json";
  std::string metric_file_path = std::string(base_dir_path) + dir_path +
      "metrics_pid" + std::to_string(getpid()) + ".json";
  std::string timing_data_file_path = std::string(base_dir_path) + dir_path +
      "timing_data_pid" + std::to_string(getpid()) + ".json";

  auto events_file = fopen(event_file_path.c_str(), "w");
  fprintf(events_file, "{\"displayTimeUnit\": \"ns\", \"traceEvents\": [\n");

  auto metrics_file = fopen(metric_file_path.c_str(), "w");

  auto timing_data_file = fopen(timing_data_file_path.c_str(), "w");
  fprintf(timing_data_file, "{\n");
  fprintf(timing_data_file, "  \"metadata\": {\n");
  fprintf(timing_data_file, "    \"pid\": %u\n", getpid());
  fprintf(timing_data_file, "  },\n");
  // Find first event, timewise.
  uint64_t tsc_base = std::numeric_limits<uint64_t>::max();
  for (int pipeline_stage = 1; pipeline_stage < NUM_OF_PIPELINE_STAGES - 1;
       pipeline_stage++) {
    current_index[pipeline_stage] =
        this->events_counter[pipeline_stage].load(std::memory_order_acquire);
  }
  for (int pipeline_stage = 1; pipeline_stage < NUM_OF_PIPELINE_STAGES - 1;
       pipeline_stage++) {
    this->events_counter[pipeline_stage].load(std::memory_order_acquire);
    for (int i = 0; i < current_index[pipeline_stage]; ++i) {
      auto& event = this->events_table[pipeline_stage][i];
      tsc_base = std::min(tsc_base, event.timestamp);
    }
  }
  std::unordered_map<
      int,
      std::unordered_map<
          std::string,
          std::unordered_map<std::string, std::pair<uint64_t, uint32_t>>>>
      stage_op_aggregate;
  std::unordered_map<std::string, std::unordered_set<std::string>> error_events;
  std::
      unordered_map<int, std::unordered_map<std::string, std::vector<uint64_t>>>
          stage_times;
  std::
      unordered_map<int, std::unordered_map<std::string, std::vector<uint64_t>>>
          queue_lengths;
  std::unordered_map<int, std::vector<uint64_t>> device_queue_lengths;
  std::unordered_map<
      int,
      std::unordered_map<std::string, std::pair<uint64_t, uint32_t>>>
      stage_op_aggregate_pipeline_level;
  std::unordered_map<int, std::vector<std::tuple<std::string, uint64_t, uint64_t, uint32_t>>> stage_single_events;
  std::unordered_map<int, std::unordered_map<std::string, std::vector<uint64_t>>> stage_op_histograms;
  for (int pipeline_stage = 1; pipeline_stage < NUM_OF_PIPELINE_STAGES - 1;
       pipeline_stage++) {
    if (current_index[pipeline_stage] > 0) {
      for (int i = 0; i < current_index[pipeline_stage]; ++i) {
        auto& event = this->events_table[pipeline_stage][i];
        auto tsc_diff = event.timestamp - tsc_base;
        auto time_ns = static_cast<uint64_t>(
            static_cast<double>(tsc_diff) / this->ticks_per_ns_ratio);
        if (event.is_begin) {
          // Push the event onto the stack
          event_stack[pipeline_stage].push(event);
        } else {
          if (event.jit_cache_hit_count > jit_cache_hit_count_threshold) {
            if (!event_stack[pipeline_stage].empty()) {
              // Pop the corresponding begin event from the stack
              auto begin_event = event_stack[pipeline_stage].top();
              event_stack[pipeline_stage].pop();
              // Check if the operation names match
              if (begin_event.name == event.name) {
                auto begin_event_tsc_diff = begin_event.timestamp - tsc_base;
                auto begin_event_time_ns = static_cast<uint64_t>(
                    static_cast<double>(begin_event_tsc_diff) /
                    this->ticks_per_ns_ratio);
                auto stage_time = time_ns - begin_event_time_ns;
                // Calculate wait time if the event is "PipelineQueueWaitTime"
                if (std::string(event.name) == "PipelineQueueWaitTime") {
                  wait_time = stage_time;
                } else {
                  // Deduct wait time from stage time if applicable
                  stage_time -= wait_time;
                  wait_time = 0;
                }
                event.stage_time = stage_time;
                stage_times[pipeline_stage][event.name].push_back(stage_time);
                // Initialize min_time, max_time, min_queue_len, max_queue_len
                // if not already set
                if (min_time[pipeline_stage].find(event.name) ==
                    min_time[pipeline_stage].end()) {
                  min_time[pipeline_stage][event.name] = max_default_value;
                }
                if (max_time[pipeline_stage].find(event.name) ==
                    max_time[pipeline_stage].end()) {
                  max_time[pipeline_stage][event.name] = min_default_value;
                }
                if (min_queue_len[pipeline_stage].find(event.name) ==
                    min_queue_len[pipeline_stage].end()) {
                  min_queue_len[pipeline_stage][event.name] = max_default_value;
                }
                if (max_queue_len[pipeline_stage].find(event.name) ==
                    max_queue_len[pipeline_stage].end()) {
                  max_queue_len[pipeline_stage][event.name] = min_default_value;
                }

                if (stage_time < min_time[pipeline_stage][event.name]) {
                  min_time[pipeline_stage][event.name] = stage_time;
                }
                if (stage_time > max_time[pipeline_stage][event.name]) {
                  max_time[pipeline_stage][event.name] = stage_time;
                }

                stage_op_aggregate[pipeline_stage][event.name][event.op_name]
                    .first += stage_time;
                stage_op_aggregate[pipeline_stage][event.name][event.op_name]
                    .second++;
                // needed to capture stats at pipeline level for ops timining
                // analysis
                if (std::string(event.name) != "PipelineQueueWaitTime") {
                  stage_op_aggregate_pipeline_level[pipeline_stage]
                                                   [event.op_name]
                                                       .first += stage_time;
                  stage_op_aggregate_pipeline_level[pipeline_stage]
                                                   [event.op_name]
                                                       .second++;
                  stage_single_events[pipeline_stage].emplace_back(
                      event.op_name,
                      stage_time,
                      time_ns - stage_time,
                      event.thread_id);
                  stage_op_histograms[pipeline_stage][event.op_name].push_back(
                      stage_time);
                }
                stage_total_time[pipeline_stage][event.name] += stage_time;
                auto queue_length = event.pipeline_queue_length;
                queue_lengths[pipeline_stage][event.name].push_back(
                    queue_length);
                if (queue_length < min_queue_len[pipeline_stage][event.name]) {
                  min_queue_len[pipeline_stage][event.name] = queue_length;
                }
                if (queue_length > max_queue_len[pipeline_stage][event.name]) {
                  max_queue_len[pipeline_stage][event.name] = queue_length;
                }
                pipeline_queue_length[pipeline_stage][event.name] +=
                    queue_length;
                if (pipeline_stage ==
                    static_cast<int>(
                        LOP::PipelineStageID::
                            PIPELIE_STAGE_EXECUTE_ID)) { // Collect samples only
                                                         // at execute stage
                  auto device_queue_length = event.device_queue_length;
                  device_queue_lengths[pipeline_stage].push_back(
                      device_queue_length);
                  if (device_queue_length < min_device_queue_len) {
                    min_device_queue_len = device_queue_length;
                  } else if (device_queue_length > max_device_queue_len) {
                    max_device_queue_len = device_queue_length;
                  }
                  device_total_queue_length += device_queue_length;
                  total_events_execute_stage += 1;
                }
                stage_counter[pipeline_stage][event.name] += 1;
              } else {
                error_events["Mismatched End"].insert(event.name);
              }
            } else {
              error_events["End Without Begin"].insert(event.name);
            }
          } else {
            if (!event_stack[pipeline_stage].empty()) {
              event_stack[pipeline_stage]
                  .pop(); // this for specific case when begin event of nested
                          // event is missing
            }
          }
        }
      }

      for (const auto& [event_name, count] : stage_counter[pipeline_stage]) {
        if (count > 0) {
          stage_mean_time[pipeline_stage][event_name] =
              stage_total_time[pipeline_stage][event_name] / count;
          mean_queue_length[pipeline_stage][event_name] =
              pipeline_queue_length[pipeline_stage][event_name] / count;
          const auto& saved_stage_times =
              stage_times[pipeline_stage][event_name];
          const auto& saved_queue_lengths =
              queue_lengths[pipeline_stage][event_name];
          for (uint64_t saved_stage_time : saved_stage_times) {
            int64_t time_diff =
                saved_stage_time - stage_mean_time[pipeline_stage][event_name];
            stage_time_variance[pipeline_stage][event_name] +=
                time_diff * time_diff;
          }
          for (uint64_t saved_queue_length : saved_queue_lengths) {
            int64_t queue_length_diff = saved_queue_length -
                mean_queue_length[pipeline_stage][event_name];
            stage_queue_length_variance[pipeline_stage][event_name] +=
                queue_length_diff * queue_length_diff;
          }

          stage_time_variance[pipeline_stage][event_name] /= (count - 1);
          stage_time_std[pipeline_stage][event_name] =
              static_cast<int64_t>(sqrt(static_cast<double>(
                  stage_time_variance[pipeline_stage][event_name])));
          stage_queue_length_variance[pipeline_stage][event_name] /=
              (count - 1);
          stage_queue_length_std[pipeline_stage][event_name] =
              static_cast<int64_t>(sqrt(static_cast<double>(
                  stage_queue_length_variance[pipeline_stage][event_name])));
        }
      }

      // Device queue is handled at pipeline stage level not event level
      if (pipeline_stage == static_cast<int>(LOP::PipelineStageID::PIPELIE_STAGE_EXECUTE_ID)) {
        if (total_events_execute_stage > 0) {
          mean_device_queue_length =
              device_total_queue_length / total_events_execute_stage;
          const auto& saved_device_queue_lengths =
              device_queue_lengths[pipeline_stage];
          for (uint64_t saved_device_queue_length :
               saved_device_queue_lengths) {
            int64_t device_queue_length_diff =
                saved_device_queue_length - mean_device_queue_length;
            device_queue_length_variance +=
                device_queue_length_diff * device_queue_length_diff;
          }

          device_queue_length_variance =
              device_queue_length_variance / (total_events_execute_stage - 1);
          device_queue_length_std = static_cast<int64_t>(
              sqrt(static_cast<double>(device_queue_length_variance)));
        }
      }
    }
  }

   fprintf(timing_data_file, "}\n");
   fclose(timing_data_file);

  if (all_stages_empty(stage_counter)) {
    printf(
        "ProfilerEngine::Profiling failed. Check PT_HPU_LOP_JIT_WARM_UP_STEPS / no meaningful event got captured\n");
    remove(event_file_path.c_str());
    remove(metric_file_path.c_str());
    remove(timing_data_file_path.c_str());
  } else {
    // If any meaningful event gets captured then only further de-dup ops cal
    // will happen
    std::unordered_map<
        int,
        std::unordered_map<std::string, std::vector<OpGroup>>>
        sorted_ops_by_stage;
    for (int pipeline_stage = 1; pipeline_stage < NUM_OF_PIPELINE_STAGES - 1;
         pipeline_stage++) {
      if (current_index[pipeline_stage] == 0)
        continue;

      for (const auto& [event_name, ops_map] :
           stage_op_aggregate[pipeline_stage]) {
        std::vector<OpGroup> aggregated_ops;
        for (const auto& [op_name, stats] : ops_map) {
          aggregated_ops.emplace_back(op_name, stats.first, stats.second);
        }

        std::sort(
            aggregated_ops.begin(),
            aggregated_ops.end(),
            [](const OpGroup& a, const OpGroup& b) {
              return a.avg_time > b.avg_time;
            });

        sorted_ops_by_stage[pipeline_stage][event_name] =
            std::move(aggregated_ops);
      }
    }

    // Dumping into timing-data file
   timing_data_file = fopen(timing_data_file_path.c_str(), "a");
   for (int pipeline_stage = 1; pipeline_stage < NUM_OF_PIPELINE_STAGES - 1; pipeline_stage++) {
     if (current_index[pipeline_stage] == 0) continue;

     const char* stage_name = "";
     if (pipeline_stage == static_cast<int>(LOP::PipelineStageID::PIPELIE_STAGE_LOWERING_ID)) {
       stage_name = "lowering";
     } else if (pipeline_stage == static_cast<int>(LOP::PipelineStageID::PIPELIE_STAGE_COMPILE_ID)) {
       stage_name = "compile";
     } else {
       stage_name = "execute";
     }
     fprintf(timing_data_file, "  \"%s\": {\n", stage_name);
     // 1. sort_by_avg_time
     fprintf(timing_data_file, "    \"sort_by_avg_time\": [\n");
     std::vector<std::pair<std::string, std::pair<uint64_t, uint32_t>>>
         avg_time_ops(
             stage_op_aggregate_pipeline_level[pipeline_stage].begin(),
             stage_op_aggregate_pipeline_level[pipeline_stage].end());
     std::sort(
         avg_time_ops.begin(),
         avg_time_ops.end(),
         [](const auto& a, const auto& b) {
           return (a.second.first / a.second.second) >
               (b.second.first / b.second.second);
         });
     for (size_t i = 0; i < avg_time_ops.size(); ++i) {
       const auto& op = avg_time_ops[i];
       fprintf(timing_data_file, "      {\"op_name\": \"%s\", \"avg_time_ns\": %lu, \"count\": %u, \"total_time\": %lu}",
               op.first.c_str(), op.second.first / op.second.second, op.second.second, op.second.first);
       if (i != avg_time_ops.size() - 1) fprintf(timing_data_file, ",");
       fprintf(timing_data_file, "\n");
     }
     fprintf(timing_data_file, "    ],\n");
     // 2. sort_by_single_time
     fprintf(timing_data_file, "    \"sort_by_single_time\": [\n");
     auto& single_events = stage_single_events[pipeline_stage];
     size_t report_count =
         std::min<size_t>(NUM_TIMING_DATA_CNTS, single_events.size());
     std::partial_sort(
         single_events.begin(),
         single_events.begin() + report_count,
         single_events.end(),
         [](const auto& a, const auto& b) {
           return std::get<1>(a) > std::get<1>(b);
         });
     for (size_t i = 0; i < report_count; ++i) {
       const auto& event = single_events[i];
       fprintf(timing_data_file, "      {\"op_name\": \"%s\", \"time_ns\": %lu, \"tid\": %u, \"start_ns\": %lu}",
               std::get<0>(event).c_str(), std::get<1>(event), std::get<3>(event), std::get<2>(event));
       if (i != report_count - 1) fprintf(timing_data_file, ",");
       fprintf(timing_data_file, "\n");
     }
     fprintf(timing_data_file, "    ],\n");
     // 3. sort_by_total_time
     fprintf(timing_data_file, "    \"sort_by_total_time\": [\n");
     for (size_t i = 0; i < avg_time_ops.size(); ++i) {
       const auto& op = avg_time_ops[i];
       fprintf(timing_data_file, "      {\"op_name\": \"%s\", \"total_time_ns\": %lu, \"count\": %u}",
               op.first.c_str(), op.second.first, op.second.second);
       if (i != avg_time_ops.size() - 1) fprintf(timing_data_file, ",");
       fprintf(timing_data_file, "\n");
     }
     fprintf(timing_data_file, "    ],\n");
     // 4. all_ops_histogram
     fprintf(timing_data_file, "    \"all_ops_histogram\": {\n");
     bool first_op = true;
     for (const auto& [op_name, times] : stage_op_histograms[pipeline_stage]) {
       if (!first_op) fprintf(timing_data_file, ",\n");
       first_op = false;
       fprintf(timing_data_file, "      \"%s\": [", op_name.c_str());
       // sampling at most NUM_TIMING_DATA_CNTS, which has a default value of 100.
       size_t sample_count = std::min<size_t>(NUM_TIMING_DATA_CNTS, times.size());
       size_t step = times.size() / sample_count;
       if (step == 0) step = 1;

       for (size_t i = 0; i < times.size(); i += step) {
         if (i > 0) fprintf(timing_data_file, ", ");
         fprintf(timing_data_file, "%lu", times[i]);
         if (i + step >= times.size()) break;
       }
       fprintf(timing_data_file, "]");
     }
     fprintf(timing_data_file, "\n    }\n");

     // end current stage
     if (pipeline_stage != NUM_OF_PIPELINE_STAGES - 2) {
       fprintf(timing_data_file, "  },\n");
     } else {
       fprintf(timing_data_file, "  }\n");
     }
   }

    // Metrics are getting dumped to metric jSON file
    fprintf(
        metrics_file,
        " Total number of events = %lu \n",
        this->events_counter[1].load(std::memory_order_acquire) +
            this->events_counter[2].load(std::memory_order_acquire) +
            this->events_counter[3].load(std::memory_order_acquire));
    std::string stage_name;
    for (int pipeline_stage = 1; pipeline_stage < NUM_OF_PIPELINE_STAGES - 1;
         pipeline_stage++) {
      if (!stage_counter[pipeline_stage].empty()) {
        if (pipeline_stage ==
            static_cast<int>(LOP::PipelineStageID::PIPELIE_STAGE_LOWERING_ID)) {
          stage_name = "Lowering";
          fprintf(metrics_file, "\n LOWERING STAGE \n");
        } else if (
            pipeline_stage ==
            static_cast<int>(LOP::PipelineStageID::PIPELIE_STAGE_COMPILE_ID)) {
          stage_name = "Compile";
          fprintf(metrics_file, "\n COMPILE STAGE \n");
        } else {
          stage_name = "Execute";
          fprintf(metrics_file, "\n EXECUTE STAGE \n");
        }
        fprintf(metrics_file, " ============== \n");
        for (const auto& [event_name, count] : stage_counter[pipeline_stage]) {
          if (count > 0) {
            fprintf(metrics_file, "   %s :\n", event_name.c_str());
            fprintf(
                metrics_file,
                "     average time(ns) = %lu \n",
                stage_total_time[pipeline_stage][event_name] / count);
            fprintf(
                metrics_file,
                "     max time(ns) = %lu \n",
                max_time[pipeline_stage][event_name]);
            fprintf(
                metrics_file,
                "     min time(ns) = %lu \n",
                min_time[pipeline_stage][event_name]);
            fprintf(
                metrics_file,
                "     standard deviation time(ns) = %lu \n",
                stage_time_std[pipeline_stage][event_name]);
            fprintf(
                metrics_file,
                "     average queue length = %lu \n",
                mean_queue_length[pipeline_stage][event_name]);
            fprintf(
                metrics_file,
                "     max queue length = %lu \n",
                max_queue_len[pipeline_stage][event_name]);
            fprintf(
                metrics_file,
                "     min queue length = %lu \n",
                min_queue_len[pipeline_stage][event_name]);
            fprintf(
                metrics_file,
                "     standard deviation queue length = %lu \n",
                stage_queue_length_std[pipeline_stage][event_name]);
            fprintf(metrics_file, "     Top 5 ops with highest time: \n");
            for (size_t i = 0;
                 i <
                 std::min<size_t>(
                     NUM_TOP_OPS,
                     sorted_ops_by_stage[pipeline_stage][event_name].size());
                 ++i) {
              const auto& op =
                  sorted_ops_by_stage[pipeline_stage][event_name][i];
              fprintf(
                  metrics_file,
                  "        Op Name: %s, Avg Time: %.1f, Count: %u\n",
                  op.name.c_str(),
                  op.avg_time,
                  op.count);
            }
            fprintf(
                metrics_file,
                "     Histogram for %s Stage\n",
                stage_name.c_str());
            print_histogram(
                min_time[pipeline_stage][event_name],
                max_time[pipeline_stage][event_name],
                this->events_table[pipeline_stage],
                event_name,
                metrics_file);
          }
        }

        if (pipeline_stage ==
            static_cast<int>(LOP::PipelineStageID::PIPELIE_STAGE_EXECUTE_ID)) {
          fprintf(metrics_file, "\n   DEVICE QUEUE \n");
          fprintf(metrics_file, "   ============ \n");
          fprintf(
              metrics_file,
              "   Min device queue length = %lu \n",
              min_device_queue_len);
          fprintf(
              metrics_file,
              "   Max device queue length = %lu \n",
              max_device_queue_len);
          fprintf(
              metrics_file,
              "   Mean device queue length = %lu \n",
              mean_device_queue_length);
          fprintf(
              metrics_file,
              "   Standard deviation device queue length = %lu \n",
              device_queue_length_std);
          print_device_queue_histogram(
              min_device_queue_len,
              max_device_queue_len,
              this->events_table[pipeline_stage],
              metrics_file);
        }
      }
    }

    // Events are getting dumped to event jSON file
    if (GET_ENV_FLAG_NEW(PT_HPU_ENABLE_LOP_TRACES_COLLECTION) ||
        this->enable_traces) {
    uint64_t time_base_ns = static_cast<uint64_t>(
        static_cast<double>(tsc_base) / this->ticks_per_ns_ratio);
    auto pid = getpid();
    for (int pipeline_stage = 1; pipeline_stage < NUM_OF_PIPELINE_STAGES - 1;
         pipeline_stage++) {
      for (int i = 0; i < current_index[pipeline_stage]; ++i) {
        auto& event = this->events_table[pipeline_stage][i];
        auto tsc_diff = event.timestamp - tsc_base;
        auto time_ns = static_cast<uint64_t>(
            static_cast<double>(tsc_diff) / this->ticks_per_ns_ratio);

        std::string name;
        if (event.name) {
          name = event.name + std::string("_") + event.op_name;
        } else {
          name = "custom event id " +
              std::to_string(
                     event.user_event_id); // no where user_event_id is set
        }

        if (name.find("_internal_counter_event") != std::string::npos) {
          std::string counter_name = std::to_string(event.thread_id);

          fprintf(
              events_file,
              "{"
              "\"tid\":%s,"
              "\"pid\":\"counters\","
              "\"ts\":%lu.%03lu,"
              "\"name\":\"%s\","
              "\"ph\":\"C\","
              "\"args\":{"
              "\"ctr\":%u"
              "}"
              "},\n",
              counter_name.c_str(),
              time_ns / 1000,
              time_ns % 1000,
              counter_name.c_str(),
              event.user_event_id);
        } else if (event.is_begin) {
          if (i == 0 && pipeline_stage == 1) {
              fprintf(
                  events_file,
                  "{"
                  "\"tid\":%lu,"
                  "\"pid\":%u,"
                  "\"ts\":%lu.%03lu,"
                  "\"name\":\"%s\","
                  "\"ph\":\"B\","
                  "\"args\":{"
                  "\"begin_cpu\":%u,"
                  "\"time_base\":%lu"
                  "}"
                  "},\n",
                  event.thread_id,
                  pid,
                  time_ns / 1000,
                  time_ns % 1000,
                  name.c_str(),
                  event.cpu_id,
                  time_base_ns);
          } else {
              fprintf(
                  events_file,
                  "{"
                  "\"tid\":%lu,"
                  "\"pid\":%u,"
                  "\"ts\":%lu.%03lu,"
                  "\"name\":\"%s\","
                  "\"ph\":\"B\","
                  "\"args\":{"
                  "\"begin_cpu\":%u"
                  "}"
                  "},\n",
                  event.thread_id,
                  pid,
                  time_ns / 1000,
                  time_ns % 1000,
                  name.c_str(),
                  event.cpu_id);
          }
        } else {
          fprintf(
              events_file,
              "{"
              "\"tid\":%lu,"
              "\"pid\":%u,"
              "\"ts\":%lu.%03lu,"
              "\"name\":\"%s\","
              "\"ph\":\"E\","
              "\"args\":{"
              "\"end_cpu\":%u"
              "}"
              "},\n",
              event.thread_id,
              pid,
              time_ns / 1000,
              time_ns % 1000,
              name.c_str(),
              event.cpu_id);
        }
      }
    }
    fprintf(events_file, "{}]}");
    }
  }
 fclose(metrics_file);
 fclose(events_file);
 this->flushed = true;
 if (!error_events.empty()) {
    for (const auto& category : error_events) {
    const std::string& category_name = category.first;
    const std::unordered_set<std::string>& events = category.second;
    printf("Category: %s\n", category_name.c_str());
    printf("Total Error Events: %zu\n", events.size());
    printf("Event Names: ");
    for (const auto& event_name : events) {
      printf("%s ", event_name.c_str());
    }
    printf("\n\n");
    }
 }
 printf("ProfilerEngine::flush finished\n");
 fflush(stdout);
}

ProfilerEngine::~ProfilerEngine() {
  printf("ProfilerEngine::~ProfilerEngine at PID:%u\n", getpid());
  fflush(stdout);

  if (!this->flushed) {
    this->flush();
  }

  printf("ProfilerEngine::~ProfilerEngine finished\n");
  fflush(stdout);
}

ProfilerEngine& ProfilerEngine::get_inst(bool dump_traces) {
  static ProfilerEngine inst;
  if (dump_traces) {
    inst.enable_traces = true; // Set enable_traces based on dump_traces
  } else {
    inst.enable_traces = false;
  }
  return inst;
}

void emit_event_fast(
    bool is_begin,
    const char* name,
    std::string_view op_name,
    int32_t pipe_stage_id,
    uint64_t queue_length,
    uint64_t jit_key,
    uint64_t jit_cache_hit_count,
    uint64_t device_queue_length) {
  bool enable_lop_collection =
      GET_ENV_FLAG_NEW(PT_HPU_ENABLE_LOP_METRICS_COLLECTION) ||
      GET_ENV_FLAG_NEW(PT_HPU_ENABLE_LOP_TRACES_COLLECTION);

  if (!enable_lop_collection) {
    return;
  }

  auto& profiler_engine_instance = ProfilerEngine::get_inst();
  if (!profiler_engine_instance.is_enabled()) {
    return;
  }

  // This function is faster because it use lighter RDTSC instead of RDTSCP.
  // It's latency is lower but it also does not make partial barrier like RDTSCP
  // does. Caveat is that we no longer know the CPUs on which we took TSC. It
  // also does not check whether event buffer is correct (so it's less
  // reliable). It does not support log levels.
  std::lock_guard<std::mutex> lock(
      profiler_engine_instance.events_mutex[pipe_stage_id]);
  int64_t current_index =
      profiler_engine_instance.events_counter[pipe_stage_id].load(
          std::memory_order_acquire);
  if (current_index < GET_ENV_FLAG_NEW(PT_HPU_EVENT_TABLE_SIZE)) {
    uint64_t tsc = _rdtsc();

    Event& new_event =
        profiler_engine_instance.events_table[pipe_stage_id][current_index];
    new_event.timestamp = tsc;
    new_event.name = name;
    new_event.op_name = op_name;
    new_event.thread_id = pthread_self();
    new_event.is_begin = is_begin;
    new_event.pipeline_stage_id = pipe_stage_id;
    new_event.pipeline_queue_length = queue_length;
    new_event.jit_cache_key = jit_key;
    new_event.jit_cache_hit_count = jit_cache_hit_count;
    new_event.device_queue_length = device_queue_length;

    profiler_engine_instance.events_counter[pipe_stage_id].fetch_add(
        1, std::memory_order_release);
  }
}
}; // namespace LOP
