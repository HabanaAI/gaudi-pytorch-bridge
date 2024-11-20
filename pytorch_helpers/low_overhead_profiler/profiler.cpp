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
#include <immintrin.h>
#include <stdint.h>
#include <stdio.h>
#include <unistd.h>
#include <x86intrin.h>
#include <climits>
#include <cmath>
#include <cstring>
#include <iostream>
#include <list>
#include <map>
#include <mutex>
#include <queue>
#include <string>
#include <string_view>
#include <thread>
#include <tuple>
#include <vector>

#include "backend/synapse_helpers/env_flags.h"
#include "profiler.h"

// RING_SIZE HAVE TO BE POWER OF 2 - due to algorithm used later.
#define EVENT_TABLE_SIZE 10000000
#define STAGE_INITIALIZER \
  { 0, 0, 0, 0, 0 }
#define NUM_BUCKETS 5

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

struct Compare {
  bool operator()(
      const std::tuple<std::string, int>& a,
      const std::tuple<std::string, int>& b) {
    return std::get<1>(a) < std::get<1>(b);
  }
};

void print_histogram(
    uint64_t min_time,
    uint64_t max_time,
    const std::vector<Event>& events,
    FILE* metrics_file) {
  uint64_t bucket_size = (max_time - min_time) / NUM_BUCKETS;
  std::vector<uint64_t> buckets(NUM_BUCKETS, 0);
  uint64_t jit_cache_hit_count_threshold =
      GET_ENV_FLAG_NEW(PT_HPU_LOP_JIT_WARM_UP_STEPS);
  for (const auto& event : events) {
    if (event.jit_cache_hit_count > jit_cache_hit_count_threshold) {
      if (event.is_begin)
        continue;
      int bucket_index = (event.stage_time - min_time) / bucket_size;
      if (bucket_index >= NUM_BUCKETS) {
        bucket_index = NUM_BUCKETS - 1;
      }
      buckets[bucket_index]++;
    }
  }

  fprintf(metrics_file, " ------------------------------------------\n");
  fprintf(metrics_file, " Time Range (ns)\tFrequency\n");
  fprintf(metrics_file, " ------------------------------------------\n");

  for (int i = 0; i < NUM_BUCKETS; ++i) {
    uint64_t bucket_min = min_time + i * bucket_size;
    uint64_t bucket_max = bucket_min + bucket_size;
    fprintf(
        metrics_file, " [%lu, %lu]\t%lu\n", bucket_min, bucket_max, buckets[i]);
  }
  fprintf(metrics_file, " ------------------------------------------\n");
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

  fprintf(metrics_file, " ------------------------------------------\n");
  fprintf(metrics_file, " Device Queue Depth Percentage");
  fprintf(metrics_file, " ------------------------------------------\n");

  for (uint64_t i = max_device_queue_len; i >= min_device_queue_len; i--) {
    float device_queue_percent =
        (static_cast<float>(device_queue_len[i]) / total_events) * 100;
    fprintf(metrics_file, " %lu\t %f\n", i, device_queue_percent);
    if (i == 0) {
      break;
    }
  }
  fprintf(metrics_file, " ------------------------------------------\n");
}

void ProfilerEngine::flush() {
  printf("ProfilerEngine::flush at PID:%u\n", getpid());
  fflush(stdout);

  if (this->flushed) {
    printf("Tried to flush already flushed LOP. Doing nothing.");
    return;
  }

  this->enabled = false;
  uint64_t stage_time_start[NUM_OF_PIPELINE_STAGES] = STAGE_INITIALIZER;
  uint64_t stage_total_time[NUM_OF_PIPELINE_STAGES] = STAGE_INITIALIZER;
  uint64_t stage_counter[NUM_OF_PIPELINE_STAGES] = STAGE_INITIALIZER;
  uint64_t pipeline_queue_length[NUM_OF_PIPELINE_STAGES] = STAGE_INITIALIZER;
  uint64_t device_total_queue_length = 0;
  uint64_t max_default_value = std::numeric_limits<uint64_t>::max();
  uint64_t min_default_value = std::numeric_limits<uint64_t>::min();
  uint64_t min_time[NUM_OF_PIPELINE_STAGES] = {
      max_default_value,
      max_default_value,
      max_default_value,
      max_default_value,
      max_default_value};
  uint64_t max_time[NUM_OF_PIPELINE_STAGES] = {
      min_default_value,
      min_default_value,
      min_default_value,
      min_default_value,
      min_default_value};
  uint64_t min_queue_len[NUM_OF_PIPELINE_STAGES] = {
      max_default_value,
      max_default_value,
      max_default_value,
      max_default_value,
      max_default_value};
  uint64_t max_queue_len[NUM_OF_PIPELINE_STAGES] = {
      min_default_value,
      min_default_value,
      min_default_value,
      min_default_value,
      min_default_value};
  std::priority_queue<
      std::tuple<std::string, uint64_t>,
      std::vector<std::tuple<std::string, uint64_t>>,
      Compare>
      top_num_buckets_ops_time[NUM_OF_PIPELINE_STAGES];
  uint64_t min_device_queue_len = 0;
  uint64_t max_device_queue_len = 0;
  uint64_t stage_mean_time[NUM_OF_PIPELINE_STAGES] = STAGE_INITIALIZER;
  uint64_t mean_queue_length[NUM_OF_PIPELINE_STAGES] = STAGE_INITIALIZER;
  uint64_t mean_device_queue_length = 0;
  int64_t stage_time_variance[NUM_OF_PIPELINE_STAGES] = STAGE_INITIALIZER;
  int64_t stage_time_std[NUM_OF_PIPELINE_STAGES] = STAGE_INITIALIZER;
  int64_t stage_queue_length_variance[NUM_OF_PIPELINE_STAGES] =
      STAGE_INITIALIZER;
  int64_t current_index[NUM_OF_PIPELINE_STAGES] = STAGE_INITIALIZER;
  int64_t stage_queue_length_std[NUM_OF_PIPELINE_STAGES] = STAGE_INITIALIZER;
  int64_t device_queue_length_variance = 0;
  int64_t device_queue_length_std = 0;
  char name[50];
  sprintf(name, "events_pid%u.json", getpid());
  auto file = fopen(name, "w");
  fprintf(file, "{\"displayTimeUnit\": \"ns\", \"traceEvents\": [\n");
  sprintf(name, "metrics_pid%u.json", getpid());
  auto metrics_file = fopen(name, "w");
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
      if (event.timestamp < tsc_base)
        tsc_base = event.timestamp;
    }
  }
  for (int pipeline_stage = 1; pipeline_stage < NUM_OF_PIPELINE_STAGES - 1;
       pipeline_stage++) {
    if (current_index[pipeline_stage] > 0) {
      for (int i = 0; i < current_index[pipeline_stage]; ++i) {
        auto& event = this->events_table[pipeline_stage][i];
        auto tsc_diff = event.timestamp - tsc_base;
        auto time_ns = static_cast<uint64_t>(
            static_cast<double>(tsc_diff) / this->ticks_per_ns_ratio);
        uint64_t jit_cache_hit_count_threshold =
            GET_ENV_FLAG_NEW(PT_HPU_LOP_JIT_WARM_UP_STEPS);
        if (event.is_begin) {
          stage_time_start[pipeline_stage] = time_ns;
        } else {
          if (event.jit_cache_hit_count > jit_cache_hit_count_threshold) {
            auto stage_time = time_ns - stage_time_start[pipeline_stage];
            event.stage_time = stage_time;
            if (stage_time < min_time[pipeline_stage]) {
              min_time[pipeline_stage] = stage_time;
            } else if (stage_time > max_time[pipeline_stage]) {
              max_time[pipeline_stage] = stage_time;
            }

            top_num_buckets_ops_time[pipeline_stage].emplace(
                event.op_name, stage_time);
            if (top_num_buckets_ops_time[pipeline_stage].size() > NUM_BUCKETS)
              top_num_buckets_ops_time[pipeline_stage].pop();

            stage_total_time[pipeline_stage] += stage_time;
            auto queue_length = event.pipeline_queue_length;
            if (queue_length < min_queue_len[pipeline_stage]) {
              min_queue_len[pipeline_stage] = queue_length;
            } else if (queue_length > max_queue_len[pipeline_stage]) {
              max_queue_len[pipeline_stage] = queue_length;
            }

            auto device_queue_length = event.device_queue_length;
            if (device_queue_length < min_device_queue_len) {
              min_device_queue_len = device_queue_length;
            } else if (device_queue_length > max_device_queue_len) {
              max_device_queue_len = device_queue_length;
            }

            pipeline_queue_length[pipeline_stage] += queue_length;
            device_total_queue_length += device_queue_length;
            stage_counter[pipeline_stage] += 1;
          }
        }
      }

      stage_mean_time[pipeline_stage] =
          stage_total_time[pipeline_stage] / stage_counter[pipeline_stage];
      mean_queue_length[pipeline_stage] =
          pipeline_queue_length[pipeline_stage] / stage_counter[pipeline_stage];
      mean_device_queue_length =
          device_total_queue_length / stage_counter[pipeline_stage];

      for (int i = 0; i < current_index[pipeline_stage]; ++i) {
        auto& event = this->events_table[pipeline_stage][i];
        auto tsc_diff = event.timestamp - tsc_base;
        auto time_ns = static_cast<uint64_t>(
            static_cast<double>(tsc_diff) / this->ticks_per_ns_ratio);
        uint64_t jit_cache_hit_count_threshold =
            GET_ENV_FLAG_NEW(PT_HPU_LOP_JIT_WARM_UP_STEPS);
        int64_t queue_length = event.pipeline_queue_length;
        int64_t device_queue_length = event.device_queue_length;
        if (event.is_begin) {
          stage_time_start[pipeline_stage] = time_ns;
        } else {
          if (event.jit_cache_hit_count > jit_cache_hit_count_threshold) {
            auto stage_time = time_ns - stage_time_start[pipeline_stage];
            int64_t queue_length_diff =
                queue_length - mean_queue_length[pipeline_stage];
            stage_queue_length_variance[pipeline_stage] +=
                queue_length_diff * queue_length_diff;
            int64_t device_queue_length_diff =
                device_queue_length - mean_device_queue_length;
            device_queue_length_variance +=
                device_queue_length_diff * device_queue_length_diff;
            int64_t time_diff = stage_time - stage_mean_time[pipeline_stage];
            stage_time_variance[pipeline_stage] += time_diff * time_diff;
          }
        }
      }

      stage_time_variance[pipeline_stage] =
          stage_time_variance[pipeline_stage] /
          (stage_counter[pipeline_stage] - 1);
      stage_time_std[pipeline_stage] =
          (int64_t)sqrt((double)stage_time_variance[pipeline_stage]);
      stage_queue_length_variance[pipeline_stage] =
          stage_queue_length_variance[pipeline_stage] /
          (stage_counter[pipeline_stage] - 1);
      stage_queue_length_std[pipeline_stage] =
          (int64_t)sqrt((double)stage_queue_length_variance[pipeline_stage]);

      device_queue_length_variance =
          device_queue_length_variance / (stage_counter[pipeline_stage] - 1);
      device_queue_length_std =
          (int64_t)sqrt((double)device_queue_length_variance);
    }
  }
  fprintf(
      metrics_file,
      " Total number of events = %lu \n",
      this->events_counter[1].load(std::memory_order_acquire) +
          this->events_counter[2].load(std::memory_order_acquire) +
          this->events_counter[3].load(std::memory_order_acquire));
  fprintf(metrics_file, "\n LOWERING STAGE \n");
  fprintf(metrics_file, " ============== \n");
  fprintf(
      metrics_file,
      " Lowering stage average time(ns) = %lu \n",
      stage_total_time[1] / stage_counter[1]);
  fprintf(metrics_file, " Lowering stage max time(ns) = %lu \n", max_time[1]);
  fprintf(metrics_file, " Lowering stage min time(ns) = %lu \n", min_time[1]);
  fprintf(
      metrics_file,
      " Lowering stage standard deviation time(ns) = %lu \n",
      stage_time_std[1]);
  fprintf(
      metrics_file,
      " Lowering stage average queue length = %lu \n",
      mean_queue_length[1]);
  fprintf(
      metrics_file,
      " Lowering stage max queue length = %lu \n",
      max_queue_len[1]);
  fprintf(
      metrics_file,
      " Lowering stage min queue length = %lu \n",
      min_queue_len[1]);
  fprintf(
      metrics_file,
      " Lowering stage standard deviation queue length = %lu \n",
      stage_queue_length_std[1]);
  fprintf(metrics_file, "\n Top 5 ops with highest lowering time: \n");
  const auto& lowering_index =
      static_cast<size_t>(LOP::PipelineStageID::PIPELIE_STAGE_LOWERING_ID);
  while (!top_num_buckets_ops_time[lowering_index].empty()) {
    fprintf(
        metrics_file,
        " Op Name: %s, Time: %lu\n",
        std::get<0>(top_num_buckets_ops_time[lowering_index].top()).c_str(),
        std::get<1>(top_num_buckets_ops_time[lowering_index].top()));
    top_num_buckets_ops_time[lowering_index].pop();
  }
  fprintf(metrics_file, "\n Histogram for Lowering Stage\n");
  print_histogram(
      min_time[1], max_time[1], this->events_table[1], metrics_file);
  fprintf(metrics_file, "\n COMPILE STAGE \n");
  fprintf(metrics_file, " ============= \n");
  fprintf(
      metrics_file,
      " Compile stage average time(ns) = %lu \n",
      stage_total_time[2] / stage_counter[2]);
  fprintf(metrics_file, " Compile stage max time(ns) = %lu \n", max_time[2]);
  fprintf(metrics_file, " Compile stage min time(ns) = %lu \n", min_time[2]);
  fprintf(
      metrics_file,
      " Compile stage standard deviation time(ns) = %lu \n",
      stage_time_std[2]);
  fprintf(
      metrics_file,
      " Compile stage average queue length = %lu \n",
      mean_queue_length[2]);
  fprintf(
      metrics_file,
      " Compile stage max queue length = %lu \n",
      max_queue_len[2]);
  fprintf(
      metrics_file,
      " Compile stage min queue length = %lu \n",
      min_queue_len[2]);
  fprintf(
      metrics_file,
      " Compile stage queue standard deviation length = %lu \n",
      stage_queue_length_std[2]);
  fprintf(metrics_file, "\n Top 5 ops with highest compile time: \n");
  const auto& compile_index =
      static_cast<size_t>(LOP::PipelineStageID::PIPELIE_STAGE_COMPILE_ID);
  while (!top_num_buckets_ops_time[compile_index].empty()) {
    fprintf(
        metrics_file,
        " Op Name: %s, Time: %lu\n",
        std::get<0>(top_num_buckets_ops_time[compile_index].top()).c_str(),
        std::get<1>(top_num_buckets_ops_time[compile_index].top()));
    top_num_buckets_ops_time[compile_index].pop();
  }
  fprintf(metrics_file, "\n Histogram for Compile Stage\n");
  print_histogram(
      min_time[2], max_time[2], this->events_table[2], metrics_file);
  fprintf(metrics_file, "\n EXECUTE STAGE \n");
  fprintf(metrics_file, " ============= \n");
  fprintf(
      metrics_file,
      " Execute stage average time(ns) = %lu \n",
      stage_total_time[3] / stage_counter[3]);
  fprintf(metrics_file, " Execute stage max time(ns) = %lu \n", max_time[3]);
  fprintf(metrics_file, " Execute stage min time(ns) = %lu \n", min_time[3]);
  fprintf(
      metrics_file,
      " Execute stage standard deviation time(ns) = %lu \n",
      stage_time_std[3]);
  fprintf(
      metrics_file,
      " Execute stage average queue length = %lu \n",
      mean_queue_length[3]);
  fprintf(
      metrics_file,
      " Execute stage max queue length = %lu \n",
      max_queue_len[3]);
  fprintf(
      metrics_file,
      " Execute stage min queue length = %lu \n",
      min_queue_len[3]);
  fprintf(
      metrics_file,
      " Execute stage standard deviation queue length = %lu \n",
      stage_queue_length_std[3]);
  fprintf(metrics_file, "\n Top 5 ops with highest execute time: \n");
  const auto& execute_index =
      static_cast<size_t>(LOP::PipelineStageID::PIPELIE_STAGE_EXECUTE_ID);
  while (!top_num_buckets_ops_time[execute_index].empty()) {
    fprintf(
        metrics_file,
        " Op Name: %s, Time: %lu\n",
        std::get<0>(top_num_buckets_ops_time[execute_index].top()).c_str(),
        std::get<1>(top_num_buckets_ops_time[execute_index].top()));
    top_num_buckets_ops_time[execute_index].pop();
  }
  fprintf(metrics_file, "\n Histogram for Execute Stage\n");
  print_histogram(
      min_time[3], max_time[3], this->events_table[3], metrics_file);
  fprintf(metrics_file, "\n DEVICE QUEUE \n");
  fprintf(metrics_file, " ============ \n");
  fprintf(
      metrics_file, " Min device queue length = %lu \n", min_device_queue_len);
  fprintf(
      metrics_file, " Max device queue length = %lu \n", max_device_queue_len);
  fprintf(
      metrics_file,
      " Mean device queue length = %lu \n",
      mean_device_queue_length);
  fprintf(
      metrics_file,
      " Standard deviation device queue length = %lu \n",
      device_queue_length_std);
  print_device_queue_histogram(
      min_device_queue_len,
      max_device_queue_len,
      this->events_table[3],
      metrics_file);

  uint64_t time_base_ns = static_cast<uint64_t>(
      static_cast<double>(tsc_base) / this->ticks_per_ns_ratio);
  auto pid = getpid();

  if (GET_ENV_FLAG_NEW(PT_HPU_ENABLE_LOP_TRACES_COLLECTION)) {
    for (int pipeline_stage = 1; pipeline_stage < NUM_OF_PIPELINE_STAGES - 1;
         pipeline_stage++) {
      for (int i = 0; i < current_index[pipeline_stage]; ++i) {
        auto& event = this->events_table[pipeline_stage][i];
        auto tsc_diff = event.timestamp - tsc_base;
        auto time_ns = static_cast<uint64_t>(
            static_cast<double>(tsc_diff) / this->ticks_per_ns_ratio);

        std::string name;
        if (event.name) {
          name = event.name;
        } else {
          name = "custom event id " + std::to_string(event.user_event_id);
        }

        if (name == "_internal_counter_event") {
          std::string counter_name = std::to_string(event.thread_id);

          fprintf(
              file,
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
                file,
                "{"
                "\"tid\":%u,"
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
                file,
                "{"
                "\"tid\":%u,"
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
              file,
              "{"
              "\"tid\":%u,"
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
    fprintf(file, "{}]}");
  }
  fclose(file);
  this->flushed = true;
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

ProfilerEngine& ProfilerEngine::get_inst() {
  static ProfilerEngine inst;
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

  // This function is faster because it use lighter RDTSC instead of RDTSCP.
  // It's latency is lower but it also does not make partial barrier like RDTSCP
  // does. Caveat is that we no longer know the CPUs on which we took TSC. It
  // also does not check whether event buffer is correct (so it's less
  // reliable). It does not support log levels.
  int64_t current_index =
      profiler_engine_instance.events_counter[pipe_stage_id].load(
          std::memory_order_acquire);
  if (enable_lop_collection && profiler_engine_instance.is_enabled() &&
      (current_index < GET_ENV_FLAG_NEW(PT_HPU_EVENT_TABLE_SIZE))) {
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
