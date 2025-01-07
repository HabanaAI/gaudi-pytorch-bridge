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
#pragma once

#define TIME_MEASURE_ENABLE

#ifdef TIME_MEASURE_ENABLE
#include <chrono>
#define TIME_MEASURE_VARS \
  std::chrono::time_point<std::chrono::steady_clock> start_time, end_time
#define START_TIME_MEASURE start_time = std::chrono::steady_clock::now()
#define END_TIME_MEASURE(MSG)                                \
  end_time = std::chrono::steady_clock::now();               \
  PT_SYNHELPER_DEBUG(                                        \
      MSG,                                                   \
      " ",                                                   \
      std::chrono::duration_cast<std::chrono::milliseconds>( \
          end_time - start_time)                             \
          .count(),                                          \
      " ms")
#define END_TIME_MEASURE2(MSG, start_time)                   \
  PT_SYNHELPER_DEBUG(                                        \
      MSG,                                                   \
      " ",                                                   \
      std::chrono::duration_cast<std::chrono::milliseconds>( \
          std::chrono::steady_clock::now() - start_time)     \
          .count(),                                          \
      " ms")
#else
#define TIME_MEASURE_VARS
#define START_TIME_MEASURE
#define END_TIME_MEASURE(MSG)
#define END_TIME_MEASURE2(MSG, start_time)
#endif