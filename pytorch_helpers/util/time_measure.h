/******************************************************************************
 * Copyright (C) 2020 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
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