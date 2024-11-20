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

#include <chrono>
#include "stat_singleton.h"
/**************************************/
/****** Statistics collection *********/
/**************************************/

namespace timeTools {
using stdTime = std::chrono::time_point<std::chrono::steady_clock>;

inline stdTime timeNow() {
  return std::chrono::steady_clock::now();
}
inline uint64_t getDurationInMicroseconds(stdTime t) {
  return std::chrono::duration_cast<std::chrono::microseconds>(
             std::chrono::steady_clock::now() - t)
      .count();
}
inline uint64_t getDurationInNanoseconds(stdTime t) {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             std::chrono::steady_clock::now() - t)
      .count();
}
} // namespace timeTools

class timeDiffHelper {
 public:
  timeDiffHelper() : m_start(timeTools::timeNow()) {}
  void collect(globalStatPtsEnum point) {
    uint64_t diff = timeTools::getDurationInNanoseconds(m_start);
    StatisticsSingleton::Instance().getStats().collect(point, diff);
  }

 private:
  timeTools::stdTime m_start;
};

#define STAT_START(x) timeDiffHelper x;
#define STAT_COLLECT_TIME(x, point) x.collect(point);
#define STAT_ADD_ATTRIBUTE(point, key, value)               \
  StatisticsSingleton::Instance().getStats().add_attribute( \
      point, std::string(key), std::string(value));

#define STAT_TO_LOG(msg) \
  StatisticsSingleton::Instance().getStats().printToLog(msg, false, true);
#define STAT_COLLECT(n, point)                          \
  do {                                                  \
    StatisticsSingleton::Instance().getStats().collect( \
        globalStatPtsEnum::point, n);                   \
  } while (0)

#define STAT_COLLECT_COND(n, flag, point1, point2) \
  do {                                             \
    if (flag)                                      \
      STAT_COLLECT(n, point1);                     \
    else                                           \
      STAT_COLLECT(n, point2);                     \
  } while (0)
