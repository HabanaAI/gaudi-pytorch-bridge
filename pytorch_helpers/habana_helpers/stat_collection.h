/******************************************************************************
 * Copyright (C) 2021 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
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
