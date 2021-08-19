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

#include "stats.h"

class StatisticsSingleton {
 public:
  static StatisticsSingleton& Instance() {
    static StatisticsSingleton singleton;
    return singleton;
  }
  Stats<globalStatPtsEnum>& getStats() {
    return m_stats;
  }

  // Other non-static member functions
 private:
  StatisticsSingleton() {
    m_stats.updateEnableGlbl();
  }
  ~StatisticsSingleton() {}
  StatisticsSingleton(const StatisticsSingleton&) = delete;
  StatisticsSingleton& operator=(const StatisticsSingleton&) = delete;
  Stats<globalStatPtsEnum> m_stats;
};