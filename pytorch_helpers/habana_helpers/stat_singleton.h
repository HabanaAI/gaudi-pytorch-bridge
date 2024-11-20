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