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

//**********************************************************************
//   This file includes a class (Statistics) that should help when you need to
//   collect statistics. ATM, we work with **global** stats as a member of a
//   singleton. The following addresses both a use of a local and global
//   instance of the class. We ease the use of this class throughout the code
//   using macros and env vars, detailed below.
//
//  General Guidelines:
//   1) Add a member of class Statistics to your class.
//          For example: Statistics m_stat;
//          ===> In the global case, it is already a member of a singleton.
//   2) Add an emum in your class with the points you want to collect.
//          For example: enum statPoints {p1, p2, p3};
//          ===> In the global case, add it to stat_points.h
//   3) In the constructor of your class, construnt m_stat. You need to provide:
//      Name, description on each point, frequency (how often to push it to
//      trace. 0 means only on destruct)
//          For example: m_stat("my Name", {{p1, "this is p1"}, {p2, "and this
//          is p2"}, {p3, "p3"}}, 1);
//          ===> In the global case, this is already done.
//   4) include stat_collection.h, and add collection where you need.
//   For example, if p1 collects size of something, use
//          For example: m_stat.collect(p1, sizeToCollect);
//          ===> You can use a macro:
//          STAT_COLLECT(count,globalStatPtsEnum::your_point)
//   5) If you want to add time, do:
//          For example: timeTools::stdTime start = timeTools::timeNow();
//                       ....Some code to measure....
//                       m_stat.collect(p2, timeTools::timeFrom(start)
//          ===> You can use macros: STAT_START(some_name_you_chose);
//          ===> You can use macros:
//          STAT_COLLECT_TIME(some_name_you_chose,globalStatPtsEnum::your_point)
//
//    6) If you want to save additional attributes for a certain stat point
//    (i.e. recipe name for recipe compilation):
//       - make sure to use PT_HPU_PRINT_STATS_DUMP_FREQ=1
//       - use
//
//  Print Modes:
//    to enable printing of statistics, use env vars:
//  PT_HPU_PRINT_STATS=1 : to enable stats print on destruction of object (
//  Default: 0=Disabled )
//
//  PT_HPU_PRINT_STATS_DUMP_FREQ=<uint:NUM> : will dump prints every NUM calls
//  the first listed stat point ( Default:0= dtor only)
//                                           ===> Includes enabling statistics,
//                                           no need to use both flags
//
//  PT_HPU_PRINT_STATS_TABLE=1 : will print data as table. ( Default: 0= OFF)
//                               ===> Includes enabling statistics, no need to
//                               use both flags
//
//************************************************************************

#pragma once

#include <atomic>
#include <iostream>
#include <map>
#include <memory>
#include <vector>

template <class T>
struct statEnumMsg {
  T pointEnum;
  std::string pointName;
};

enum class globalStatPtsEnum {
#define ENUM_TXT_COL(enum, txt) enum,
#include "stat_points.h"
#undef ENUM_TXT_COL
};

using PointAttrMap = std::map<std::string, std::string>;

class StatsBase {
 protected:
  virtual ~StatsBase();

  void init(
      std::string statName,
      std::vector<std::string>& names,
      uint32_t dumpFreq,
      bool disable);

  inline void collect(int point, uint64_t sum) {
    if (!m_enabled)
      return;
    m_pPointData[point].last_measurement = sum;
    m_pPointData[point].sum += sum;
    m_pPointData[point].count++;

    uint64_t cnt = m_pPointData[point].count;

    // To Avoid print on all points: only when first stat point is hit,
    // call PrintToLog() to output last available data.
    // if clear=True on call, all previous stats for this point are erased.
    if ((point == 0) && (m_dumpFreq != 0) && ((cnt % m_dumpFreq) == 0) &&
        (cnt > 0)) {
      printToLog(
          " periodic " + std::to_string(cnt),
          false /*dump all*/,
          false /*clear*/);
    }
  };

  // Per-stat point attributes, only available when using
  // PT_HPU_PRINT_STATS_DUMP_FREQ > 0
  inline void add_attribute(
      int point,
      std::string attr_key,
      std::string attr_val) {
    if (!m_enabled)
      return;
    m_pointAttributes[point][attr_key] = attr_val;
  };

  StatsBase();
  StatsBase(const StatsBase& other);
  StatsBase& operator=(const StatsBase& other) = delete;

 public:
  void printToLog(
      std::string msg = "",
      bool dumpAll = false,
      bool clear = true);
  void updateEnable();
  void updateEnableGlbl();

 private:
  void clearAll();
  void outputHeader();

  struct sumCollectData {
    std::atomic<uint64_t> sum;
    std::atomic<uint64_t> count;
    std::atomic<uint64_t> last_measurement;
  };

  std::string m_statName;
  int m_maxEnum;
  uint32_t m_dumpFreq;
  bool m_enabled{false};
  bool m_headerPrinted;
  bool m_isTbl;
  std::unique_ptr<sumCollectData[]> m_pPointData;
  std::unique_ptr<std::string[]> m_pointMsg;
  std::unique_ptr<PointAttrMap[]> m_pointAttributes;

  const static std::string m_grep; // something to grep and cut on
};

template <class T>
class Stats : public StatsBase {
 public:
  Stats();
  ~Stats(){};

  Stats(
      std::string statName,
      std::vector<statEnumMsg<T>> enumNamePoints,
      uint32_t dumpFreq = 0,
      bool disable = false) {
    std::vector<std::string> names;
    for (int i = 0; i < (int)enumNamePoints.size(); i++) {
      names.push_back(enumNamePoints[i].pointName);
    }
    StatsBase::init(statName, names, dumpFreq, disable);
  }

  void collect(T point, uint64_t sum) {
    StatsBase::collect((int)point, sum);
  }
  void add_attribute(T point, std::string attr_key, std::string attr_val) {
    StatsBase::add_attribute((int)point, attr_key, attr_val);
  }
};
