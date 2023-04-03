/*******************************************************************************
 * Copyright (C) 2021-2023 Habana Labs, Ltd. an Intel Company
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
#include "stats.h"
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include "backend/synapse_helpers/env_flags.h"
#include "logging.h"

/**************************************/
/****** Statistics collection *********/
/**************************************/
static std::vector<statEnumMsg<globalStatPtsEnum>> globalStatPoints = {
#define ENUM_TXT_COL(enum, txt) {globalStatPtsEnum::enum, txt},
#include "stat_points.h"
#undef ENUM_TXT_COL
};

template <class T>
Stats<T>::Stats() : Stats<T>("Global", globalStatPoints, /*dumpFreq*/ 0){};
template class Stats<globalStatPtsEnum>;

const std::string StatsBase::m_grep = "zzz~";
StatsBase::~StatsBase() {
  if (!m_enabled)
    return;
  // For table format, print only if at least one member is not 0
  if (m_isTbl) {
    bool allZero = true;
    for (int i = 0; i < m_maxEnum; i++) {
      uint64_t count = m_pPointData[i].count.load();
      if (count != 0) {
        allZero = false;
        break;
      }
    }
    if (allZero)
      return;
  }

  printToLog("During exit", true);
}

void StatsBase::init(
    std::string statName,
    std::vector<std::string>& names,
    uint32_t dumpFreq,
    bool disable) {
  m_statName = statName;
  m_maxEnum = names.size();
  m_dumpFreq = dumpFreq;
  if (!disable)
    updateEnable(); // sets m_enabled
  m_headerPrinted = false;

  m_pPointData.reset(new sumCollectData[m_maxEnum]{});
  m_pointMsg.reset(new std::string[m_maxEnum]{});
  m_pointAttributes.reset(new PointAttrMap[m_maxEnum]{});

  for (int i = 0; i < m_maxEnum; i++) {
    m_pointMsg[i] = names[i];
  }
}

StatsBase::StatsBase()
    : m_maxEnum(0),
      m_dumpFreq(0),
      m_enabled(false),
      m_headerPrinted(false),
      m_isTbl(false) {}

StatsBase::StatsBase(const StatsBase& other)
    : m_headerPrinted(false), m_isTbl(false) {
  m_statName = other.m_statName + " Cloned";
  m_maxEnum = other.m_maxEnum;
  m_dumpFreq = other.m_dumpFreq;
  m_enabled = other.m_enabled;

  m_pPointData.reset(new sumCollectData[m_maxEnum]{});
  m_pointMsg.reset(new std::string[m_maxEnum]{});
  m_pointAttributes.reset(new PointAttrMap[m_maxEnum]{});
  for (int i = 0; i < m_maxEnum; i++) {
    m_pointMsg[i] = other.m_pointMsg[i];
    m_pointAttributes[i] = other.m_pointAttributes[i];
  }
}

void StatsBase::printToLog(std::string msg, bool dumpAll, bool clear) {
  if (m_isTbl) // output in table format
  {
    dumpAll = true;
    if (!m_headerPrinted) {
      m_headerPrinted = true;
      outputHeader();
    }
  }

  bool first = true;
  std::stringstream out;
  out << m_grep;
  std::string msgOut = "   -----  " + m_statName + " " + msg;
  for (int i = 0; i < m_maxEnum; i++) {
    uint64_t count = m_pPointData[i].count.load();
    uint64_t sum = m_pPointData[i].sum.load();
    uint64_t last_meas = m_pPointData[i].last_measurement.load();

    if ((count == 0) && !dumpAll)
      continue;

    if (m_pointAttributes[i].size() > 0 && !dumpAll) { // Print point attributes
      PT_PROFILE_DUMP(m_pointMsg[i] + " Attributes");
      for (const auto& attr : m_pointAttributes[i]) {
        PT_PROFILE_DUMP(attr.first + " : " + attr.second);
      }
    }

    uint64_t averg = 0;
    if (count != 0) {
      averg = sum / count;
    }
    if (m_isTbl) {
      out << sum << "," << count << "," << averg << ",";
    } else {
      PT_PROFILE_DUMP(
          m_pointMsg[i],
          " |last ",
          last_meas,
          " [ns] |sum ",
          sum,
          " [ns] |count ",
          count,
          "|average ",
          averg,
          " [ns]",
          msgOut);
    }
    if (first && !m_isTbl) {
      msgOut = "";
      first = false;
    }
  }
  if (m_isTbl) {
    PT_PROFILE_DUMP(out.str(), msgOut);
  }
  if (clear) {
    clearAll();
  }
}

void StatsBase::outputHeader() {
  std::stringstream out;
  out << m_grep;
  for (int i = 0; i < m_maxEnum; i++) {
    out << m_pointMsg[i] << "-sum,";
    out << m_pointMsg[i] << "-count,";
    out << m_pointMsg[i] << "-average,";
  }
  out << m_statName;
  PT_PROFILE_DUMP(out.str());
}

// This is needed for syn_singleton class. It constructs the stats before GCFG
// are set, so we call this function to update the enable state after GCFG is
// set.
void StatsBase::updateEnable() {
  bool stats_enabled = GET_ENV_FLAG_NEW(PT_HPU_PRINT_STATS);
  unsigned dump_freq = GET_ENV_FLAG_NEW(PT_HPU_PRINT_STATS_DUMP_FREQ);
  bool table_enabled = GET_ENV_FLAG_NEW(PT_HPU_PRINT_STATS_TABLE);
  m_isTbl = table_enabled;
  m_enabled = stats_enabled || table_enabled || (dump_freq > 0);
}

void StatsBase::updateEnableGlbl() {
  updateEnable();

  unsigned stats_dump_freq = GET_ENV_FLAG_NEW(PT_HPU_PRINT_STATS_DUMP_FREQ);
  if (stats_dump_freq > 0) {
    m_dumpFreq = 1;
  }

  if (m_enabled) {
    hl_logger::setLoggingLevel(
        HlLogger::LoggerType::PT_RECIPE_STATS, HLLOG_LEVEL_DEBUG);
  }
}

void StatsBase::clearAll() {
  for (int i = 0; i < m_maxEnum; i++) {
    m_pPointData[i].count = 0;
    m_pPointData[i].sum = 0;
  }
}
