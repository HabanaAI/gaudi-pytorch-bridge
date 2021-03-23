/******************************************************************************
 * Copyright (C) 2021 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly
 * prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */
#include <ATen/core/interned_strings.h>
#include <mutex>
#include <unordered_map>

#include "habana_helpers/logging.h"

#pragma once
namespace habana {
class HpuFallbackStatistics {
  std::unordered_map<c10::Symbol, size_t> m_op_count;
  std::mutex m_mutex;

 public:
  ~HpuFallbackStatistics() {
    print();
  }

  void print() {
    std::lock_guard<std::mutex> lock(m_mutex);
    if (m_op_count.empty()) {
      return;
    }

    // Sort ops by frequency of occurrence
    using op_count_type = std::pair<c10::Symbol, size_t>;
    std::vector<op_count_type> ops_sorted = {
        m_op_count.begin(), m_op_count.end()};
    std::sort(
        ops_sorted.begin(),
        ops_sorted.end(),
        [](const op_count_type& a, const op_count_type& b) {
          return a.second > b.second;
        });

    std::stringstream ss;
    ss << "Frequency of op and op name that were executed on CPU: "
          "(Set env var PT_HABANA_LOG_TYPE_MASK=0 to disable this print)\n";
    for (const auto& oc : ops_sorted) {
      // TODO use setw
      ss << oc.second << "\t" << oc.first.toQualString() << "\n";
    }

    // By default, warnings are printed but can be disabled by setting env
    // variable PT_HABANA_LOG_TYPE_MASK=0
    PT_FALLBACK_WARN(ss.str());
  }

  void increment_count(const std::string& op) {
    std::lock_guard<std::mutex> lock(m_mutex);
    m_op_count[at::Symbol::fromQualString(op)]++;
  }
};
} // namespace habana
#define HPU_FALLBACK_COUNTER(op) stat.increment_count(op)
