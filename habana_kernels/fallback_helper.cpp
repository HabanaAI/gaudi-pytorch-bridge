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
#include "fallback_helper.h"

namespace habana {

void HpuFallbackHelper::enumerate_fallback() {
  const char* fallback_list = std::getenv("PT_HPU_PLACE_ON_CPU");

  if (fallback_list) {
    std::stringstream ss(fallback_list);
    while (ss.good()) {
      std::string substr;
      std::getline(ss, substr, ',');
      if (substr == "none") {
        enable_fallback = false;
        return;
      }
      m_ops_placed_on_cpu.insert(substr);
    }
  }
}

void HpuFallbackHelper::print_fallback_freq() const {
  std::lock_guard<std::mutex> lock(m_mutex);
  if (m_op_count.empty()) {
    return;
  }

  // Sort ops by frequency of occurrence
  using op_count_type = std::pair<std::string, size_t>;
  std::vector<op_count_type> ops_sorted = {
      m_op_count.begin(), m_op_count.end()};
  std::sort(
      ops_sorted.begin(),
      ops_sorted.end(),
      [](const op_count_type& a, const op_count_type& b) {
        return a.second > b.second;
      });

  std::stringstream ss;
  ss << "CPU Fallback stats:\n";
  for (const auto& oc : ops_sorted) {
    ss << oc.second << "\t" << oc.first << "\n";
  }

  const auto& logger = PtLogger::getLogger()->GetOpLogger();
  logger->warn(ss.str());
}
} // namespace habana
