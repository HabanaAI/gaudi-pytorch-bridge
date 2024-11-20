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
#include <c10/util/Exception.h>
#include <mutex>
#include <unordered_map>
#include <unordered_set>

#include "habana_helpers/logging_pt.h"

#pragma once
namespace habana {
class HpuFallbackHelper {
 private:
  HpuFallbackHelper() {
    enumerate_fallback();
  }

  ~HpuFallbackHelper() = default;

 public:
  static HpuFallbackHelper* get() {
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
    static HpuFallbackHelper* singleton = new HpuFallbackHelper();
    return singleton;
  }

  const std::unordered_map<std::string, size_t>& get_op_count() const {
    return m_op_count;
  }

  void enumerate_fallback();

  void print_fallback_freq() const;

  void increment_count(const std::string& op) {
    std::lock_guard<std::mutex> lock(m_mutex);
    m_op_count[op]++;
    PT_IRGRAPH_DEBUG("step marker due to cpu fallback ", op);
  }

  void check_fallback_allowed(const std::string& op) const {
    TORCH_CHECK(
        enable_fallback || m_ops_placed_on_cpu.count(op),
        op,
        " is not yet supported on HPU.")
  }

  bool is_placed_on_cpu(const std::string& op) const {
    return m_ops_placed_on_cpu.count(op);
  }

 private:
  std::unordered_map<std::string, size_t> m_op_count;
  mutable std::mutex m_mutex;
  std::unordered_set<std::string> m_ops_placed_on_cpu;
  bool enable_fallback = true;
};

} // namespace habana
