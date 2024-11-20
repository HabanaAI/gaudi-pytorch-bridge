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
#include "hpu_lazy_tensors.h"

// The Side-By-Side (SBS) Debug Tool is a debug capability for comparing
// between tensors that are calculated by HPU to tensors that are calculated
// by CPU.
// Run it by adding the env var PT_SBS with one of the enum values described
// here: debug_utils.h :: SBSModes
// See more here:
// https://confluence.habana-labs.com/display/SYN/Side-By-Side+Debug+Tool

namespace TensorComparison_pt {
class TensorValidator;
}

namespace habana_lazy {

class SBSDebug {
 public:
  static SBSDebug& getInstance() {
    static SBSDebug instance;
    instance.init();
    return instance;
  }

  void CompareTensors(std::vector<HbLazyTensor>& tensors);

  bool LogError(
      const std::string& op_name,
      const std::string& message_short,
      const std::string& message_detailed = "");

  size_t GetNumberOfReportLines();
  size_t GetNumberOfErrorLines();
  size_t GetNumberOfCompareLines();

  size_t GetNumberOfAccumulatedOps() {
    return m_number_of_accumulated_ops;
  }

  size_t GetNumberOfAccumulatedOpOutputTensors() {
    return m_number_of_accumulated_op_output_tensors;
  }

  void IncreaseOpsAndTensors(size_t tensor_count) {
    ++m_number_of_accumulated_ops;
    m_number_of_accumulated_op_output_tensors += tensor_count;
  }

  void reset();

 private:
  SBSDebug();

  static bool NeedToCompare(const HbLazyTensor& hb_tensor, bool update = false);

  void init();
  void report(const std::string& log_message, size_t& log_counter);

  void compare_tensors_cos(
      const at::Tensor& hpu_res,
      const at::Tensor& cpu_res,
      const std::string& op_type);

  const std::string m_report_file_name = "sbs_tensor_compare.csv";
  const std::string m_error_file_name = "sbs_error.csv";
  std::ofstream m_error_file;
  std::shared_ptr<TensorComparison_pt::TensorValidator> mp_tc;
  size_t m_number_of_successful_compares;
  size_t m_number_of_errors;
  size_t m_number_of_accumulated_ops;
  size_t m_number_of_accumulated_op_output_tensors;
  bool m_is_initialized;

 public:
  SBSDebug(SBSDebug const&) = delete;
  void operator=(SBSDebug const&) = delete;
};

} // namespace habana_lazy