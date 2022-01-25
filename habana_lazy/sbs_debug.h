/******************************************************************************
 * Copyright (C) 2022 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */
#pragma once
#include "hpu_lazy_tensors.h"
#include "tensor_comparator.hpp"

// The Side-By-Side (SBS) Debug Tool is a debug capability for comparing
// between tensors that are calculated by HPU to tensors that are calculated
// by CPU.
// Run it by adding the env var PT_SBS with one of the enum values described
// here: sbs_runner.h :: SBSModes
// See more here:
// https://confluence.habana-labs.com/display/SYN/Side-By-Side+Debug+Tool

namespace habana_lazy {

class SBSDebug {
 public:
  static SBSDebug& getInstance() {
    static SBSDebug instance;
    return instance;
  }

  void CompareTensors(std::vector<HbLazyTensor>& tensors);

  bool LogError(
      const std::string& op_name,
      const std::string& message_short,
      const std::string& message_detailed = "");

 private:
  SBSDebug();

  void compare_tensors_cos(
      at::Tensor hpu_res,
      at::Tensor cpu_res,
      const std::string& op_type);

  const std::string m_report_file_name = "sbs_tensor_compare.csv";
  const std::string m_error_file_name = "sbs_error.csv";
  std::ofstream m_error_file;
  TensorComparison::TensorValidator m_tc;

 public:
  SBSDebug(SBSDebug const&) = delete;
  void operator=(SBSDebug const&) = delete;
};

} // namespace habana_lazy