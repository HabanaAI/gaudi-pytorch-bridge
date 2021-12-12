/******************************************************************************
 * Copyright (C) 2020 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */
#pragma once
#include "aten_lazy_bridge.h"
#include "hpu_lazy_tensors.h"
#include "sbs_runner.h"
#include "synapse_helpers/env_flags.h"
#include "tensor_impl.h"
// namespace habana_lazy
namespace habana_lazy {

/**
 * @brief Class to provide utilities for dumping lazy-tensor graphs
 * in text/dot format.
 */
class IrGraphDumpUtil {
 public:
  static std::string ToDot(std::vector<ir::NodePtr> nodes);

  static std::string PostOrderToDot(
      const std::vector<ir::NodePtr>& post_order,
      const std::vector<ir::NodePtr>& roots,
      const bool use_ir_names = true);

  static std::string ToText(std::vector<ir::NodePtr> nodes);

  static std::string PostOrderToText(
      const std::vector<ir::NodePtr>& post_order,
      const std::vector<ir::NodePtr>& roots,
      const bool use_ir_names = true,
      const bool print_ir_graph_info = false);
};

class DebugHelper {
 public:
  static DebugHelper& getInstance() {
    static DebugHelper instance;
    return instance;
  }

  size_t getCurrentAccumulatedOps() {
    return curr_number_of_accumulated_ops;
  }

  void resetCurrentAccumulatedOps() {
    curr_number_of_accumulated_ops = 0;
  }

  void incrementAccumulatedOps() {
    curr_number_of_accumulated_ops++;
  }

  bool isExceededMaxAccumlatedSize() {
    return (curr_number_of_accumulated_ops >= max_number_of_accumulated_ops);
  }

 private:
  DebugHelper()
      : curr_number_of_accumulated_ops(0),
        max_number_of_accumulated_ops(GET_ENV_FLAG_NEW(PT_HPU_MAX_ACCUM_SIZE)) {
  }
  ~DebugHelper() {}
  DebugHelper(const DebugHelper&);
  DebugHelper& operator=(const DebugHelper&);
  std::atomic<size_t> curr_number_of_accumulated_ops;
  const size_t max_number_of_accumulated_ops;
};

class SBSDebug {
 public:
  static SBSDebug& getInstance() {
    static SBSDebug instance;
    return instance;
  }

  void compare_tensors_cos(
      at::Tensor hpu_res,
      at::Tensor cpu_res,
      const std::string& op_type);

  void CompareTensors(std::vector<HbLazyTensor>& tensors);

 private:
  SBSDebug();

  const std::string m_report_file_name = "sbs_tensor_compare.csv";

 public:
  SBSDebug(SBSDebug const&) = delete;
  void operator=(SBSDebug const&) = delete;
};
} // namespace habana_lazy
