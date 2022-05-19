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
#include "synapse_helpers/env_flags.h"
#include "tensor_impl.h"

#define INT64_T_MAX std::numeric_limits<int64_t>::max();

// namespace habana_lazy
namespace habana_lazy {

// The Side-By-Side (SBS) Debug Tool is a debug capability for comparing
// between tensors that are calculated by HPU to tensors that are calculated
// by CPU.
// Run it by adding the env var PT_SBS with one of the enum values described
// here: debug_utils.h :: SBSModes
// See more here:
// https://confluence.habana-labs.com/display/SYN/Side-By-Side+Debug+Tool
enum SBSModes : unsigned {
  SBS_MODE_DISABLED = 0,
  SBS_MODE_STANDALONE = 1,
  SBS_MODE_USE_CPU_INPUT = 2,
  SBS_MODE_USE_HPU_INPUT = 3
};

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

  void incrementAccumulatedOps() {
    curr_number_of_accumulated_ops++;
  }

  bool isExceededMaxAccumlatedSize() {
    return (curr_number_of_accumulated_ops >= max_number_of_accumulated_ops);
  }

  void incrementCompoundOps() {
    curr_number_of_compound_ops++;
  }

  bool isExceededMaxCompoundSize() {
    return (curr_number_of_compound_ops >= max_number_of_compound_ops);
  }

  inline int64_t find_limit(
      const int64_t& current_max_value,
      const int64_t& max_value) {
    int64_t max_limit = std::min(current_max_value, max_value);
    max_limit = (max_limit < 0) ? max_value : max_limit;
    return max_limit;
  }

  void resetCurrentAccumulatedOps() {
    curr_number_of_accumulated_ops = 0;
    // Resetting to -1 to handle PT_HPU_MAX_COMPOUND_OP_SIZE=1,
    // otherwise StepMarker always cause max_number_of_compound_ops
    curr_number_of_compound_ops = -1;

    if (enable_stage_submission) {
      if (is_stage_submission) {
        max_number_of_compound_ops = find_limit(
            2 * max_number_of_compound_ops,
            GET_ENV_FLAG_NEW(PT_HPU_MAX_COMPOUND_OP_SIZE));
      } else {
        max_number_of_compound_ops =
            GET_ENV_FLAG_NEW(PT_HPU_MAX_COMPOUND_OP_SIZE);
      }
    }
  }

  void setStageSubmissionFlow() {
    if (enable_stage_submission) {
      is_stage_submission = true;
      max_number_of_compound_ops =
          GET_ENV_FLAG_NEW(PT_HPU_MAX_COMPOUND_OP_SIZE_SS);
    }
  }

  void resetStageSubmissionFlow() {
    is_stage_submission = false;
  }

 private:
  DebugHelper()
      : curr_number_of_accumulated_ops(0),
        curr_number_of_compound_ops(0),
        max_number_of_accumulated_ops(GET_ENV_FLAG_NEW(PT_HPU_MAX_ACCUM_SIZE)),
        max_number_of_compound_ops(
            GET_ENV_FLAG_NEW(PT_HPU_MAX_COMPOUND_OP_SIZE)),
        is_stage_submission(0),
        enable_stage_submission(
            GET_ENV_FLAG_NEW(PT_HPU_ENABLE_STAGE_SUBMISSION)) {
  }
  ~DebugHelper() {}
  DebugHelper(const DebugHelper&);
  DebugHelper& operator=(const DebugHelper&);
  std::atomic<size_t> curr_number_of_accumulated_ops;
  std::atomic<int64_t> curr_number_of_compound_ops;
  const size_t max_number_of_accumulated_ops;
  std::atomic<int64_t> max_number_of_compound_ops;
  bool is_stage_submission;
  const bool enable_stage_submission;
};

} // namespace habana_lazy
