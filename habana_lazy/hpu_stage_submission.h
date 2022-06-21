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
#include "synapse_helpers/env_flags.h"

// namespace habana_lazy
namespace habana_lazy {
class StageSubmission {
 public:
  static StageSubmission& getInstance() {
    static StageSubmission instance;
    return instance;
  }

  size_t getCurrentAccumulatedOps() {
    return curr_number_of_accumulated_ops;
  }
  size_t getCurrentCompoundOps() {
    return curr_number_of_compound_ops;
  }
  size_t getMaxAccumulatedOps() {
    return max_number_of_accumulated_ops;
  }
  size_t getMaxCompoundOps() {
    return max_number_of_compound_ops;
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
    curr_number_of_compound_ops = 0;

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
  StageSubmission()
      : curr_number_of_accumulated_ops(0),
        curr_number_of_compound_ops(0),
        max_number_of_accumulated_ops(GET_ENV_FLAG_NEW(PT_HPU_MAX_ACCUM_SIZE)),
        max_number_of_compound_ops(
            GET_ENV_FLAG_NEW(PT_HPU_MAX_COMPOUND_OP_SIZE)),
        is_stage_submission(0),
        enable_stage_submission(
            GET_ENV_FLAG_NEW(PT_HPU_ENABLE_STAGE_SUBMISSION)) {}
  ~StageSubmission() {}
  StageSubmission(const StageSubmission&);
  StageSubmission& operator=(const StageSubmission&);
  std::atomic<size_t> curr_number_of_accumulated_ops;
  std::atomic<size_t> curr_number_of_compound_ops;
  const size_t max_number_of_accumulated_ops;
  std::atomic<size_t> max_number_of_compound_ops;
  bool is_stage_submission;
  const bool enable_stage_submission;
};

class PTOpTrace {
 public:
  PTOpTrace();
  ~PTOpTrace();
  void increment_compound_ops();
};

#define PT_OP_TRACE habana_lazy::PTOpTrace pt_op_trace;

} // namespace habana_lazy
