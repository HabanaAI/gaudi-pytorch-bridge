/*******************************************************************************
 * Copyright (C) 2020-2023 Habana Labs, Ltd. an Intel Company
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
#pragma once

#include <atomic>
#include "backend/synapse_helpers/env_flags.h"

// namespace habana_lazy
namespace habana_lazy {
class StageSubmission {
 public:
  enum Mode {
    DO_NOT_RESET = 0b1,
    SET_WHEN_NON_INFERABLE = 0b10,
    SET_WHEN_CPU_FALLBACK = 0b100,
    SET_WHEN_D2H_COPY = 0b1000,
    SET_WHEN_ANY_ITEM_CALL = 0b10000
  };

  static StageSubmission& getInstance() {
    static StageSubmission instance;
    return instance;
  }

  size_t getCurrentOpCount() {
    return curr_op_count;
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
  void incrementOpCount() {
    curr_op_count++;
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
    curr_op_count = 0;
    curr_number_of_accumulated_ops = 0;
    curr_number_of_compound_ops = 0;

    if (!enable_stage_submission) {
      return;
    }

    if (is_stage_submission) {
      max_number_of_compound_ops = find_limit(
          2 * max_number_of_compound_ops,
          GET_ENV_FLAG_NEW(PT_HPU_MAX_COMPOUND_OP_SIZE));
    } else {
      max_number_of_compound_ops =
          GET_ENV_FLAG_NEW(PT_HPU_MAX_COMPOUND_OP_SIZE);
    }
  }

  void setStageSubmissionFlow(Mode mode = Mode::SET_WHEN_NON_INFERABLE) {
    if (!enable_stage_submission) {
      return;
    }

    if (!is_mode_set_to(mode)) {
      return;
    }

    is_stage_submission = true;
    max_number_of_compound_ops =
        GET_ENV_FLAG_NEW(PT_HPU_MAX_COMPOUND_OP_SIZE_SS);
  }

  void resetStageSubmissionFlow() {
    if (is_mode_set_to(DO_NOT_RESET)) {
      return;
    }

    is_stage_submission = false;
  }

  inline bool is_mode_set_to(Mode mode) {
    return this->mode & mode;
  }

 private:
  StageSubmission()
      : curr_op_count(0),
        curr_number_of_accumulated_ops(0),
        curr_number_of_compound_ops(0),
        max_number_of_accumulated_ops(GET_ENV_FLAG_NEW(PT_HPU_MAX_ACCUM_SIZE)),
        max_number_of_compound_ops(
            GET_ENV_FLAG_NEW(PT_HPU_MAX_COMPOUND_OP_SIZE)),
        is_stage_submission(0),
        enable_stage_submission(
            GET_ENV_FLAG_NEW(PT_HPU_ENABLE_STAGE_SUBMISSION)),
        mode(
            static_cast<Mode>(GET_ENV_FLAG_NEW(PT_HPU_STAGE_SUBMISSION_MODE))) {
  }
  ~StageSubmission() {}
  StageSubmission(const StageSubmission&);
  StageSubmission& operator=(const StageSubmission&);
  std::atomic<size_t> curr_op_count;
  std::atomic<size_t> curr_number_of_accumulated_ops;
  std::atomic<size_t> curr_number_of_compound_ops;
  const size_t max_number_of_accumulated_ops;
  std::atomic<size_t> max_number_of_compound_ops;
  bool is_stage_submission;
  const bool enable_stage_submission;
  const Mode mode;
};

class PTOpTrace {
 public:
  PTOpTrace();
  ~PTOpTrace();
  void increment_compound_ops();
};

#define PT_LAZY_OP_TRACE habana_lazy::PTOpTrace pt_op_trace;

} // namespace habana_lazy
