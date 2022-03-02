/*******************************************************************************
 * Copyright (C) 2021 Habana Labs, Ltd. an Intel Company
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
#include <absl/container/flat_hash_set.h>
#include <absl/strings/str_cat.h>
#include <nlohmann/json.hpp>
#include <atomic>
#include <memory>
#include "dynamic_bucket_info.h"

namespace habana_helpers {

/**
 * @brief Compilation statistics data handling.
 * This class allows collecting and dumping compilation statistics.
 *
 */
class CompilationStatistics {
 public:
  /**
   * @brief Factory method used to create CompilationStatistics object if
   * PT_COMPILATION_STATS_PATH variable is available. Additionally creates
   * PT_COMPILATION_STATS_PATH directory.
   *
   * @param cluster_name name of cluster - will be used as a file name
   * @return std::unique_ptr<CompilationStatistics> Compilation statistics
   * object
   */
  static std::unique_ptr<CompilationStatistics> Create(
      size_t id,
      uint64_t count);

  virtual ~CompilationStatistics();

  /**
   * @brief Add single input shape information. Call this function multiple
   * times for each input.
   *
   * @param index index of input in range between 0 and context->num_inputs()
   * @param shape the shape to be logged
   * @param kind type of input shape
   * @param step iteration where this data belongs, leave to 0 and data will be
   * assigned to current iteration
   */
  virtual void LogShape(
      int index,
      const habana_helpers::TensorShape& shape,
      const std::string& kind,
      uint64_t step = 0);
  /**
   * @brief Adds compilation details to iteration compilation list
   *
   * @param min_policy min policy used to with this compilation
   * @param max_policy max policy used to with this compilation
   * @param ranges min-max ranges of inputs
   * @param signature recipe signature (hash)
   * @param result the result of compilation, should be mapped from
   * context.ToString()
   * @param last_compilation_pass last attempted compilation pass
   * @param step iteration where this data belongs, leave to 0 and data will be
   * assigned to current iteration
   */
  virtual void LogCompilation(
      DynamicDimsPolicy min_policy,
      DynamicDimsPolicy max_policy,
      ResultShapes ranges,
      uint64_t signature,
      const std::string& result,
      CompilationPass last_compilation_pass,
      uint64_t step = 0);
  /**
   * @brief Add used bucket details
   *
   * @param id bucket identifier from DynamicBucketInfo::GetBucketId
   * @param ranges ranges connected to this bucket
   * @param refine_candidate is the bucket still considered as a refine
   * candidate
   * @param step iteration where this data belongs, leave to 0 and data will be
   * assigned to current iteration
   */
  virtual void LogUsedBucket(
      int id,
      ResultShapes ranges,
      bool refine_candidate,
      uint64_t step = 0);

  /**
   * @brief Add selected recipe information
   *
   * @param signature recipe signature (hash)
   * @param step iteration where this data belongs, leave to 0 and data will be
   * assigned to current iteration
   */
  virtual void LogSelectedRecipe(uint64_t signature, uint64_t step = 0);

  /**
   * @brief Adds synLaunch time measurement
   *
   * @param ms synLaunch time in miliseconds
   * @param step iteration where this data belongs, leave to 0 and data will be
   * assigned to current iteration
   */
  virtual void LogLaunch(uint64_t ms, uint64_t step = 0);

  /**
   * @brief Adds refine compilation details
   *
   * @param ranges ranges connected to the refined bucket
   * @param signature refined recipe signature (hash)
   * @param bucket refined bucket identifier
   * @param step iteration where this data belongs, leave to 0 and data will be
   * assigned to current iteration
   */
  virtual void LogRefineCompilation(
      ResultShapes ranges,
      uint64_t signature,
      uint64_t bucket,
      uint64_t step = 0);

  /**
   * @brief Adds refine compilation result
   *
   * @param result the result of compilation, should be mapped from
   * context.ToString()
   * @param step iteration where this data belongs, leave to 0 and data will be
   * assigned to current iteration.
   */
  virtual void LogRefineResult(const std::string& result, uint64_t step = 0);

  /**
   * @brief Get the Current Step number
   *
   * @return uint64_t Current step number
   */
  virtual uint64_t GetCurrentStep();

  /**
   * @brief Dump current json data and increase internal step counter
   *
   */
  virtual void DumpAndNextStep();

 protected:
  std::string path_;
  std::atomic<uint64_t> step_;
  nlohmannV340::json json_file_;

  std::string GetStep(uint64_t step);
  nlohmannV340::json GetRanges(ResultShapes ranges);
  CompilationStatistics(absl::string_view path, uint64_t global_count);

  // TF_DISALLOW_COPY_AND_ASSIGN(CompilationStatistics);
};

/**
 * @brief Used to RAII CompilationStatistics object
 * Destructor calls CompilationStatistics::DumpAndNextStep;
 *
 */
class CompilationStatisticsScope {
 public:
  CompilationStatisticsScope(
      std::shared_ptr<CompilationStatistics>& compilation_statistics)
      : compilation_statistics(compilation_statistics) {}
  ~CompilationStatisticsScope() {
    compilation_statistics->DumpAndNextStep();
  }

 private:
  std::shared_ptr<CompilationStatistics>& compilation_statistics;
  // TF_DISALLOW_COPY_AND_ASSIGN(CompilationStatisticsScope);
};
}; // namespace habana_helpers
