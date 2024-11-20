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
#include <absl/container/flat_hash_set.h>
#include <absl/strings/str_cat.h>
#include <nlohmann/json.hpp>
#include <torch/csrc/jit/ir/ir.h>
#include <atomic>
#include <memory>
#include "dynamic_bucket_info.h"
#include "dynamic_bucket_info_utils.h"

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
      const std::string& id,
      uint64_t count,
      size_t hash_code);

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
      std::string index,
      const habana_helpers::TensorShape& shape,
      const std::string& kind,
      uint64_t step = 0);

  virtual void LogShapes(
      std::shared_ptr<torch::jit::Graph> jit_ir_graph,
      InpTensorShapes&,
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
      const std::string& jit_ir,
      std::shared_ptr<torch::jit::Graph> jit_ir_graph,
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
      std::shared_ptr<torch::jit::Graph> jit_ir_graph,
      ResultShapes ranges,
      bool refine_candidate,
      uint64_t step = 0);

  virtual void LogFallback(
      std::string,
      DynamicDimsPolicy,
      std::string error,
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
   * @param ns synLaunch time in nanoseconds
   * @param step iteration where this data belongs, leave to 0 and data will be
   * assigned to current iteration
   */
  virtual void LogLaunchBase(uint64_t ns, uint64_t step = 0);
  virtual void LogLaunch(uint64_t ns, uint64_t step = 0);
  virtual void LogLaunchPerf(uint64_t base_ns, uint64_t ns, uint64_t step = 0);
  virtual void LogRecipeMemory(
      synapse_helpers::graph::recipe_handle& recipe,
      uint64_t step = 0);

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
      std::shared_ptr<torch::jit::Graph> jit_ir_graph,
      uint64_t signature,
      uint64_t bucket,
      const std::string& result_str,
      uint64_t step = 0);

  /**
   * @brief Get the Current Step number
   *
   * @return uint64_t Current step number
   */
  virtual uint64_t GetCurrentStep();

  /**
   * @brief Set the parent bucket id used for current refinement
   *
   * @param bucket id considered for current refinement
   */
  virtual void SetCurrentParentBucketID(size_t bucket_id);

  /**
   * @brief Get the parent bucket id used for current refinement
   *
   * @return size_t current parent bucket id
   */
  virtual size_t GetCurrentParentBucketID();

  /**
   * @brief Set the last step in which the parent bucket have used
   * before the refinement
   *
   * @param last step the parent bucket was used
   */
  virtual void SetCurrentParentLastStep(size_t step);

  /**
   * @brief Get the last step in which the parent bucket have used
   * before the refinement
   *
   * @return size_t last step the parent bucket was used
   */
  virtual size_t GetCurrentParentLastStep();

  /**
   * @brief Dump current json data and increase internal step counter
   *
   */
  virtual void DumpAndNextStep();
  void GetDigest(
      size_t graph_key,
      size_t bucket_id,
      uint64_t token,
      size_t recipe_key,
      bool cache_hit);

  void Serialize(std::ostream& os) const;
  CompilationStatistics(std::istream& is);

 protected:
  std::string path_;
  std::atomic<uint64_t> step_;
  size_t refine_init_step_;
  size_t curr_parent_bucket_id_;
  size_t curr_parent_last_step_;
  nlohmannV340::json json_file_;
  std::ofstream file_handle;
  std::string GetStep(uint64_t step);
  nlohmannV340::json GetRanges(
      habana_helpers::ResultShapes ranges,
      std::shared_ptr<torch::jit::Graph> jit_ir_graph);
  CompilationStatistics(std::string path, uint64_t global_count);
  CompilationStatistics(const CompilationStatistics&) = delete;
  void operator=(const CompilationStatistics&) = delete;
  std::mutex json_file_mutex_;
};
}; // namespace habana_helpers
