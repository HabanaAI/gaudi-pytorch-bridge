/******************************************************************************
 * Copyright (C) 2020 Habana Labs, Ltd. an Intel Company
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

#include <cstdint>
#include <map>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
#include "synapse_helpers/habana_tensor.h"
#include "tensor_shape.h"

namespace habana_helpers {
enum class CompilationPass {
  DYNAMIC_MIN,
  DYNAMIC_MAX,
  DYNAMIC_CURRENT,
  STATIC
};

using DynamicDims =
    std::map<int64_t, std::map<int64_t, int64_t>>; // input_idx => {dim_idx =>
                                                   // range_idx}
using DynamicRanges = std::vector<std::pair<int64_t, int64_t>>;

struct InputOutputShapes {
  habana_helpers::TensorShape input;
  habana_helpers::TensorShape output;
};
using PadShapes = std::unordered_map<int64_t, InputOutputShapes>;
enum class DynamicDimsPolicy {
  DEFAULT,
  CALCULATED,
  HISTORIC,
  FLATTENED,
  CURRENT
};
class Bucket {
 public:
  Bucket(
      DynamicRanges&& ranges,
      DynamicDims dynamic_dims,
      bool is_refine_allowed);
  bool IsInRange(
      const std::vector<int64_t>& dims,
      const std::set<int64_t>& skipped_ranges) const;
  uint64_t getScore() const {
    return score_;
  }
  DynamicRanges& ranges() {
    return ranges_;
  }
  size_t getDynamiDimsCount() const {
    return ranges_.size();
  }
  const DynamicDims& getDynamicDims() const {
    return dynamic_dims_;
  }

  void IncStats(const std::vector<int64_t>& dims);
  uint64_t getCount() const {
    return count_;
  }
  void ResetCount();
  Bucket CreateNewBucket();

 private:
  static constexpr uint64_t max_number_of_dimensions = 20;
  uint64_t score_ = 0;

  uint64_t count_ = 0;
  std::vector<int64_t> split_count_;

  DynamicRanges ranges_;
  DynamicDims dynamic_dims_;
};

class DynamicBucketInfo {
 public:
  DynamicBucketInfo();

  // using SynapseShapes = std::unordered_map<int64_t,
  // synapse_helpers::tensor::dynamic_shape_t>;
  using TensorShapes = std::unordered_map<int64_t, habana_helpers::TensorShape>;
  using InpTensorShapes = std::map<int64_t, habana_helpers::TensorShape>;
  using DimMultipliers =
      std::map<int64_t, std::map<int64_t, std::pair<int64_t, int64_t>>>;
  using DimSizes = std::map<int64_t, int64_t>;

  bool AreDynamicDimsContained() const {
    return buckets_.size() > 1;
  }
  struct ResultShapes {
    TensorShapes min_shapes;
    TensorShapes max_shapes;
    // SynapseShapes syn_shapes;
    std::string DebugString();
  };

  ResultShapes CalculateShapes(uint64_t bucket);

  uint64_t GetBucketId(
      const InpTensorShapes& shapes,
      const PadShapes& pad_shapes = PadShapes{});
  absl::optional<uint64_t> CheckForSplitBucket();
  void CollectDynamicDims(const InpTensorShapes& shapes);
  std::unordered_set<int64_t> GetDynamicInputs() const;
  bool IsConsistentDynamicDimsCount();
  bool UpdateBucketingPolicy(
      uint64_t bucket_id,
      const InpTensorShapes& shapes,
      const PadShapes& pad_shapes,
      DynamicDimsPolicy min_policy,
      DynamicDimsPolicy max_policy);

 private:
  DynamicRanges GetDynamicRanges();
  std::vector<int64_t> ExtractDynamicDimsValue(
      const InpTensorShapes& shapes) const;
  bool IsInRangeStaticDims(const std::vector<int64_t>& dims, int64_t num) const;
  int64_t GetMaxMultiplier(const PadShapes& pad_shapes);
  DynamicRanges CalculateRanges(
      const InpTensorShapes& shapes,
      const PadShapes& pad_shapes);
  DimMultipliers CalculateFlattenedMultipliers(
      const InpTensorShapes& shapes,
      int64_t max_multiplier);
  std::vector<int64_t> CalculateHistoricMin(const InpTensorShapes& shapes);

  static constexpr int64_t default_max_multiplier_ = 2;
  static constexpr int64_t default_min_value_ = 2;
  static constexpr uint64_t max_buckets_number_ = 20;
  static constexpr uint64_t min_iterations_to_split_ = 100;
  static constexpr float density_coefficient_ = 0.75;

  std::vector<Bucket> buckets_;
  uint64_t global_count = 0;
  InpTensorShapes shapes_;
  size_t prev_dynamic_dims_{};
  DynamicDimsPolicy min_policy_{DynamicDimsPolicy::CALCULATED};
  DynamicDimsPolicy max_policy_{DynamicDimsPolicy::CALCULATED};
  std::vector<std::vector<int64_t>> dim_history_;
  bool refine_enabled_ = true;

  struct DynamicDimsElement {
    int64_t num;
    int64_t pos;
    int64_t previous_val;
    DynamicDimsElement(int64_t n, int64_t p, int64_t v)
        : num(n), pos(p), previous_val(v){};
  };
  using DynamicDimsFlat = std::vector<DynamicDimsElement>;

  struct DynamicDimsHelper {
    DynamicDims dd_;
    DimSizes rem_size_;
    DynamicDimsFlat flat_dd_;
    void FindOrAdd(int64_t num, int64_t pos, int64_t val);
  } dynamic_dims_;
};

}; // namespace habana_helpers
