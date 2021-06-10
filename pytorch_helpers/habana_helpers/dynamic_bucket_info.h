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

#include <atomic>
#include <map>
#include <mutex>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "pytorch_helpers/synapse_helpers/habana_tensor.h"
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

inline std::ostream& operator<<(std::ostream& O, const DynamicDimsPolicy& d) {
  switch (d) {
    case DynamicDimsPolicy::DEFAULT:
      return O << "DEFAULT";
    case DynamicDimsPolicy::CALCULATED:
      return O << "CALCULATED";
    case DynamicDimsPolicy::HISTORIC:
      return O << "HISTORIC";
    case DynamicDimsPolicy::FLATTENED:
      return O << "FLATTENED";
    case DynamicDimsPolicy::CURRENT:
      return O << "CURRENT";
  }
}

inline std::ostream& operator<<(std::ostream& O, const DynamicDims& d) {
  O << "dynamic dims ::" << '\n';
  for (const auto& r : d) {
    O << "  " << r.first << " -> ";
    for (const auto& a : r.second) {
      O << ' ' << '(' << a.first << " -> " << a.second << ')';
    }
    O << '\n';
  }

  return O;
}

inline std::ostream& operator<<(
    std::ostream& O,
    const std::map<int64_t, habana_helpers::TensorShape>& t) {
  for (const auto& a : t) {
    O << "  " << a.first << " -> " << a.second << '\n';
  }
  return O;
}

inline std::ostream& operator<<(
    std::ostream& O,
    const std::unordered_map<int64_t, habana_helpers::TensorShape>& t) {
  for (const auto& a : t) {
    O << "  " << a.first << " -> " << a.second << '\n';
  }
  return O;
}

class Bucket {
 public:
  Bucket(
      DynamicRanges&& ranges,
      DynamicDims dynamic_dims,
      bool is_refine_allowed);
  bool IsInRange(
      const std::vector<int64_t>& dims,
      const std::set<int64_t>& skipped_ranges) const;
  void IncStats(const std::vector<int64_t>& dims);
  Bucket CreateNewBucket();
  uint64_t getCount() const {
    return count_;
  }
  void ResetCount() {
    count_ = 0;
    std::fill(split_count_.begin(), split_count_.end(), 0);
  }
  uint64_t getScore() const {
    return score_;
  }
  uint64_t getToken() const {
    return token_;
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

  friend inline std::ostream& operator<<(std::ostream& O, const Bucket& b) {
    O << "bucket ::"
      << " score " << b.score_ << ',' << " count " << b.count_ << ','
      << " token " << b.token_ << '\n'
      << "split_count " << at::IntArrayRef(b.split_count_) << '\n';
    O << "ranges :";
    for (const auto& a : b.ranges_) {
      O << ' ' << '(' << a.first << ", " << a.second << ')';
    }
    O << '\n';
    O << "dims :" << '\n' << b.dynamic_dims_;
    return O;
  }

  static constexpr uint64_t uninitialized_token = 1000000006;

 private:
  static constexpr uint64_t max_number_of_dimensions = 20;

  uint64_t score_{0};
  uint64_t count_{0};
  uint64_t token_{uninitialized_token};

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

    bool empty() {
      return (min_shapes.empty() && max_shapes.empty());
    }
    // SynapseShapes syn_shapes;
    std::string DebugString();
    friend inline std::ostream& operator<<(
        std::ostream& O,
        const ResultShapes& r) {
      O << "Min shapes ::" << '\n' << r.min_shapes;
      O << "Max shapes ::" << '\n' << r.max_shapes;
      return O;
    }
  };

  ResultShapes CalculateShapes(uint64_t bucket);

  std::unordered_set<int64_t> GetDynamicInputs() const;
  void CollectDynamicDims(const InpTensorShapes& shapes);

  uint64_t GetBucketId(
      const InpTensorShapes& shapes,
      const PadShapes& pad_shapes = PadShapes{});
  absl::optional<uint64_t> CheckForSplitBucket();
  bool IsConsistentDynamicDimsCount();
  bool UpdateBucketingPolicy(
      uint64_t bucket_id,
      const InpTensorShapes& shapes,
      const PadShapes& pad_shapes,
      DynamicDimsPolicy min_policy,
      DynamicDimsPolicy max_policy);

  uint64_t GetTokenForBucketId(uint64_t bidx) {
    TORCH_CHECK(
        bidx < buckets_.size(),
        "Invalid bucket index ",
        bidx,
        " encountered, should be less than ",
        buckets_.size());

    TORCH_CHECK(
        buckets_[bidx].getToken() != Bucket::uninitialized_token,
        "Token is uninitialized for bucket index ",
        bidx);

    return buckets_[bidx].getToken();
  }

  bool exists_token_for_input_shapes(size_t key) {
    return (input_token_map_.count(key) != 0);
  }
  uint64_t get_token_for_input_shapes(size_t key) {
    return input_token_map_[key];
  }
  void add_token_for_input_shapes(size_t key, size_t val) {
    input_token_map_.emplace(key, val);
  }
  friend inline std::ostream& operator<<(
      std::ostream& O,
      const DynamicBucketInfo& d) {
    O << "DynamicBucketInfo :: "
      << " global_count=" << d.global_count
      << ", prev_dynamic_dims=" << d.prev_dynamic_dims_
      << ", min policy=" << d.min_policy_ << ", max policy=" << d.max_policy_
      << ", refine_enabled=" << std::boolalpha << d.refine_enabled_ << '\n';
    O << "Input tensor shapes ::" << '\n';
    for (const auto& a : d.shapes_) {
      O << "  " << a.first << " -> " << a.second << '\n';
    }
    O << "Buckets ::" << '\n';
    for (const auto& a : d.buckets_) {
      O << a;
    }
    O << "Dim history : len=" << d.dim_history_.size()
      << ", contents ::" << '\n';
    bool skipped{false};
    for (size_t i = 0; i < d.dim_history_.size(); i++) {
      const auto& a = d.dim_history_[i];
      if (i > 0 && a == d.dim_history_[i - 1]) {
        skipped = true;
        continue;
      }
      if (skipped) {
        skipped = false;
        O << "  "
          << "..." << '\n';
      }
      O << "  " << i << " : " << at::IntArrayRef(a) << '\n';
    }
    if (skipped) {
      skipped = false;
      O << "  "
        << "..." << '\n';
    }
    O << '\n';
    return O;
  }

  static uint64_t min_iterations_to_split() {
    return min_iterations_to_split_;
  }

 private:
  std::vector<int64_t> ExtractDynamicDimsValue(
      const InpTensorShapes& shapes) const;
  bool IsInRangeStaticDims(const std::vector<int64_t>& dims, int64_t num) const;
  int64_t GetMaxMultiplier(const PadShapes& pad_shapes);
  DimMultipliers CalculateFlattenedMultipliers(
      const InpTensorShapes& shapes,
      int64_t max_multiplier);
  std::vector<int64_t> CalculateHistoricMin(const InpTensorShapes& shapes);
  DynamicRanges CalculateRanges(
      const InpTensorShapes& shapes,
      const PadShapes& pad_shapes);

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
    friend inline std::ostream& operator<<(
        std::ostream& O,
        const DynamicDimsElement& d) {
      return O << '(' << d.num << ", " << d.pos << ", " << d.previous_val
               << ')';
    }
  };
  using DynamicDimsFlat = std::vector<DynamicDimsElement>;

  struct DynamicDimsHelper {
    DynamicDims dd_;
    DimSizes rem_size_;
    DynamicDimsFlat flat_dd_;
    void FindOrAdd(int64_t num, int64_t pos, int64_t val);
    friend inline std::ostream& operator<<(
        std::ostream& O,
        const DynamicDimsHelper& d) {
      O << "dd :" << '\n' << d.dd_;
      O << "rem_size :";
      for (const auto& a : d.rem_size_) {
        O << ' ' << '(' << a.first << " -> " << a.second << ')';
      }
      O << "flat dd :";
      for (const auto& a : d.flat_dd_) {
        O << ' ' << a;
      }

      return O;
    }
  } dynamic_dims_;

  // Following map is only populated for exact shapes
  std::unordered_map<size_t, uint64_t> input_token_map_;
};

class UniqueTokenGenerator {
 public:
  static UniqueTokenGenerator& get_gen() {
    std::lock_guard<std::mutex> lg(mutex_);
    if (!instance_) {
      instance_ = new UniqueTokenGenerator();
    }
    return *instance_;
  }

  uint64_t token() {
    return ++current_token_;
  }

 private:
  UniqueTokenGenerator() = default;
  ~UniqueTokenGenerator() = default;
  UniqueTokenGenerator(const UniqueTokenGenerator&) = delete;
  UniqueTokenGenerator& operator=(const UniqueTokenGenerator&) = delete;

  static std::mutex mutex_;
  static UniqueTokenGenerator* instance_;
  static std::atomic_uint64_t current_token_;
};

}; // namespace habana_helpers
