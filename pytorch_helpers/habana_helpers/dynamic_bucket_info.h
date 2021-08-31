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
enum class SplitPolicy { UNSPECIFIED, DEFAULT, DYNAMIC };

enum class CompilationPass {
  DYNAMIC_MIN,
  DYNAMIC_MAX,
  DYNAMIC_CURRENT,
  STATIC
};
enum class DynamicDimsPolicy {
  DEFAULT,
  CALCULATED,
  HISTORIC,
  FLATTENED,
  CURRENT
};

template <typename T, typename A>
inline std::ostream& operator<<(std::ostream& O, const std::vector<T, A>& V) {
  if (V.empty()) {
    O << "empty";
  } else {
    bool is_first(true);
    for (auto a : V) {
      O << (is_first ? "" : " ") << a;
      is_first = false;
    }
  }
  return O;
}

inline std::ostream& operator<<(std::ostream& O, const std::vector<bool>& V) {
  if (V.empty()) {
    O << "empty";
  } else {
    for (auto a : V) {
      O << a;
    }
  }
  return O;
}

template <typename T, typename U>
inline std::ostream& operator<<(
    std::ostream& O,
    const std::vector<std::pair<T, U>>& V) {
  if (V.empty()) {
    O << "empty";
  } else {
    bool is_first(true);
    for (const auto& a : V) {
      O << (is_first ? "" : " ") << '(' << a.first << ", " << a.second << ')';
      is_first = false;
    }
  }
  return O;
}

inline std::ostream& operator<<(std::ostream& O, const SplitPolicy& p) {
  switch (p) {
    case SplitPolicy::UNSPECIFIED:
      O << "UNSPECIFIED";
      break;
    case SplitPolicy::DEFAULT:
      O << "DEFAULT";
      break;
    case SplitPolicy::DYNAMIC:
      O << "DYNAMIC";
      break;
  }
  return O;
}

inline std::string DebugString(const DynamicDimsPolicy& d) {
  switch (d) {
    case DynamicDimsPolicy::DEFAULT:
      return std::string("DEFAULT");
    case DynamicDimsPolicy::CALCULATED:
      return std::string("CALCULATED");
    case DynamicDimsPolicy::HISTORIC:
      return std::string("HISTORIC");
    case DynamicDimsPolicy::FLATTENED:
      return std::string("FLATTENED");
    case DynamicDimsPolicy::CURRENT:
      return std::string("CURRENT");
  }
  return std::string();
}

inline std::ostream& operator<<(std::ostream& O, const DynamicDimsPolicy& d) {
  return O << DebugString(d);
}

using DynamicDims =
    std::map<int64_t, std::map<int64_t, int64_t>>; // input_idx => {dim_idx =>
                                                   // range_idx}
using DynamicRanges = std::vector<std::pair<int64_t, int64_t>>;

inline std::ostream& operator<<(std::ostream& O, const DynamicDims& d) {
  O << "dynamic dims ::";
  if (d.empty()) {
    O << ' ' << "empty" << '\n';
  } else {
    O << '\n';
    for (const auto& r : d) {
      O << "  " << r.first << " -> ";
      for (const auto& a : r.second) {
        O << ' ' << '(' << a.first << " -> " << a.second << ')';
      }
      O << '\n';
    }
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

struct SplitStatImplBase {
  SplitStatImplBase(SplitPolicy sp = SplitPolicy::UNSPECIFIED)
      : split_policy_(sp) {}
  SplitPolicy get_policy() {
    return split_policy_;
  }

  virtual void Increment(
      const DynamicRanges& ranges,
      const std::vector<int64_t>& dims) = 0;
  virtual void CalculateNewRanges(
      const DynamicRanges& ranges,
      DynamicRanges& new_ranges) = 0;
  virtual void Reset() = 0;
  virtual ~SplitStatImplBase() = 0;

  SplitPolicy split_policy_{SplitPolicy::UNSPECIFIED};
};

struct SplitStatImplDefault : public SplitStatImplBase {
  SplitStatImplDefault(size_t m = 0) : SplitStatImplBase(SplitPolicy::DEFAULT) {
    split_count_.resize(1 << m);
  }
  void Increment(const DynamicRanges& ranges, const std::vector<int64_t>& dims)
      override;
  void CalculateNewRanges(
      const DynamicRanges& ranges,
      DynamicRanges& new_ranges) override;
  void Reset() override {
    std::fill(split_count_.begin(), split_count_.end(), 0);
  }
  std::vector<int64_t> split_count_;
};

struct SplitStatImplDynamic : public SplitStatImplBase {
  SplitStatImplDynamic(size_t m = 1) : SplitStatImplBase(SplitPolicy::DYNAMIC) {
    num_dyn_ranges_ = m;
    max_pos_.resize(m, 0);
  }
  void Increment(const DynamicRanges& ranges, const std::vector<int64_t>& dims)
      override;
  void CalculateNewRanges(
      const DynamicRanges& ranges,
      DynamicRanges& new_ranges) override;
  void Reset() override {
    max_count_ = 0;
    std::fill(max_pos_.begin(), max_pos_.end(), 0);
    split_stat_impl_.clear();
  }

  size_t num_dyn_ranges_{1};
  uint64_t max_count_{0};
  std::vector<bool> max_pos_;
  std::unordered_map<std::vector<bool>, uint64_t> split_stat_impl_;
};

inline std::ostream& operator<<(
    std::ostream& O,
    const std::shared_ptr<SplitStatImplDefault>& s) {
  O << "split_count [" << s->split_count_ << "]";
  return O;
}

inline std::ostream& operator<<(
    std::ostream& O,
    const std::shared_ptr<SplitStatImplDynamic>& s) {
  O << "num dyn ranges " << s->num_dyn_ranges_ << ", max_count "
    << s->max_count_ << ", max_pos [" << s->max_pos_ << "]" << '\n';
  O << " pos to count map :";
  if (s->split_stat_impl_.empty()) {
    O << ' ' << "empty";
  } else {
    for (const auto& a : s->split_stat_impl_) {
      O << ' ' << '(' << a.first << " -> " << a.second << ')';
    }
  }

  return O;
}

inline std::ostream& operator<<(
    std::ostream& O,
    const std::shared_ptr<SplitStatImplBase>& s) {
  if (s) {
    O << "split policy " << s->split_policy_ << ", ";
    switch (s->split_policy_) {
      case SplitPolicy::UNSPECIFIED:
        O << " invalid SplitStatImpl used" << '\n';
        break;
      case SplitPolicy::DEFAULT: {
        std::shared_ptr<SplitStatImplDefault> spsh =
            std::dynamic_pointer_cast<SplitStatImplDefault>(s);
        O << spsh;
      } break;
      case SplitPolicy::DYNAMIC: {
        std::shared_ptr<SplitStatImplDynamic> spsh =
            std::dynamic_pointer_cast<SplitStatImplDynamic>(s);
        O << spsh;
      } break;
    }
  } else {
    O << "uninstantiated";
  }
  O << '\n';
  return O;
}

inline std::ostream& operator<<(std::ostream& O, const SplitStatImplBase& S);

struct InputOutputShapes {
  habana_helpers::TensorShape input;
  habana_helpers::TensorShape output;
};
using PadShapes = std::unordered_map<int64_t, InputOutputShapes>;

class Bucket {
 public:
  Bucket(
      DynamicRanges&& ranges,
      DynamicDims dynamic_dims,
      bool is_refine_allowed,
      SplitPolicy sp);
  bool IsInRange(
      const std::vector<int64_t>& dims,
      const std::set<int64_t>& skipped_ranges) const;
  void IncStats(const std::vector<int64_t>& dims);
  void SetIndex(size_t i) {
    idx = i;
  }
  Bucket CreateNewBucket(SplitPolicy sp);
  uint64_t getCount() const {
    return count_;
  }
  void ResetCount() {
    count_ = 0;
    if (split_stat_impl_) {
      split_stat_impl_->Reset();
    }
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
    O << "Bucket " << b.idx << " ::"
      << " score " << b.score_ << ',' << " count " << b.count_ << ','
      << " token " << b.token_ << '\n';
    O << " split stat impl : " << b.split_stat_impl_;
    O << " " << b.dynamic_dims_;
    O << " ranges : " << b.ranges_ << '\n';
    return O;
  }

  static constexpr uint64_t uninitialized_token = 1000000006;

 private:
  static constexpr uint64_t max_number_of_dims = 20;
  static constexpr uint64_t max_number_of_dims_fixed = sizeof(uint64_t);

  uint64_t score_{0};
  uint64_t count_{0};
  uint64_t token_{uninitialized_token};
  size_t idx{0};

  DynamicRanges ranges_;
  DynamicDims dynamic_dims_;

  std::shared_ptr<SplitStatImplBase> split_stat_impl_{nullptr};

  void CreateSplitStatImpl(SplitPolicy sp);
};

class DynamicBucketInfo {
 public:
  DynamicBucketInfo(
      DynamicDimsPolicy min_policy = DynamicDimsPolicy::HISTORIC,
      SplitPolicy sp = SplitPolicy::DEFAULT);

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
    O << "DynamicBucketInfo ::" << '\n'
      << " global_count=" << d.global_count
      << ", prev_dynamic_dims=" << d.prev_dynamic_dims_
      << ", refine_enabled=" << std::boolalpha << d.refine_enabled_
      << std::noboolalpha << '\n';
    O << " min policy=" << d.min_policy_ << ", max policy=" << d.max_policy_
      << ", split policy=" << d.split_policy_ << '\n';
    O << "Input tensor shapes ::" << '\n';
    for (const auto& a : d.shapes_) {
      O << "  " << a.first << " -> " << a.second << '\n';
    }
    O << "List of buckets ::" << '\n' << ' ' << d.buckets_;
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
    O << d.dynamic_dims_ << '\n';
    O << '\n';
    return O;
  }

  static uint64_t min_iterations_to_split() {
    return min_iterations_to_split_;
  }

  static int64_t default_min_value() {
    return default_min_value_;
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
  // TODO: Check default_min_value_
  static constexpr int64_t default_min_value_ = 2;
  static constexpr uint64_t max_buckets_number_ = 20;
  static constexpr uint64_t min_iterations_to_split_ = 100;
  static constexpr float density_coefficient_ = 0.75;

  // NOTE : Current implementation assumes immutability of indivual bucket.
  // Once created, individual buckets should not be copied to a local variable.
  // All modifications to the bucket should be done through the handler
  // functions in DynamicBucketInfo class.
  // If this semantics needs to be changed, create a vector of shared_ptr of
  // buckets.
  std::vector<Bucket> buckets_;
  uint64_t global_count = 0;
  InpTensorShapes shapes_;
  size_t prev_dynamic_dims_{};
  DynamicDimsPolicy min_policy_{DynamicDimsPolicy::HISTORIC};
  DynamicDimsPolicy max_policy_{DynamicDimsPolicy::CALCULATED};
  std::vector<std::vector<int64_t>> dim_history_;
  bool refine_enabled_ = true;
  SplitPolicy split_policy_{SplitPolicy::DEFAULT};

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
      O << "dd : " << d.dd_;
      O << "rem_size :" << '\n';
      for (const auto& a : d.rem_size_) {
        O << "   " << '(' << a.first << " -> " << a.second << ')' << '\n';
      }
      O << "flat dd :" << '\n';
      for (const auto& a : d.flat_dd_) {
        O << "   " << a << '\n';
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
