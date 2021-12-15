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
#include <iostream>
#include <limits>
#include <map>
#include <mutex>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "habana_helpers/tensor_shape.h"
#include "synapse_helpers/habana_tensor.h"
#include "synapse_helpers/stream.h"
#include "synapse_helpers/time_slot.h"
#include "torch/csrc/jit/ir/ir.h"

namespace habana_helpers {
const size_t max_elements_to_print = 64;
enum class SplitPolicy { UNSPECIFIED, DYNAMIC };

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

constexpr DynamicDimsPolicy MIN_POLICY_DEFAULT{DynamicDimsPolicy::HISTORIC};
constexpr DynamicDimsPolicy MAX_POLICY_DEFAULT{DynamicDimsPolicy::CALCULATED};

template <typename T, typename A>
inline std::ostream& operator<<(std::ostream& O, const std::vector<T, A>& V) {
  if (V.empty()) {
    O << "empty";
  } else {
    if (V.size() <= max_elements_to_print) {
      bool is_first(true);
      O << '[';
      for (auto a : V) {
        O << (is_first ? "" : " ") << a;
        is_first = false;
      }
      O << ']';
    } else {
      O << "has " << V.size() << " elements which is greater than"
        << " max_elements_to_print=" << max_elements_to_print
        << ", will skip printing";
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

// DynamicRanges: vector of <low, high> representing ranges
// This is a flat array, containing all ranges.
using DynamicRanges = std::vector<std::pair<int64_t, int64_t>>;
// DynamicDims : input_idx => {dim_idx => range_idx in DynamicRanges}
using DynamicDims = std::map<int64_t, std::map<int64_t, int64_t>>;
// Example
// Invocation 1: T0=[10,40, 45], T1=[30,60]
// Invocation 1: T0=[20,40, 55], T1=[30,80]
// For the above invocations DynamicRanges: <10,20>, <45,55>, <60,80>
// DynamicDims : [0->[0->0,
//                    2->1],
//                1-[1->2]]

// DimsHistoryElement : input_idx => {dim_idx => dim_val}
using DimsHistoryElement = std::map<int64_t, std::map<int64_t, int64_t>>;

// Only use for reference tensor shape
inline std::string DebugString(const DimsHistoryElement& d) {
  std::ostringstream O;
  for (auto tensor_it : d) {
    O << '\n' << " [";
    bool is_first{true};
    for (auto dim_it : tensor_it.second) {
      O << (is_first ? "" : ",");
      O << dim_it.second;
      is_first = false;
    }
    O << "]";
  }
  return O.str();
}

inline std::string DebugString(
    const DimsHistoryElement& d,
    const DimsHistoryElement& ref) {
  std::ostringstream O;
  for (auto tensor_it : ref) {
    const auto& tensor_idx{tensor_it.first};
    O << '\n' << " [";
    bool is_first{true};
    for (auto dim_it : tensor_it.second) {
      const auto& dim_idx{dim_it.first};
      auto dim_val{dim_it.second};
      if (d.count(tensor_idx) && d.at(tensor_idx).count(dim_idx)) {
        dim_val = d.at(tensor_idx).at(dim_idx);
      }
      O << (is_first ? "" : ",") << dim_val;
      is_first = false;
    }
    O << "]";
  }
  return O.str();
}

inline std::ostream& operator<<(std::ostream& O, const DynamicDims& d) {
  O << "dynamic dims ::";
  if (d.empty()) {
    O << ' ' << "empty";
  } else {
    for (const auto& r : d) {
      O << "  " << r.first << "->";
      bool is_first{true};
      O << '(';
      for (const auto& a : r.second) {
        O << (is_first ? "" : ",");
        O << a.first << "->" << a.second;
        is_first = false;
      }
      O << ')';
    }
  }
  O << '\n';

  return O;
}

inline std::ostream& operator<<(
    std::ostream& O,
    const std::map<int64_t, habana_helpers::TensorShape>& t) {
  for (const auto& a : t) {
    O << '\n' << " " << a.second;
  }
  return O;
}

inline std::ostream& operator<<(
    std::ostream& O,
    const std::unordered_map<int64_t, habana_helpers::TensorShape>& t) {
  std::vector<int64_t> tensor_idx_vec;
  tensor_idx_vec.reserve(t.size());
  for (const auto& a : t) {
    tensor_idx_vec.push_back(a.first);
  }
  for (const auto i : tensor_idx_vec) {
    O << "  " << i << ":" << t.at(i);
  }
  O << '\n';
  return O;
}

class TimeStat {
 public:
  TimeStat() = default;
  void Update(uint64_t elapsed_time) {
    total_time_ += elapsed_time;
    num_samples_++;
    average_time_ = total_time_ / num_samples_;
    min_time_ = std::min(min_time_, elapsed_time);
    max_time_ = std::max(max_time_, elapsed_time);
  }
  uint64_t GetAvgTime() const {
    return average_time_;
  }
  uint64_t GetMinTime() const {
    return min_time_;
  }
  uint64_t GetMaxTime() const {
    return max_time_;
  }

  friend inline std::ostream& operator<<(std::ostream& O, const TimeStat& t) {
    O << "<#samples=" << t.num_samples_ << " min="
      << (t.min_time_ == std::numeric_limits<uint64_t>::max() ? 0 : t.min_time_)
      << " max=" << t.max_time_ << " avg=" << t.average_time_
      << " total=" << t.total_time_ << '>';
    return O;
  }

 private:
  uint64_t total_time_{};
  uint64_t average_time_{};
  uint64_t min_time_{std::numeric_limits<uint64_t>::max()};
  uint64_t max_time_{};
  uint64_t num_samples_{};
};

struct SplitStatImplBase {
  SplitStatImplBase(SplitPolicy sp) : split_policy_(sp) {}
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
  virtual void ResetMax() = 0;
  virtual ~SplitStatImplBase() = 0;

  SplitPolicy split_policy_{SplitPolicy::UNSPECIFIED};
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
  void ResetMax() override {
    split_stat_impl_.erase(max_pos_);
    max_count_ = 0;
    for (auto& a : split_stat_impl_) {
      if (max_count_ < a.second) {
        max_count_ = a.second;
        max_pos_ = a.first;
      }
    }
  }

  size_t num_dyn_ranges_{1};
  uint64_t max_count_{0};
  std::vector<bool> max_pos_;
  std::unordered_map<std::vector<bool>, uint64_t> split_stat_impl_;
};

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

inline bool IsInRange(
    const DynamicRanges& ranges,
    const std::vector<int64_t>& dims,
    const std::set<int64_t>& skipped_ranges) {
  TORCH_CHECK(
      ranges.size() <= dims.size(),
      "wrong dynamic dims size ",
      dims.size(),
      ", expected greater or equal to ",
      ranges.size());

  for (size_t i = 0; i < ranges.size(); ++i) {
    if (skipped_ranges.find(i) != skipped_ranges.end()) {
      continue;
    }
    if (dims[i] < ranges[i].first || ranges[i].second < dims[i]) {
      return false;
    }
  }
  return true;
}

class Bucket {
 public:
  Bucket(
      DynamicRanges&& ranges,
      DynamicDims dynamic_dims,
      bool is_refine_enabled,
      SplitPolicy sp,
      const uint64_t base_time = 0);
  bool IsInRange(
      const std::vector<int64_t>& dims,
      const std::set<int64_t>& skipped_ranges) const;
  void IncStats(const std::vector<int64_t>& dims);
  void SetIndex(size_t i) {
    idx_ = i;
  }
  Bucket CreateNewBucket(SplitPolicy sp);
  uint64_t GetRunCount() const {
    return run_count_;
  }
  void ResetRunCount() {
    hit_count_ = 0;
    run_count_ = 0;
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
  const DynamicRanges& getRanges() const {
    return ranges_;
  }
  void setRanges(const DynamicRanges& r) {
    ranges_ = r;
  }
  size_t getDynamiDimsCount() const {
    return ranges_.size();
  }
  const DynamicDims& getDynamicDims() const {
    return dynamic_dims_;
  }
  // Stats related functions
  bool GetKeepRunTime() const {
    return keep_time_;
  };
  void SetKeepRunTime(bool flag) {
    keep_time_ = flag;
  };

  size_t GetRecipeKey() {
    return recipe_key_;
  };
  void SetRecipeKey(size_t key) {
    recipe_key_ = key;
  };

  bool IsStatic() const {
    return (idx_ == 0);
  }
  bool IsRefinementCandidate() const {
    // Bucket with idx_ 0 is always a static bucket
    return (!IsStatic() && refine_candidate_);
  };
  void UpdateCompileTime(uint64_t t_ns) {
    compile_time_ += t_ns;
  }
  void UpdateRunTime(uint64_t t_ns);
  void IncrementHitCount() {
    hit_count_++;
    cumu_hit_count_++;
  }
  void IncrementRunCount() {
    run_count_++;
    cumu_run_count_++;
  }

  inline std::string digest_str() const {
    // Present summary stats
    std::ostringstream O;
    O << " recipe key: " << recipe_key_ << '\n'
      << " hit count: " << cumu_hit_count_ << '\n'
      << " miss count: " << (cumu_run_count_ - cumu_hit_count_) << '\n';

    if (GET_ENV_FLAG_NEW(PT_ENABLE_SYNLAUNCH_TIME_CAPTURE)) {
      O << " compile time stat: " << compile_time_ << '\n'
        << " base_time: " << base_time_ << '\n'
        << " run time stat: " << run_time_stat_ << '\n';
    }

    return O.str();
  }

  static constexpr uint64_t uninitialized_token = 1000000006;

 private:
  static constexpr uint64_t max_number_of_dims = 20;
  static constexpr uint64_t max_number_of_dims_fixed = sizeof(uint64_t);

  static constexpr double time_improve_factor_ = 0.90;
  static constexpr double polarization_factor_ = 0.75;

  uint64_t score_{0};
  // hit_count_ tracks the number of cache hits for the associated recipe
  uint64_t hit_count_{0};
  // run_count_ tracks the number of launches for the associated recipe
  uint64_t run_count_{0};
  uint64_t token_{uninitialized_token};
  size_t idx_{0};
  size_t recipe_key_{};

  DynamicRanges ranges_;
  DynamicDims dynamic_dims_;
  std::shared_ptr<SplitStatImplBase> split_stat_impl_{nullptr};

  void CreateSplitStatImpl(SplitPolicy sp);

  // Stats related data members
  uint64_t compile_time_{};
  uint64_t cumu_hit_count_{0};
  uint64_t cumu_run_count_{0};
  TimeStat run_time_stat_;
  uint64_t base_time_{};

  bool keep_time_{true};
  bool refine_candidate_{true};
};

class DynamicBucketInfo {
 public:
  DynamicBucketInfo(
      DynamicDimsPolicy min_policy = MIN_POLICY_DEFAULT,
      DynamicDimsPolicy max_policy = MAX_POLICY_DEFAULT,
      SplitPolicy sp = SplitPolicy::DYNAMIC)
      : min_policy_(min_policy), max_policy_(max_policy), split_policy_(sp) {
    refine_enabled_ = GET_ENV_FLAG_NEW(PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES);
  };

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

    bool empty() const {
      return (min_shapes.empty() && max_shapes.empty());
    }
    // SynapseShapes syn_shapes;
    std::string DebugString();
    std::string DebugString(const InpTensorShapes& inp_shapes);
  };

  ResultShapes CalculateShapes(uint64_t bucket);

  void CollectDynamicDims(const InpTensorShapes& shapes);

  uint64_t GetBucketId(
      const InpTensorShapes& shapes,
      const PadShapes& pad_shapes = PadShapes{});
  absl::optional<uint64_t> CheckForSplitBucket();
  bool UpdateBucketWithPolicy(
      uint64_t bucket_id,
      const InpTensorShapes& shapes,
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
  std::string digest_str() const;
  friend inline std::ostream& operator<<(
      std::ostream& O,
      const DynamicBucketInfo& d) {
    O << d.digest_str();
    O << "Dims history len: " << d.dims_history_.size()
      << ", contents :" << '\n';
    bool skipped{false};
    size_t i{0};
    O << " history[" << i << "]:" << DebugString(d.ref_tensor_shapes_) << '\n';
    i += 1;
    for (; i < d.dims_history_.size(); i++) {
      const auto& a = d.dims_history_[i];
      if (a == d.dims_history_[i - 1]) {
        skipped = true;
        continue;
      }
      if (skipped) {
        skipped = false;
        O << "  "
          << "..." << '\n';
      }
      O << " history[" << i << "]:" << DebugString(a, d.ref_tensor_shapes_)
        << '\n';
    }
    if (skipped) {
      skipped = false;
      O << "  "
        << "..." << '\n';
    }
    O << "--------------------" << '\n';
    return O;
  }

  // Compile and launch related stats
  void UpdateCompileTime(uint64_t t_ns, uint64_t bucket_idx) {
    cumu_compile_time_stat_.Update(t_ns);
    cumu_compile_count_++;
    buckets_.at(bucket_idx).UpdateCompileTime(t_ns);
  }
  bool NeedRunTimeSlot(uint64_t bucket_idx);
  void RegisterTimeSlot(
      const std::shared_ptr<synapse_helpers::TimeSlotBase>& ts,
      int bucket);
  void UpdateRunTimes();

  size_t GetRecipeKeyForBucket(size_t bucket_idx) {
    return buckets_.at(bucket_idx).GetRecipeKey();
  };
  void SetRecipeKeyForBucket(size_t bucket_idx, size_t key) {
    buckets_.at(bucket_idx).SetRecipeKey(key);
  };
  void SetMinPolicy(DynamicDimsPolicy policy) {
    min_policy_ = policy;
  }
  void SetMaxPolicy(DynamicDimsPolicy policy) {
    max_policy_ = policy;
  }
  void SetDefaultPolicy() {
    min_policy_ = MIN_POLICY_DEFAULT;
    max_policy_ = MAX_POLICY_DEFAULT;
  }
  DynamicDimsPolicy GetMinPolicy() {
    return min_policy_;
  }
  DynamicDimsPolicy GetMaxPolicy() {
    return max_policy_;
  }
  void SetJitIRGraphPtr(std::shared_ptr<torch::jit::Graph> jirpsh) {
    jit_ir_pwk = jirpsh;
  }
  void IncrementHitCount(size_t bucket_idx) {
    cumu_hit_count_++;
    buckets_.at(bucket_idx).IncrementHitCount();
  }

  static uint64_t min_iterations_to_split() {
    return min_iterations_to_split_;
  }

  static int64_t default_min_value() {
    return default_min_value_;
  }

  static constexpr int64_t default_max_multiplier_ = 2;
  static constexpr int64_t default_min_value_ = 2;
  static constexpr uint64_t max_buckets_number_ = 20;
  static constexpr uint64_t min_iterations_to_split_ = 5;
  static constexpr float density_coefficient_ = 0.75;

 private:
  void UpdateMFUBucketDetails(uint64_t bucket_id);
  std::vector<int64_t> ExtractDynamicDimsValue(const InpTensorShapes& shapes);
  bool IsInRangeStaticDims(const std::vector<int64_t>& dims, int64_t num) const;
  int64_t GetMaxMultiplier(const PadShapes& pad_shapes);
  DimMultipliers CalculateFlattenedMultipliers(
      const InpTensorShapes& shapes,
      int64_t max_multiplier);

  // Following function determines the historic min/max depending on the
  // comparator and initial min/max value.
  // xin is used as a shortened form of max or min in valiable names.
  size_t CalculateHistoric(
      const InpTensorShapes& shapes,
      std::string xin_name,
      std::function<bool(int64_t, int64_t)> comp,
      int64_t xin_val);
  size_t CalculateHistoricMin(const InpTensorShapes& shapes) {
    return CalculateHistoric(
        shapes,
        std::string("Min"),
        std::greater<int64_t>(),
        std::numeric_limits<int64_t>::max());
  }
  size_t CalculateHistoricMax(const InpTensorShapes& shapes) {
    return CalculateHistoric(
        shapes,
        std::string("Max"),
        std::less<int64_t>(),
        std::numeric_limits<int64_t>::min());
  }

  DynamicRanges CalculateRanges(
      const InpTensorShapes& shapes,
      const PadShapes& pad_shapes);

  // NOTE : Current implementation assumes immutability of indivual bucket.
  // Once created, individual buckets should not be copied to a local variable.
  // All modifications to the bucket should be done through the handler
  // functions in DynamicBucketInfo class.
  std::vector<Bucket> buckets_;
  std::queue<
      std::pair<std::shared_ptr<synapse_helpers::TimeSlotBase>, uint64_t>>
      run_time_states;

  uint64_t global_count = 0;
  uint64_t mfu_bucket_id{0};
  uint64_t mfu_bucket_run_count{0};

  InpTensorShapes shapes_;
  DimsHistoryElement ref_tensor_shapes_;
  size_t prev_dynamic_dims_{};
  DynamicDimsPolicy min_policy_{MIN_POLICY_DEFAULT};
  DynamicDimsPolicy max_policy_{MAX_POLICY_DEFAULT};
  std::vector<DimsHistoryElement> dims_history_;
  bool refine_enabled_ = true;
  SplitPolicy split_policy_{SplitPolicy::DYNAMIC};

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
      O << "flatened view: " << (d.flat_dd_.size() ? "" : " empty");
      for (const auto& a : d.flat_dd_) {
        O << a;
      }
      O << '\n';

      O << d.dd_;
      O << "rem_size: " << (d.rem_size_.size() ? "" : " empty");
      for (const auto& a : d.rem_size_) {
        O << "  "
          << "Tensor" << a.first << ":" << a.second;
      }
      O << '\n';

      return O;
    }
  };

  DynamicDimsHelper dynamic_dims_;

  // Corresponding JIT IR graph
  std::weak_ptr<torch::jit::Graph> jit_ir_pwk;

  // TimeStat across all buckets
  TimeStat cumu_run_time_stat_;
  TimeStat cumu_compile_time_stat_;

  uint64_t cumu_compile_count_{};
  uint64_t cumu_run_count_{};
  uint64_t cumu_hit_count_{};

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
