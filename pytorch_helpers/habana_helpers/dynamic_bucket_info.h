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
using DimsHistory = std::vector<std::vector<int>>;

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
  uint64_t getTime() const {
    return average_time_;
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
  uint64_t getRunCount() const {
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
  DynamicRanges& ranges() {
    return ranges_;
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
  bool IsRefineCandidate() const {
    // Bucket with idx_ 0 is always a static bucket
    return (!IsStatic() && refine_candidate_);
  };
  void UpdateCompileTime(uint64_t t_ns) {
    compile_time_ += t_ns;
  }
  void UpdateRunTime(uint64_t t_ns);
  void IncrementHitCount() {
    hit_count_++;
  }
  void IncrementCumuHitCount() {
    cumu_hit_count_++;
  }
  void IncrementRunCount() {
    run_count_++;
    cumu_run_count_++;
  }

  inline std::string digest_str() const {
    // Present summary stats
    std::ostringstream O;
    O << "Bucket " << idx_ << '\n';
    if (idx_) {
      O << " " << dynamic_dims_ << " ranges : " << ranges_ << '\n';
    } else {
      O << " static dims" << '\n';
    }
    O << " hit count " << cumu_hit_count_ << ", miss count "
      << (cumu_run_count_ - cumu_hit_count_) << '\n'
      << " compile time stat : " << compile_time_ << '\n'
      << " run time stat     : " << run_time_stat_ << '\n';

    return O.str();
  }

  friend inline std::ostream& operator<<(std::ostream& O, const Bucket& b) {
    O << b.digest_str() << " score " << b.score_ << ',' << " recipe key "
      << b.recipe_key_ << ',' << " base_time " << b.base_time_ << ','
      << " token " << b.token_ << '\n';
    O << " split stat impl : " << b.split_stat_impl_;
    return O;
  }

  static constexpr uint64_t uninitialized_token = 1000000006;

 private:
  static constexpr uint64_t max_number_of_dims = 20;
  static constexpr uint64_t max_number_of_dims_fixed = sizeof(uint64_t);

  static constexpr double time_improve_factor_ = 1.0;
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
      DynamicDimsPolicy min_policy = DynamicDimsPolicy::HISTORIC,
      SplitPolicy sp = SplitPolicy::DYNAMIC);

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
      O << '\n' << "min" << r.min_shapes << "max" << r.max_shapes;
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
  inline std::string digest_str() const {
    // Present summary stats
    std::ostringstream O;
    O << "DynamicBucketInfo digest (times are in nano second)::" << '\n'
      << " [number of run times stats collected can be lesser than the total number of runs]"
      << '\n'
      << " hit count " << cumu_hit_count_ << ", miss count "
      << (cumu_run_count_ - cumu_hit_count_) << '\n'
      << " compile time stat : " << cumu_compile_time_stat_ << '\n'
      << " run time stat     : " << cumu_run_time_stat_ << '\n';
    O << "Individual bucket-wise digest ::" << '\n';
    for (const auto& b : buckets_) {
      O << b.digest_str();
    }
    return O.str();
  }
  friend inline std::ostream& operator<<(
      std::ostream& O,
      const DynamicBucketInfo& d) {
    O << d.digest_str() << "DynamicBucketInfo details::" << '\n'
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
  void SetJitIRGraphPtr(std::shared_ptr<torch::jit::Graph> jirpsh) {
    jit_ir_pwk = jirpsh;
  }
  void IncrementHitCount(size_t bucket_idx) {
    cumu_hit_count_++;
    buckets_.at(bucket_idx).IncrementHitCount();
    buckets_.at(bucket_idx).IncrementCumuHitCount();
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
  static constexpr uint64_t min_iterations_to_split_ = 100;
  static constexpr float density_coefficient_ = 0.75;

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

  // NOTE : Current implementation assumes immutability of indivual bucket.
  // Once created, individual buckets should not be copied to a local variable.
  // All modifications to the bucket should be done through the handler
  // functions in DynamicBucketInfo class.
  std::vector<Bucket> buckets_;
  std::queue<
      std::pair<std::shared_ptr<synapse_helpers::TimeSlotBase>, uint64_t>>
      run_time_states;
  std::pair<uint64_t, uint64_t> bucket_comparison;
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
