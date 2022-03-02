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

#include "pytorch_helpers/habana_helpers/dynamic_bucket_info_utils.h"

#include <cstdint>

#include <algorithm>
#include <atomic>
#include <iostream>
#include <limits>
#include <mutex>

#include "torch/csrc/jit/ir/ir.h"

#include "pytorch_helpers/synapse_helpers/habana_tensor.h"
#include "pytorch_helpers/synapse_helpers/stream.h"
#include "pytorch_helpers/synapse_helpers/time_slot.h"

namespace habana {
class RecipeValueSpec;
}

namespace habana_helpers {
enum class SplitPolicy { UNSPECIFIED, DYNAMIC };

enum class CompilationPass {
  DYNAMIC_MIN,
  DYNAMIC_MAX,
  DYNAMIC_CURRENT,
  STATIC
};
enum class DynamicDimsPolicy {
  DEFAULT,
  CURRENT,
  CALCULATED,
  HISTORIC,
  FLATTENED
};

constexpr DynamicDimsPolicy MIN_POLICY_DEFAULT{DynamicDimsPolicy::HISTORIC};
constexpr DynamicDimsPolicy MAX_POLICY_DEFAULT{DynamicDimsPolicy::CALCULATED};

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
  size_t GetIndex() {
    return idx_;
  }
  void SetIndex(size_t i) {
    idx_ = i;
  }
  Bucket CreateNewBucket(SplitPolicy sp);

  uint64_t GetRunCount() const {
    return run_count_;
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
  uint64_t GetTime() const {
    return run_time_stat_.GetAvgTime();
  };
  size_t GetRecipeKey() {
    return recipe_key_;
  };
  void SetRecipeKey(size_t key) {
    recipe_key_ = key;
  };

  bool IsStatic() const {
    return (ranges_.empty());
  }
  bool IsRefinementCandidate() const {
    return (!IsStatic() && refine_candidate_ && time_improvement_met_);
  };
  void UpdateCompileTime(uint64_t t_ns) {
    compile_time_ += t_ns;
  }
  void UpdateRunTime(uint64_t t_ns);
  void ResetBaseLine(const HistoryItemLog& hist);
  void ResetRunCount() {
    run_count_ = input_hist_idxes_.size();
    if (split_stat_impl_) {
      split_stat_impl_->Reset();
    }
  }
  void IncrementHitCount() {
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
      << " total launch count: " << cumu_run_count_ << '\n'
      << " current launch count: " << run_count_ << '\n';

    O << "Historical input indices:" << '\n'
      << " inherited: " << inherited_input_hist_idxes_ << '\n'
      << " accumulated: " << input_hist_idxes_ << '\n';

    if (GET_ENV_FLAG_NEW(PT_ENABLE_SYNLAUNCH_TIME_CAPTURE)) {
      O << " compile time stat: " << compile_time_ << '\n'
        << " base_time: " << base_time_ << '\n'
        << " run time stat: " << run_time_stat_ << '\n';
    }

    return O.str();
  }

  void SetSynapseRecipePtr(std::shared_ptr<habana::RecipeValueSpec> rvpsh) {
    rvpwk_ = rvpsh;
  }
  std::shared_ptr<habana::RecipeValueSpec> GetSynapseRecipePtr() {
    return rvpwk_.lock();
  }

  void AppendInputHistIndex(size_t hidx) {
    input_hist_idxes_.push_back(hidx);
  }

  std::vector<size_t>& GetInputHistIdxes() {
    return input_hist_idxes_;
  };
  void SetInputHistIdxes(std::vector<size_t>& v) {
    std::swap(input_hist_idxes_, v);
    // input_hist_idxes_.clear();
    // input_hist_idxes_ = v;
  };

  std::vector<size_t>& GetInheritedInputHistIdxes() {
    return inherited_input_hist_idxes_;
  };
  void SetInheritedInputHistIdxes(std::vector<size_t>& v) {
    std::swap(inherited_input_hist_idxes_, v);
    // inherited_input_hist_idxes_.clear();
    // inherited_input_hist_idxes_ = v;
  };

  static constexpr uint64_t uninitialized_token = 1000000006;

 private:
  static constexpr double time_improve_factor_ = 0.90;
  static constexpr double polarization_factor_ = 0.75;

  uint64_t score_{0};
  uint64_t run_count_{0}; // tracks the number of launches

  uint64_t token_{uninitialized_token};
  size_t idx_{0};
  size_t parent_idx_{ULONG_MAX};
  size_t recipe_key_{0};
  bool is_first_launch_{true};

  DynamicRanges ranges_;
  DynamicDims dynamic_dims_;
  std::shared_ptr<SplitStatImplBase> split_stat_impl_{nullptr};

  void CreateSplitStatImpl(SplitPolicy sp);

  // Stats related data members
  uint64_t base_time_{0};
  uint64_t compile_time_{0};
  uint64_t cumu_hit_count_{0};
  uint64_t cumu_run_count_{0};

  TimeStat run_time_stat_;

  // Refinement related
  bool keep_time_{true};
  bool refine_candidate_{true};
  bool time_improvement_met_{true};
  std::vector<size_t> input_hist_idxes_;
  std::vector<size_t> inherited_input_hist_idxes_;

  std::weak_ptr<habana::RecipeValueSpec> rvpwk_;
};

class DynamicBucketInfo {
 public:
  DynamicBucketInfo();

  DynamicBucketInfo(DynamicDimsPolicy min_policy, DynamicDimsPolicy max_policy)
      : min_policy_(min_policy),
        max_policy_(max_policy),
        split_policy_(SplitPolicy::DYNAMIC) {
    refine_enabled_ = GET_ENV_FLAG_NEW(PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES);
  };

  using DimMultipliers =
      std::map<int64_t, std::map<int64_t, std::pair<int64_t, int64_t>>>;
  using DimSizes = std::map<int64_t, int64_t>;

  bool AreDynamicDimsContained() const {
    return buckets_.size() > 1;
  }

  ResultShapes CalculateShapes(uint64_t bucket);

  void CollectDynamicDims(const InpTensorShapes& shapes);

  size_t GetBucketId(
      const InpTensorShapes& shapes,
      const PadShapes& pad_shapes = PadShapes{});
  absl::optional<uint64_t> CheckForSplitBucket();
  Bucket ConstructNewBucket(
      ResultShapes& result_computed,
      const Bucket& mfu_bucket,
      size_t min_dist_idx,
      bool choose_lower);

  bool UpdateBucketWithPolicy(
      size_t bucket_id,
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
  std::string bucket_range_str(const Bucket& bucket, bool is_first = false)
      const;
  std::string digest_str() const;
  std::string history_str() const;
  friend inline std::ostream& operator<<(
      std::ostream& O,
      const DynamicBucketInfo& d) {
    O << d.digest_str();
    O << d.history_str();
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
      uint64_t bucket);
  void UpdateRunTimes();
  uint64_t GetTime(uint64_t bucket_idx) const {
    return buckets_.at(bucket_idx).GetTime();
  };
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
  size_t GetGraphKey() {
    return graph_key_;
  }
  void SetGraphKey(size_t key) {
    graph_key_ = key;
  }
  DynamicDimsPolicy GetMinPolicy() {
    return min_policy_;
  }
  DynamicDimsPolicy GetMaxPolicy() {
    return max_policy_;
  }
  std::shared_ptr<torch::jit::Graph> GetJitIRGraphPtr() {
    return jitirpwk_.lock();
  }
  void SetJitIRGraphPtr(std::shared_ptr<torch::jit::Graph> jirpsh) {
    jitirpwk_ = jirpsh;
  }
  void SetSynapseRecipePtr(
      size_t bidx,
      std::shared_ptr<habana::RecipeValueSpec> rvpsh) {
    buckets_.at(bidx).SetSynapseRecipePtr(rvpsh);
  }
  std::shared_ptr<habana::RecipeValueSpec> GetSynapseRecipePtr(size_t bidx) {
    return buckets_.at(bidx).GetSynapseRecipePtr();
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

  uint64_t getCount() {
    return global_count;
  }

  void split_history(
      const std::vector<size_t>& input_hist_idxes,
      const ResultShapes& new_result,
      std::vector<size_t>& input_hist_move,
      std::vector<size_t>& input_hist_retain);

  static constexpr int64_t default_max_multiplier_ = 2;
  static constexpr int64_t default_min_value_ = 2;
  static constexpr uint64_t max_buckets_number_ = 20;
  static constexpr uint64_t min_iterations_to_split_ = 5;
  static constexpr float density_coefficient_ = 0.75;

 private:
  void ComputeMFUBucketDetails();
  void UpdateMFUBucketDetails(size_t bucket_id);
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
  // std::queue<
  // std::pair<std::shared_ptr<synapse_helpers::TimeSlotBase>, uint64_t>>
  // run_time_states;
  // The following queue is used to store the time events for capturing
  // the time spent by a recipe in the compute stream
  // Each element is a tuple of time_slot, bucket_id, input_history_idx
  std::queue<
      std::
          tuple<std::shared_ptr<synapse_helpers::TimeSlotBase>, size_t, size_t>>
      run_time_q_;

  uint64_t global_count = 0;
  uint64_t mfu_bucket_id{0};
  uint64_t mfu_bucket_run_count{0};
  uint64_t current_run_count{0};

  InpTensorShapes shapes_;
  DynamicDimsPolicy min_policy_{MIN_POLICY_DEFAULT};
  DynamicDimsPolicy max_policy_{MAX_POLICY_DEFAULT};
  bool refine_enabled_ = true;
  SplitPolicy split_policy_{SplitPolicy::DYNAMIC};

  HistoryItemLog input_history_;
  size_t current_input_idx_{ULONG_MAX};

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

  DynamicDimsHelper dynamic_dims_helper_;

  // Corresponding JIT IR graph
  size_t graph_key_{};
  std::weak_ptr<torch::jit::Graph> jitirpwk_;

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

} // namespace habana_helpers
