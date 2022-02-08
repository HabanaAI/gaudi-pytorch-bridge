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
#include "dynamic_bucket_info.h"

#include <cmath>
#include <cstddef>

#include <algorithm>
#include <limits>
#include <memory>

#include <absl/types/variant.h>

#include "habana_bridge/kernel/ds_graph_recompile.h"
#include "habana_bridge/kernel/hpu_habana_cache.h"

#include "pytorch_helpers/habana_device/HPUCheck.h"
#include "pytorch_helpers/habana_helpers/compilation_statistics.h"
#include "pytorch_helpers/synapse_helpers/env_flags.h"

using namespace synapse_helpers;
namespace habana_helpers {

constexpr int64_t DynamicBucketInfo::default_min_value_;
constexpr uint64_t DynamicBucketInfo::min_iterations_to_split_;
constexpr uint64_t DynamicBucketInfo::max_buckets_number_;

std::mutex UniqueTokenGenerator::mutex_;
UniqueTokenGenerator* UniqueTokenGenerator::instance_{nullptr};
std::atomic_uint64_t UniqueTokenGenerator::current_token_{
    Bucket::uninitialized_token};

SplitStatImplBase::~SplitStatImplBase() {}

void Bucket::CreateSplitStatImpl(SplitPolicy sp) {
  switch (sp) {
    case SplitPolicy::UNSPECIFIED:
      TORCH_CHECK(false, "Can not create Bucket with policy : ", sp);
      break;
    case SplitPolicy::DYNAMIC:
      split_stat_impl_ = std::make_shared<SplitStatImplDynamic>(ranges_.size());
      break;
  }
}

void SplitStatImplDynamic::Increment(
    const DynamicRanges& ranges,
    const std::vector<int64_t>& dims) {
  if (0 == num_dyn_ranges_)
    return;

  TORCH_CHECK(
      ranges.size() <= dims.size(),
      "wrong dynamic dims size ",
      dims.size(),
      ", expected ",
      ranges.size());

  std::vector<bool> pos(num_dyn_ranges_, 0);
  for (size_t i = 0; i < ranges.size(); ++i) {
    auto mid{(ranges[i].second + ranges[i].first) / 2};
    pos[i] = (dims[i] > mid ? 1 : 0);
  }

  if (split_stat_impl_.count(pos) == 0) {
    split_stat_impl_.emplace(pos, 0);
  }
  split_stat_impl_.at(pos) += 1;
  if (max_count_ < split_stat_impl_.at(pos)) {
    max_count_ = split_stat_impl_.at(pos);
    max_pos_ = pos;
  }
}

void SplitStatImplDynamic::CalculateNewRanges(
    const DynamicRanges& ranges,
    DynamicRanges& new_ranges) {
  TORCH_CHECK(
      ranges.size() == num_dyn_ranges_,
      "wrong dynamic dims size ",
      ranges.size(),
      ", expected ",
      num_dyn_ranges_);

  for (size_t i = 0; i < ranges.size(); i++) {
    auto& el = ranges[i];
    int64_t mid = (el.second + el.first) / 2;
    if (max_pos_[i] == 0) {
      new_ranges.emplace_back(el.first, mid);
    } else {
      new_ranges.emplace_back(mid, el.second);
    }
  }
}

Bucket::Bucket(
    DynamicRanges&& ranges,
    DynamicDims dynamic_dims,
    bool is_refine_enabled,
    SplitPolicy sp,
    const uint64_t base_time)
    : ranges_(std::move(ranges)),
      dynamic_dims_(std::move(dynamic_dims)),
      base_time_(base_time),
      refine_candidate_(is_refine_enabled) {
  if (is_refine_enabled) {
    CreateSplitStatImpl(sp);
  }
  for (auto& el : ranges_)
    score_ += el.second - el.first;
  token_ = habana_helpers::UniqueTokenGenerator::get_gen().token();
}

bool Bucket::IsInRange(
    const std::vector<int64_t>& dims,
    const std::set<int64_t>& skipped_ranges) const {
  TORCH_CHECK(
      ranges_.size() <= dims.size(),
      "wrong dynamic dims size ",
      dims.size(),
      ", expected greater or equal to ",
      ranges_.size());

  for (size_t i = 0; i < ranges_.size(); ++i) {
    if (skipped_ranges.find(i) != skipped_ranges.end()) {
      continue;
    }
    if (dims[i] < ranges_[i].first || ranges_[i].second < dims[i]) {
      return false;
    }
  }
  return true;
}

void Bucket::UpdateRunTime(uint64_t elapsed_time) {
  if (is_first_launch_) {
    is_first_launch_ = false;
    return;
  }

  run_time_stat_.Update(elapsed_time);
  if (base_time_ > 0) {
    uint64_t time_to_beat = static_cast<uint64_t>(
        static_cast<double>(base_time_) * time_improve_factor_);

    auto cur_avg_time{run_time_stat_.GetAvgTime()};
    time_improvement_met_ = (cur_avg_time < time_to_beat);
  }
};

void Bucket::IncStats(const std::vector<int64_t>& dims) {
  IncrementRunCount();
  if (ranges_.empty() || nullptr == split_stat_impl_) {
    return;
  }

  split_stat_impl_->Increment(ranges_, dims);
}

Bucket Bucket::CreateNewBucket(SplitPolicy sp) {
  TORCH_CHECK(
      nullptr != split_stat_impl_, "Dynamic bucket : Refine stage is disabled");

  DynamicRanges new_ranges;
  split_stat_impl_->CalculateNewRanges(ranges_, new_ranges);
  split_stat_impl_->ResetMax();
  return Bucket(std::move(new_ranges), dynamic_dims_, true, sp);
}

void Bucket::ResetBaseLine(const HistoryItemLog& hist) {
  // Recompute the base_time based on the content of:
  // 1. input_hist_idxes_, and
  // 2. inherited_input_hist_idxes_

  uint64_t total_run_time{0};
  uint64_t cnt{0};

  for (auto i : inherited_input_hist_idxes_) {
    auto hist_run_time{hist[i].run_time()};
    if (hist_run_time > 0) {
      total_run_time += hist_run_time;
      cnt++;
    }
  }

  base_time_ = 0;
  if (cnt > 0) {
    base_time_ = total_run_time / cnt;
  }

  run_time_stat_.Reset();
  for (auto i : input_hist_idxes_) {
    auto hist_run_time{hist[i].run_time()};
    if (hist_run_time > 0) {
      run_time_stat_.Update(hist_run_time);
    }
  }

  ResetRunCount();
}

DynamicBucketInfo::DynamicBucketInfo()
    : min_policy_(DynamicDimsPolicy::HISTORIC),
      max_policy_(DynamicDimsPolicy::CALCULATED),
      split_policy_(SplitPolicy::DYNAMIC) {
  refine_enabled_ = GET_ENV_FLAG_NEW(PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES);
  if (GET_ENV_FLAG_NEW(PT_HPU_ENABLE_MIN_MAX_AS_CURRENT)) {
    min_policy_ = DynamicDimsPolicy::CURRENT;
    max_policy_ = DynamicDimsPolicy::CURRENT;
    return;
  }
  auto min_policy_num = GET_ENV_FLAG_NEW(PT_HPU_DYNAMIC_MIN_POLICY_DEFAULT);
  switch (min_policy_num) {
    case 1:
      min_policy_ = DynamicDimsPolicy::CURRENT;
      break;
    case 3:
      min_policy_ = DynamicDimsPolicy::HISTORIC;
      break;
    default:
      PT_DYNAMIC_SHAPE_WARN(
          "Invalid min policy value ",
          min_policy_num,
          " specified\n.",
          "  Supported values are 1:CURRENT, 3:HISTORIC\n",
          "  Default min policy ",
          min_policy_,
          " will be used");
  }
  auto max_policy_num = GET_ENV_FLAG_NEW(PT_HPU_DYNAMIC_MAX_POLICY_DEFAULT);
  switch (max_policy_num) {
    case 1:
      max_policy_ = DynamicDimsPolicy::CURRENT;
      break;
    case 2:
      max_policy_ = DynamicDimsPolicy::CALCULATED;
      break;
    case 3:
      max_policy_ = DynamicDimsPolicy::HISTORIC;
      break;
    default:
      PT_DYNAMIC_SHAPE_WARN(
          "Invalid max policy value ",
          max_policy_num,
          " specified\n.",
          "  Supported values are 1:CURRENT, 2:CALCULATED, 3:HISTORIC\n",
          "  Default max policy ",
          max_policy_,
          " will be used");
  }
}

ResultShapes DynamicBucketInfo::CalculateShapes(uint64_t bucket) {
  ResultShapes result(shapes_);
  TORCH_CHECK(
      bucket < buckets_.size(),
      "Invalid bucket index ",
      bucket,
      " encountered, should be less than ",
      buckets_.size());

  auto& ranges = buckets_[bucket].getRanges();
  auto& dynamic_dims = buckets_[bucket].getDynamicDims();

  for (auto& input : dynamic_dims) {
    auto shape_min = shapes_.at(input.first);
    auto shape_max = shape_min;

    for (auto dim : input.second) {
      shape_min.set_dim(dim.first, ranges[dim.second].first);
      shape_max.set_dim(dim.first, ranges[dim.second].second);
    }
    result.min_shapes[input.first] = shape_min;
    result.max_shapes[input.first] = shape_max;
  }

  return result;
}

void DynamicBucketInfo::CollectDynamicDims(const InpTensorShapes& new_shapes) {
  if (shapes_.empty()) {
    shapes_ = new_shapes;
    auto& ref_tshapes{input_history_.ref_tshapes()};
    // Populate refrerence DimsHistoryElement
    for (auto tensor_it = shapes_.cbegin(); tensor_it != shapes_.cend();
         tensor_it++) {
      auto tensor_idx{tensor_it->first};
      ref_tshapes.emplace(tensor_idx, std::map<int64_t, int64_t>());
      for (size_t dim_idx = 0; dim_idx < tensor_it->second.dims(); dim_idx++) {
        auto dim_val{tensor_it->second.dim_size(dim_idx)};
        ref_tshapes[tensor_idx].emplace(dim_idx, dim_val);
      }
    }
  }

  TORCH_CHECK(
      shapes_.size() == new_shapes.size(),
      "new input shapes size ",
      new_shapes.size(),
      " is not matching with existing shapes size ",
      shapes_.size());
  for (auto tensor_it_ref = shapes_.cbegin(),
            tensor_it_new = new_shapes.cbegin();
       tensor_it_ref != shapes_.cend();
       ++tensor_it_ref, ++tensor_it_new) {
    dynamic_dims_helper_.rem_size_[tensor_it_ref->first] = 1;
    for (size_t i = 0; i < tensor_it_ref->second.dims(); ++i) {
      if (tensor_it_ref->second.dim_size(i) !=
          tensor_it_new->second.dim_size(i)) {
        dynamic_dims_helper_.FindOrAdd(
            tensor_it_ref->first,
            i,
            shapes_.at(tensor_it_ref->first).dim_size(i));
      } else if (tensor_it_ref->second.dim_size(i) > 1) {
        // For avoiding the dimension with value 0
        dynamic_dims_helper_.rem_size_[tensor_it_ref->first] *=
            tensor_it_ref->second.dim_size(i);
      }
    }
  }
}

void DynamicBucketInfo::ComputeMFUBucketDetails() {
  current_run_count = 0;
  mfu_bucket_run_count = 0;
  mfu_bucket_id = 0;
  for (auto& b : buckets_) {
    // Skip the refinement of the static bucket
    if (b.IsStatic()) {
      PT_TEST_DEBUG(__FUNCTION__, ": Skipping a static bucket.");
      continue;
    }
    uint64_t cur_bucket_run_count{b.GetRunCount()};
    if (mfu_bucket_run_count < cur_bucket_run_count) {
      mfu_bucket_run_count = cur_bucket_run_count;
      mfu_bucket_id = b.GetIndex();
    }
  }
}

void DynamicBucketInfo::UpdateMFUBucketDetails(size_t bucket_id) {
  // Skip the refinement of the static bucket
  if (buckets_[bucket_id].IsStatic()) {
    PT_TEST_DEBUG(__FUNCTION__, ": Skipping a static bucket.");
    return;
  }
  current_run_count += 1;
  uint64_t cur_bucket_run_count{buckets_[bucket_id].GetRunCount()};
  if (mfu_bucket_run_count < cur_bucket_run_count) {
    mfu_bucket_run_count = cur_bucket_run_count;
    mfu_bucket_id = bucket_id;
  }
}

size_t DynamicBucketInfo::GetBucketId(
    const InpTensorShapes& shapes,
    const PadShapes& pad_shapes) {
  TORCH_CHECK(shapes_.size() == shapes.size(), "Shapes dont match");

  cumu_run_count_++;
  if (buckets_.empty()) {
    buckets_.emplace_back(DynamicRanges{}, DynamicDims{}, true, split_policy_);
    auto& new_bucket = buckets_.back();
    new_bucket.IncStats({});

    uint64_t new_bucket_id = buckets_.size() - 1;
    new_bucket.SetIndex(new_bucket_id);

    // Start the history log
    input_history_.hist_items().emplace_back(
        DimsHistoryElement{}, new_bucket_id, 0);
    current_input_idx_ = 0;

    buckets_[new_bucket_id].AppendInputHistIndex(current_input_idx_);

    return new_bucket_id;
  }
  global_count++;
  auto dims = ExtractDynamicDimsValue(shapes);

  absl::optional<uint64_t> best_bucket{};
  std::set<int64_t> skipped_ranges;
  for (uint i = 0; i < dynamic_dims_helper_.flat_dd_.size(); i++) {
    if (pad_shapes.find(dynamic_dims_helper_.flat_dd_[i].num) !=
        pad_shapes.end()) {
      skipped_ranges.insert(i);
    }
  }

  for (size_t i = 0; i < buckets_.size(); i++) {
    bool in_range = buckets_[i].IsInRange(dims, skipped_ranges) &&
        IsInRangeStaticDims(dims, buckets_[i].getDynamiDimsCount());
    // Choose a box with lower score meaning narrower ranges
    if (in_range &&
        ((best_bucket.has_value() &&
          buckets_[best_bucket.value()].getScore() > buckets_[i].getScore()) ||
         !best_bucket.has_value()))
      best_bucket = i;
  }

  if (best_bucket.has_value()) {
    uint64_t best_bucket_id = best_bucket.value();
    buckets_[best_bucket_id].IncStats(dims);
    UpdateMFUBucketDetails(best_bucket_id);

    buckets_[best_bucket_id].AppendInputHistIndex(current_input_idx_);
    input_history_.hist_items_[current_input_idx_].bucket_index_ =
        best_bucket_id;
    return best_bucket_id;
  }

  // Create new bucket
  auto ranges = CalculateRanges(shapes, pad_shapes);
  buckets_.emplace_back(
      std::move(ranges),
      dynamic_dims_helper_.dd_,
      refine_enabled_,
      split_policy_);
  auto& new_bucket = buckets_.back();
  new_bucket.IncStats(dims);

  uint64_t new_bucket_id = buckets_.size() - 1;
  new_bucket.SetIndex(new_bucket_id);
  buckets_[new_bucket_id].AppendInputHistIndex(current_input_idx_);
  // Update the bucket id of the history item
  input_history_.hist_items_[current_input_idx_].bucket_index_ = new_bucket_id;
  UpdateMFUBucketDetails(new_bucket_id);

  return buckets_.size() - 1;
}

absl::optional<uint64_t> DynamicBucketInfo::CheckForSplitBucket() {
  PT_TEST_DEBUG("Checking buckets for refinement");
  if (refine_enabled_ == false) {
    PT_TEST_DEBUG("Refinement is not enabled");
    return {};
  }
  if (buckets_.size() >= max_buckets_number_) {
    PT_TEST_DEBUG(
        "Maxed out total number=", max_buckets_number_, " of buckets");
    return {};
  }
  if (current_run_count < min_iterations_to_split_) {
    PT_TEST_DEBUG(
        "Yet to reach ",
        min_iterations_to_split_,
        " for graph, currently at ",
        current_run_count);
    return {};
  }
  if (mfu_bucket_run_count < min_iterations_to_split_) {
    PT_TEST_DEBUG(
        "Yet to reach ",
        min_iterations_to_split_,
        " for mfu bucket, currently at ",
        mfu_bucket_run_count);
    return {};
  }

  if (mfu_bucket_id == 0) {
    PT_TEST_DEBUG("Can not refine static bucket");
    return {};
  }

  auto& mfu_bucket = buckets_[mfu_bucket_id];
  if (mfu_bucket.IsRefinementCandidate() == false) {
    PT_TEST_DEBUG(
        "Bucket ", mfu_bucket_id, " is not a candidate for refinement");
    return {};
  }

  PT_TEST_DEBUG("Current mfu bucket is eligible for refinement");
  auto rvpsh = mfu_bucket.GetSynapseRecipePtr();
  if (nullptr == rvpsh) {
    PT_TEST_DEBUG("Recipe for mfu bucket is null");
    return {};
  }

  size_t min_dist_idx{};
  bool choose_lower{};
  std::tie(min_dist_idx, choose_lower) =
      input_history_.FindMidPoint(mfu_bucket.GetInputHistIdxes());
  PT_TEST_DEBUG_TH(
      "Nearest history item from mid point is history[",
      min_dist_idx,
      "], choose lower: ",
      choose_lower);

  // Use the split history input as lo or hi depending on choose_lower
  ResultShapes result_computed(shapes_);
  Bucket new_bucket_computed = ConstructNewBucket(
      result_computed, mfu_bucket, min_dist_idx, choose_lower);
  PT_TEST_DEBUG_TH(
      "With new method, input range for new bucket:\n",
      "Min\n",
      result_computed.min_shapes,
      "Max\n",
      result_computed.max_shapes,
      "--------------------");

  Bucket& new_bucket_candidate{new_bucket_computed};
  ResultShapes& new_range{result_computed};
  bool is_compiled{false};
  size_t new_recipe_key{0};
  try {
    is_compiled = habana::CompileGraphWithRange(
        rvpsh, new_range, new_bucket_candidate, new_recipe_key);
  } catch (std::exception& e) {
    PT_TEST_DEBUG("Recipe compilation failed with exception '", e.what(), "'");
    return {};
  }
  PT_TEST_DEBUG(
      "Recipe compilation for new bucket: ",
      (is_compiled ? "successful" : "failed"));

  // Only push this bucket if the compilation is successful
  if (is_compiled) {
    PT_TEST_DEBUG(
        "Compiled new bucket with input range:\n",
        "Min\n",
        new_range.min_shapes,
        "Max\n",
        new_range.max_shapes,
        "--------------------");
    // Move the history
    // Find the previous hits
    auto& input_hist_idxes{mfu_bucket.GetInputHistIdxes()};
    std::vector<size_t> input_hist_move;
    std::vector<size_t> input_hist_retain;
    split_history(
        input_hist_idxes, new_range, input_hist_move, input_hist_retain);

    auto& inherited_input_hist_idxes{mfu_bucket.GetInheritedInputHistIdxes()};
    std::vector<size_t> inherited_input_hist_move;
    std::vector<size_t> inherited_input_hist_retain;
    split_history(
        inherited_input_hist_idxes,
        new_range,
        inherited_input_hist_move,
        inherited_input_hist_retain);

    mfu_bucket.SetInputHistIdxes(input_hist_retain);
    mfu_bucket.SetInheritedInputHistIdxes(inherited_input_hist_retain);
    mfu_bucket.ResetBaseLine(input_history_);

    buckets_.push_back(new_bucket_candidate);
    uint64_t new_bucket_id = buckets_.size() - 1;
    auto& new_bucket = buckets_.back();

    // Append the input to be moved to inherited input of the new bucket
    inherited_input_hist_move.insert(
        inherited_input_hist_move.end(),
        input_hist_move.begin(),
        input_hist_move.end());
    new_bucket.SetInheritedInputHistIdxes(inherited_input_hist_move);
    new_bucket.SetIndex(new_bucket_id);
    new_bucket.ResetBaseLine(input_history_);
    SetRecipeKeyForBucket(new_bucket_id, new_recipe_key);

    PT_TEST_DEBUG(
        "Bucket with id ",
        mfu_bucket_id,
        " is split and new bucket is created with id ",
        new_bucket_id);

    // Reset MFU bucket details
    mfu_bucket_id = 0;
    mfu_bucket_run_count = 0;
    ComputeMFUBucketDetails();

    return new_bucket_id;
  }

  return {};
}

Bucket DynamicBucketInfo::ConstructNewBucket(
    ResultShapes& result_computed,
    const Bucket& mfu_bucket,
    size_t min_dist_idx,
    bool choose_lower) {
  auto& ranges = mfu_bucket.getRanges();
  auto& dynamic_dims = mfu_bucket.getDynamicDims();
  const DimsHistoryElement& distr_split{input_history_[min_dist_idx].tshapes()};
  const DimsHistoryElement& ref{input_history_.ref_tshapes()};
  DynamicRanges new_ranges{ranges};
  PT_TEST_DEBUG_TH(
      "Before computing new_ranges",
      ", ranges: ",
      ranges,
      ", new_ranges: ",
      new_ranges);

  for (auto& input : dynamic_dims) {
    auto tensor_idx{input.first};
    auto shape_min = shapes_.at(tensor_idx);
    auto shape_max = shape_min;

    for (auto dim : input.second) {
      auto dim_idx{dim.first};
      auto split_dim_val{ref.at(tensor_idx).at(dim_idx)};
      if (distr_split.count(tensor_idx) &&
          distr_split.at(tensor_idx).count(dim_idx)) {
        split_dim_val = distr_split.at(tensor_idx).at(dim_idx);
      }
      auto range_idx{dim.second};

      int64_t lo{(choose_lower ? ranges[range_idx].first : split_dim_val)};
      int64_t hi{(choose_lower ? split_dim_val : ranges[range_idx].second)};
      shape_min.set_dim(dim.first, lo);
      shape_max.set_dim(dim.first, hi);
      new_ranges[range_idx] = std::make_pair(lo, hi);
    }

    result_computed.min_shapes[input.first] = shape_min;
    result_computed.max_shapes[input.first] = shape_max;
  }

  PT_TEST_DEBUG_TH(
      "After computing new_ranges",
      ", ranges: ",
      ranges,
      ", new_ranges: ",
      new_ranges);

  return Bucket(std::move(new_ranges), dynamic_dims, true, split_policy_);
}

void DynamicBucketInfo::split_history(
    const std::vector<size_t>& input_hist_idxes,
    const ResultShapes& new_result,
    std::vector<size_t>& input_hist_move,
    std::vector<size_t>& input_hist_retain) {
  for (auto i : input_hist_idxes) {
    auto& history_item{input_history_[i]};
    bool is_in_range = history_item.IsInRange(new_result);
    if (is_in_range) {
      input_hist_move.push_back(i);
    } else {
      input_hist_retain.push_back(i);
    }
  }
}

bool DynamicBucketInfo::UpdateBucketWithPolicy(
    size_t bucket_id,
    const InpTensorShapes& shapes,
    DynamicDimsPolicy min_policy,
    DynamicDimsPolicy max_policy) {
  if (min_policy == min_policy_ && max_policy == max_policy_) {
    return false;
  } else {
    min_policy_ = min_policy;
    max_policy_ = max_policy;
    if (min_policy != DynamicDimsPolicy::DEFAULT ||
        max_policy != DynamicDimsPolicy::DEFAULT) {
      const PadShapes& pad_shapes = PadShapes{};
      buckets_[bucket_id].setRanges(CalculateRanges(shapes, pad_shapes));
    }
    return true;
  }
}

std::vector<int64_t> DynamicBucketInfo::ExtractDynamicDimsValue(
    const InpTensorShapes& shapes) {
  std::vector<int64_t> dims;
  std::vector<int64_t> dims_new;
  DimsHistoryElement dims_he;
  for (auto& el : dynamic_dims_helper_.flat_dd_) {
    auto dim_val = shapes.at(el.num).dim_size(el.pos);
    dims.push_back(dim_val);

    auto it_dhe = dims_he.find(el.num);
    if (it_dhe == dims_he.end()) {
      dims_he.emplace(el.num, std::map<int64_t, int64_t>{{el.pos, dim_val}});
    } else {
      it_dhe->second.emplace(el.pos, dim_val);
    }
  }

  for (auto& el : dynamic_dims_helper_.flat_dd_) {
    auto dim_val{dims_he.at(el.num).at(el.pos)};
    dims_new.push_back(dim_val);
  }

  TORCH_CHECK(
      dims == dims_new,
      "dims ",
      dims,
      " is not matching with dims_new ",
      dims_new);

  // Append to the history log
  input_history_.hist_items().emplace_back(std::move(dims_he), 0, 0);
  current_input_idx_++;

  return dims;
}

bool DynamicBucketInfo::IsInRangeStaticDims(
    const std::vector<int64_t>& dims,
    int64_t num) const {
  TORCH_CHECK(
      dynamic_dims_helper_.flat_dd_.size() >= dims.size(),
      "wrong dynamic dims size",
      dims.size(),
      " expected ",
      dynamic_dims_helper_.flat_dd_.size());
  for (size_t i = num; i < dims.size(); ++i)
    if (dynamic_dims_helper_.flat_dd_[i].previous_val != dims[i])
      return false;
  return true;
}

int64_t DynamicBucketInfo::GetMaxMultiplier(const PadShapes& pad_shapes) {
  int64_t max_multiplier = default_max_multiplier_;
  // by default MAX size is calculated as current * multiplier
  for (auto& pad_shape : pad_shapes) {
    int64_t num_dyn_dims_in_tensor =
        dynamic_dims_helper_.dd_.at(pad_shape.first).size();
    if (pad_shape.second.input.num_elements() *
            std::pow(default_max_multiplier_, num_dyn_dims_in_tensor) >
        pad_shape.second.output.num_elements()) {
      // unless MAX size would exceed the static Pad output size, then fall back
      // to MAX=current
      max_multiplier = 1;
      break;
    }
  }
  return max_multiplier;
}

DynamicBucketInfo::DimMultipliers DynamicBucketInfo::
    CalculateFlattenedMultipliers(
        const InpTensorShapes& shapes,
        int64_t max_multiplier) {
  DimMultipliers dim_multipliers;
  // Check how many dynamic dims to account for across all inputs
  int64_t max_num_dynamic_dims_for_max = 1;
  int64_t max_num_dynamic_dims_for_min = 0;
  for (auto& el : dynamic_dims_helper_.dd_) {
    max_num_dynamic_dims_for_max =
        std::max(size_t(max_num_dynamic_dims_for_max), el.second.size());
    // for MIN pass don't modify dim sizes < default_min_size_
    int64_t num_dynamic_dims_for_min = 0;
    for (auto& dim : el.second) {
      size_t dim_size = shapes.at(el.first).dim_size(dim.first);
      if (dim_size >= default_min_value_) {
        num_dynamic_dims_for_min++;
      }
    }
    max_num_dynamic_dims_for_min =
        std::max(max_num_dynamic_dims_for_min, num_dynamic_dims_for_min);
  }
  // Prapare individual multipliers for each dim
  for (auto& input : dynamic_dims_helper_.dd_) {
    auto& input_multipliers = dim_multipliers[input.first];
    for (auto& dyn_dim : input.second) {
      input_multipliers.emplace(
          dyn_dim.first, std::make_pair<int64_t, int64_t>(1, 1));
    }
  }
  // All dim multipliers are initialized to 1, now calculate their proper values
  // so that each tensor total size is multiplied the same number of times
  for (auto& input : dim_multipliers) {
    int64_t num_dynamic_dims_for_max = max_num_dynamic_dims_for_max;
    while (num_dynamic_dims_for_max > 0) {
      for (auto& dim : input.second) {
        dim.second.second *= max_multiplier; // max_multiplier
        num_dynamic_dims_for_max--;
        if (num_dynamic_dims_for_max == 0)
          break;
      }
    }
    int64_t num_dynamic_dims_for_min = max_num_dynamic_dims_for_min;
    while (num_dynamic_dims_for_min > 0) {
      for (auto& dim : input.second) {
        size_t dim_size = shapes.at(input.first).dim_size(dim.first);
        // Again, skip updating min_values for dim sizes < default_min_size_
        if (dim_size >= default_min_value_) {
          dim.second.first *= default_min_value_; // min_value
          num_dynamic_dims_for_min--;
        } else {
          dim.second.first = dim_size; // min_value
        }
        if (num_dynamic_dims_for_min == 0)
          break;
      }
      if (num_dynamic_dims_for_min == max_num_dynamic_dims_for_min)
        break;
    }
  }
  return dim_multipliers;
}

size_t DynamicBucketInfo::CalculateHistoric(
    const InpTensorShapes& shapes,
    std::string xin_name,
    std::function<bool(int64_t, int64_t)> comp,
    int64_t xin_val) {
  auto& dims_history{input_history_.hist_items()};
  TORCH_CHECK(!dims_history.empty(), "dims history is empty");

  size_t xin_idx = 0;
  std::vector<int64_t> xin_hist_dims;
  bool is_xin_found{false};
  auto& ref_tshapes{input_history_.ref_tshapes()};

  for (size_t history_idx{}; history_idx < dims_history.size(); history_idx++) {
    const auto& history_element = dims_history[history_idx].tshapes();
    int64_t history_element_xin_size{};
    bool is_fit_history_element{true};

    for (auto dynamic_dims{dynamic_dims_helper_.dd_.begin()};
         dynamic_dims != dynamic_dims_helper_.dd_.end() &&
         is_fit_history_element;
         dynamic_dims++) {
      int64_t dynamic_input_size =
          dynamic_dims_helper_.rem_size_[dynamic_dims->first];
      auto tensor_idx = dynamic_dims->first;
      TORCH_CHECK(
          ref_tshapes.count(tensor_idx),
          "tensor index=",
          tensor_idx,
          " is missing");
      auto& ref_tensor_dim_map{ref_tshapes.at(tensor_idx)};

      for (auto curr_dim{dynamic_dims->second.begin()};
           curr_dim != dynamic_dims->second.end() && is_fit_history_element;
           curr_dim++) {
        auto dim_idx = curr_dim->first;
        int64_t current_dim_val =
            shapes.at(dynamic_dims->first).dim_size(curr_dim->first);

        // Check for error condition
        TORCH_CHECK(
            ref_tensor_dim_map.find(dim_idx) != ref_tensor_dim_map.end(),
            "Missing dim_index=",
            dim_idx,
            " in ref_tensor_dim_map");

        // Start by setting historic_dim_val to reference value of the
        // corresponding dim
        TORCH_CHECK(
            ref_tshapes.count(tensor_idx),
            "dimension index=",
            dim_idx,
            " of tensor index=",
            tensor_idx,
            " is missing");
        int64_t historic_dim_val{ref_tshapes.at(tensor_idx).at(dim_idx)};
        if (history_element.find(tensor_idx) != history_element.end() &&
            history_element.at(tensor_idx).find(dim_idx) !=
                history_element.at(tensor_idx).end()) {
          auto updated_val{history_element.at(tensor_idx).at(dim_idx)};
          historic_dim_val = updated_val;
        }
        if ((1 == current_dim_val && current_dim_val != historic_dim_val) ||
            (1 != current_dim_val && comp(historic_dim_val, current_dim_val))) {
          is_fit_history_element = false;
          break;
        }
        dynamic_input_size *= historic_dim_val;
      }
      history_element_xin_size += dynamic_input_size;
    }

    // Only update the min if the history_element is a valid fit
    if (is_fit_history_element &&
        false == comp(history_element_xin_size, xin_val)) {
      xin_idx = history_idx;
      xin_val = history_element_xin_size;
      is_xin_found = true;
    }
  }
  TORCH_CHECK(
      is_xin_found,
      "CalculateHistoric",
      xin_name,
      " could not find a valid historic ",
      xin_name);

  return xin_idx;
}

DynamicRanges DynamicBucketInfo::CalculateRanges(
    const InpTensorShapes& shapes,
    const PadShapes& pad_shapes) {
  DynamicRanges result;
  auto& dims_history{input_history_.hist_items()};
  result.reserve(dynamic_dims_helper_.flat_dd_.size());

  int64_t max_multiplier = GetMaxMultiplier(pad_shapes);
  DimMultipliers dim_multipliers;
  DimsHistoryElement min_dim_shapes;
  DimsHistoryElement max_dim_shapes;
  if (min_policy_ == DynamicDimsPolicy::FLATTENED ||
      max_policy_ == DynamicDimsPolicy::FLATTENED) {
    dim_multipliers = CalculateFlattenedMultipliers(shapes, max_multiplier);
  }

  if (min_policy_ == DynamicDimsPolicy::HISTORIC) {
    auto dims_history_idx{CalculateHistoricMin(shapes)};
    min_dim_shapes = dims_history.at(dims_history_idx).tshapes();
  }

  if (max_policy_ == DynamicDimsPolicy::HISTORIC) {
    auto dims_history_idx{CalculateHistoricMax(shapes)};
    max_dim_shapes = dims_history.at(dims_history_idx).tshapes();
  }

  for (size_t i{}; i < dynamic_dims_helper_.flat_dd_.size(); i++) {
    auto& el = dynamic_dims_helper_.flat_dd_[i];
    auto tensor_idx = el.num;
    auto dim_idx = el.pos;
    auto ref_dim_val = el.previous_val;
    auto max_value = ref_dim_val;

    TORCH_CHECK(
        shapes.find(el.num) != shapes.end(),
        "Tensor index ",
        el.num,
        " not found in input shapes\n",
        shapes);
    const auto& current_shape = shapes.at(el.num);
    // size 1 is treated as a special case
    int64_t min_value = default_min_value_;
    int64_t dim_max_multiplier = max_multiplier;
    switch (min_policy_) {
      case DynamicDimsPolicy::DEFAULT:
        TORCH_CHECK(0, "Unrecognized condition");
        break;
      case DynamicDimsPolicy::HISTORIC:
        min_value = ref_dim_val;
        if (min_dim_shapes.count(tensor_idx)) {
          auto& dim_map = min_dim_shapes.at(tensor_idx);
          if (dim_map.count(dim_idx)) {
            min_value = dim_map.at(dim_idx);
          }
        }
        // Not allowed to go from non-0 to 0
        if (min_value == 0)
          min_value = current_shape.dim_size(el.pos);
        break;
      case DynamicDimsPolicy::CURRENT:
        min_value = current_shape.dim_size(el.pos);
        break;
      case DynamicDimsPolicy::FLATTENED:
        min_value = dim_multipliers.at(el.num).at(el.pos).first;
        break;
      case DynamicDimsPolicy::CALCULATED:
        // leave default min
        break;
    }
    switch (max_policy_) {
      case DynamicDimsPolicy::DEFAULT:
        TORCH_CHECK(0, "Unrecognized condition");
        break;
      case DynamicDimsPolicy::HISTORIC:
        if (max_dim_shapes.count(tensor_idx)) {
          auto& dim_map = max_dim_shapes.at(tensor_idx);
          if (dim_map.count(dim_idx)) {
            max_value = dim_map.at(dim_idx);
          }
        }
        break;
      case DynamicDimsPolicy::CURRENT:
        dim_max_multiplier = 1;
        break;
      case DynamicDimsPolicy::FLATTENED:
        dim_max_multiplier = dim_multipliers.at(el.num).at(el.pos).second;
        break;
      case DynamicDimsPolicy::CALCULATED:
        // leave default max multiplier
        break;
    }

    // determine min
    int64_t min = shapes.at(el.num).dim_size(el.pos) >= default_min_value_
        ? min_value
        : int64_t(shapes.at(el.num).dim_size(el.pos));

    // using the current as max
    // max_policy_ only determines the multiplication factor
    int64_t max =
        ((max_policy_ == DynamicDimsPolicy::HISTORIC)
             ? max_value
             : int64_t(shapes.at(el.num).dim_size(el.pos)) *
                 dim_max_multiplier);
    // Check if this is a dynamic paddings input
    if (pad_shapes.count(el.num) > 0) {
      // paddings contain a pair of (before, after) num of pad elements for each
      // dimension
      TORCH_CHECK(
          pad_shapes.at(el.num).output.dims() * 2 == shapes.at(el.num).dims(),
          "Incorect padding shape");
      int64_t pad_output_dim_size =
          pad_shapes.at(el.num).output.dim_size(el.pos / 2);
      int64_t pad_input_dim_size =
          pad_shapes.at(el.num).input.dim_size(el.pos / 2);
      int64_t current_paddings_dim_size = shapes.at(el.num).dim_size(el.pos);
      min = pad_output_dim_size -
          (pad_input_dim_size >= default_min_value_ ? min_value
                                                    : pad_input_dim_size);
      max = pad_output_dim_size -
          (pad_output_dim_size - current_paddings_dim_size) *
              dim_max_multiplier;
    }
    result.emplace_back(std::make_pair(min, max));
  }
  return result;
}

void DynamicBucketInfo::RegisterTimeSlot(
    const std::shared_ptr<synapse_helpers::TimeSlotBase>& ts,
    uint64_t bucket_id) {
  UpdateRunTimes();
  run_time_q_.emplace(ts, bucket_id, current_input_idx_);
}

void DynamicBucketInfo::UpdateRunTimes() {
  while (!run_time_q_.empty()) {
    std::shared_ptr<synapse_helpers::TimeSlotBase> tsbpsh;
    size_t bucket_id;
    size_t input_hist_idx;
    std::tie(tsbpsh, bucket_id, input_hist_idx) = run_time_q_.front();
    auto time_opt = tsbpsh->getTime();
    if (false == time_opt.has_value()) {
      break;
    }
    auto t_ns{time_opt.value()};
    buckets_[bucket_id].UpdateRunTime(t_ns);
    input_history_.hist_items_[input_hist_idx].run_time_ = t_ns;
    cumu_run_time_stat_.Update(t_ns);
    run_time_q_.pop();
  }
}

bool DynamicBucketInfo::NeedRunTimeSlot(uint64_t bucket) {
  return bucket < buckets_.size() && buckets_[bucket].GetKeepRunTime();
}

std::string DynamicBucketInfo::bucket_range_str(
    const Bucket& bucket,
    bool is_first) const {
  auto& ref_tshapes{input_history_.ref_tshapes()};
  if (is_first) {
    return DebugString(ref_tshapes);
  }

  std::ostringstream O;
  const auto& ranges{bucket.getRanges()};
  const auto& dynamic_dims{bucket.getDynamicDims()};
  for (auto tensor_it : ref_tshapes) {
    const auto& tensor_idx{tensor_it.first};
    std::string tensor_str_lo;
    std::string tensor_str_hi;
    O << '\n';
    tensor_str_lo += " [";
    tensor_str_hi += " [";
    bool is_first{true};
    for (auto dim_it : tensor_it.second) {
      const auto& dim_idx{dim_it.first};
      auto dim_lo{dim_it.second};
      auto dim_hi{dim_it.second};
      if (dynamic_dims.count(tensor_idx) &&
          dynamic_dims.at(tensor_idx).count(dim_idx)) {
        auto range_idx = dynamic_dims.at(tensor_idx).at(dim_idx);
        dim_lo = ranges.at(range_idx).first;
        dim_hi = ranges.at(range_idx).second;
      }
      tensor_str_lo += (is_first ? "" : ",") + std::to_string(dim_lo);
      tensor_str_hi += (is_first ? "" : ",") + std::to_string(dim_hi);
      is_first = false;
    }
    tensor_str_lo += "]";
    tensor_str_hi += "]";
    O << tensor_str_lo << " -" << tensor_str_hi;
  }

  return O.str();
}

std::string DynamicBucketInfo::digest_str() const {
  // Present summary stats
  std::ostringstream O;
  O << "DynamicBucketInfo details:" << '\n'
    << " min policy: " << min_policy_ << '\n'
    << " max policy: " << max_policy_ << '\n'
    << " hit count: " << cumu_hit_count_ << '\n'
    << " miss count: " << (cumu_run_count_ - cumu_hit_count_) << '\n';

  if (GET_ENV_FLAG_NEW(PT_ENABLE_SYNLAUNCH_TIME_CAPTURE)) {
    O << " [reported times are in nano seconds]" << '\n';
  }
  if (GET_ENV_FLAG_NEW(PT_ENABLE_SYNLAUNCH_TIME_CAPTURE)) {
    O << " compile time stat : " << cumu_compile_time_stat_ << '\n'
      << " run time stat     : " << cumu_run_time_stat_ << '\n';
  }
  O << "Bucket details:" << '\n';

  for (size_t idx = 0; idx < buckets_.size(); idx++) {
    const auto& bucket{buckets_.at(idx)};
    O << "Bucket id: " << idx << '\n';
    O << bucket.digest_str();
    O << "Ranges:";
    O << bucket_range_str(bucket, (0 == idx));
    O << '\n' << "--------------------" << '\n';
  }
  return O.str();
}

std::string DynamicBucketInfo::history_str() const {
  std::ostringstream O;
  auto& ref_tshapes{input_history_.ref_tshapes()};
  auto& dims_history{input_history_.hist_items()};
  O << "Number of historical inputs: " << dims_history.size() << '\n';
  bool skipped{false};
  size_t i{0};
  O << "Input[" << i << "]:" << DebugString(ref_tshapes) << '\n';
  i += 1;
  for (; i < dims_history.size(); i++) {
    const auto& a = dims_history[i].tshapes();
    if (a == dims_history[i - 1].tshapes()) {
      skipped = true;
      continue;
    }
    if (skipped) {
      skipped = false;
      O << "  "
        << "..." << '\n';
    }
    O << "Input[" << i << "]:" << DebugString(a, ref_tshapes) << '\n';
  }
  if (skipped) {
    skipped = false;
    O << "  "
      << "..." << '\n';
  }
  O << "--------------------" << '\n';
  O << "History:\n"
    << DebugString(input_history_) << "--------------------" << std::endl;
  return O.str();
}

void DynamicBucketInfo::DynamicDimsHelper::FindOrAdd(
    int64_t num,
    int64_t pos,
    int64_t val) {
  auto it_dd = dd_.find(num);
  if (it_dd == dd_.end()) {
    dd_.emplace(num, std::map<int64_t, int64_t>{{pos, flat_dd_.size()}});
    DynamicDimsElement item(num, pos, val);
    flat_dd_.emplace_back(num, pos, val);
    return;
  }

  if (it_dd->second.find(pos) == it_dd->second.end()) {
    it_dd->second.emplace(pos, flat_dd_.size());
    DynamicDimsElement item(num, pos, val);
    flat_dd_.emplace_back(num, pos, val);
  }
}

std::shared_ptr<habana_helpers::CompilationStatistics> DynamicBucketInfo::
    get_statistics() {
  return statistics_;
}

void DynamicBucketInfo::create_statistics(
    std::unique_ptr<habana_helpers::CompilationStatistics> sptr) {
  statistics_ = std::move(sptr);
}

} // namespace habana_helpers
