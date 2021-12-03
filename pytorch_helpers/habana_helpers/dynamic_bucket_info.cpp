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

#include "habana_device/HPUCheck.h"
#include "pytorch_helpers/synapse_helpers/env_flags.h"

using namespace synapse_helpers;
namespace habana_helpers {

constexpr int64_t DynamicBucketInfo::default_min_value_;

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
  run_time_stat_.Update(elapsed_time);
  if (base_time_ > 0) {
    uint64_t time_to_beat = static_cast<uint64_t>(
        static_cast<double>(base_time_) * time_improve_factor_);

    auto cur_avg_time{run_time_stat_.GetAvgTime()};

    refine_candidate_ = (time_to_beat > cur_avg_time);
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
  uint64_t cur_avg_time = (idx_ == 0 ? 0 : run_time_stat_.GetAvgTime());
  return Bucket(std::move(new_ranges), dynamic_dims_, true, sp, cur_avg_time);
}

DynamicBucketInfo::ResultShapes DynamicBucketInfo::CalculateShapes(
    uint64_t bucket) {
  ResultShapes result;
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
    // skip if MIN shape == MAX shape (== current)
    if (shape_min != shape_max) {
      result.min_shapes[input.first] = shape_min;
      result.max_shapes[input.first] = shape_max;
    }
  }

  return result;
}

void DynamicBucketInfo::CollectDynamicDims(const InpTensorShapes& new_shapes) {
  if (shapes_.empty()) {
    shapes_ = new_shapes;
    // Populate refrerence DimsHistoryElement
    for (auto tensor_it = shapes_.cbegin(); tensor_it != shapes_.cend();
         tensor_it++) {
      auto tensor_idx{tensor_it->first};
      ref_tensor_shapes_.emplace(tensor_idx, std::map<int64_t, int64_t>());
      for (size_t dim_idx = 0; dim_idx < tensor_it->second.dims(); dim_idx++) {
        auto dim_val{tensor_it->second.dim_size(dim_idx)};
        ref_tensor_shapes_[tensor_idx].emplace(dim_idx, dim_val);
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
    dynamic_dims_.rem_size_[tensor_it_ref->first] = 1;
    for (size_t i = 0; i < tensor_it_ref->second.dims(); ++i) {
      if (tensor_it_ref->second.dim_size(i) !=
          tensor_it_new->second.dim_size(i)) {
        dynamic_dims_.FindOrAdd(
            tensor_it_ref->first,
            i,
            shapes_.at(tensor_it_ref->first).dim_size(i));
      } else if (tensor_it_ref->second.dim_size(i) > 1) {
        // For avoiding the dimension with value 0
        dynamic_dims_.rem_size_[tensor_it_ref->first] *=
            tensor_it_ref->second.dim_size(i);
      }
    }
  }
}

void DynamicBucketInfo::UpdateMFUBucketDetails(uint64_t bucket_id) {
  uint64_t cur_bucket_run_count{buckets_[bucket_id].GetRunCount()};
  if (mfu_bucket_run_count < cur_bucket_run_count) {
    mfu_bucket_run_count = cur_bucket_run_count;
    mfu_bucket_id = bucket_id;
  }
}

uint64_t DynamicBucketInfo::GetBucketId(
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
    UpdateMFUBucketDetails(new_bucket_id);
    dims_history_.emplace_back(DimsHistoryElement{});

    return new_bucket_id;
  }
  global_count++;
  auto dims = ExtractDynamicDimsValue(shapes);

  absl::optional<uint64_t> best_bucket{};
  std::set<int64_t> skipped_ranges;
  for (uint i = 0; i < dynamic_dims_.flat_dd_.size(); i++) {
    if (pad_shapes.find(dynamic_dims_.flat_dd_[i].num) != pad_shapes.end()) {
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

    return best_bucket_id;
  }

  // Create new bucket
  prev_dynamic_dims_ = dynamic_dims_.flat_dd_.size();

  auto ranges = CalculateRanges(shapes, pad_shapes);
  buckets_.emplace_back(
      std::move(ranges), dynamic_dims_.dd_, refine_enabled_, split_policy_);
  auto& new_bucket = buckets_.back();
  new_bucket.IncStats(dims);

  uint64_t new_bucket_id = buckets_.size() - 1;
  new_bucket.SetIndex(new_bucket_id);
  UpdateMFUBucketDetails(new_bucket_id);

  return buckets_.size() - 1;
}

absl::optional<uint64_t> DynamicBucketInfo::CheckForSplitBucket() {
  if (refine_enabled_ == false || buckets_.size() >= max_buckets_number_ ||
      mfu_bucket_run_count < min_iterations_to_split_ || mfu_bucket_id == 0) {
    return {};
  }

  auto mfu_bucket = buckets_.at(mfu_bucket_id);

  if (mfu_bucket.IsRefinementCandidate() == false) {
    return {};
  }
  buckets_.push_back(mfu_bucket.CreateNewBucket(split_policy_));

  mfu_bucket.ResetRunCount();

  uint64_t new_bucket_id = buckets_.size() - 1;
  auto& new_bucket = buckets_.back();
  new_bucket.SetIndex(new_bucket_id);

  PT_DYNAMIC_SHAPE_DEBUG(
      "Bucket with id ",
      mfu_bucket_id,
      " is split and ",
      " new bucket is created with id ",
      new_bucket_id);

  // Reset MFU bucket details
  mfu_bucket_id = 0;
  mfu_bucket_run_count = 0;
  UpdateMFUBucketDetails(new_bucket_id);

  return new_bucket_id;
}

bool DynamicBucketInfo::UpdateBucketWithPolicy(
    uint64_t bucket_id,
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

std::string DynamicBucketInfo::ResultShapes::DebugString() {
  std::string result;
  TORCH_CHECK(
      min_shapes.size() == max_shapes.size(),
      "max and min have different shapes");
  for (auto imin = min_shapes.cbegin(); imin != min_shapes.cend(); imin++) {
    auto tensor_idx = imin->first;
    auto imax = max_shapes.find(tensor_idx);
    TORCH_CHECK(
        imax != max_shapes.end() && imin->second.dims() == imax->second.dims(),
        "max and min have different number of input dimensions");
    result += "  Tensor" + std::to_string(tensor_idx) + ":";
    bool is_first{true};
    result += "(";
    for (size_t dim = 0; dim < imin->second.dims(); dim++) {
      result += (is_first ? "" : ",");
      is_first = false;
      result += "Dim" + std::to_string(dim) + ":[";
      result += std::to_string(imin->second.dim_size(dim)) + ",";
      result += std::to_string(imax->second.dim_size(dim));
      result += "]";
    }
    result += ")";
  }
  result += "\n";
  return result;
}

std::string DynamicBucketInfo::ResultShapes::DebugString(
    const InpTensorShapes& inp_shapes) {
  std::string result;
  for (auto tshape_it : inp_shapes) {
    const auto& tshape_idx{tshape_it.first};
    std::string tshape_str_lo;
    std::string tshape_str_hi;
    result += '\n';
    tshape_str_lo += " [";
    tshape_str_hi += " [";
    bool is_first{true};
    const auto& dims{tshape_it.second.get_dims()};
    for (size_t i = 0; i < tshape_it.second.dims(); i++) {
      auto dim{dims.at(i)};
      auto dim_lo{dim};
      auto dim_hi{dim};
      if (min_shapes.count(tshape_idx) && max_shapes.count(tshape_idx)) {
        dim_lo = min_shapes.at(tshape_idx).get_dims().at(i);
        dim_hi = max_shapes.at(tshape_idx).get_dims().at(i);
      }
      tshape_str_lo += (is_first ? "" : ",") + std::to_string(dim_lo);
      tshape_str_hi += (is_first ? "" : ",") + std::to_string(dim_hi);
      is_first = false;
    }
    tshape_str_lo += "]";
    tshape_str_hi += "]";
    result += tshape_str_lo + " -" + tshape_str_hi;
  }
  return result;
}

std::vector<int64_t> DynamicBucketInfo::ExtractDynamicDimsValue(
    const InpTensorShapes& shapes) {
  std::vector<int64_t> dims;
  std::vector<int64_t> dims_new;
  DimsHistoryElement dims_he;
  for (auto& el : dynamic_dims_.flat_dd_) {
    auto dim_val = shapes.at(el.num).dim_size(el.pos);
    dims.push_back(dim_val);

    auto it_dhe = dims_he.find(el.num);
    if (it_dhe == dims_he.end()) {
      dims_he.emplace(el.num, std::map<int64_t, int64_t>{{el.pos, dim_val}});
    } else {
      it_dhe->second.emplace(el.pos, dim_val);
    }
  }

  for (auto& el : dynamic_dims_.flat_dd_) {
    auto dim_val{dims_he.at(el.num).at(el.pos)};
    dims_new.push_back(dim_val);
  }

  TORCH_CHECK(
      dims == dims_new,
      "dims ",
      dims,
      " is not matching with dims_new ",
      dims_new);

  if (!dims_he.empty()) {
    dims_history_.push_back(dims_he);
  }

  return dims;
}

bool DynamicBucketInfo::IsInRangeStaticDims(
    const std::vector<int64_t>& dims,
    int64_t num) const {
  TORCH_CHECK(
      dynamic_dims_.flat_dd_.size() >= dims.size(),
      "wrong dynamic dims size",
      dims.size(),
      " expected ",
      dynamic_dims_.flat_dd_.size());
  for (size_t i = num; i < dims.size(); ++i)
    if (dynamic_dims_.flat_dd_[i].previous_val != dims[i])
      return false;
  return true;
}

int64_t DynamicBucketInfo::GetMaxMultiplier(const PadShapes& pad_shapes) {
  int64_t max_multiplier = default_max_multiplier_;
  // by default MAX size is calculated as current * multiplier
  for (auto& pad_shape : pad_shapes) {
    int64_t num_dyn_dims_in_tensor =
        dynamic_dims_.dd_.at(pad_shape.first).size();
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
  for (auto& el : dynamic_dims_.dd_) {
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
  for (auto& input : dynamic_dims_.dd_) {
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
  TORCH_CHECK(!dims_history_.empty(), "dims history is empty");

  size_t xin_idx = 0;
  std::vector<int64_t> xin_hist_dims;
  bool is_xin_found{false};

  for (size_t history_idx{}; history_idx < dims_history_.size();
       history_idx++) {
    const auto& history_element = dims_history_[history_idx];
    int64_t history_element_xin_size{};
    bool is_fit_history_element{true};

    for (auto dynamic_dims{dynamic_dims_.dd_.begin()};
         dynamic_dims != dynamic_dims_.dd_.end() && is_fit_history_element;
         dynamic_dims++) {
      int64_t dynamic_input_size = dynamic_dims_.rem_size_[dynamic_dims->first];
      auto tensor_idx = dynamic_dims->first;
      auto& ref_tensor_dim_map{ref_tensor_shapes_.at(tensor_idx)};

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
        int64_t historic_dim_val{ref_tensor_shapes_.at(tensor_idx).at(dim_idx)};
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
  result.reserve(dynamic_dims_.flat_dd_.size());

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
    min_dim_shapes = dims_history_.at(dims_history_idx);
  }

  if (max_policy_ == DynamicDimsPolicy::HISTORIC) {
    auto dims_history_idx{CalculateHistoricMax(shapes)};
    max_dim_shapes = dims_history_.at(dims_history_idx);
  }

  for (size_t i{}; i < dynamic_dims_.flat_dd_.size(); i++) {
    auto& el = dynamic_dims_.flat_dd_[i];
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
    if (min == default_min_value_) {
      PT_BRIDGE_WARN("[Dyn WARN] Using default_min_value_ = ", min, " as min");
    }
    result.emplace_back(std::make_pair(min, max));
  }
  return result;
}

void DynamicBucketInfo::RegisterTimeSlot(
    const std::shared_ptr<synapse_helpers::TimeSlotBase>& ts,
    int bucket) {
  UpdateRunTimes();
  run_time_states.emplace(ts, bucket);
}

void DynamicBucketInfo::UpdateRunTimes() {
  while (!run_time_states.empty()) {
    auto time = run_time_states.front().first->getTime();
    if (false == time.has_value()) {
      break;
    }
    auto t_ns{time.value()};
    buckets_[run_time_states.front().second].UpdateRunTime(t_ns);
    cumu_run_time_stat_.Update(t_ns);
    run_time_states.pop();
  }
}

bool DynamicBucketInfo::NeedRunTimeSlot(uint64_t bucket) {
  return bucket < buckets_.size() && buckets_[bucket].GetKeepRunTime();
}

std::string DynamicBucketInfo::digest_str() const {
  // Present summary stats
  std::ostringstream O;
  O << "DynamicBucketInfo details:" << '\n'
    << " min policy: " << min_policy_ << '\n'
    << " max policy: " << max_policy_ << '\n'
    << " hit count: " << cumu_hit_count_ << '\n'
    << " miss count: " << (cumu_run_count_ - cumu_hit_count_) << '\n';

  if (GET_ENV_FLAG(PT_ENABLE_SYNLAUNCH_TIME_CAPTURE)) {
    O << " [number of run times stats collected can be lesser than the total number of runs]"
      << '\n'
      << " [times are in nano seconds]" << '\n';
  }
  if (GET_ENV_FLAG(PT_ENABLE_SYNLAUNCH_TIME_CAPTURE)) {
    O << " compile time stat : " << cumu_compile_time_stat_ << '\n'
      << " run time stat     : " << cumu_run_time_stat_ << '\n';
  }
  O << "Bucket details:" << '\n';
  // for (const auto& b : buckets_) {
  for (size_t idx = 0; idx < buckets_.size(); idx++) {
    const auto& bucket{buckets_.at(idx)};
    O << "Bucket id: " << idx << '\n';
    O << bucket.digest_str();
    O << "Ranges:";
    if (idx) {
      const auto& ranges{bucket.getRanges()};
      const auto& dynamic_dims{bucket.getDynamicDims()};
      for (auto tensor_it : ref_tensor_shapes_) {
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
    } else {
      O << DebugString(ref_tensor_shapes_);
    }
    O << '\n' << "--------------------" << '\n';
  }
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

} // namespace habana_helpers
