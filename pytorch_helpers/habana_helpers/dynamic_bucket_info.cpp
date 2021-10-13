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

#include <limits>
#include <memory>

#include <absl/types/variant.h>

#include "habana_device/HPUCheck.h"
#include "pytorch_helpers/synapse_helpers/env_flags.h"

using namespace synapse_helpers;
namespace habana_helpers {

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
    case SplitPolicy::DEFAULT:
      split_stat_impl_ = std::make_shared<SplitStatImplDefault>(ranges_.size());
      break;
    case SplitPolicy::DYNAMIC:
      split_stat_impl_ = std::make_shared<SplitStatImplDynamic>(ranges_.size());
      break;
  }
}

void SplitStatImplDefault::Increment(
    const DynamicRanges& ranges,
    const std::vector<int64_t>& dims) {
  if (split_count_.empty())
    return;

  TORCH_CHECK(
      ranges.size() == dims.size(),
      "wrong dynamic dims size ",
      dims.size(),
      ", expected ",
      ranges.size());

  uint64_t pos = 0;
  for (size_t i = 0; i < ranges.size(); ++i) {
    int n = (dims[i] < (ranges[i].second - ranges[i].first) / 2) ? 0 : 1;
    pos = pos | (n << i);
  }

  TORCH_CHECK(
      pos < split_count_.size(),
      "out of range value for pos ",
      pos,
      " with dims size ",
      dims.size(),
      ", split count size ",
      split_count_.size());

  split_count_.at(pos)++;
}

void SplitStatImplDefault::CalculateNewRanges(
    const DynamicRanges& ranges,
    DynamicRanges& new_ranges) {
  uint64_t max_idx = 1;
  for (size_t i = 1; i < split_count_.size(); i++) {
    if (split_count_[i] > split_count_[max_idx])
      max_idx = i;
  }
  uint64_t pos = max_idx;
  for (auto& el : ranges) {
    int64_t mid = (el.second - el.first) / 2;
    if ((pos & 1) == 0)
      new_ranges.emplace_back(el.first, mid);
    else
      new_ranges.emplace_back(mid, el.second);
    pos >>= 1;
  }
}

void SplitStatImplDynamic::Increment(
    const DynamicRanges& ranges,
    const std::vector<int64_t>& dims) {
  if (0 == num_dyn_ranges_)
    return;

  TORCH_CHECK(
      ranges.size() == dims.size(),
      "wrong dynamic dims size ",
      dims.size(),
      ", expected ",
      ranges.size());

  std::vector<bool> pos(num_dyn_ranges_, 0);
  for (size_t i = 0; i < ranges.size(); ++i) {
    pos[i] = (dims[i] < (ranges[i].second - ranges[i].first) / 2) ? 0 : 1;
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
    int64_t mid = (el.second - el.first) / 2;
    if (max_pos_[i] == 0)
      new_ranges.emplace_back(el.first, mid);
    else
      new_ranges.emplace_back(mid, el.second);
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
    const auto m = max_number_of_dims;
    TORCH_CHECK(
        ranges_.size() <= m,
        "We don't support this much dimension ",
        ranges_.size(),
        ", max ",
        m);
    // split_count_.resize(1 << ranges_.size());
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
    refine_candidate_ = time_to_beat > run_time_stat_.getTime();
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
  return Bucket(std::move(new_ranges), dynamic_dims_, true, sp);
}

DynamicBucketInfo::DynamicBucketInfo(
    DynamicDimsPolicy min_policy,
    SplitPolicy sp) {
  refine_enabled_ = GET_ENV_FLAG(PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES);
  split_policy_ = sp;
  min_policy_ = min_policy;
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

  auto& ranges = buckets_[bucket].ranges();
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

  // TODO: Check if this can be removed or not required
  // for (auto& input : dynamic_dims) {
  //   if (result.min_shapes.count(input.first) == 0) {
  //     HABANA_ASSERT(result.max_shapes.count(input.first) == 0);
  //     continue;
  //   }
  //   synapse_helpers::tensor::dynamic_shape_t
  //   dynamicity{synapse_helpers::tensor::dynamic_shape_t{}}; if
  //   (result.min_shapes.at(input.first).dims() <= SYN_GAUDI_MAX_TENSOR_DIM &&
  //       result.max_shapes.at(input.first).dims() <= SYN_GAUDI_MAX_TENSOR_DIM)
  //       {
  //     auto syn_shape_min = absl::get<graph_builder::TensorShape>(
  //         graph_builder::adjust_shape_tf_to_syn(result.min_shapes.at(input.first)));
  //     auto syn_shape_max = absl::get<graph_builder::TensorShape>(
  //         graph_builder::adjust_shape_tf_to_syn(result.max_shapes.at(input.first)));
  //     dynamicity.set_rank(static_cast<synapse_helpers::tensor::shape_t::dimension_count_t>(syn_shape_min.dims()));

  //     for (int64_t dim = 0; dim < syn_shape_min.dims(); dim++)
  //       dynamicity.set_dim(dim, syn_shape_min.dim_size(dim),
  //       syn_shape_max.dim_size(dim));
  //   }

  //   result.syn_shapes.emplace(input.first, dynamicity);
  // }

  return result;
}

std::unordered_set<int64_t> DynamicBucketInfo::GetDynamicInputs() const {
  std::unordered_set<int64_t> inputs;
  for (auto& el : dynamic_dims_.dd_)
    inputs.insert(el.first);
  return inputs;
}

void DynamicBucketInfo::CollectDynamicDims(const InpTensorShapes& shapes) {
  if (shapes_.empty()) {
    shapes_ = shapes;
  }
  TORCH_CHECK(
      shapes_.size() == shapes.size(),
      "input shapes size ",
      shapes.size(),
      " is not matching with existing shapes size ",
      shapes_.size());
  for (auto it1 = shapes_.cbegin(), it2 = shapes.cbegin();
       it1 != shapes_.cend();
       ++it1, ++it2) {
    dynamic_dims_.rem_size_[it1->first] = 1;
    for (size_t i = 0; i < it1->second.dims(); ++i) {
      if (it1->second.dim_size(i) != it2->second.dim_size(i)) {
        dynamic_dims_.FindOrAdd(
            it1->first, i, shapes_.at(it1->first).dim_size(i));
      } else if (it1->second.dim_size(i) > 1)
        dynamic_dims_.rem_size_[it1->first] *= it1->second.dim_size(i);
    }
  }
}

uint64_t DynamicBucketInfo::GetBucketId(
    const InpTensorShapes& shapes,
    const PadShapes& pad_shapes) {
  TORCH_CHECK(shapes_.size() == shapes.size(), "Shapes dont match");
  if (buckets_.empty()) {
    buckets_.emplace_back(DynamicRanges{}, DynamicDims{}, true, split_policy_);
    auto& new_bucket = buckets_.back();
    new_bucket.IncStats({});
    new_bucket.SetIndex(buckets_.size() - 1);
    return 0;
  }
  global_count++;
  auto dims = ExtractDynamicDimsValue(shapes);
  if (!dims.empty())
    dim_history_.push_back(dims);

  absl::optional<uint64_t> best_bucket_id{};
  std::set<int64_t> skipped_ranges;
  for (uint i = 0; i < dynamic_dims_.flat_dd_.size(); i++) {
    if (pad_shapes.find(dynamic_dims_.flat_dd_[i].num) != pad_shapes.end()) {
      skipped_ranges.insert(i);
    }
  }

  for (size_t i = 0; i < buckets_.size(); i++) {
    if (i > 0 && dims.size() != buckets_[i].ranges().size()) {
      continue;
    }
    bool in_range = buckets_[i].IsInRange(dims, skipped_ranges) &&
        IsInRangeStaticDims(dims, buckets_[i].getDynamiDimsCount());
    // Choose a box with lower score meaning narrower ranges
    if (in_range &&
        ((best_bucket_id.has_value() &&
          buckets_[best_bucket_id.value()].getScore() >
              buckets_[i].getScore()) ||
         !best_bucket_id.has_value()))
      best_bucket_id = i;
  }
  if (best_bucket_id.has_value()) {
    buckets_[best_bucket_id.value()].IncStats(dims);

    return best_bucket_id.value();
  }

  // Create new bucket
  if (dynamic_dims_.flat_dd_.size() != prev_dynamic_dims_) {
    // In case new dynamic dims detected reset default bucketing policy
    if (min_policy_ != DynamicDimsPolicy::HISTORIC) {
      min_policy_ = DynamicDimsPolicy::CALCULATED;
    }
    max_policy_ = DynamicDimsPolicy::CALCULATED;
  }
  prev_dynamic_dims_ = dynamic_dims_.flat_dd_.size();

  auto ranges = CalculateRanges(shapes, pad_shapes);
  buckets_.emplace_back(
      std::move(ranges), dynamic_dims_.dd_, refine_enabled_, split_policy_);
  auto& new_bucket = buckets_.back();
  new_bucket.IncStats(dims);
  new_bucket.SetIndex(buckets_.size() - 1);

  return buckets_.size() - 1;
}

absl::optional<uint64_t> DynamicBucketInfo::CheckForSplitBucket() {
  if (!refine_enabled_ || global_count < min_iterations_to_split_ ||
      buckets_.size() > max_buckets_number_) {
    return {};
  }

  auto freq_used_bucket = std::max_element(
      buckets_.begin(), buckets_.end(), [](const Bucket& a, const Bucket& b) {
        return a.getRunCount() < b.getRunCount();
      });

  if (freq_used_bucket == buckets_.begin()) {
    return {};
  }

  auto target_count{
      static_cast<decltype(global_count)>(density_coefficient_ * global_count)};
  if (target_count < freq_used_bucket->getRunCount()) {
    buckets_.push_back(freq_used_bucket->CreateNewBucket(split_policy_));
    auto& new_bucket = buckets_.back();
    new_bucket.SetIndex(buckets_.size() - 1);

    for (auto& el : buckets_)
      el.ResetRunCount();
    global_count = 0;
    return buckets_.size() - 1;
  } else {
    return {};
  }
}

bool DynamicBucketInfo::IsConsistentDynamicDimsCount() {
  int dynamic_dim_count = -1;
  for (const auto& entry : dynamic_dims_.dd_) {
    if (dynamic_dim_count == -1) {
      dynamic_dim_count = int(entry.second.size());
    }
    if (dynamic_dim_count != int(entry.second.size())) {
      return false;
    }
  }
  return true;
}

bool DynamicBucketInfo::UpdateBucketingPolicy(
    uint64_t bucket_id,
    const InpTensorShapes& shapes,
    const PadShapes& pad_shapes,
    DynamicDimsPolicy min_policy,
    DynamicDimsPolicy max_policy) {
  if (min_policy == min_policy_ && max_policy == max_policy_) {
    return false;
  } else {
    min_policy_ =
        min_policy == DynamicDimsPolicy::DEFAULT ? min_policy_ : min_policy;
    max_policy_ =
        max_policy == DynamicDimsPolicy::DEFAULT ? max_policy_ : max_policy;
    if (min_policy != DynamicDimsPolicy::DEFAULT ||
        max_policy != DynamicDimsPolicy::DEFAULT) {
      buckets_[bucket_id].ranges() = CalculateRanges(shapes, pad_shapes);
    }
    return true;
  }
}

std::string DynamicBucketInfo::ResultShapes::DebugString() {
  std::string result;
  TORCH_CHECK(
      min_shapes.size() == max_shapes.size(),
      "max and min have different shapes");
  for (auto imin = min_shapes.cbegin(), imax = max_shapes.cbegin();
       imin != min_shapes.cend();
       imin++, imax++) {
    TORCH_CHECK(
        imin->second.dims() == imax->second.dims(),
        "max and min have different number of input dimensions");
    result += std::to_string(imin->first) + ":[";
    for (size_t dim = 0; dim < imin->second.dims(); dim++) {
      result += std::to_string(imin->second.dim_size(dim)) + ",";
    }
    result += "]-[";
    for (size_t dim = 0; dim < imax->second.dims(); dim++) {
      result += std::to_string(imax->second.dim_size(dim)) + ",";
    }
    result += "] ";
  }
  return result;
}

std::vector<int64_t> DynamicBucketInfo::ExtractDynamicDimsValue(
    const InpTensorShapes& shapes) const {
  std::vector<int64_t> dims;
  for (auto& el : dynamic_dims_.flat_dd_)
    dims.push_back(shapes.at(el.num).dim_size(el.pos));
  return dims;
}

bool DynamicBucketInfo::IsInRangeStaticDims(
    const std::vector<int64_t>& dims,
    int64_t num) const {
  TORCH_CHECK(
      dynamic_dims_.flat_dd_.size() == dims.size(),
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

std::vector<int64_t> DynamicBucketInfo::CalculateHistoricMin(
    const InpTensorShapes& shapes) {
  TORCH_CHECK(!dim_history_.empty(), "historic min is empty");

  int64_t min_size = std::numeric_limits<int64_t>::max();
  size_t min_idx = 0;

  for (size_t i{}; i < dim_history_.size(); i++) {
    const auto& candidate = dim_history_[i];
    int64_t candidate_min_size{};
    bool history_item_fit{true};

    for (auto dynamic_dims{dynamic_dims_.dd_.begin()};
         dynamic_dims != dynamic_dims_.dd_.end() && history_item_fit;
         dynamic_dims++) {
      int64_t dynamic_input_size = dynamic_dims_.rem_size_[dynamic_dims->first];
      for (auto curr_dim{dynamic_dims->second.begin()};
           curr_dim != dynamic_dims->second.end() && history_item_fit;
           curr_dim++) {
        dynamic_input_size *= candidate[curr_dim->second];
        if (shapes.at(dynamic_dims->first).dim_size(curr_dim->first) <
            candidate[curr_dim->second])
          history_item_fit = false;
      }
      candidate_min_size += dynamic_input_size;
    }
    if (history_item_fit && candidate_min_size < min_size) {
      min_idx = i;
      min_size = candidate_min_size;
    }
  }
  return dim_history_[min_idx];
}

DynamicRanges DynamicBucketInfo::CalculateRanges(
    const InpTensorShapes& shapes,
    const PadShapes& pad_shapes) {
  DynamicRanges result;
  result.reserve(dynamic_dims_.flat_dd_.size());

  int64_t max_multiplier = GetMaxMultiplier(pad_shapes);
  DimMultipliers dim_multipliers;
  std::vector<int64_t> min_dim_shapes;
  if (min_policy_ == DynamicDimsPolicy::FLATTENED ||
      max_policy_ == DynamicDimsPolicy::FLATTENED) {
    dim_multipliers = CalculateFlattenedMultipliers(shapes, max_multiplier);
  }
  if (min_policy_ == DynamicDimsPolicy::HISTORIC) {
    min_dim_shapes = CalculateHistoricMin(shapes);
  }

  for (size_t i{}; i < dynamic_dims_.flat_dd_.size(); i++) {
    auto& el = dynamic_dims_.flat_dd_[i];
    const auto& current_shape = shapes.at(el.num);
    // size 1 is treated as a special case
    int64_t min_value = default_min_value_;
    int64_t dim_max_multiplier = max_multiplier;
    switch (min_policy_) {
      case DynamicDimsPolicy::DEFAULT:
        TORCH_CHECK(0, "Unrecognized condition");
        break;
      case DynamicDimsPolicy::HISTORIC:
        min_value = min_dim_shapes[i];
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
      case DynamicDimsPolicy::HISTORIC:
      case DynamicDimsPolicy::DEFAULT:
        TORCH_CHECK(0, "Unrecognized condition");
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
    int64_t max =
        int64_t(shapes.at(el.num).dim_size(el.pos)) * dim_max_multiplier;
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

void DynamicBucketInfo::DynamicDimsHelper::FindOrAdd(
    int64_t num,
    int64_t pos,
    int64_t val) {
  auto it_dd = dd_.find(num);
  if (it_dd == dd_.end()) {
    dd_.emplace(num, std::map<int64_t, int64_t>{{pos, flat_dd_.size()}});
    flat_dd_.emplace_back(num, pos, val);
    return;
  }

  if (it_dd->second.find(pos) == it_dd->second.end()) {
    it_dd->second.emplace(pos, flat_dd_.size());
    flat_dd_.emplace_back(num, pos, val);
  }
}

} // namespace habana_helpers
