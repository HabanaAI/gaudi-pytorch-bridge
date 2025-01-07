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
#include "dynamic_bucket_info.h"

#include <cmath>
#include <cstddef>

#include <algorithm>
#include <functional>
#include <limits>
#include <memory>
#include <variant>

#include "backend/kernel/ds_graph_recompile.h"
#include "backend/kernel/hpu_habana_cache.h"

#include "backend/helpers/compilation_statistics.h"
#include "backend/synapse_helpers/env_flags.h"
#include "habana_helpers/logging.h"

using namespace synapse_helpers;
namespace habana_helpers {

constexpr int64_t DynamicBucketInfo::default_min_value_;
constexpr uint64_t DynamicBucketInfo::min_iterations_to_split_;
constexpr uint64_t DynamicBucketInfo::max_buckets_number_;

size_t DynamicBucketInfo::original_recipe_count_{0};
size_t DynamicBucketInfo::refined_recipe_count_{0};
size_t DynamicBucketInfo::refined_recipe_wirt_count_{0};
size_t DynamicBucketInfo::num_original_recipe_hits_{0};
size_t DynamicBucketInfo::num_refined_recipe_hits_{0};
size_t DynamicBucketInfo::num_refined_recipe_wirt_hits_{0};

uint64_t DynamicBucketInfo::total_syn_runtime_{0};
uint64_t DynamicBucketInfo::original_syn_runtime_{0};
uint64_t DynamicBucketInfo::refined_syn_runtime_{0};

std::unordered_map<uint64_t, bool> DynamicBucketInfo::improvement_map_;

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

Bucket::Bucket(
    DynamicRanges&& ranges,
    DynamicDims dynamic_dims,
    bool is_refine_enabled,
    SplitPolicy sp,
    const InpTensorShapes& shapes_,
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
  token_ = habana_helpers::UniqueTokenGenerator::get_gen().token(
      ranges_, dynamic_dims_, shapes_);
}
void Bucket::setToken(const InpTensorShapes& shapes_) {
  token_ = habana_helpers::UniqueTokenGenerator::get_gen().token(
      ranges_, dynamic_dims_, shapes_);
}

uint64_t UniqueTokenGenerator::token(
    DynamicRanges& ranges,
    DynamicDims& dynamic_dims,
    const InpTensorShapes& shapes_) {
  size_t seed = 0;
  // Use all shapes information to create a md5sum
  for (auto const& ent1 : shapes_) {
    seed = at::hash_combine(seed, ent1.first);
    auto vec = ent1.second.get_dims();
    for (auto itr : vec) {
      seed = at::hash_combine(seed, itr);
    }
  }
  for (auto const& ent1 : ranges) {
    seed = at::hash_combine(seed, ent1.first);
    seed = at::hash_combine(seed, ent1.second);
  }
  for (auto const& ent1 : dynamic_dims) {
    seed = at::hash_combine(seed, ent1.first);
    for (auto const& ent2 : ent1.second) {
      seed = at::hash_combine(seed, ent2.first);
      seed = at::hash_combine(seed, ent2.second);
    }
  }

  return seed;
}

void Bucket::Serialize(std::ostream& os) const {
  using namespace serialization;
  serialize(os, score_);
  serialize(os, run_count_);
  serialize(os, token_);
  serialize(os, idx_);
  serialize(os, recipe_key_);
  serialize(os, is_first_launch_);
  serialize(os, ranges_);
  serialize(os, dynamic_dims_);
  serialize(os, base_time_);
  serialize(os, compile_time_);
  serialize(os, cumu_hit_count_);
  serialize(os, cumu_run_count_);
  run_time_stat_.Serialize(os);
  serialize(os, created_by_refinement_);
  serialize(os, keep_time_);
  serialize(os, refine_candidate_);
  serialize(os, time_improvement_met_);
  serialize(os, input_hist_idxes_);
  serialize(os, inherited_input_hist_idxes_);
}

Bucket::Bucket(std::istream& is) {
  using namespace serialization;
  deserialize(is, score_);
  deserialize(is, run_count_);
  deserialize(is, token_);
  deserialize(is, idx_);
  deserialize(is, recipe_key_);
  deserialize(is, is_first_launch_);
  deserialize(is, ranges_);
  deserialize(is, dynamic_dims_);
  deserialize(is, base_time_);
  deserialize(is, compile_time_);
  deserialize(is, cumu_hit_count_);
  deserialize(is, cumu_run_count_);
  run_time_stat_ = TimeStat(is);
  deserialize(is, created_by_refinement_);
  deserialize(is, keep_time_);
  deserialize(is, refine_candidate_);
  deserialize(is, time_improvement_met_);
  deserialize(is, input_hist_idxes_);
  deserialize(is, inherited_input_hist_idxes_);
  UniqueTokenGenerator::get_gen().set_token(token_);
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

    if (true == GET_ENV_FLAG_NEW(PT_HPU_ENABLE_ZERO_MIN)) {
      if ((ranges_[i].first == ranges_[i].second) &&
          (ranges_[i].first != dims[i])) {
        return false;
      }
      // Since GC assuming min = 0 we just check if input[i] > max
      // for bucket match
      if (dims[i] > ranges_[i].second) {
        return false;
      }
    } else {
      if (dims[i] < ranges_[i].first || ranges_[i].second < dims[i]) {
        return false;
      }
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
  if (created_by_refinement_ == false) {
    DynamicBucketInfo::inc_original_syn_runtime(elapsed_time);
  } else {
    DynamicBucketInfo::inc_refined_syn_runtime(elapsed_time);
  }
  if (base_time_ > 0) {
    uint32_t time_improve_threshold =
        GET_ENV_FLAG_NEW(PT_HPU_DS_TIME_IMPROVE_THRESHOLD_PERCENT);
    double time_improve_factor = (100.0 - time_improve_threshold) / 100.0;
    uint64_t time_to_beat = static_cast<uint64_t>(
        static_cast<double>(base_time_) * time_improve_factor);

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

DynamicBucketInfo::DynamicBucketInfo(size_t key)
    : min_policy_(DynamicDimsPolicy::HISTORIC),
      max_policy_(DynamicDimsPolicy::CALCULATED),
      split_policy_(SplitPolicy::DYNAMIC) {
  SetGraphKey(key);
  if (GET_ENV_FLAG_NEW(PT_HPU_ENABLE_MIN_MAX_AS_CURRENT)) {
    min_policy_ = DynamicDimsPolicy::CURRENT;
    max_policy_ = DynamicDimsPolicy::CURRENT;
    return;
  }
  std::string min_policy_seq =
      GET_ENV_FLAG_NEW(PT_HPU_DYNAMIC_MIN_POLICY_ORDER);
  std::string max_policy_seq =
      GET_ENV_FLAG_NEW(PT_HPU_DYNAMIC_MAX_POLICY_ORDER);
  min_policy_ = getPolicy(min_policy_seq.at(0) - zero_offset);
  max_policy_ = getPolicy(max_policy_seq.at(0) - zero_offset);
}

bool DynamicBucketInfo::IsBucketMember(int64_t tensor_idx, uint64_t bucket) {
  auto& dynamic_dims = buckets_[bucket].getDynamicDims();
  return dynamic_dims.count(tensor_idx);
}

void DynamicBucketInfo::UpdateShapes(
    uint64_t bucket,
    int64_t tensor_idx,
    int64_t dim_idx,
    int64_t new_val) {
  // This function can update both min/max values based on current pass;
  HABANA_ASSERT(
      ((habana::ShapeInference::GetCurrentPass() ==
        habana::ShapeInfo::InferencePass::MAX_SHAPE) ||
       (habana::ShapeInference::GetCurrentPass() ==
        habana::ShapeInfo::InferencePass::MIN_SHAPE)),
      "UpdateShape is supported from min/max passes only")

  auto& dynamic_dims = buckets_[bucket].getDynamicDims();
  if (dynamic_dims.count(tensor_idx) &&
      dynamic_dims.at(tensor_idx).count(dim_idx)) {
    auto range_idx = dynamic_dims.at(tensor_idx).at(dim_idx);
    auto& ranges = buckets_[bucket].getRanges();
    HABANA_ASSERT(
        (range_idx < static_cast<int64_t>(ranges.size())),
        "UpdateShapes ranges exceed the index");

    std::pair<int64_t, int64_t> new_minmax;
    bool isMax =
        (habana::ShapeInference::GetCurrentPass() ==
         habana::ShapeInfo::InferencePass::MAX_SHAPE);

    if (isMax) {
      new_minmax = std::make_pair(ranges[range_idx].first, new_val);
    } else {
      new_minmax = std::make_pair(new_val, ranges[range_idx].second);
    }
    buckets_[bucket].updateRanges(range_idx, new_minmax);
  }
}

ResultShapes DynamicBucketInfo::CalculateShapes(uint64_t bucket) {
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
      local_min_history_tensor_shapes_.emplace(
          tensor_idx, std::map<int64_t, int64_t>());
      local_max_history_tensor_shapes_.emplace(
          tensor_idx, std::map<int64_t, int64_t>());
      min_user_shapes_.emplace(tensor_idx, std::map<int64_t, int64_t>());
      max_user_shapes_.emplace(tensor_idx, std::map<int64_t, int64_t>());
      local_pt_history_tensor_shapes_[0].emplace(
          tensor_idx, std::map<int64_t, int64_t>());
      local_pt_history_tensor_shapes_[1].emplace(
          tensor_idx, std::map<int64_t, int64_t>());
      for (size_t dim_idx = 0; dim_idx < tensor_it->second.dims(); dim_idx++) {
        auto dim_val{tensor_it->second.dim_size(dim_idx)};
        ref_tshapes[tensor_idx].emplace(dim_idx, dim_val);
        local_min_history_tensor_shapes_[tensor_idx].emplace(
            dim_idx, (dim_val == 1) ? INT_MAX : dim_val);
        local_max_history_tensor_shapes_[tensor_idx].emplace(dim_idx, dim_val);
        local_pt_history_tensor_shapes_[0][tensor_idx].emplace(
            dim_idx, (dim_val == 1) ? INT_MAX : dim_val);
        local_pt_history_tensor_shapes_[1][tensor_idx].emplace(
            dim_idx, dim_val);
        min_user_shapes_[tensor_idx].emplace(dim_idx, dim_val);
        max_user_shapes_[tensor_idx].emplace(dim_idx, dim_val);
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
      continue;
    }
    uint64_t cur_bucket_run_count{b.GetRunCount()};
    if (mfu_bucket_run_count <= cur_bucket_run_count) {
      mfu_bucket_run_count = cur_bucket_run_count;
      mfu_bucket_id = b.GetIndex();
    }
  }
}

void DynamicBucketInfo::UpdateMFUBucketDetails(size_t bucket_id) {
  // Skip the refinement of the static bucket
  if (buckets_[bucket_id].IsStatic()) {
    return;
  }
  current_run_count += 1;
  uint64_t cur_bucket_run_count{buckets_[bucket_id].GetRunCount()};
  if (mfu_bucket_run_count <= cur_bucket_run_count) {
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
    buckets_.emplace_back(
        DynamicRanges{}, DynamicDims{}, true, split_policy_, shapes_);
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
      split_policy_,
      shapes_);
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

absl::optional<uint64_t> DynamicBucketInfo::CheckForSplitBucket(
    std::shared_ptr<habana_helpers::DynamicBucketInfo> dbipsh) {
  PT_DYNAMIC_SHAPE_DEBUG("Checking buckets for refinement");
  if (refine_enabled_ == false) {
    PT_DYNAMIC_SHAPE_DEBUG("Refinement is not enabled");
    return {};
  }
  if (buckets_.size() >= max_buckets_number_) {
    PT_DYNAMIC_SHAPE_DEBUG(
        "Maxed out total number=", max_buckets_number_, " of buckets");
    return {};
  }
  if (current_run_count < min_iterations_to_split_) {
    PT_DYNAMIC_SHAPE_DEBUG(
        "Yet to reach ",
        min_iterations_to_split_,
        " for graph, currently at ",
        current_run_count);
    return {};
  }
  if (mfu_bucket_run_count < min_iterations_to_split_) {
    PT_DYNAMIC_SHAPE_DEBUG(
        "Yet to reach ",
        min_iterations_to_split_,
        " for mfu bucket, currently at ",
        mfu_bucket_run_count);
    return {};
  }

  size_t curr_mfu_id = mfu_bucket_id;
  if (curr_mfu_id == 0) {
    PT_DYNAMIC_SHAPE_DEBUG("Can not refine static bucket");
    return {};
  }

  statistics_->SetCurrentParentBucketID(curr_mfu_id);
  statistics_->SetCurrentParentLastStep(
      buckets_[curr_mfu_id].GetLastUsedStep());
  if (buckets_[curr_mfu_id].IsRefinementCandidate() == false) {
    PT_DYNAMIC_SHAPE_DEBUG(
        "Bucket ", curr_mfu_id, " is not a candidate for refinement");
    return {};
  }

  bool isRuntimeImproved{buckets_[curr_mfu_id].IsRuntimeImproved()};

  PT_DYNAMIC_SHAPE_DEBUG(
      "Current mfu bucket %d is eligible for refinement", curr_mfu_id);
  auto rvpsh = buckets_[curr_mfu_id].GetSynapseRecipePtr();
  if (nullptr == rvpsh) {
    PT_DYNAMIC_SHAPE_DEBUG("Recipe for mfu bucket is null");
    return {};
  }

  bool is_valid_split{};
  size_t min_dist_idx{};
  bool choose_lower{};
  std::vector<size_t> bucket_hist_idxes(
      buckets_[curr_mfu_id].GetInheritedInputHistIdxes());
  bucket_hist_idxes.insert(
      bucket_hist_idxes.end(),
      buckets_[curr_mfu_id].GetInputHistIdxes().begin(),
      buckets_[curr_mfu_id].GetInputHistIdxes().end());
  std::tie(is_valid_split, min_dist_idx, choose_lower) =
      input_history_.FindMidPoint(bucket_hist_idxes);
  if (is_valid_split == false) {
    PT_DYNAMIC_SHAPE_DEBUG(
        "Range mid point is matching with one of the endpoints.",
        " Abandoning refinement.");
    return {};
  }

  // Use the split history input as lo or hi depending on choose_lower
  ResultShapes result_computed(shapes_);
  Bucket new_bucket_computed = ConstructNewBucket(
      result_computed, buckets_[curr_mfu_id], min_dist_idx, choose_lower);

  std::unordered_map<uint64_t, habana::ShapeTensorStruct>& input_metadata =
      buckets_[curr_mfu_id].GetInputMetaData();
  Bucket& new_bucket_candidate{new_bucket_computed};
  uint64_t new_bucket_candidate_id = buckets_.size();
  new_bucket_candidate.SetIndex(new_bucket_candidate_id);
  ResultShapes& new_range{result_computed};
  bool is_compiled{false};
  size_t new_recipe_key{0};
  try {
    is_compiled = habana::CompileGraphWithRange(
        rvpsh,
        input_metadata,
        new_range,
        new_bucket_candidate,
        new_recipe_key,
        statistics_,
        dbipsh);
  } catch (std::exception& e) {
    PT_DYNAMIC_SHAPE_WARN(
        "Recipe compilation failed with exception '", e.what(), "'");
    return {};
  }
  PT_DYNAMIC_SHAPE_DEBUG(
      "Recipe compilation for new bucket: ",
      (is_compiled ? "successful" : "failed"));

  // Only push this bucket if the compilation is successful
  if (is_compiled) {
    std::lock_guard<std::mutex> lg(refine_mutex_);

    // Move the history
    // Find the previous hits
    auto& input_hist_idxes{buckets_[curr_mfu_id].GetInputHistIdxes()};
    std::vector<size_t> input_hist_move;
    std::vector<size_t> input_hist_retain;
    split_history(
        input_hist_idxes, new_range, input_hist_move, input_hist_retain);

    auto& inherited_input_hist_idxes{
        buckets_[curr_mfu_id].GetInheritedInputHistIdxes()};
    std::vector<size_t> inherited_input_hist_move;
    std::vector<size_t> inherited_input_hist_retain;
    split_history(
        inherited_input_hist_idxes,
        new_range,
        inherited_input_hist_move,
        inherited_input_hist_retain);

    buckets_[curr_mfu_id].SetInputHistIdxes(input_hist_retain);
    buckets_[curr_mfu_id].SetInheritedInputHistIdxes(
        inherited_input_hist_retain);
    buckets_[curr_mfu_id].ResetBaseLine(input_history_);

    buckets_.push_back(new_bucket_candidate);
    uint64_t new_bucket_id = buckets_.size() - 1;
    auto& new_bucket = buckets_.back();

    std::string result_str{"OK"};
    uint64_t current_step{statistics_->GetCurrentStep()};
    statistics_->LogRefineCompilation(
        new_range,
        rvpsh->jit_graph_,
        new_recipe_key,
        new_bucket_id,
        result_str,
        current_step);

    // Append the input to be moved to inherited input of the new bucket
    inherited_input_hist_move.insert(
        inherited_input_hist_move.end(),
        input_hist_move.begin(),
        input_hist_move.end());
    new_bucket.SetInheritedInputHistIdxes(inherited_input_hist_move);
    new_bucket.SetIndex(new_bucket_id);
    new_bucket.ResetBaseLine(input_history_);
    new_bucket.SetCreatedByRefinement();
    SetRecipeKeyForBucket(new_bucket_id, new_recipe_key);
    inc_refined_recipe_count();
    new_bucket.GetSynapseRecipePtr()->set_refined();
    if (isRuntimeImproved) {
      inc_refined_recipe_wirt_count();
      new_bucket.GetSynapseRecipePtr()->set_refined_wirt();
    }

    PT_DYNAMIC_SHAPE_DEBUG(
        "Bucket with id ",
        curr_mfu_id,
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

size_t DynamicBucketInfo::GetUserBucketId(
    const InpTensorShapes& shapes,
    std::vector<habana_helpers::RangeInfo>& range_infos) {
  TORCH_CHECK(shapes_.size() == shapes.size(), "Shapes dont match");
  PT_DYNAMIC_SHAPE_DEBUG("Creating bucket with user ranges");
  cumu_run_count_++;
  global_count++;

  // Create new bucket
  DynamicRanges ranges;
  DynamicDims dd;
  // DynamicDims : input_idx => {dim_idx => range_idx in DynamicRanges}
  TORCH_CHECK(!max_user_shapes_.empty(), "User max structure is empty");
  TORCH_CHECK(!min_user_shapes_.empty(), "User min structure is empty");
  for (auto dynamic_dims{max_user_shapes_.begin()};
       dynamic_dims != max_user_shapes_.end();
       dynamic_dims++) {
    auto tensor_idx = dynamic_dims->first;
    auto range_info = range_infos[tensor_idx];
    auto tensor_sizes_min = range_info.min_shape;
    auto tensor_sizes_max = range_info.max_shape;
    std::map<int64_t, int64_t> dim_range_map;
    for (auto curr_dim{dynamic_dims->second.begin()};
         curr_dim != dynamic_dims->second.end();
         curr_dim++) {
      auto dim_idx = curr_dim->first;
      int64_t min_dim_val = tensor_sizes_min[dim_idx];
      int64_t max_dim_val = tensor_sizes_max[dim_idx];
      min_user_shapes_.at(tensor_idx).at(dim_idx) = min_dim_val;
      max_user_shapes_.at(tensor_idx).at(dim_idx) = max_dim_val;
      if (min_dim_val != max_dim_val) {
        ranges.emplace_back(std::make_pair(min_dim_val, max_dim_val));
        dim_range_map[dim_idx] = ranges.size() - 1;
        dynamic_dims_helper_.FindOrAdd(
            tensor_idx, dim_idx, shapes_.at(tensor_idx).dim_size(dim_idx));
      }
    }
    if (!dim_range_map.empty()) {
      dd[tensor_idx] = dim_range_map;
    }
  }

  buckets_.emplace_back(
      std::move(ranges), dd, refine_enabled_, split_policy_, shapes_);
  auto& new_bucket = buckets_.back();
  uint64_t new_bucket_id = buckets_.size() - 1;
  new_bucket.SetIndex(new_bucket_id);
  // Start the history log
  input_history_.hist_items().emplace_back(
      DimsHistoryElement{}, new_bucket_id, 0);
  current_input_idx_ = 0;
  buckets_[new_bucket_id].AppendInputHistIndex(current_input_idx_);

  return buckets_.size() - 1;
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

  return Bucket(
      std::move(new_ranges), dynamic_dims, true, split_policy_, shapes_);
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

void DynamicBucketInfo::Serialize(std::ostream& os) const {
  using namespace serialization;
  serialize(os, static_cast<int>(buckets_.size()));
  for (auto& bucket : buckets_) {
    bucket.Serialize(os);
  }
  serialize(os, global_count);
  serialize(os, mfu_bucket_id);
  serialize(os, mfu_bucket_run_count);
  serialize(os, current_run_count);
  statistics_->Serialize(os);
  serialize(os, static_cast<int>(shapes_.size()));
  for (auto& shape : shapes_) {
    serialize(os, shape.first);
    shape.second.Serialize(os);
  }
  serialize(os, local_min_history_tensor_shapes_);
  serialize(os, local_min_history_success_shapes_);
  serialize(os, local_max_history_tensor_shapes_);
  serialize(os, local_max_history_success_shapes_);
  serialize(os, max_user_shapes_);
  serialize(os, min_user_shapes_);
  for (auto& element : local_pt_history_tensor_shapes_) {
    serialize(os, element);
  }
  for (auto& element : local_pt_history_success_shapes_) {
    serialize(os, element);
  }
  serialize(os, min_policy_);
  serialize(os, max_policy_);
  serialize(os, refine_enabled_);
  input_history_.Serialize(os);
  serialize(os, current_input_idx_);
  dynamic_dims_helper_.Serialize(os);
  serialize(os, graph_key_);
  cumu_run_time_stat_.Serialize(os);
  cumu_compile_time_stat_.Serialize(os);
  serialize(os, cumu_compile_count_);
  serialize(os, cumu_run_count_);
  serialize(os, cumu_hit_count_);
  serialize(os, input_token_map_);
  serialize(os, original_recipe_count_);
  serialize(os, refined_recipe_count_);
  serialize(os, refined_recipe_wirt_count_);
  serialize(os, num_original_recipe_hits_);
  serialize(os, num_refined_recipe_hits_);
  serialize(os, num_refined_recipe_wirt_hits_);
  serialize(os, total_syn_runtime_);
  serialize(os, original_syn_runtime_);
  serialize(os, refined_syn_runtime_);
  serialize(os, improvement_map_);
}

DynamicBucketInfo::DynamicBucketInfo(std::istream& is) {
  using namespace serialization;
  int bucket_size = 0;
  deserialize(is, bucket_size);
  for (int i = 0; i < bucket_size; ++i) {
    buckets_.emplace_back(Bucket(is));
  }
  deserialize(is, global_count);
  deserialize(is, mfu_bucket_id);
  deserialize(is, mfu_bucket_run_count);
  deserialize(is, current_run_count);
  statistics_ = std::make_shared<habana_helpers::CompilationStatistics>(is);
  int num_of_tensors = 0;
  deserialize(is, num_of_tensors);
  for (int i = 0; i < num_of_tensors; ++i) {
    int64_t key = 0;
    deserialize(is, key);
    auto value = habana_helpers::TensorShape(is);
    shapes_[key] = value;
  }
  deserialize(is, local_min_history_tensor_shapes_);
  deserialize(is, local_min_history_success_shapes_);
  deserialize(is, local_max_history_tensor_shapes_);
  deserialize(is, local_max_history_success_shapes_);
  deserialize(is, max_user_shapes_);
  deserialize(is, min_user_shapes_);
  for (int i = 0; i < 2; ++i) {
    deserialize(is, local_pt_history_tensor_shapes_[i]);
  }
  for (int i = 0; i < 2; ++i) {
    deserialize(is, local_pt_history_success_shapes_[i]);
  }
  deserialize(is, min_policy_);
  deserialize(is, max_policy_);
  deserialize(is, refine_enabled_);
  input_history_.Deserialize(is);
  deserialize(is, current_input_idx_);
  dynamic_dims_helper_.Deserialize(is);
  deserialize(is, graph_key_);
  cumu_run_time_stat_ = TimeStat(is);
  cumu_compile_time_stat_ = TimeStat(is);
  deserialize(is, cumu_compile_count_);
  deserialize(is, cumu_run_count_);
  deserialize(is, cumu_hit_count_);
  deserialize(is, input_token_map_);
  deserialize(is, original_recipe_count_);
  deserialize(is, refined_recipe_count_);
  deserialize(is, refined_recipe_wirt_count_);
  deserialize(is, num_original_recipe_hits_);
  deserialize(is, num_refined_recipe_hits_);
  deserialize(is, num_refined_recipe_wirt_hits_);
  deserialize(is, total_syn_runtime_);
  deserialize(is, original_syn_runtime_);
  deserialize(is, refined_syn_runtime_);
  deserialize(is, improvement_map_);
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
      buckets_[bucket_id].setToken(shapes_);
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
        if (((1 == current_dim_val || 1 == historic_dim_val) &&
             current_dim_val != historic_dim_val) ||
            (1 != current_dim_val && 1 != historic_dim_val &&
             comp(historic_dim_val, current_dim_val))) {
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

void DynamicBucketInfo::CalculateLocalHistoricPerTensor(
    const InpTensorShapes& shapes,
    bool isMin) {
  auto idx = isMin ? 0 : 1;
  auto& local_history_success_shapes_ = local_pt_history_success_shapes_[idx];
  auto& local_history_tensor_shapes_ = local_pt_history_tensor_shapes_[idx];
  TORCH_CHECK(
      !local_history_tensor_shapes_.empty(),
      "Local per tensor history is empty");
  local_history_success_shapes_ = local_history_tensor_shapes_;
  for (auto dynamic_dims{local_history_tensor_shapes_.begin()};
       dynamic_dims != local_history_tensor_shapes_.end();
       dynamic_dims++) {
    auto tensor_idx = dynamic_dims->first;

    // No update to history required if a tensor does not have a dynamic dim
    if (dynamic_dims_helper_.dd_.find(tensor_idx) ==
        dynamic_dims_helper_.dd_.end()) {
      continue;
    }

    std::map<int64_t, int64_t> local_vals, current_vals;
    for (auto curr_dim{dynamic_dims->second.begin()};
         curr_dim != dynamic_dims->second.end();
         curr_dim++) {
      auto dim_idx = curr_dim->first;
      int64_t current_dim_val = shapes.at(tensor_idx).dim_size(dim_idx);
      current_vals[dim_idx] = current_dim_val;
      local_vals[dim_idx] =
          local_history_tensor_shapes_.at(tensor_idx).at(dim_idx);
    }

    auto y_min = [](std::map<int64_t, int64_t> first,
                    std::map<int64_t, int64_t> second) {
      auto it1 = first.cbegin();
      auto it2 = second.cbegin();
      // if any dim value within current_val_tensor is less than
      // historic_val_tensor, update vals. Having a stricter check, i.e., update
      // vals only if all dim vals are less does not work in cases where dim
      // vals are changing in opposite directions.
      // E.g. historic_vals = {1, 1056, 811008} current_vals = {1, 800, 870400}
      // we end up creating range {1056 - 1600} which will fail in patching.
      for (; it1 != first.end(); it1++, it2++) {
        if (it1->second < it2->second) {
          return true;
        }
      }
      return false;
    };

    auto y_max = [](std::map<int64_t, int64_t> first,
                    std::map<int64_t, int64_t> second) {
      auto it1 = first.cbegin();
      auto it2 = second.cbegin();
      // if any dim value within current_val_tensor is greater than
      // historic_val_tensor, update vals. Having a stricter check, i.e., update
      // vals only if all dim vals are greater does not work in cases where dim
      // vals are changing in opposite directions.
      for (; it1 != first.end(); it1++, it2++) {
        if (it1->second > it2->second) {
          return true;
        }
      }
      return false;
    };

    // Check if input recieved is lower than already stored,
    // If yes replace the input stored with recieved
    if (isMin) {
      for (auto curr_vals_local{current_vals.begin()};
           curr_vals_local != current_vals.end();
           curr_vals_local++) {
        // if value on a dim is 1, then skip comparison for that dim
        if (curr_vals_local->second == 1) {
          curr_vals_local->second = local_vals.at(curr_vals_local->first);
        }
      }
      if (y_min(current_vals, local_vals)) {
        local_history_tensor_shapes_.at(tensor_idx) = current_vals;
      }
    }
    // Check if input recieved is higher than already stored,
    // If yes replace the input stored with recieved
    if (!isMin && y_max(current_vals, local_vals)) {
      local_history_tensor_shapes_.at(tensor_idx) = current_vals;
    }
  }
}

void DynamicBucketInfo::CalculateLocalHistoricMin(
    const InpTensorShapes& shapes) {
  TORCH_CHECK(
      !local_min_history_tensor_shapes_.empty(), "Local min history is empty");
  local_min_history_success_shapes_ = local_min_history_tensor_shapes_;
  for (auto dynamic_dims{local_min_history_tensor_shapes_.begin()};
       dynamic_dims != local_min_history_tensor_shapes_.end();
       dynamic_dims++) {
    auto tensor_idx = dynamic_dims->first;
    for (auto curr_dim{dynamic_dims->second.begin()};
         curr_dim != dynamic_dims->second.end();
         curr_dim++) {
      auto dim_idx = curr_dim->first;
      int64_t current_dim_val = shapes.at(tensor_idx).dim_size(dim_idx);
      // Check if input recieved is lower than already stored,
      // If yes replace the input stored with recieved
      if ((local_min_history_tensor_shapes_.at(tensor_idx).at(dim_idx) >
           current_dim_val) &&
          (current_dim_val != 1)) {
        local_min_history_tensor_shapes_.at(tensor_idx).at(dim_idx) =
            current_dim_val;
      }
    }
  }
}

void DynamicBucketInfo::CalculateLocalHistoricMax(
    const InpTensorShapes& shapes) {
  TORCH_CHECK(
      !local_max_history_tensor_shapes_.empty(), "Local max history is empty");
  local_max_history_success_shapes_ = local_max_history_tensor_shapes_;
  for (auto dynamic_dims{local_max_history_tensor_shapes_.begin()};
       dynamic_dims != local_max_history_tensor_shapes_.end();
       dynamic_dims++) {
    auto tensor_idx = dynamic_dims->first;
    for (auto curr_dim{dynamic_dims->second.begin()};
         curr_dim != dynamic_dims->second.end();
         curr_dim++) {
      auto dim_idx = curr_dim->first;
      int64_t current_dim_val = shapes.at(tensor_idx).dim_size(dim_idx);
      // Check if input recieved is lower than already stored,
      // If yes replace the input stored with recieved
      if (local_max_history_tensor_shapes_.at(tensor_idx).at(dim_idx) <
          current_dim_val) {
        local_max_history_tensor_shapes_.at(tensor_idx).at(dim_idx) =
            current_dim_val;
      }
    }
  }
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

  if (min_policy_ == DynamicDimsPolicy::LOCAL_HISTORIC) {
    CalculateLocalHistoricMin(shapes);
    min_dim_shapes = local_min_history_tensor_shapes_;
  }

  if (min_policy_ == DynamicDimsPolicy::LOCAL_HIST_PER_TSR) {
    CalculateLocalHistoricPerTensor(shapes, true);
    min_dim_shapes = local_pt_history_tensor_shapes_[0];
  }

  if (max_policy_ == DynamicDimsPolicy::HISTORIC) {
    auto dims_history_idx{CalculateHistoricMax(shapes)};
    max_dim_shapes = dims_history.at(dims_history_idx).tshapes();
  }
  if (max_policy_ == DynamicDimsPolicy::LOCAL_HISTORIC) {
    CalculateLocalHistoricMax(shapes);
    max_dim_shapes = local_max_history_tensor_shapes_;
  }
  if (max_policy_ == DynamicDimsPolicy::LOCAL_HIST_PER_TSR) {
    CalculateLocalHistoricPerTensor(shapes, false);
    max_dim_shapes = local_pt_history_tensor_shapes_[1];
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
      case DynamicDimsPolicy::LOCAL_HISTORIC:
      case DynamicDimsPolicy::LOCAL_HIST_PER_TSR:
        if (GET_ENV_FLAG_NEW(PT_HPU_ENABLE_BROADCAST_BUCKET_HANDLING) &&
            (shapes.at(el.num).dim_size(el.pos) == 1 ||
             ref_dim_val == INT_MAX)) {
          min_value = shapes.at(el.num).dim_size(el.pos);
        } else {
          min_value = ref_dim_val;
          if (min_dim_shapes.count(tensor_idx)) {
            auto& dim_map = min_dim_shapes.at(tensor_idx);
            if (dim_map.count(dim_idx)) {
              min_value = dim_map.at(dim_idx);
            }
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
        TORCH_CHECK(
            false,
            "Policy FLATTENED is currently unsupported for choosing min");
        min_value = dim_multipliers.at(el.num).at(el.pos).first;
        break;
      case DynamicDimsPolicy::CALCULATED:
        if (1 != current_shape.dim_size(el.pos)) {
          min_value = current_shape.dim_size(el.pos) * 0.5;
        }
        break;
    }

    // min == 1 means that the corresponding dimension is a broadcast dimension
    // max will have to be 1 whenever min == 1
    // historic should return max as 1 whenever min == 1
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
      case DynamicDimsPolicy::LOCAL_HISTORIC:
      case DynamicDimsPolicy::LOCAL_HIST_PER_TSR:
        if (GET_ENV_FLAG_NEW(PT_HPU_ENABLE_BROADCAST_BUCKET_HANDLING) &&
            1 == min_value) {
          max_value = 1;
        } else {
          auto& dim_map = max_dim_shapes.at(tensor_idx);
          if (dim_map.count(dim_idx)) {
            max_value = dim_map.at(dim_idx);
          }
        }
        break;
      case DynamicDimsPolicy::CURRENT:
        max_value = int64_t(shapes.at(el.num).dim_size(el.pos));
        break;
      case DynamicDimsPolicy::FLATTENED:
        TORCH_CHECK(
            false,
            "Policy FLATTENED is currently unsupported for choosing max");
        dim_max_multiplier = dim_multipliers.at(el.num).at(el.pos).second;
        max_value =
            int64_t(shapes.at(el.num).dim_size(el.pos)) * dim_max_multiplier;
        break;
      case DynamicDimsPolicy::CALCULATED:
        // use default max multiplier
        if (GET_ENV_FLAG_NEW(PT_HPU_ENABLE_BROADCAST_BUCKET_HANDLING) &&
            1 == min_value) {
          max_value = 1;
        } else {
          max_value =
              int64_t(shapes.at(el.num).dim_size(el.pos)) * dim_max_multiplier;
        }
        break;
    }

    // determine min
    int64_t min = shapes.at(el.num).dim_size(el.pos) >= default_min_value_
        ? min_value
        : int64_t(shapes.at(el.num).dim_size(el.pos));

    int64_t max{max_value};

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
    if (GET_ENV_FLAG_NEW(PT_HPU_ENABLE_BROADCAST_BUCKET_HANDLING)) {
      TORCH_CHECK(
          min != 1 || max == 1,
          "with min policy: ",
          min_policy_,
          ", and max policy: ",
          max_policy_,
          '\n',
          "Incompatible min=",
          min,
          " and max=",
          max,
          " values are computed. Broadcast may break");
    }

    result.emplace_back(std::make_pair(min, max));
  }
  return result;
}

void DynamicBucketInfo::RegisterTimeSlot(
    const std::shared_ptr<synapse_helpers::TimeSlotBase>& ts,
    uint64_t bucket_id) {
  UpdateRunTimes();
  run_time_q_.emplace_back(ts, bucket_id, current_input_idx_);
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
    inc_total_syn_runtime(t_ns);
    // Ignore the first launch runtime
    if (!buckets_[bucket_id].IsFirstLaunch())
      input_history_.hist_items_[input_hist_idx].run_time_ = t_ns;
    TORCH_CHECK(
        bucket_id < buckets_.size(),
        "invalid bucket index access in UpdateRunTimes at ",
        __FILE__,
        " : ",
        __LINE__);
    buckets_[bucket_id].UpdateRunTime(t_ns);
    cumu_run_time_stat_.Update(t_ns);
    run_time_q_.pop_front();
  }
}

bool DynamicBucketInfo::NeedRunTimeSlot(uint64_t bucket) {
  TORCH_CHECK(
      bucket < buckets_.size(),
      "Invalid bucket index access in NeedRunTimeSlot ",
      __FILE__,
      " : ",
      __LINE__);
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
    O << bucket_range_str(bucket, bucket.IsStatic());
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

void DynamicBucketInfo::DynamicDimsHelper::Serialize(std::ostream& os) const {
  using namespace serialization;
  serialize(os, dd_);
  serialize(os, rem_size_);
  serialize(os, static_cast<int>(flat_dd_.size()));
  for (auto& ele : flat_dd_) {
    ele.Serialize(os);
  }
}

void DynamicBucketInfo::DynamicDimsHelper::Deserialize(std::istream& is) {
  using namespace serialization;
  deserialize(is, dd_);
  deserialize(is, rem_size_);
  int flat_dd_size = 0;
  deserialize(is, flat_dd_size);
  for (int i = 0; i < flat_dd_size; ++i) {
    flat_dd_.emplace_back(DynamicDimsElement(is));
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

void DynamicBucketInfo::DumpDynamicRecipeStat() {
  size_t launch_time_increase_cnt{0};
  for (auto k : improvement_map_) {
    if (k.second == false)
      launch_time_increase_cnt += 1;
  }
  PT_REFINEMENT_DEBUG(
      "  #original_recipes=",
      DynamicBucketInfo::original_recipe_count_,
      ", #refined_recipes=",
      DynamicBucketInfo::refined_recipe_count_,
      ", #refined_recipes_wirt=",
      DynamicBucketInfo::refined_recipe_wirt_count_,
      ", #original_recipe_hits=",
      DynamicBucketInfo::num_original_recipe_hits_,
      ", #refined_recipe_hits=",
      DynamicBucketInfo::num_refined_recipe_hits_,
      ", #refined_recipe_wirt_hits=",
      DynamicBucketInfo::num_refined_recipe_wirt_hits_,
      ", #total_syn_runtime=",
      DynamicBucketInfo::total_syn_runtime_,
      ", #total_syn_runtime=",
      DynamicBucketInfo::original_syn_runtime_,
      ", #refined_syn_runtime=",
      DynamicBucketInfo::refined_syn_runtime_,
      ", (",
      launch_time_increase_cnt,
      " | ",
      improvement_map_.size(),
      ")");
}

void DynamicBucketInfo::DisableBucketRefinement() {
  SET_ENV_FLAG_NEW(PT_HPU_ENABLE_COMPILE_THREAD, false, 1);
}

} // namespace habana_helpers
