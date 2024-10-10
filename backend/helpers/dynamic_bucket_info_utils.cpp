/******************************************************************************
 * Copyright (C) 2022 Habana Labs, Ltd. an Intel Company
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

#include "backend/helpers/dynamic_bucket_info.h"

namespace habana_helpers {
std::string ResultShapes::DebugString() {
  TORCH_CHECK(
      min_shapes.size() == max_shapes.size(),
      "max and min have different shapes");
  std::ostringstream O;
  std::vector<int64_t> tensor_idx_vec;
  tensor_idx_vec.reserve(min_shapes.size());
  for (const auto& a : min_shapes) {
    tensor_idx_vec.push_back(a.first);
  }
  std::sort(tensor_idx_vec.begin(), tensor_idx_vec.end());
  for (const auto i : tensor_idx_vec) {
    O << " " << i << " : " << min_shapes.at(i) << " - " << max_shapes.at(i)
      << '\n';
  }
  return O.str();
}

std::string ResultShapes::DebugString(const InpTensorShapes& inp_shapes) {
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
    result += std::to_string(tshape_idx) + " : " + tshape_str_lo + " -" +
        tshape_str_hi;
  }
  return result;
}

bool HistoryItem::IsInRange(const ResultShapes& r) {
  bool isInRange{true};
  for (const auto& a : tshapes_) {
    auto& tidx{a.first};
    auto& tshape{a.second};

    TORCH_CHECK(
        r.min_shapes.count(tidx) && r.max_shapes.count(tidx),
        "Tensor index ",
        tidx,
        " is missing from ResultShapes");

    auto& tshape_min{r.min_shapes.at(tidx)};
    auto& tshape_max{r.max_shapes.at(tidx)};

    for (auto& p : tshape) {
      auto& dim_idx{p.first};
      auto& dim_val{p.second};

      if (tshape_min.dim_size(dim_idx) > dim_val ||
          tshape_max.dim_size(dim_idx) < dim_val) {
        isInRange = false;
        break;
      }
    }
    if (!isInRange) {
      break;
    }
  }

  return isInRange;
}

void HistoryItem::Serialize(std::ostream& os) const {
  using namespace serialization;
  serialize(os, tshapes_);
  serialize(os, bucket_index_);
  serialize(os, run_time_);
}

HistoryItem::HistoryItem(std::istream& is) {
  using namespace serialization;
  deserialize(is, tshapes_);
  deserialize(is, bucket_index_);
  deserialize(is, run_time_);
}

std::tuple<bool, size_t, bool> HistoryItemLog::FindMidPoint(
    const std::vector<size_t>& bucket_input_hist_idxes) {
  // Compute the distribution midpoint of input_hist_idxes_
  DimsHistoryElement distr_lo{clone_ref_with(LONG_MAX)};
  DimsHistoryElement distr_hi{clone_ref_with(0)};
  DimsHistoryElement distr_mid{clone_ref_with(LONG_MAX)};

  const DimsHistoryElement& ref{ref_tshapes_};
  for (auto i : bucket_input_hist_idxes) {
    const DimsHistoryElement& d{hist_items_[i].tshapes_};
    for (auto tensor_it : ref) {
      const auto& tensor_idx{tensor_it.first};
      for (auto dim_it : tensor_it.second) {
        const auto& dim_idx{dim_it.first};
        auto dim_val{dim_it.second};
        if (d.count(tensor_idx) && d.at(tensor_idx).count(dim_idx)) {
          dim_val = d.at(tensor_idx).at(dim_idx);
        }
        distr_lo[tensor_idx][dim_idx] =
            std::min(distr_lo[tensor_idx][dim_idx], dim_val);
        distr_hi[tensor_idx][dim_idx] =
            std::max(distr_hi[tensor_idx][dim_idx], dim_val);
      }
    }
  }

  for (auto tensor_it : ref) {
    const auto& tensor_idx{tensor_it.first};
    for (auto dim_it : tensor_it.second) {
      const auto& dim_idx{dim_it.first};
      auto dim_min{distr_lo[tensor_idx][dim_idx]};
      auto dim_max{distr_hi[tensor_idx][dim_idx]};
      distr_mid[tensor_idx][dim_idx] = (dim_min + dim_max) / 2;
    }
  }

  // Find the nearest from mid point
  uint64_t min_dist{ULONG_MAX};
  size_t min_dist_idx{ULONG_MAX};
  for (auto i : bucket_input_hist_idxes) {
    const DimsHistoryElement& d{hist_items_[i].tshapes_};
    uint64_t cur_dist{0};
    for (auto tensor_it : ref) {
      const auto& tensor_idx{tensor_it.first};
      for (auto dim_it : tensor_it.second) {
        const auto& dim_idx{dim_it.first};
        auto dim_val{dim_it.second};
        if (d.count(tensor_idx) && d.at(tensor_idx).count(dim_idx)) {
          dim_val = d.at(tensor_idx).at(dim_idx);
        }
        auto dim_mid{distr_mid[tensor_idx][dim_idx]};
        auto dim_diff = (dim_mid - dim_val) * (dim_mid - dim_val);
        cur_dist += dim_diff;
      }
    }
    if (cur_dist < min_dist) {
      min_dist = cur_dist;
      min_dist_idx = i;
    }
  }

  if (min_dist == 0 && (distr_mid == distr_hi || distr_mid == distr_lo)) {
    return std::make_tuple(false, 0, false);
  }

  const DimsHistoryElement& distr_split{hist_items_[min_dist_idx].tshapes_};

  DimsHistoryElement distr_sp_copy{ref};
  for (auto tensor_it : ref) {
    const auto& tensor_idx{tensor_it.first};
    for (auto dim_it : tensor_it.second) {
      const auto& dim_idx{dim_it.first};
      if (distr_split.count(tensor_idx) &&
          distr_split.at(tensor_idx).count(dim_idx)) {
        distr_sp_copy[tensor_idx][dim_idx] =
            distr_split.at(tensor_idx).at(dim_idx);
      }
    }
  }

  if (distr_sp_copy == distr_hi || distr_sp_copy == distr_lo) {
    return std::make_tuple(false, 0, false);
  }

  // Count the hits at lower as well as upper half based on the nearest mid
  // point
  auto lo_hit_cnt =
      WithinRangeCount(distr_lo, distr_split, bucket_input_hist_idxes);
  auto hi_hit_cnt =
      WithinRangeCount(distr_split, distr_hi, bucket_input_hist_idxes);
  bool choose_lower{(lo_hit_cnt > hi_hit_cnt)};

  return std::make_tuple(true, min_dist_idx, choose_lower);
}

size_t HistoryItemLog::WithinRangeCount(
    const DimsHistoryElement& lo,
    const DimsHistoryElement& hi,
    const std::vector<size_t>& bucket_input_hist_idxes) {
  const DimsHistoryElement& ref{ref_tshapes_};
  size_t within_range_cnt{0};
  for (auto i : bucket_input_hist_idxes) {
    const DimsHistoryElement& d{hist_items_[i].tshapes_};
    bool within_range{true};
    for (auto tensor_it : ref) {
      const auto& tensor_idx{tensor_it.first};
      for (auto dim_it : tensor_it.second) {
        const auto& dim_idx{dim_it.first};
        auto dim_val{dim_it.second};
        auto dim_min{dim_it.second};
        auto dim_max{dim_it.second};

        if (d.count(tensor_idx) && d.at(tensor_idx).count(dim_idx)) {
          dim_val = d.at(tensor_idx).at(dim_idx);
        }
        if (lo.count(tensor_idx) && lo.at(tensor_idx).count(dim_idx)) {
          dim_min = lo.at(tensor_idx).at(dim_idx);
        }
        if (hi.count(tensor_idx) && hi.at(tensor_idx).count(dim_idx)) {
          dim_max = hi.at(tensor_idx).at(dim_idx);
        }
        if (dim_min > dim_val || dim_val > dim_max) {
          within_range = false;
          break;
        }
      }
      if (!within_range) {
        break;
      }
    }

    if (within_range) {
      within_range_cnt++;
    }
  }

  return within_range_cnt;
}

void HistoryItemLog::Serialize(std::ostream& os) const {
  using namespace serialization;
  serialize(os, ref_tshapes_);
  serialize(os, static_cast<int>(hist_items_.size()));
  for (auto& hist_item : hist_items_) {
    hist_item.Serialize(os);
  }
}

void HistoryItemLog::Deserialize(std::istream& is) {
  using namespace serialization;
  deserialize(is, ref_tshapes_);
  int hist_items_size = 0;
  deserialize(is, hist_items_size);
  for (int i = 0; i < hist_items_size; ++i) {
    hist_items_.emplace_back(HistoryItem(is));
  }
}

} // namespace habana_helpers
