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

#pragma once

#include <climits>

#include <map>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include "backend/helpers/tensor_shape.h"
#include "habana_helpers/habana_serialization/include/habana_serialization/deserializers.h"
#include "habana_helpers/habana_serialization/include/habana_serialization/serializers.h"

namespace habana_helpers {
constexpr size_t max_elements_to_print = 64;

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
    O << '\n' << " " << a.first << " : " << a.second;
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
  std::sort(tensor_idx_vec.begin(), tensor_idx_vec.end());
  for (const auto i : tensor_idx_vec) {
    O << "  " << i << ":" << t.at(i);
    O << '\n';
  }
  return O;
}

inline std::ostream& operator<<(
    std::ostream& O,
    const std::unordered_map<uint64_t, habana_helpers::TensorShape>& t) {
  std::vector<uint64_t> tensor_idx_vec;
  tensor_idx_vec.reserve(t.size());
  for (const auto& a : t) {
    tensor_idx_vec.push_back(a.first);
  }
  std::sort(tensor_idx_vec.begin(), tensor_idx_vec.end());
  for (const auto i : tensor_idx_vec) {
    O << "  " << i << ":" << t.at(i);
    O << '\n';
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
  uint64_t GetAvgTime() const {
    return average_time_;
  }
  uint64_t GetMinTime() const {
    return min_time_;
  }
  uint64_t GetMaxTime() const {
    return max_time_;
  }

  void Reset() {
    total_time_ = 0;
    average_time_ = 0;
    min_time_ = {std::numeric_limits<uint64_t>::max()};
    max_time_ = 0;
    num_samples_ = 0;
  }

  friend inline std::ostream& operator<<(std::ostream& O, const TimeStat& t) {
    O << "<#samples=" << t.num_samples_ << " min="
      << (t.min_time_ == std::numeric_limits<uint64_t>::max() ? 0 : t.min_time_)
      << " max=" << t.max_time_ << " avg=" << t.average_time_
      << " total=" << t.total_time_ << '>';
    return O;
  }

  void Serialize(std::ostream& os) const {
    using namespace serialization;
    serialize(os, total_time_);
    serialize(os, average_time_);
    serialize(os, min_time_);
    serialize(os, max_time_);
    serialize(os, num_samples_);
  }

  TimeStat(std::istream& is) {
    using namespace serialization;
    deserialize(is, total_time_);
    deserialize(is, average_time_);
    deserialize(is, min_time_);
    deserialize(is, max_time_);
    deserialize(is, num_samples_);
  }

 private:
  uint64_t total_time_{};
  uint64_t average_time_{};
  uint64_t min_time_{std::numeric_limits<uint64_t>::max()};
  uint64_t max_time_{};
  uint64_t num_samples_{};
};

using InpTensorShapes = std::map<int64_t, habana_helpers::TensorShape>;
using TensorShapes = std::unordered_map<int64_t, habana_helpers::TensorShape>;

struct ResultShapes {
  TensorShapes min_shapes;
  TensorShapes max_shapes;

  ResultShapes() = default;
  ResultShapes(const InpTensorShapes& inp_shapes)
      : min_shapes(inp_shapes.begin(), inp_shapes.end()),
        max_shapes(inp_shapes.begin(), inp_shapes.end()) {}

  bool empty() const {
    return (min_shapes.empty() && max_shapes.empty());
  }
  // SynapseShapes syn_shapes;
  std::string DebugString();
  std::string DebugString(const InpTensorShapes& inp_shapes);
};

struct HistoryItem {
  DimsHistoryElement tshapes_;
  size_t bucket_index_{ULONG_MAX};
  uint64_t run_time_{0};

  size_t Size() const {
    size_t size = sizeof(*this);
    for (auto& s : tshapes_) {
      size += sizeof(decltype(tshapes_)::key_type);
      size += s.second.size() *
          (sizeof(decltype(s.second)::key_type) +
           sizeof(decltype(s.second)::mapped_type));
    }
    return size;
  }

  HistoryItem(DimsHistoryElement&& e, size_t i, uint64_t t)
      : tshapes_(std::move(e)), bucket_index_(i), run_time_(t) {}

  bool IsInRange(const ResultShapes& r);
  uint64_t run_time() const {
    return run_time_;
  }
  const DimsHistoryElement& tshapes() const {
    return tshapes_;
  }
  DimsHistoryElement& tshapes() {
    return tshapes_;
  }

  void Serialize(std::ostream& os) const;
  HistoryItem(std::istream& is);
};

inline std::string DebugString(
    const HistoryItem& h,
    const DimsHistoryElement& ref) {
  std::ostringstream O;
  O << " Input shape:" << DebugString(h.tshapes_, ref) << '\n'
    << " Bucket id: " << h.bucket_index_ << '\n'
    << " Runtime: " << h.run_time_;
  return O.str();
}

struct HistoryItemLog {
  DimsHistoryElement ref_tshapes_;
  std::vector<HistoryItem> hist_items_;

  size_t Size() const {
    size_t size = sizeof(*this);
    for (auto& s : ref_tshapes_) {
      size += sizeof(decltype(ref_tshapes_)::key_type);
      size += s.second.size() *
          (sizeof(decltype(s.second)::key_type) +
           sizeof(decltype(s.second)::mapped_type));
    }
    for (auto& h : hist_items_) {
      size += h.Size();
    }
    return size;
  }

  DimsHistoryElement clone_ref_with(int64_t val = 0) const {
    DimsHistoryElement d{ref_tshapes_};
    for (auto tensor_it : d) {
      const auto& tensor_idx{tensor_it.first};
      for (auto dim_it : tensor_it.second) {
        const auto& dim_idx{dim_it.first};
        d[tensor_idx][dim_idx] = val;
      }
    }

    return d;
  }
  void clear() {
    ref_tshapes_.clear();
    hist_items_.clear();
  }

  const DimsHistoryElement& ref_tshapes() const {
    return ref_tshapes_;
  }
  DimsHistoryElement& ref_tshapes() {
    return ref_tshapes_;
  }

  const std::vector<HistoryItem>& hist_items() const {
    return hist_items_;
  }
  std::vector<HistoryItem>& hist_items() {
    return hist_items_;
  }

  size_t size() {
    return hist_items_.size();
  }
  const HistoryItem& operator[](size_t i) const {
    return hist_items_[i];
  }
  HistoryItem& operator[](size_t i) {
    return hist_items_[i];
  }
  std::tuple<bool, size_t, bool> FindMidPoint(
      const std::vector<size_t>& bucket_input_hist_idxes);
  size_t WithinRangeCount(
      const DimsHistoryElement& lo,
      const DimsHistoryElement& hi,
      const std::vector<size_t>& bucket_input_hist_idxes);
  void Serialize(std::ostream& os) const;
  void Deserialize(std::istream& is);
};

inline std::string DebugString(const HistoryItemLog h) {
  std::ostringstream O;
  for (size_t i{}; i < h.hist_items_.size(); i++) {
    O << "History [" << i << "]:\n"
      << DebugString(h.hist_items_.at(i), h.ref_tshapes_) << '\n';
  }
  return O.str();
}

} // namespace habana_helpers
