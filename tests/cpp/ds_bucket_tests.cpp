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

#include <cstdlib>

#include <algorithm>
#include <iostream>
#include <random>
#include <stdexcept>
#include <tuple>
#include <unordered_map>

#include <gtest/gtest.h>
#include <torch/torch.h>

#include "backend/helpers/dynamic_bucket_info.h"
#include "backend/synapse_helpers/env_flags.h"
#include "habana_lazy_test_infra.h"

TEST(DS_TensorShapeTest, Simple) {
  const int H = 3;
  const int W = 3;
  const int C = 16;
  const int K = 16;
  torch::Tensor c0 = torch::randn({K, C, W, H}, torch::requires_grad(false));

  at::IntArrayRef shape_0(c0.sizes()), strides_0(c0.strides());
  c10::ScalarType type(c10::ScalarType::Long);

  habana_helpers::TensorShape tshape(shape_0, type), tstrides(strides_0, type);
  PT_TEST_DEBUG("tshape : ", tshape, " tstrides : ", tstrides);
  PT_TEST_DEBUG("expected shape : ", shape_0, " exp strides : ", strides_0);

  auto shape_1(tshape.get_dims()), strides_1(tstrides.get_dims());
  PT_TEST_DEBUG(
      "actual shape : ",
      at::IntArrayRef(shape_1),
      " exp strides : ",
      at::IntArrayRef(strides_1));

  EXPECT_EQ(tshape.get_dims(), shape_0);
  EXPECT_EQ(tstrides.get_dims(), strides_0);
}

class InpShapeGen {
 public:
  static inline void print_dbi(
      const habana_helpers::DynamicBucketInfo& bucket_info,
      size_t input_idx,
      habana_helpers::InpTensorShapes& input_shapes,
      size_t bidx) {
    PT_TEST_DEBUG(
        "====================\n",
        Logger::_str_wrapper(bucket_info),
        "Collect info with input shapes",
        "[",
        input_idx,
        "]:",
        "\n",
        input_shapes,
        "Returned bucket id : ",
        bidx);
  }
  static inline void print_dyn_dimvals(size_t d1, size_t d0) {
    PT_TEST_DEBUG(
        "Using min_dim = ", min_dim, ", dyn_dims shape [ ", d1, " ", d0, " ]");
  }
  static inline void print_input_shapes(
      const std::vector<habana_helpers::InpTensorShapes>& input_shapes_vec) {
    PT_TEST_DEBUG("Will use the following input tensor shapes:");
    size_t in_idx{0};
    for (auto a : input_shapes_vec) {
      PT_TEST_DEBUG("input shape[", in_idx++, "]", a);
    }
  }

  std::vector<habana_helpers::InpTensorShapes> get_input_shapes_vec(
      std::vector<std::vector<std::vector<int64_t>>>& dyn_dimvals_arg) {
    print_dyn_dimvals(dyn_dimvals_arg.size(), dyn_dimvals_arg[0].size());

    c10::ScalarType typ(c10::ScalarType::Long);
    std::vector<habana_helpers::InpTensorShapes> input_shapes_vec;
    for (auto inputs : dyn_dimvals_arg) {
      int64_t input_idx{0};
      habana_helpers::InpTensorShapes input_shapes;
      for (auto dimvals : inputs) {
        habana_helpers::TensorShape ts(dimvals, typ);
        input_shapes.emplace(input_idx, ts);
        input_idx++;
      }
      input_shapes_vec.push_back(input_shapes);
    }

    print_input_shapes(input_shapes_vec);

    return std::move(input_shapes_vec);
  }

  static int64_t min_dim;

  static std::vector<std::vector<std::vector<int64_t>>> dyn_dimvals_sw63966;
  static std::vector<std::vector<std::vector<int64_t>>> dyn_dimvals_sw60162;
  static std::vector<std::vector<std::vector<int64_t>>> dyn_dimvals_sw61032;
  static std::vector<std::vector<std::vector<int64_t>>> dyn_dimvals_sw57731;
  static std::vector<std::vector<std::vector<int64_t>>> dyn_dimvals;

  static std::unordered_map<size_t, uint64_t> exp_result_map_sw63966;
  static std::unordered_map<size_t, uint64_t> exp_result_map_sw60162;
  static std::unordered_map<size_t, uint64_t> exp_result_map_sw61032;
  static std::unordered_map<size_t, uint64_t> exp_result_map_sw57731;
  static std::unordered_map<size_t, uint64_t> exp_result_map_cal;
  static std::unordered_map<size_t, uint64_t> exp_result_map_cur;
  static std::unordered_map<size_t, uint64_t> exp_result_map_his;
};

int64_t InpShapeGen::min_dim =
    habana_helpers::DynamicBucketInfo::default_min_value();

std::vector<std::vector<std::vector<int64_t>>>
    InpShapeGen::dyn_dimvals_sw63966 = {
        {{40, 50, 60}, {20, 30}},
        {{40, 50, 70}, {20, 25}},
        {{35, 50, 80}, {20, 20}},
        {{35, 50, 90}, {20, 15}},
};
std::unordered_map<size_t, uint64_t> InpShapeGen::exp_result_map_sw63966 = {
    {0, 0},
    {1, 1},
    {2, 2},
    {3, 3},
};

std::vector<std::vector<std::vector<int64_t>>>
    InpShapeGen::dyn_dimvals_sw60162 = {
        {{280000}, {85, 5200}},
        {{280000}, {30, 5200}},
        {{180000}, {35, 3200}},
        {{180000}, {25, 3200}},
        {{220000}, {95, 4800}},
};
std::unordered_map<size_t, uint64_t> InpShapeGen::exp_result_map_sw60162 = {
    {0, 0},
    {1, 1},
    {2, 2},
    {3, 3},
    {4, 4},
};

std::vector<std::vector<std::vector<int64_t>>>
    InpShapeGen::dyn_dimvals_sw61032 = {
        {{1, 200}},
        {{482, 1}},
        {{1, 200}},
        {{482, 1}},
        {{1, 1}},
        {{482, 200}},
};
std::unordered_map<size_t, uint64_t> InpShapeGen::exp_result_map_sw61032 = {
    {0, 0},
    {1, 1},
    {2, 0},
    {3, 1},
    {4, 2},
    {5, 3},
};

std::vector<std::vector<std::vector<int64_t>>>
    InpShapeGen::dyn_dimvals_sw57731 = {
        {{32768, 1024}},
        {{1024, 1024}},
        {{1024, 1024}},
        {{4096, 1024}},
        {{1024, 4096}},
};
std::unordered_map<size_t, uint64_t> InpShapeGen::exp_result_map_sw57731 =
    {{0, 0}, {1, 1}, {2, 1}, {3, 2}, {4, 3}};

std::vector<std::vector<std::vector<int64_t>>> InpShapeGen::dyn_dimvals = {
    {{10}, {20, 30}, {40, 50, 60}},
    {{10}, {20, 15}, {40, 50, 60}},
    {{10}, {15, 25}, {35, 50, 60}},
    {{10}, {15, 12}, {35, 50, 60}},
    {{10}, {17, 33}, {37, 55, 60}},
    {{10}, {20, 22}, {40, 50, 60}},
};
std::unordered_map<size_t, uint64_t> InpShapeGen::exp_result_map_cal =
    {{0, 0}, {1, 1}, {2, 2}, {3, 3}, {4, 4}, {5, 1}};
std::unordered_map<size_t, uint64_t> InpShapeGen::exp_result_map_cur = {
    {0, 0},
    {1, 1},
    {2, 2},
    {3, 3},
    {4, 4},
    {5, 5},
};
std::unordered_map<size_t, uint64_t> InpShapeGen::exp_result_map_his =
    {{0, 0}, {1, 1}, {2, 2}, {3, 3}, {4, 4}, {5, 1}};

class DynamicDimsTest : public ::testing::TestWithParam<std::tuple<
                            std::vector<std::vector<std::vector<int64_t>>>,
                            std::unordered_map<size_t, uint64_t>,
                            habana_helpers::DynamicDimsPolicy>>,
                        public InpShapeGen {};

INSTANTIATE_TEST_SUITE_P(
    DS_Ranges,
    DynamicDimsTest,
    ::testing::Values(
        std::make_tuple(
            InpShapeGen::dyn_dimvals_sw63966,
            InpShapeGen::exp_result_map_sw63966,
            habana_helpers::DynamicDimsPolicy::HISTORIC),
        std::make_tuple(
            InpShapeGen::dyn_dimvals_sw60162,
            InpShapeGen::exp_result_map_sw60162,
            habana_helpers::DynamicDimsPolicy::CURRENT),
        std::make_tuple(
            InpShapeGen::dyn_dimvals_sw60162,
            InpShapeGen::exp_result_map_sw60162,
            habana_helpers::DynamicDimsPolicy::CALCULATED),
        std::make_tuple(
            InpShapeGen::dyn_dimvals_sw61032,
            InpShapeGen::exp_result_map_sw61032,
            habana_helpers::DynamicDimsPolicy::HISTORIC),
        std::make_tuple(
            InpShapeGen::dyn_dimvals_sw57731,
            InpShapeGen::exp_result_map_sw57731,
            habana_helpers::DynamicDimsPolicy::CURRENT),
        std::make_tuple(
            InpShapeGen::dyn_dimvals_sw57731,
            InpShapeGen::exp_result_map_sw57731,
            habana_helpers::DynamicDimsPolicy::CALCULATED),
        std::make_tuple(
            InpShapeGen::dyn_dimvals,
            InpShapeGen::exp_result_map_his,
            habana_helpers::DynamicDimsPolicy::HISTORIC),
        std::make_tuple(
            InpShapeGen::dyn_dimvals,
            InpShapeGen::exp_result_map_cur,
            habana_helpers::DynamicDimsPolicy::CURRENT),
        std::make_tuple(
            InpShapeGen::dyn_dimvals,
            InpShapeGen::exp_result_map_cal,
            habana_helpers::DynamicDimsPolicy::CALCULATED)));

TEST_P(DynamicDimsTest, BucketingPolicy) {
  bool refine_enabled = habana_helpers::GetRefineDynamicShapeStatus();
  if (!refine_enabled) {
    habana_helpers::EnableRefineDynamicShape();
  }

  std::vector<std::vector<std::vector<int64_t>>> input_dimvals;
  std::unordered_map<size_t, uint64_t> exp_result_map;

  auto min_policy = habana_helpers::DynamicDimsPolicy::HISTORIC;
  habana_helpers::DynamicDimsPolicy max_policy;

  std::tie(input_dimvals, exp_result_map, max_policy) = GetParam();
  auto input_shapes_vec(get_input_shapes_vec(input_dimvals));

  PT_TEST_DEBUG(
      "Running with min policy=", min_policy, ", and max_policy=", max_policy);

  habana_helpers::DynamicBucketInfo bucket_info(min_policy, max_policy);

  auto get_and_check_bucket{[&](size_t ddim_idx) {
    auto& input_shapes = input_shapes_vec.at(ddim_idx);
    PT_TEST_DEBUG(
        "====================\n",
        "CollectDynamicDims with input shapes",
        "[",
        ddim_idx,
        "]:",
        input_shapes,
        '\n',
        "--------------------",
        "\n");
    bucket_info.CollectDynamicDims(input_shapes);
    auto bidx = bucket_info.GetBucketId(input_shapes);

    auto ranges = bucket_info.CalculateShapes(bidx);
    PT_TEST_DEBUG(
        "--------------------\n",
        Logger::_str_wrapper(bucket_info),
        "--------------------\n",
        "For input shapes",
        "[",
        ddim_idx,
        "]:",
        input_shapes,
        '\n',
        "Returned bucket id : ",
        bidx,
        "\nDynamic ranges:",
        ranges.DebugString(input_shapes),
        '\n',
        "--------------------\n\n");
    ASSERT_EQ(bidx, exp_result_map.at(ddim_idx));
  }};

  for (size_t i{}; i < input_dimvals.size(); i++) {
    // for (size_t i{}; i < 2; i++) {
    get_and_check_bucket(i);
  }

  if (!refine_enabled) {
    habana_helpers::DisableRefineDynamicShape();
  }
}

habana_helpers::InpTensorShapes get_shape(int64_t d1, int64_t d2) {
  c10::ScalarType t(c10::ScalarType::Long);
  return habana_helpers::InpTensorShapes{
      {0, {{d1, 10, 8, 9}, t}}, {1, {{10, 20, 30, d2}, t}}};
};

class DynamicBucketInfoTest
    : public ::testing::TestWithParam<habana_helpers::DynamicDimsPolicy> {};

struct PrintToStringParamName {
  template <class ParamType>
  std::string operator()(
      const ::testing::TestParamInfo<ParamType>& info) const {
    auto p = static_cast<habana_helpers::DynamicDimsPolicy>(info.param);
    return std::string{habana_helpers::DebugString(p)};
  }
};

INSTANTIATE_TEST_SUITE_P(
    DS_Bucket,
    DynamicBucketInfoTest,
    ::testing::Values(
        habana_helpers::DynamicDimsPolicy::CALCULATED,
        habana_helpers::DynamicDimsPolicy::HISTORIC),
    PrintToStringParamName());

TEST_P(DynamicBucketInfoTest, MinShape) {
  bool refine_enabled = habana_helpers::GetRefineDynamicShapeStatus();
  if (!refine_enabled) {
    habana_helpers::EnableRefineDynamicShape();
  }

  int64_t min_dim{habana_helpers::DynamicBucketInfo::default_min_value()};

  std::cout << "PTI_DBG :: "
            << "Using min_dim = " << min_dim << '\n';
  std::vector<std::vector<int64_t>> dyn_dims = {
      {min_dim * 2, min_dim * 2},
      {min_dim * 3, min_dim * 3},
      {min_dim * 4, min_dim * 4},
      {min_dim * 8, min_dim * 16},
      {min_dim * 10, min_dim * 25},
      {min_dim * 12, min_dim * 22},
      {min_dim * 6, min_dim * 15},
  };
  std::vector<habana_helpers::InpTensorShapes> s;
  s.reserve(dyn_dims.size());
  for (auto dim : dyn_dims) {
    s.push_back(get_shape(dim[0], dim[1]));
  }

  std::cout << "PTI_DBG :: "
            << "Will use the following input tensor shapes:" << '\n';
  size_t in_idx{0};
  for (auto a : s) {
    std::cout << "PTI_DBG :: "
              << "input shape[" << in_idx++ << "]" << '\n'
              << a;
  }
  auto min_policy{GetParam()};
  std::cout << "PTI_DBG :: "
            << "Running with min policy " << min_policy << '\n';

  habana_helpers::DynamicBucketInfo bucket_info(
      min_policy, habana_helpers::DynamicDimsPolicy::CALCULATED);

  auto get_and_check_bucket{
      [&](size_t ddim_idx, uint64_t exp_bidx, bool dbg_print = true) {
        bucket_info.CollectDynamicDims(s[ddim_idx]);
        auto bidx = bucket_info.GetBucketId(s[ddim_idx]);

        if (dbg_print) {
          std::cout << '\n' << "====================" << '\n';
          std::cout << "PTI_DBG :: " << bucket_info;
          std::cout << "PTI_DBG :: "
                    << "Collect info with tensor shapes:" << '\n'
                    << s[ddim_idx];
          std::cout << "PTI_DBG :: "
                    << "Returned bucket id : " << bidx << '\n';
          auto ranges = bucket_info.CalculateShapes(bidx);
          if (!ranges.empty()) {
            habana_helpers::InpTensorShapes min_intshapes;
            habana_helpers::InpTensorShapes max_intshapes;
            min_intshapes.insert(
                ranges.min_shapes.begin(), ranges.min_shapes.end());
            max_intshapes.insert(
                ranges.max_shapes.begin(), ranges.max_shapes.end());
            std::cout << "PTI_DBG :: "
                      << "Min shape\n"
                      << min_intshapes;
            std::cout << "PTI_DBG :: "
                      << "Max shape\n"
                      << max_intshapes;
          } else {
            std::cout << "PTI_DBG :: "
                      << "Empty range returned\n";
          }
          std::cout << "--------------------" << '\n';
        }
        ASSERT_EQ(bidx, exp_bidx);
      }};

  get_and_check_bucket(0, 0);
  get_and_check_bucket(1, 1);
  get_and_check_bucket(0, 0);
  get_and_check_bucket(2, 1);
  get_and_check_bucket(3, 2);

  if (!refine_enabled) {
    habana_helpers::DisableRefineDynamicShape();
  }
}
