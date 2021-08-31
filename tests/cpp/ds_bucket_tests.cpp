/******************************************************************************
 * Copyright (C) 2020 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */

#include <iostream>
#include <stdexcept>
#include <unordered_set>

#include <gtest/gtest.h>
#include <torch/csrc/jit/ir/irparser.h>
#include <torch/torch.h>

#include "habana_lazy_test_infra.h"
#include "pytorch_helpers/habana_helpers/dynamic_bucket_info.h"
#include "pytorch_helpers/synapse_helpers/env_flags.h"

TEST(DS_TensorShapeTest, Simple) {
  const int H = 3;
  const int W = 3;
  const int C = 16;
  const int K = 16;
  torch::Tensor c0 = torch::randn({K, C, W, H}, torch::requires_grad(false));

  at::IntArrayRef shape_0(c0.sizes()), strides_0(c0.strides());
  c10::ScalarType type(c10::ScalarType::Long);

  habana_helpers::TensorShape tshape(shape_0, type), tstrides(strides_0, type);
  // std::cout << "PTI_DBG ::" << " tshape : " << tshape << " tstrides : " <<
  // tstrides << '\n'; std::cout << "PTI_DBG ::" << " exp shape : " << shape_0
  // << " exp strides : " << strides_0 << '\n';

  auto shape_1(tshape.get_dims()), strides_1(tstrides.get_dims());
  // std::cout << "PTI_DBG ::" << " act shape : " << at::IntArrayRef(shape_1)
  //<< " act strides : " << at::IntArrayRef(strides_1) << '\n';

  EXPECT_EQ(tshape.get_dims(), shape_0);
  EXPECT_EQ(tstrides.get_dims(), strides_0);
}

habana_helpers::DynamicBucketInfo::InpTensorShapes get_shape(
    int64_t d1,
    int64_t d2) {
  c10::ScalarType t(c10::ScalarType::Long);
  return habana_helpers::DynamicBucketInfo::InpTensorShapes{
      {0, {{d1, 10, 8, 9}, t}}, {1, {{10, 20, 30, d2}, t}}};
};

class DynamicBucketInfoTest
    : public ::testing::TestWithParam<habana_helpers::DynamicDimsPolicy> {};

struct PrintToStringParamName {
  template <class ParamType>
  std::string operator()(
      const ::testing::TestParamInfo<ParamType>& info) const {
    auto p = static_cast<habana_helpers::DynamicDimsPolicy>(info.param);
    return habana_helpers::DebugString(p);
  }
};

INSTANTIATE_TEST_SUITE_P(
    DSBucket,
    DynamicBucketInfoTest,
    ::testing::Values(
        habana_helpers::DynamicDimsPolicy::CALCULATED,
        habana_helpers::DynamicDimsPolicy::HISTORIC),
    PrintToStringParamName());

TEST_P(DynamicBucketInfoTest, MinShape) {
  bool refine_enabled = GET_ENV_FLAG(PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES);
  if (!refine_enabled) {
    setenv("PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES", "1", 1);
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
  std::vector<habana_helpers::DynamicBucketInfo::InpTensorShapes> s;
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

  std::cout << "PTI_DBG :: "
            << "Running with min policy " << GetParam() << '\n';
  habana_helpers::DynamicBucketInfo bucket_info(
      habana_helpers::DynamicDimsPolicy::CALCULATED,
      habana_helpers::SplitPolicy::DEFAULT);

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
            habana_helpers::DynamicBucketInfo::InpTensorShapes min_intshapes;
            habana_helpers::DynamicBucketInfo::InpTensorShapes max_intshapes;
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

  uint64_t iter_cnt{0};
  uint64_t max_iter_cnt{
      habana_helpers::DynamicBucketInfo::min_iterations_to_split() * 2};
  while (iter_cnt++ < max_iter_cnt) {
    get_and_check_bucket(4, 2, false);
  }

  std::cout << "PTI_DBG :: "
            << "Collected info with the following for " << max_iter_cnt
            << " times with input tensor shapes ::" << '\n'
            << s[4];
  get_and_check_bucket(4, 2, true);

  auto new_bucket = bucket_info.CheckForSplitBucket();

  ASSERT_TRUE(new_bucket.has_value());
  ASSERT_EQ(new_bucket.value(), 3);

  std::cout << "PTI_DBG :: "
            << "New bucket id : " << new_bucket.value();
  std::cout << '\n' << "====================" << '\n';
  std::cout << "PTI_DBG :: " << bucket_info;
  std::cout << "--------------------" << '\n';

  get_and_check_bucket(5, 3);
  get_and_check_bucket(6, 2);

  if (!refine_enabled) {
    unsetenv("PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES");
  }
}

TEST(DS_DynamicBucketInfoTest, SplitStatImpl) {
  bool refine_enabled = GET_ENV_FLAG(PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES);
  if (!refine_enabled) {
    setenv("PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES", "1", 1);
  }

  int64_t min_dim{habana_helpers::DynamicBucketInfo::default_min_value()};

  std::cout << "PTI_DBG :: "
            << "Using min_dim = " << min_dim << '\n';
  std::vector<std::vector<int64_t>> dyn_dims = {
      {min_dim, min_dim},
      {min_dim + 3, 30},
      {min_dim, 10},
      {50, 100},
      {60, 150},
      {70, 130},
      {45, 90}};
  std::vector<habana_helpers::DynamicBucketInfo::InpTensorShapes> s;
  s.reserve(dyn_dims.size());
  for (auto dim : dyn_dims) {
    s.push_back(get_shape(dim[0], dim[1]));
  }

  for (auto a : s) {
    std::cout << "PTI_DBG :: "
              << "Will use input tensor shapes:" << '\n'
              << a;
  }

  size_t bidx_default{}, bidx_dynamic;

  habana_helpers::DynamicBucketInfo binfo_default(
      habana_helpers::DynamicDimsPolicy::CALCULATED,
      habana_helpers::SplitPolicy::DEFAULT);
  binfo_default.CollectDynamicDims(s[0]);
  bidx_default = binfo_default.GetBucketId(s[0]);
  ASSERT_EQ(bidx_default, 0);

  habana_helpers::DynamicBucketInfo binfo_dynamic(
      habana_helpers::DynamicDimsPolicy::CALCULATED,
      habana_helpers::SplitPolicy::DYNAMIC);
  binfo_dynamic.CollectDynamicDims(s[0]);
  bidx_dynamic = binfo_dynamic.GetBucketId(s[0]);
  ASSERT_EQ(bidx_dynamic, bidx_default);

  std::cout << "PTI_DBG :: "
            << "Collect info with input tensor shapes:" << '\n'
            << s[0];
  std::cout << "PTI_DBG :: "
            << "Returned default bucket id : " << bidx_default << '\n';
  std::cout << '\n' << binfo_default;
  std::cout << "PTI_DBG :: "
            << "Returned dynamic bucket id : " << bidx_dynamic << '\n';
  std::cout << '\n' << binfo_dynamic;

  binfo_default.CollectDynamicDims(s[1]);
  bidx_default = binfo_default.GetBucketId(s[1]);
  ASSERT_EQ(bidx_default, 1);

  binfo_dynamic.CollectDynamicDims(s[1]);
  bidx_dynamic = binfo_dynamic.GetBucketId(s[1]);
  ASSERT_EQ(bidx_dynamic, bidx_default);

  std::cout << "PTI_DBG :: "
            << "Collect info with input tensor shapes:" << '\n'
            << s[1];
  std::cout << "PTI_DBG :: "
            << "Returned default bucket id : " << bidx_default << '\n';
  std::cout << '\n' << binfo_default;
  std::cout << "PTI_DBG :: "
            << "Returned dynamic bucket id : " << bidx_dynamic << '\n';
  std::cout << '\n' << binfo_dynamic;

  binfo_default.CollectDynamicDims(s[2]);
  bidx_default = binfo_default.GetBucketId(s[2]);
  ASSERT_EQ(bidx_default, 1);

  binfo_dynamic.CollectDynamicDims(s[2]);
  bidx_dynamic = binfo_dynamic.GetBucketId(s[2]);
  ASSERT_EQ(bidx_dynamic, bidx_default);

  std::cout << "PTI_DBG :: "
            << "Collect info with input tensor shapes:" << '\n'
            << s[2];
  std::cout << "PTI_DBG :: "
            << "Returned default bucket id : " << bidx_default << '\n';
  std::cout << '\n' << binfo_default;
  std::cout << "PTI_DBG :: "
            << "Returned dynamic bucket id : " << bidx_dynamic << '\n';
  std::cout << '\n' << binfo_dynamic;

  binfo_default.CollectDynamicDims(s[3]);
  bidx_default = binfo_default.GetBucketId(s[3]);
  ASSERT_EQ(bidx_default, 2);

  binfo_dynamic.CollectDynamicDims(s[3]);
  bidx_dynamic = binfo_dynamic.GetBucketId(s[3]);
  ASSERT_EQ(bidx_dynamic, bidx_default);

  std::cout << "PTI_DBG :: "
            << "Collect info with input tensor shapes:" << '\n'
            << s[3];
  std::cout << "PTI_DBG :: "
            << "Returned default bucket id : " << bidx_default << '\n';
  std::cout << '\n' << binfo_default;
  std::cout << "PTI_DBG :: "
            << "Returned dynamic bucket id : " << bidx_dynamic << '\n';
  std::cout << '\n' << binfo_dynamic;

  uint64_t iter_cnt{0};
  uint64_t max_iter_cnt{
      habana_helpers::DynamicBucketInfo::min_iterations_to_split() * 2};
  while (iter_cnt++ < max_iter_cnt) {
    binfo_default.CollectDynamicDims(s[4]);
    bidx_default = binfo_default.GetBucketId(s[4]);
    binfo_dynamic.CollectDynamicDims(s[4]);
    bidx_dynamic = binfo_dynamic.GetBucketId(s[4]);
    ASSERT_EQ(bidx_default, 2);
    ASSERT_EQ(bidx_dynamic, bidx_default);
  }

  std::cout << "PTI_DBG :: "
            << "Collected info with the following for " << max_iter_cnt
            << " times with input tensor shapes ::" << '\n'
            << s[4];
  std::cout << "PTI_DBG :: "
            << "Returned default bucket id : " << bidx_default << '\n';
  std::cout << '\n' << binfo_default;
  std::cout << "PTI_DBG :: "
            << "Returned dynamic bucket id : " << bidx_dynamic << '\n';
  std::cout << '\n' << binfo_dynamic;

  auto new_bucket_default = binfo_default.CheckForSplitBucket();
  ASSERT_TRUE(new_bucket_default.has_value());
  ASSERT_EQ(new_bucket_default.value(), 3);

  auto new_bucket_dynamic = binfo_dynamic.CheckForSplitBucket();
  ASSERT_TRUE(new_bucket_dynamic.has_value());
  ASSERT_EQ(new_bucket_dynamic.value(), new_bucket_default.value());

  std::cout << "PTI_DBG :: "
            << "New default bucket id : " << new_bucket_default.value();
  std::cout << '\n' << binfo_default;
  std::cout << "PTI_DBG :: "
            << "New dynamic bucket id : " << new_bucket_dynamic.value();
  std::cout << '\n' << binfo_dynamic;

  binfo_default.CollectDynamicDims(s[5]);
  bidx_default = binfo_default.GetBucketId(s[5]);
  ASSERT_EQ(bidx_default, 3);

  binfo_dynamic.CollectDynamicDims(s[5]);
  bidx_dynamic = binfo_dynamic.GetBucketId(s[5]);
  ASSERT_EQ(bidx_dynamic, bidx_default);

  std::cout << "PTI_DBG :: "
            << "Collect info with input tensor shapes:" << '\n'
            << s[5];
  std::cout << "PTI_DBG :: "
            << "Returned default bucket id : " << bidx_default;
  std::cout << '\n' << binfo_default;
  std::cout << "PTI_DBG :: "
            << "Returned dynamic bucket id : " << bidx_dynamic;
  std::cout << '\n' << binfo_dynamic;

  if (!refine_enabled) {
    unsetenv("PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES");
  }
}
