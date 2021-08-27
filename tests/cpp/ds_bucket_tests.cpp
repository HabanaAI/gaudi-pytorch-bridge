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

TEST(DS_DynamicBucketInfoTest, Simple) {
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

  size_t bidx{};
  habana_helpers::DynamicBucketInfo bucket_info;
  bucket_info.CollectDynamicDims(s[0]);
  bidx = bucket_info.GetBucketId(s[0]);
  ASSERT_EQ(bidx, 0);

  std::cout << "PTI_DBG :: "
            << "Collect info with input tensor shapes:" << '\n'
            << s[0];
  std::cout << "PTI_DBG :: "
            << "Returned bucket id : " << bidx << '\n';
  std::cout << '\n' << bucket_info;

  bucket_info.CollectDynamicDims(s[1]);
  bidx = bucket_info.GetBucketId(s[1]);
  ASSERT_EQ(bidx, 1);

  std::cout << "PTI_DBG :: "
            << "Collect info with input tensor shapes:" << '\n'
            << s[1];
  std::cout << "PTI_DBG :: "
            << "Returned bucket id : " << bidx << '\n';
  std::cout << '\n' << bucket_info;

  bucket_info.CollectDynamicDims(s[0]);
  bidx = bucket_info.GetBucketId(s[0]);
  ASSERT_EQ(bidx, 0);

  std::cout << "PTI_DBG :: "
            << "Collect info with input tensor shapes:" << '\n'
            << s[0];
  std::cout << "PTI_DBG :: "
            << "Returned bucket id : " << bidx << '\n';
  std::cout << '\n' << bucket_info;

  bucket_info.CollectDynamicDims(s[2]);
  bidx = bucket_info.GetBucketId(s[2]);
  ASSERT_EQ(bidx, 1);

  std::cout << "PTI_DBG :: "
            << "Collect info with input tensor shapes:" << '\n'
            << s[2];
  std::cout << "PTI_DBG :: "
            << "Returned bucket id : " << bidx << '\n';
  std::cout << '\n' << bucket_info;

  bucket_info.CollectDynamicDims(s[3]);
  bidx = bucket_info.GetBucketId(s[3]);
  ASSERT_EQ(bidx, 2);

  std::cout << "PTI_DBG :: "
            << "Collect info with input tensor shapes:" << '\n'
            << s[3];
  std::cout << "PTI_DBG :: "
            << "Returned bucket id : " << bidx << '\n';
  std::cout << '\n' << bucket_info;

  uint64_t iter_cnt{0};
  uint64_t max_iter_cnt{
      habana_helpers::DynamicBucketInfo::min_iterations_to_split() * 2};
  while (iter_cnt++ < max_iter_cnt) {
    bucket_info.CollectDynamicDims(s[4]);
    bidx = bucket_info.GetBucketId(s[4]);
  }
  ASSERT_EQ(bidx, 2);

  std::cout << "PTI_DBG :: "
            << "Collected info with the following for " << max_iter_cnt
            << " times with input tensor shapes ::" << '\n'
            << s[4];
  std::cout << "PTI_DBG :: "
            << "Last returned bucket id : " << bidx << '\n';
  std::cout << '\n' << bucket_info;

  auto new_bucket = bucket_info.CheckForSplitBucket();
  ASSERT_TRUE(new_bucket.has_value());
  ASSERT_EQ(new_bucket.value(), 3);

  std::cout << "PTI_DBG :: "
            << "New bucket id : " << new_bucket.value();
  std::cout << '\n' << bucket_info;

  bucket_info.CollectDynamicDims(s[5]);
  bidx = bucket_info.GetBucketId(s[5]);
  ASSERT_EQ(bidx, 3);

  std::cout << "PTI_DBG :: "
            << "Collect info with input tensor shapes:" << '\n'
            << s[5];
  std::cout << "PTI_DBG :: "
            << "Returned bucket id : " << bidx << '\n';
  std::cout << '\n' << bucket_info;

  bucket_info.CollectDynamicDims(s[6]);
  bidx = bucket_info.GetBucketId(s[6]);
  ASSERT_EQ(bidx, 2);

  std::cout << "PTI_DBG :: "
            << "Collect info with input tensor shapes:" << '\n'
            << s[6];
  std::cout << "PTI_DBG :: "
            << "Returned bucket id : " << bidx << '\n';
  std::cout << '\n' << bucket_info;

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
      habana_helpers::SplitPolicy::DEFAULT);
  binfo_default.CollectDynamicDims(s[0]);
  bidx_default = binfo_default.GetBucketId(s[0]);
  ASSERT_EQ(bidx_default, 0);

  habana_helpers::DynamicBucketInfo binfo_dynamic(
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
