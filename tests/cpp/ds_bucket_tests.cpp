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

habana_helpers::DynamicBucketInfo::InpTensorShapes get_shape(int d1, int d2) {
  c10::ScalarType t(c10::ScalarType::Long);
  return habana_helpers::DynamicBucketInfo::InpTensorShapes{
      {0, {{d1, 10, 8, 9}, t}}, {1, {{10, 20, 30, d2}, t}}};
};

TEST(DS_DynamicBucketInfoTest, Simple) {
  bool refine_enabled = GET_ENV_FLAG(PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES);
  if (!refine_enabled) {
    setenv("PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES", "1", 1);
  }
  size_t bidx{};
  habana_helpers::DynamicBucketInfo::InpTensorShapes s0(get_shape(2, 2));
  habana_helpers::DynamicBucketInfo bucket_info;
  bucket_info.CollectDynamicDims(s0);
  bidx = bucket_info.GetBucketId(s0);
  ASSERT_EQ(bidx, 0);

  // std::cout << "Collect info with input tensor shapes:" << '\n' << s0;
  // std::cout << "Returned bucket id : " << bidx << '\n';
  // std::cout << '\n' << bucket_info;
  habana_helpers::DynamicBucketInfo::InpTensorShapes s1(get_shape(5, 30));
  bucket_info.CollectDynamicDims(s1);
  bidx = bucket_info.GetBucketId(s1);
  ASSERT_EQ(bidx, 1);

  // std::cout << "Collect info with input tensor shapes:" << '\n' << s1;
  // std::cout << "Returned bucket id : " << bidx << '\n';
  // std::cout << '\n' << bucket_info;

  bucket_info.CollectDynamicDims(s0);
  bidx = bucket_info.GetBucketId(s0);
  ASSERT_EQ(bidx, 0);

  // std::cout << "Collect info with input tensor shapes:" << '\n' << s0;
  // std::cout << "Returned bucket id : " << bidx << '\n';
  // std::cout << '\n' << bucket_info;

  habana_helpers::DynamicBucketInfo::InpTensorShapes s2(get_shape(2, 10));
  bucket_info.CollectDynamicDims(s2);
  bidx = bucket_info.GetBucketId(s2);
  ASSERT_EQ(bidx, 1);

  // std::cout << "Collect info with input tensor shapes:" << '\n' << s2;
  // std::cout << "Returned bucket id : " << bidx << '\n';
  // std::cout << '\n' << bucket_info;

  habana_helpers::DynamicBucketInfo::InpTensorShapes s3(get_shape(50, 100));
  bucket_info.CollectDynamicDims(s3);
  bidx = bucket_info.GetBucketId(s3);
  ASSERT_EQ(bidx, 2);

  // std::cout << "Collect info with input tensor shapes:" << '\n' << s3;
  // std::cout << "Returned bucket id : " << bidx << '\n';
  // std::cout << '\n' << bucket_info;

  habana_helpers::DynamicBucketInfo::InpTensorShapes s4(get_shape(60, 150));
  uint64_t iter_cnt{0};
  uint64_t max_iter_cnt{
      habana_helpers::DynamicBucketInfo::min_iterations_to_split() * 2};
  while (iter_cnt++ < max_iter_cnt) {
    bucket_info.CollectDynamicDims(s4);
    bidx = bucket_info.GetBucketId(s4);
  }
  ASSERT_EQ(bidx, 2);

  // std::cout << "Collected info with the following for " << max_iter_cnt
  // std::cout << " times with input tensor shapes ::" << '\n' << s4;
  // std::cout << "Last returned bucket id : " << bidx << '\n';
  // std::cout << '\n' << bucket_info;

  auto new_bucket = bucket_info.CheckForSplitBucket();
  ASSERT_TRUE(new_bucket.has_value());
  ASSERT_EQ(new_bucket.value(), 3);

  // std::cout << "New bucket id : " << new_bucket.value();
  // std::cout << '\n' << bucket_info;

  habana_helpers::DynamicBucketInfo::InpTensorShapes s5(get_shape(70, 130));
  bucket_info.CollectDynamicDims(s5);
  bidx = bucket_info.GetBucketId(s5);
  ASSERT_EQ(bidx, 3);

  // std::cout << "Collect info with input tensor shapes:" << '\n' << s5;
  // std::cout << "Returned bucket id : " << bidx << '\n';
  // std::cout << '\n' << bucket_info;

  habana_helpers::DynamicBucketInfo::InpTensorShapes s6(get_shape(45, 90));
  bucket_info.CollectDynamicDims(s6);
  bidx = bucket_info.GetBucketId(s6);
  ASSERT_EQ(bidx, 2);

  // std::cout << "Collect info with input tensor shapes:" << '\n' << s6;
  // std::cout << "Returned bucket id : " << bidx << '\n';
  // std::cout << '\n' << bucket_info;

  if (!refine_enabled) {
    unsetenv("PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES");
  }
}

TEST(DS_DynamicBucketInfoTest, SplitStatImpl) {
  bool refine_enabled = GET_ENV_FLAG(PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES);
  if (!refine_enabled) {
    setenv("PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES", "1", 1);
  }
  size_t bidx_default{}, bidx_dynamic;
  habana_helpers::DynamicBucketInfo::InpTensorShapes s0(get_shape(2, 2));

  habana_helpers::DynamicBucketInfo binfo_default(
      habana_helpers::SplitPolicy::DEFAULT);
  binfo_default.CollectDynamicDims(s0);
  bidx_default = binfo_default.GetBucketId(s0);
  ASSERT_EQ(bidx_default, 0);

  habana_helpers::DynamicBucketInfo binfo_dynamic(
      habana_helpers::SplitPolicy::DYNAMIC);
  binfo_dynamic.CollectDynamicDims(s0);
  bidx_dynamic = binfo_dynamic.GetBucketId(s0);
  ASSERT_EQ(bidx_dynamic, bidx_default);

  // std::cout << "Collect info with input tensor shapes:" << '\n' << s0;
  // std::cout << "Returned default bucket id : " << bidx_default << '\n';
  // std::cout << '\n' << binfo_default;
  // std::cout << "Returned dynamic bucket id : " << bidx_dynamic << '\n';
  // std::cout << '\n' << binfo_dynamic;

  habana_helpers::DynamicBucketInfo::InpTensorShapes s1(get_shape(5, 30));

  binfo_default.CollectDynamicDims(s1);
  bidx_default = binfo_default.GetBucketId(s1);
  ASSERT_EQ(bidx_default, 1);

  binfo_dynamic.CollectDynamicDims(s1);
  bidx_dynamic = binfo_dynamic.GetBucketId(s1);
  ASSERT_EQ(bidx_dynamic, bidx_default);

  // std::cout << "Collect info with input tensor shapes:" << '\n' << s1;
  // std::cout << "Returned default bucket id : " << bidx_default << '\n';
  // std::cout << '\n' << binfo_default;
  // std::cout << "Returned dynamic bucket id : " << bidx_dynamic << '\n';
  // std::cout << '\n' << binfo_dynamic;

  habana_helpers::DynamicBucketInfo::InpTensorShapes s2(get_shape(2, 10));

  binfo_default.CollectDynamicDims(s2);
  bidx_default = binfo_default.GetBucketId(s2);
  ASSERT_EQ(bidx_default, 1);

  binfo_dynamic.CollectDynamicDims(s2);
  bidx_dynamic = binfo_dynamic.GetBucketId(s2);
  ASSERT_EQ(bidx_dynamic, bidx_default);

  // std::cout << "Collect info with input tensor shapes:" << '\n' << s2;
  // std::cout << "Returned default bucket id : " << bidx_default << '\n';
  // std::cout << '\n' << binfo_default;
  // std::cout << "Returned dynamic bucket id : " << bidx_dynamic << '\n';
  // std::cout << '\n' << binfo_dynamic;

  habana_helpers::DynamicBucketInfo::InpTensorShapes s3(get_shape(50, 100));

  binfo_default.CollectDynamicDims(s3);
  bidx_default = binfo_default.GetBucketId(s3);
  ASSERT_EQ(bidx_default, 2);

  binfo_dynamic.CollectDynamicDims(s3);
  bidx_dynamic = binfo_dynamic.GetBucketId(s3);
  ASSERT_EQ(bidx_dynamic, bidx_default);

  // std::cout << "Collect info with input tensor shapes:" << '\n' << s3;
  // std::cout << "Returned default bucket id : " << bidx_default << '\n';
  // std::cout << '\n' << binfo_default;
  // std::cout << "Returned dynamic bucket id : " << bidx_dynamic << '\n';
  // std::cout << '\n' << binfo_dynamic;

  habana_helpers::DynamicBucketInfo::InpTensorShapes s4(get_shape(60, 150));
  uint64_t iter_cnt{0};
  uint64_t max_iter_cnt{
      habana_helpers::DynamicBucketInfo::min_iterations_to_split() * 2};
  while (iter_cnt++ < max_iter_cnt) {
    binfo_default.CollectDynamicDims(s4);
    bidx_default = binfo_default.GetBucketId(s4);
    binfo_dynamic.CollectDynamicDims(s4);
    bidx_dynamic = binfo_dynamic.GetBucketId(s4);
    ASSERT_EQ(bidx_default, 2);
    ASSERT_EQ(bidx_dynamic, bidx_default);
  }

  // std::cout << "Collected info with the following for " << max_iter_cnt
  // std::cout << " times with input tensor shapes ::" << '\n' << s4;
  // std::cout << "Returned default bucket id : " << bidx_default << '\n';
  // std::cout << '\n' << binfo_default;
  // std::cout << "Returned dynamic bucket id : " << bidx_dynamic << '\n';
  // std::cout << '\n' << binfo_dynamic;

  auto new_bucket_default = binfo_default.CheckForSplitBucket();
  ASSERT_TRUE(new_bucket_default.has_value());
  ASSERT_EQ(new_bucket_default.value(), 3);

  auto new_bucket_dynamic = binfo_dynamic.CheckForSplitBucket();
  ASSERT_TRUE(new_bucket_dynamic.has_value());
  ASSERT_EQ(new_bucket_dynamic.value(), new_bucket_default.value());

  // std::cout << "New default bucket id : " << new_bucket_default.value();
  // std::cout << '\n' << binfo_default;
  // std::cout << "New dynamic bucket id : " << new_bucket_dynamic.value();
  // std::cout << '\n' << binfo_dynamic;

  habana_helpers::DynamicBucketInfo::InpTensorShapes s5(get_shape(70, 130));

  binfo_default.CollectDynamicDims(s5);
  bidx_default = binfo_default.GetBucketId(s5);
  ASSERT_EQ(bidx_default, 3);

  binfo_dynamic.CollectDynamicDims(s5);
  bidx_dynamic = binfo_dynamic.GetBucketId(s5);
  ASSERT_EQ(bidx_dynamic, bidx_default);

  std::cout << "Collect info with input tensor shapes:" << '\n' << s5;
  std::cout << "Returned default bucket id : " << bidx_default;
  std::cout << '\n' << binfo_default;
  std::cout << "Returned dynamic bucket id : " << bidx_dynamic;
  std::cout << '\n' << binfo_dynamic;

  if (!refine_enabled) {
    unsetenv("PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES");
  }
}
