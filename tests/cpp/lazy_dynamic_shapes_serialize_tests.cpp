/******************************************************************************
 * Copyright (C) 2022 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */
#include "habana_lazy_test_infra.h"

// In this class both the pass fallback and compilation fallback are disabled
class LazyDynamicShapesSerializtionTest : public habana_lazy_test::LazyTest {
  void SetUp() override {
    SetLazyMode();

    SetSeed();

    DisableCpuFallback();

    SetDynamicMode();

    DisableDynamicPassFallback();

    habana_lazy::exec::OptPassCfg::GetInstance()->SetDefaultOptFlags();

    habana::RecipeCacheLRU::get_cache().clear();
  }

  void TearDown() override {
    habana_lazy::exec::OptPassCfg::GetInstance()->SetDefaultOptFlags();

    UnsetDynamicMode();

    RestoreDynamicPassFallback();

    RestoreMode();
  }
};

void AddNonzeroOpsTest(std::vector<int64_t> input_shape) {
  torch::Tensor input1 = torch::randn(input_shape, torch::requires_grad(false));
  torch::Tensor input2 = torch::randn(input_shape, torch::requires_grad(false));
  torch::Tensor out_add = torch::add(input1, input2);
  torch::Tensor output = torch::nonzero(out_add);

  torch::Tensor hinput1 = input1.to(torch::kHPU);
  torch::Tensor hinput2 = input2.to(torch::kHPU);
  torch::Tensor hout_add = torch::add(hinput1, hinput2);
  torch::Tensor houtput = torch::nonzero(hout_add);
  torch::Tensor h_cpu = houtput.to(torch::kCPU);
  EXPECT_EQ(allclose(output, h_cpu, 0.01, 0.01), true);
}

TEST_F(LazyDynamicShapesSerializtionTest, SerializeDeserializeDBITest) {
  std::vector<int> channel_sizes{6, 8, 10, 4};
  SET_ENV_FLAG_NEW(PT_RECIPE_TRACE_PATH, "recipe_trace.csv", 1);

  for (int i = 0; i < channel_sizes.size(); i++) {
    AddNonzeroOpsTest({4, channel_sizes[i], 3});
  }

  habana::ClearDynamicBucketRecipeInfo();
  habana_helpers::UniqueTokenGenerator::get_gen().reset();
  SET_ENV_FLAG_NEW(PT_RECIPE_TRACE_PATH, "recipe_trace_rerun.csv", 1);

  for (int i = 0; i < channel_sizes.size() - 2; i++) {
    AddNonzeroOpsTest({4, channel_sizes[i], 3});
  }
  habana::DynamicBucketInfoMap::save_ds_checkpoint("ds_checkpoint.pt");
  habana::ClearDynamicBucketRecipeInfo();
  habana::DynamicBucketInfoMap::load_ds_checkpoint("ds_checkpoint.pt");

  for (int i = 2; i < channel_sizes.size(); i++) {
    AddNonzeroOpsTest({4, channel_sizes[i], 3});
  }
  UNSET_ENV_FLAG_NEW(PT_RECIPE_TRACE_PATH);
}