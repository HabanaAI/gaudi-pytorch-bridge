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
#include <iostream>
#include "habana_lazy/hlexec.h"
#include "habana_lazy_test_infra.h"

// In this class both the pass fallback and compilation fallback are disabled
class LazyDynamicShapesSerializtionTest
    : public habana_lazy_test::LazyDynamicTest {};

std::string read_csv_file(std::string path) {
  std::string text = "", line;
  std::ifstream file(path);
  while (std::getline(file, line)) {
    text += line + "\n";
  }
  return text;
}

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
  SET_ENV_FLAG_NEW(PT_HPU_ENABLE_DISK_CACHE_FOR_DSD, true, 1);

  for (int i = 0; i < channel_sizes.size(); i++) {
    AddNonzeroOpsTest({4, channel_sizes[i], 3});
  }

  habana::ClearDynamicBucketRecipeInfo();
  habana_helpers::UniqueTokenGenerator::get_gen().reset();
  SET_ENV_FLAG_NEW(PT_RECIPE_TRACE_PATH, "recipe_trace_rerun.csv", 1);

  for (int i = 0; i < channel_sizes.size() - 2; i++) {
    AddNonzeroOpsTest({4, channel_sizes[i], 3});
  }
  habana_lazy::exec::HlExec::SaveDSCheckpoint("ds_checkpoint.pt");
  habana::ClearDynamicBucketRecipeInfo();
  habana_lazy::exec::HlExec::LoadDSCheckpoint("ds_checkpoint.pt");

  for (int i = 2; i < channel_sizes.size(); i++) {
    AddNonzeroOpsTest({4, channel_sizes[i], 3});
  }
  UNSET_ENV_FLAG_NEW(PT_RECIPE_TRACE_PATH);
  UNSET_ENV_FLAG_NEW(PT_HPU_ENABLE_DISK_CACHE_FOR_DSD);

  std::string f1 = read_csv_file("recipe_trace.csv");
  std::string f2 = read_csv_file("recipe_trace_rerun.csv");
  const bool is_match = (f1.compare(f2) == 0);

  if (is_match) {
    std::remove("ds_checkpoint.pt");
    std::remove("recipe_trace.csv");
    std::remove("recipe_trace_rerun.csv");
  } else {
    PT_DYNAMIC_SHAPE_WARN("Recipe traces are not matching");
  }
}
