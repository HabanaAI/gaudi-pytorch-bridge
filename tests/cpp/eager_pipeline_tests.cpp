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
#include <gtest/gtest.h>
#include <torch/torch.h>
#include "backend/synapse_helpers/env_flags.h"
#include "habana_lazy_test_infra.h"

using namespace habana_lazy;
using namespace at;

class EagerPipelineTest : public habana_lazy_test::LazyTest {
 protected:
  void SetUp() override {
    SetEagerMode();
    DisableRecipeCache();
    SetSeed();
  }

  void TearDown() override {
    RestoreRecipeCache();
    RestoreMode();
  }
};

void CompileTask(bool error) {
  if (error) {
    throw std::runtime_error("Compiled failed");
  }
}

void ExecTask(bool error) {
  if (error) {
    throw std::runtime_error("Exec failed");
  }
}

TEST_F(EagerPipelineTest, QueueFull) {
  auto default_queue_capacity_ =
      GET_ENV_FLAG_NEW(PT_HPU_THREAD_POOL_QUEUE_CAPACITY);
  SET_ENV_FLAG_NEW(PT_HPU_THREAD_POOL_QUEUE_CAPACITY, 1, 1);
  torch::Tensor A = torch::randn({3, 3, 3});
  auto B = A.add(1.0);
  auto C = B.add(1.0);
  auto D = C.add(1.0);
  auto E = D.add(1.0);

  auto hA = A.to(torch::kHPU);
  auto hB = hA.add(1.0);
  auto hC = hB.add(1.0);
  auto hD = hC.add(1.0);
  auto hE = hD.add(1.0);

  EXPECT_EQ(allclose(E, hE.cpu(), 0.001, 0.001), true);
  SET_ENV_FLAG_NEW(
      PT_HPU_THREAD_POOL_QUEUE_CAPACITY, default_queue_capacity_, 1);
}

TEST_F(EagerPipelineTest, CompileError) {
  auto default_queue_capacity_ =
      GET_ENV_FLAG_NEW(PT_HPU_THREAD_POOL_QUEUE_CAPACITY);
  SET_ENV_FLAG_NEW(PT_HPU_THREAD_POOL_QUEUE_CAPACITY, 1, 1);
  habana::HPUDeviceContext::compile_thread().enqueue(CompileTask, true);
  EXPECT_ANY_THROW(
      habana::HPUDeviceContext::compile_thread().waitWorkComplete());
  SET_ENV_FLAG_NEW(
      PT_HPU_THREAD_POOL_QUEUE_CAPACITY, default_queue_capacity_, 1);
}

TEST_F(EagerPipelineTest, ExecError) {
  auto default_queue_capacity_ =
      GET_ENV_FLAG_NEW(PT_HPU_THREAD_POOL_QUEUE_CAPACITY);
  SET_ENV_FLAG_NEW(PT_HPU_THREAD_POOL_QUEUE_CAPACITY, 1, 1);
  habana::HPUDeviceContext::execute_thread().enqueue(ExecTask, true);
  EXPECT_ANY_THROW(
      habana::HPUDeviceContext::execute_thread().waitWorkComplete());
  SET_ENV_FLAG_NEW(
      PT_HPU_THREAD_POOL_QUEUE_CAPACITY, default_queue_capacity_, 1);
}
