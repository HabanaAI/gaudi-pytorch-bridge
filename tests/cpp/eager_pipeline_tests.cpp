/*******************************************************************************
 * Copyright (C) 2023 Habana Labs, Ltd. an Intel Company
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
#include <gtest/gtest.h>
#include <torch/torch.h>
#include "backend/helpers/eager_pipeline.h"
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
  habana_helpers::Singleton_CompileThreadPool::getInstance().Enqueue(
      CompileTask, true);
  EXPECT_ANY_THROW(habana_helpers::Singleton_CompileThreadPool::getInstance()
                       .JoinPendingThread());
  SET_ENV_FLAG_NEW(
      PT_HPU_THREAD_POOL_QUEUE_CAPACITY, default_queue_capacity_, 1);
}

TEST_F(EagerPipelineTest, ExecError) {
  auto default_queue_capacity_ =
      GET_ENV_FLAG_NEW(PT_HPU_THREAD_POOL_QUEUE_CAPACITY);
  SET_ENV_FLAG_NEW(PT_HPU_THREAD_POOL_QUEUE_CAPACITY, 1, 1);
  habana_helpers::Singleton_ExecThreadPool::getInstance().Enqueue(
      ExecTask, true);
  EXPECT_ANY_THROW(habana_helpers::Singleton_ExecThreadPool::getInstance()
                       .JoinPendingThread());
  SET_ENV_FLAG_NEW(
      PT_HPU_THREAD_POOL_QUEUE_CAPACITY, default_queue_capacity_, 1);
}
