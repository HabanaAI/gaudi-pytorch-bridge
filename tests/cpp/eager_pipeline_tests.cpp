/**
 * Copyright (c) 2021-2025 Intel Corporation
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

void WorkTask() {
  for (int i = 0; i < 5; i++) {
    std::this_thread::yield();
  }
}

void ProducerTask() {
  for (int i = 0; i < 10; i++) {
    habana::HPUDeviceContext::compile_thread_pool().enqueue(WorkTask);
  }
}

TEST_F(EagerPipelineTest, PipelineThrottling) {
  auto default_queue_capacity =
      GET_ENV_FLAG_NEW(PT_HPU_THREAD_POOL_QUEUE_CAPACITY);

  // make sure the thread pools are initialized
  [[maybe_unused]] at::Device device =
      habana::HPUDeviceContext::get_or_create_aten_device();

  auto& lowering_thread = habana::HPUDeviceContext::lowering_thread();
  auto& compile_thread_pool = habana::HPUDeviceContext::compile_thread_pool();
  SET_ENV_FLAG_NEW(PT_HPU_THREAD_POOL_QUEUE_CAPACITY, 2, 1);
  lowering_thread.enqueue(ProducerTask);

  auto producer_task = lowering_thread.get_active_task_count();
  while (producer_task > 0) {
    auto work_task = compile_thread_pool.get_active_task_count();
    ASSERT_LE(work_task, 2) << "Number of tasks exceeds capacity.";
    std::this_thread::yield();
    producer_task = lowering_thread.get_active_task_count();
  }

  lowering_thread.waitWorkComplete();
  compile_thread_pool.waitWorkComplete();

  SET_ENV_FLAG_NEW(
      PT_HPU_THREAD_POOL_QUEUE_CAPACITY, default_queue_capacity, 1);
}

TEST_F(EagerPipelineTest, CompileError) {
  // make sure the thread pools are initialized
  [[maybe_unused]] at::Device device =
      habana::HPUDeviceContext::get_or_create_aten_device();
  auto default_queue_capacity_ =
      GET_ENV_FLAG_NEW(PT_HPU_THREAD_POOL_QUEUE_CAPACITY);
  SET_ENV_FLAG_NEW(PT_HPU_THREAD_POOL_QUEUE_CAPACITY, 1, 1);
  habana::HPUDeviceContext::compile_thread_pool().enqueue(CompileTask, true);
  EXPECT_ANY_THROW(
      habana::HPUDeviceContext::compile_thread_pool().waitWorkComplete());
  SET_ENV_FLAG_NEW(
      PT_HPU_THREAD_POOL_QUEUE_CAPACITY, default_queue_capacity_, 1);
}

TEST_F(EagerPipelineTest, ExecError) {
  // make sure the thread pools are initialized
  [[maybe_unused]] at::Device device =
      habana::HPUDeviceContext::get_or_create_aten_device();
  auto default_queue_capacity_ =
      GET_ENV_FLAG_NEW(PT_HPU_THREAD_POOL_QUEUE_CAPACITY);
  SET_ENV_FLAG_NEW(PT_HPU_THREAD_POOL_QUEUE_CAPACITY, 1, 1);
  habana::HPUDeviceContext::execute_thread().enqueue(ExecTask, true);
  EXPECT_ANY_THROW(
      habana::HPUDeviceContext::execute_thread().waitWorkComplete());
  SET_ENV_FLAG_NEW(
      PT_HPU_THREAD_POOL_QUEUE_CAPACITY, default_queue_capacity_, 1);
}
