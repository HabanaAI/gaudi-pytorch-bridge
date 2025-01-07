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

#include <atomic>
#include <chrono>
#include <iostream>
#include <thread>

#include <gtest/gtest.h>

#include "backend/synapse_helpers/env_flags.h"
#include "habana_helpers/thread_pool/acc_thread_pool.h"

TEST(ACC_ThreadPoolTest, single_task) {
  std::atomic_bool task_done{false};
  std::chrono::milliseconds sleep_time_ms{50};

  auto acc_thread_pool = habana_lazy::CreateAccThreadPool();

  acc_thread_pool->run([&task_done]() { task_done = true; });

  std::this_thread::sleep_for(sleep_time_ms);

  EXPECT_EQ(true, task_done);
}

TEST(ACC_ThreadPoolTest, single_task_async) {
  SET_ENV_FLAG_NEW(PT_HPU_SYNCHRONOUS_ACC_QUEUE_FLUSHING, 1, 1);
  std::atomic_bool task_done{false};
  std::chrono::milliseconds sleep_time_ms{50};

  auto acc_thread_pool = habana_lazy::CreateAccThreadPool();

  acc_thread_pool->run([&task_done]() { task_done = true; });

  std::this_thread::sleep_for(sleep_time_ms);

  EXPECT_EQ(false, task_done);

  acc_thread_pool->waitWorkComplete();

  EXPECT_EQ(true, task_done);

  UNSET_ENV_FLAG_NEW(PT_HPU_SYNCHRONOUS_ACC_QUEUE_FLUSHING);
}

TEST(ACC_ThreadPoolTest, single_task_async_discard) {
  SET_ENV_FLAG_NEW(PT_HPU_SYNCHRONOUS_ACC_QUEUE_FLUSHING, 1, 1);
  std::atomic_bool task_done{false};
  std::chrono::milliseconds sleep_time_ms{50};

  auto acc_thread_pool = habana_lazy::CreateAccThreadPool();

  acc_thread_pool->run([&task_done]() { task_done = true; });

  std::this_thread::sleep_for(sleep_time_ms);
  EXPECT_EQ(false, task_done);

  acc_thread_pool->discardPendingTasks();
  EXPECT_EQ(false, task_done);

  acc_thread_pool->waitWorkComplete();
  EXPECT_EQ(false, task_done);

  UNSET_ENV_FLAG_NEW(PT_HPU_SYNCHRONOUS_ACC_QUEUE_FLUSHING);
}

TEST(ACC_ThreadPoolTest, many_tasks) {
  std::atomic_bool task_one_done{false};
  std::atomic_bool task_two_done{false};
  std::atomic_bool task_three_done{false};

  std::chrono::milliseconds sleep_time_ms{50};

  auto acc_thread_pool = habana_lazy::CreateAccThreadPool();

  acc_thread_pool->run([&task_one_done]() { task_one_done = true; });
  acc_thread_pool->run([&task_two_done]() { task_two_done = true; });
  acc_thread_pool->run([&task_three_done]() { task_three_done = true; });

  std::this_thread::sleep_for(sleep_time_ms);

  EXPECT_EQ(true, task_one_done);
  EXPECT_EQ(true, task_two_done);
  EXPECT_EQ(true, task_three_done);
}

TEST(ACC_ThreadPoolTest, many_tasks_async) {
  SET_ENV_FLAG_NEW(PT_HPU_SYNCHRONOUS_ACC_QUEUE_FLUSHING, 1, 1);
  std::atomic_bool task_one_done{false};
  std::atomic_bool task_two_done{false};
  std::atomic_bool task_three_done{false};

  std::chrono::milliseconds sleep_time_ms{50};

  auto acc_thread_pool = habana_lazy::CreateAccThreadPool();

  acc_thread_pool->run([&task_one_done]() { task_one_done = true; });
  acc_thread_pool->run([&task_two_done]() { task_two_done = true; });
  acc_thread_pool->run([&task_three_done]() { task_three_done = true; });

  std::this_thread::sleep_for(sleep_time_ms);

  EXPECT_EQ(false, task_one_done);
  EXPECT_EQ(false, task_two_done);
  EXPECT_EQ(false, task_three_done);

  acc_thread_pool->waitWorkComplete();
  EXPECT_EQ(true, task_one_done);
  EXPECT_EQ(true, task_two_done);
  EXPECT_EQ(true, task_three_done);
  UNSET_ENV_FLAG_NEW(PT_HPU_SYNCHRONOUS_ACC_QUEUE_FLUSHING);
}

TEST(ACC_ThreadPoolTest, many_tasks_async_discard) {
  SET_ENV_FLAG_NEW(PT_HPU_SYNCHRONOUS_ACC_QUEUE_FLUSHING, 1, 1);
  std::atomic_bool task_one_done{false};
  std::atomic_bool task_two_done{false};
  std::atomic_bool task_three_done{false};

  std::chrono::milliseconds sleep_time_ms{50};

  auto acc_thread_pool = habana_lazy::CreateAccThreadPool();

  acc_thread_pool->run([&task_one_done]() { task_one_done = true; });
  acc_thread_pool->run([&task_two_done]() { task_two_done = true; });
  acc_thread_pool->run([&task_three_done]() { task_three_done = true; });

  std::this_thread::sleep_for(sleep_time_ms);
  EXPECT_EQ(false, task_one_done);
  EXPECT_EQ(false, task_two_done);
  EXPECT_EQ(false, task_three_done);

  acc_thread_pool->discardPendingTasks();
  EXPECT_EQ(false, task_one_done);
  EXPECT_EQ(false, task_two_done);
  EXPECT_EQ(false, task_three_done);

  acc_thread_pool->waitWorkComplete();
  EXPECT_EQ(false, task_one_done);
  EXPECT_EQ(false, task_two_done);
  EXPECT_EQ(false, task_three_done);
  UNSET_ENV_FLAG_NEW(PT_HPU_SYNCHRONOUS_ACC_QUEUE_FLUSHING);
}
