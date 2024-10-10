/*******************************************************************************
 * Copyright (C) 2022-2023 Habana Labs, Ltd. an Intel Company
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
#include <math.h>
#include <torch/torch.h>
#include <stdexcept>
#include "backend/synapse_helpers/env_flags.h"
#include "habana_kernels/lazy_kernels_declarations.h"
#include "habana_kernels/linear_kernels.h"
#include "habana_lazy/hpu_lazy_tensors.h"
#include "habana_lazy/ir_utils.h"
#include "habana_lazy_test_infra.h"

#include <functional>
#include <future>
#include <thread>
#include <unordered_set>

#include "backend/habana_device/HPUGuardImpl.h"
#include "habana_helpers/logging_pt.h"
#include "utils/check_device_type.h"

using namespace habana_lazy;

#define ASSERT_EQ_HPU(X, Y) \
  {                         \
    bool isTRUE = X == Y;   \
    ASSERT_TRUE(isTRUE);    \
  }

#define ASSERT_NE_HPU(X, Y) \
  {                         \
    bool isFALSE = X == Y;  \
    ASSERT_FALSE(isFALSE);  \
  }

class TestStream : public habana_lazy_test::LazyTest {};
/*
   Tests related to ATen streams.
   */
// Verifies streams are live through copying and moving
TEST(TestStream, CopyAndMoveTest) {
  habana::HABANAGuardImpl device_guard;
  device_guard.getDevice();
  auto& device = habana::HPURegistrar::get_device();
  auto num_hpus = device.get_count_by_current_type();
  if (num_hpus == 0)
    return;
  int32_t device_id = -1;
  synapse_helpers::hpuStream_t hpu_stream;

  // Tests that copying works as expected and preserves the stream
  c10::hpu::HPUStream copyStream = c10::hpu::getStreamFromPool();
  {
    auto s = c10::hpu::getStreamFromPool();
    device_id = s.device_index();
    hpu_stream = s.stream();

    copyStream = s;

    ASSERT_EQ_HPU(copyStream.device_index(), device_id);
    ASSERT_EQ_HPU(copyStream.stream(), hpu_stream);
  }

  ASSERT_EQ_HPU(copyStream.device_index(), device_id);
  ASSERT_EQ_HPU(copyStream.stream(), hpu_stream);

  // Tests that moving works as expected and preserves the stream
  c10::hpu::HPUStream moveStream = c10::hpu::getStreamFromPool();
  {
    auto s = c10::hpu::getStreamFromPool();
    device_id = s.device_index();
    hpu_stream = s.stream();

    moveStream = std::move(s);

    ASSERT_EQ_HPU(moveStream.device_index(), device_id);
    ASSERT_EQ_HPU(moveStream.stream(), hpu_stream);
  }

  ASSERT_EQ_HPU(moveStream.device_index(), device_id);
  ASSERT_EQ_HPU(moveStream.stream(), hpu_stream);
}

// Verifies streams are set properly
TEST(TestStream, GetAndSetTest) {
  habana::HABANAGuardImpl device_guard;
  device_guard.getDevice();
  auto& device = habana::HPURegistrar::get_device();
  auto num_hpus = device.get_count_by_current_type();
  if (num_hpus == 0)
    return;
  c10::hpu::HPUStream myStream = c10::hpu::getStreamFromPool();

  // Sets and gets
  c10::hpu::setCurrentHPUStream(myStream);
  c10::hpu::HPUStream curStream = c10::hpu::getCurrentHPUStream();

  ASSERT_EQ_HPU(myStream, curStream);

  // Gets, sets, and gets default stream
  c10::hpu::HPUStream defaultStream = c10::hpu::getDefaultHPUStream();
  c10::hpu::setCurrentHPUStream(defaultStream);
  curStream = c10::hpu::getCurrentHPUStream();

  ASSERT_NE_HPU(defaultStream, myStream);
  ASSERT_EQ_HPU(curStream, defaultStream);
}

void thread_fun(at::optional<c10::hpu::HPUStream>& cur_thread_stream) {
  auto new_stream = c10::hpu::getStreamFromPool();
  c10::hpu::HPUStream cur_stream = c10::hpu::getCurrentHPUStream();
  c10::hpu::HPUStream default_stream = c10::hpu::getDefaultHPUStream();
  c10::hpu::setCurrentHPUStream(new_stream);
  cur_thread_stream = {c10::hpu::getCurrentHPUStream()};
  ASSERT_EQ_HPU(*cur_thread_stream, new_stream);
}

// Ensures streams are thread local
TEST(TestStream, DISABLED_MultithreadGetAndSetTest) {
  habana::HABANAGuardImpl device_guard;
  device_guard.getDevice();
  auto& device = habana::HPURegistrar::get_device();
  auto num_hpus = device.get_count_by_current_type();
  if (num_hpus == 0)
    return;
  at::optional<c10::hpu::HPUStream> s0, s1;

  std::thread t0{thread_fun, std::ref(s0)};
  std::thread t1{thread_fun, std::ref(s1)};
  t0.join();
  t1.join();

  c10::hpu::HPUStream cur_stream = c10::hpu::getCurrentHPUStream();
  c10::hpu::HPUStream default_stream = c10::hpu::getDefaultHPUStream();

  if (device.type() == synDeviceGaudi) {
    ASSERT_EQ_HPU(cur_stream, default_stream);
    ASSERT_NE_HPU(cur_stream, *s0);
    ASSERT_NE_HPU(cur_stream, *s1);
    ASSERT_EQ_HPU(s0, s1);
  } else {
    ASSERT_EQ_HPU(cur_stream, default_stream);
    ASSERT_NE_HPU(cur_stream, *s0);
    ASSERT_NE_HPU(cur_stream, *s1);
    ASSERT_NE_HPU(s0, s1);
  }
}

TEST(TestStream, StreamPoolTest) {
  habana::HABANAGuardImpl device_guard;
  device_guard.getDevice();
  auto& device = habana::HPURegistrar::get_device();
  auto num_hpus = device.get_count_by_current_type();
  if (num_hpus == 0)
    return;
  std::vector<c10::hpu::HPUStream> streams{};
  for (const auto i : c10::irange(200)) {
    (void)i;
    streams.emplace_back(c10::hpu::getStreamFromPool());
  }

  std::unordered_set<synapse_helpers::hpuStream_t> stream_set{};
  bool hasDuplicates = false;
  for (const auto i : c10::irange(streams.size())) {
    synapse_helpers::hpuStream_t hpu_stream = streams[i];
    if (stream_set.find(hpu_stream) == stream_set.end()) {
      stream_set.insert(hpu_stream);
    } else {
      hasDuplicates = true;
      break;
    }
  }
  ASSERT_TRUE(!hasDuplicates);
}

TEST(TestStream, DISABLED_Use2StreamForadd) {
  habana::HABANAGuardImpl device_guard;
  device_guard.getDevice();
  auto& device = habana::HPURegistrar::get_device();
  auto num_hpus = device.get_count_by_current_type();
  if (num_hpus == 0)
    return;

  c10::hpu::HPUStream compute1 = c10::hpu::getStreamFromPool();
  c10::hpu::HPUStream compute2 = c10::hpu::getStreamFromPool();

  /*default stream */
  torch::Tensor tensor_D = torch::randn({200, 300});
  torch::Tensor tHabana_D = tensor_D.to(torch::kHPU);
  auto outHabana_D = torch::add(tHabana_D, 4.0);

  c10::hpu::setCurrentHPUStream(compute1);
  torch::Tensor tensor_A = torch::randn({200, 300});
  torch::Tensor tensor_B = torch::randn({200, 300});
  torch::Tensor tHabana_A = tensor_A.to(torch::kHPU);

  c10::hpu::setCurrentHPUStream(compute1);
  auto outHabana_A = torch::add(tHabana_A, 4.0);

  torch::Tensor tHabana_B = tensor_B.to(torch::kHPU);
  c10::hpu::setCurrentHPUStream(compute2);
  auto outHabana_B = torch::add(tHabana_B, 4.0);

  auto out_A = torch::add(tensor_A, 4.0);
  auto out_B = torch::add(tensor_B, 4.0);
  auto out_D = torch::add(tensor_D, 4.0);
  bool equal = out_A.allclose(outHabana_A.to(torch::kCPU), 1e-3, 1e-3);
  EXPECT_EQ(equal, true);
  equal = out_B.allclose(outHabana_B.to(torch::kCPU), 1e-3, 1e-3);
  EXPECT_EQ(equal, true);
  equal = out_D.allclose(outHabana_D.to(torch::kCPU), 1e-3, 1e-3);
  EXPECT_EQ(equal, true);
}

TEST(TestStream, ForceUseDefaultStream) {
  habana::HABANAGuardImpl device_guard;
  device_guard.getDevice();
  auto& device = habana::HPURegistrar::get_device();
  auto num_hpus = device.get_count_by_current_type();
  if (num_hpus == 0)
    return;

  SET_ENV_FLAG_NEW(PT_HPU_FORCE_USE_DEFAULT_STREAM, true, 1);
  c10::hpu::HPUStream compute1 = c10::hpu::getStreamFromPool();
  c10::hpu::HPUStream compute2 = c10::hpu::getStreamFromPool();

  /*default stream */
  torch::Tensor tensor_D = torch::randn({200, 300});
  torch::Tensor tHabana_D = tensor_D.to(torch::kHPU);
  auto outHabana_D = torch::add(tHabana_D, 4.0);

  c10::hpu::setCurrentHPUStream(compute1);
  torch::Tensor tensor_A = torch::randn({200, 300});
  torch::Tensor tensor_B = torch::randn({200, 300});
  torch::Tensor tHabana_A = tensor_A.to(torch::kHPU);

  c10::hpu::setCurrentHPUStream(compute1);
  auto outHabana_A = torch::add(tHabana_A, 4.0);

  torch::Tensor tHabana_B = tensor_B.to(torch::kHPU);
  c10::hpu::setCurrentHPUStream(compute2);
  auto outHabana_B = torch::add(tHabana_B, 4.0);

  auto out_A = torch::add(tensor_A, 4.0);
  auto out_B = torch::add(tensor_B, 4.0);
  auto out_D = torch::add(tensor_D, 4.0);
  bool equal = out_A.allclose(outHabana_A.to(torch::kCPU), 1e-3, 1e-3);
  EXPECT_EQ(equal, true);
  equal = out_B.allclose(outHabana_B.to(torch::kCPU), 1e-3, 1e-3);
  EXPECT_EQ(equal, true);
  equal = out_D.allclose(outHabana_D.to(torch::kCPU), 1e-3, 1e-3);
  EXPECT_EQ(equal, true);
  UNSET_ENV_FLAG_NEW(PT_HPU_FORCE_USE_DEFAULT_STREAM);
}

void thread_fun_add(bool& result) {
  auto new_stream = c10::hpu::getStreamFromPool();
  c10::hpu::setCurrentHPUStream(new_stream);
  torch::Tensor tensor_A = torch::randn({200, 300});
  torch::Tensor tHabana_A = tensor_A.to(torch::kHPU);
  auto outHabana_A = torch::add(tHabana_A, 4.0);
  auto out_A = torch::add(tensor_A, 4.0);
  bool equal = out_A.allclose(outHabana_A.to(torch::kCPU), 1e-3, 1e-3);
  result = equal;
}

TEST(TestStream, DISABLED_MultithreadStreamAddOP) {
  habana::HABANAGuardImpl device_guard;
  device_guard.getDevice();
  auto& device = habana::HPURegistrar::get_device();
  auto num_hpus = device.get_count_by_current_type();
  if (num_hpus == 0)
    return;
  bool result1, result2;

  std::thread t0{thread_fun_add, std::ref(result1)};
  std::thread t1{thread_fun_add, std::ref(result2)};
  t0.join();
  t1.join();

  EXPECT_EQ(result1, true);
  EXPECT_EQ(result2, true);
}

void kernel_add(torch::Tensor in_tensor, torch::Tensor& outtensor) {
  auto new_stream = c10::hpu::getStreamFromPool();
  c10::hpu::setCurrentHPUStream(new_stream);
  outtensor = torch::add(in_tensor, 4.0);
  HbLazyTensor::StepMarker({});
}

// Ensures streams are thread local
TEST(TestStream, MultithreadStreamKernelAdd) {
  habana::HABANAGuardImpl device_guard;
  device_guard.getDevice();
  auto& device = habana::HPURegistrar::get_device();
  auto num_hpus = device.get_count_by_current_type();
  if (num_hpus == 0)
    return;
  torch::Tensor tensor_A = torch::randn({200, 300});
  torch::Tensor tHabana_A = tensor_A.to(torch::kHPU);
  torch::Tensor tensor_B = torch::randn({200, 300});
  torch::Tensor tHabana_B = tensor_B.to(torch::kHPU);
  torch::Tensor outHabana_A, outHabana_B;

  std::thread t0{kernel_add, tensor_A, std::ref(outHabana_A)};
  std::thread t1{kernel_add, tensor_B, std::ref(outHabana_B)};
  t0.join();
  t1.join();

  auto out_A = torch::add(tensor_A, 4.0);
  auto out_B = torch::add(tensor_B, 4.0);
  bool equal1 = out_A.allclose(outHabana_A.to(torch::kCPU), 1e-3, 1e-3);
  bool equal2 = out_B.allclose(outHabana_B.to(torch::kCPU), 1e-3, 1e-3);
  EXPECT_EQ(equal1, true);
  EXPECT_EQ(equal2, true);
}

TEST(TestStream, TestStreamQuery) {
  habana::HABANAGuardImpl device_guard;
  device_guard.getDevice();
  auto& device = habana::HPURegistrar::get_device();
  auto num_hpus = device.get_count_by_current_type();
  if (num_hpus == 0)
    return;

  c10::hpu::HPUStream compute1 = c10::hpu::getStreamFromPool();
  c10::hpu::HPUStream compute2 = c10::hpu::getStreamFromPool();

  PT_TEST_DEBUG("stream query for stream1", compute1.query());
  c10::hpu::setCurrentHPUStream(compute1);
  torch::Tensor tensor_A = torch::randn({200, 300});
  torch::Tensor tensor_B = torch::randn({200, 300});
  torch::Tensor tHabana_A = tensor_A.to(torch::kHPU);
  c10::hpu::setCurrentHPUStream(compute1);
  auto outHabana_A = torch::add(tHabana_A, 4.0);
  PT_TEST_DEBUG("stream query for stream1", compute1.query());
  torch::Tensor tHabana_B = tensor_B.to(torch::kHPU);
  c10::hpu::setCurrentHPUStream(compute2);
  auto outHabana_B = torch::add(tHabana_B, 4.0);
  PT_TEST_DEBUG("stream query for stream2", compute2.query());
  auto out_A = torch::add(tensor_A, 4.0);
  auto out_B = torch::add(tensor_B, 4.0);
  bool equal = out_A.allclose(outHabana_A.to(torch::kCPU), 1e-3, 1e-3);
  EXPECT_EQ(equal, true);
  equal = out_B.allclose(outHabana_B.to(torch::kCPU), 1e-3, 1e-3);
  EXPECT_EQ(equal, true);
  PT_TEST_DEBUG("stream query for stream1", compute1.query());
}

TEST(TestStream, TestStreamSynchronize) {
  habana::HABANAGuardImpl device_guard;
  device_guard.getDevice();
  auto& device = habana::HPURegistrar::get_device();
  auto num_hpus = device.get_count_by_current_type();
  if (num_hpus == 0)
    return;

  c10::hpu::HPUStream defaultStream = c10::hpu::getDefaultHPUStream();
  defaultStream.synchronize();
  c10::hpu::HPUStream compute1 = c10::hpu::getStreamFromPool();
  c10::hpu::HPUStream compute2 = c10::hpu::getStreamFromPool();

  c10::hpu::setCurrentHPUStream(compute1);
  torch::Tensor tensor_A = torch::randn({200, 300});
  torch::Tensor tensor_B = torch::randn({200, 300});
  torch::Tensor tHabana_A = tensor_A.to(torch::kHPU);
  c10::hpu::setCurrentHPUStream(compute1);
  auto outHabana_A = torch::add(tHabana_A, 4.0);
  PT_TEST_DEBUG("STREAM:: synchronize stream1");
  compute1.synchronize();
  torch::Tensor tHabana_B = tensor_B.to(torch::kHPU);
  c10::hpu::setCurrentHPUStream(compute2);
  auto outHabana_B = torch::add(tHabana_B, 4.0);
  PT_TEST_DEBUG("STREAM:: synchronize stream2");
  compute2.synchronize();
  auto out_A = torch::add(tensor_A, 4.0);
  auto out_B = torch::add(tensor_B, 4.0);
  bool equal = out_A.allclose(outHabana_A.to(torch::kCPU), 1e-3, 1e-3);
  EXPECT_EQ(equal, true);
  equal = out_B.allclose(outHabana_B.to(torch::kCPU), 1e-3, 1e-3);
  EXPECT_EQ(equal, true);
  compute2.synchronize();
}

// simple HPUEvent Syncs with nops
TEST(TestStream, HPUEventSyncTest) {
  habana::HABANAGuardImpl device_guard;
  device_guard.getDevice();
  auto& device = habana::HPURegistrar::get_device();
  device.synchronize();
  auto num_hpus = device.get_count_by_current_type();
  if (num_hpus == 0)
    return;

  const auto stream = c10::hpu::getStreamFromPool();
  at::hpu::HPUEvent event;

  ASSERT_TRUE(event.query());

  event.recordOnce(stream);

  const auto wait_stream0 = c10::hpu::getStreamFromPool();
  const auto wait_stream1 = c10::hpu::getStreamFromPool();

  event.block(wait_stream0);
  event.block(wait_stream1);

  wait_stream0.synchronize();
  ASSERT_TRUE(event.query());
}

/// block and wait, instead of event sycn
TEST(TestStream, TestEventblockandwait) {
  habana::HABANAGuardImpl device_guard;
  device_guard.getDevice();
  auto& device = habana::HPURegistrar::get_device();
  auto num_hpus = device.get_count_by_current_type();
  if (num_hpus == 0)
    return;

  c10::hpu::HPUStream compute1 = c10::hpu::getStreamFromPool();
  c10::hpu::HPUStream compute2 = c10::hpu::getStreamFromPool();

  at::hpu::HPUEvent event1;
  c10::hpu::setCurrentHPUStream(compute1);
  torch::Tensor tensor_A = torch::randn({200, 300});
  torch::Tensor tensor_B = torch::randn({200, 300});
  torch::Tensor tHabana_A = tensor_A.to(torch::kHPU);
  c10::hpu::setCurrentHPUStream(compute1);
  auto outHabana_A = torch::add(tHabana_A, 4.0);
  event1.record(compute1);
  event1.block(compute2);
  torch::Tensor tHabana_B = tensor_B.to(torch::kHPU);
  c10::hpu::setCurrentHPUStream(compute2);
  auto outHabana_B = torch::add(tHabana_B, 4.0);
  compute2.synchronize();
  auto out_A = torch::add(tensor_A, 4.0);
  auto out_B = torch::add(tensor_B, 4.0);
  bool equal = out_A.allclose(outHabana_A.to(torch::kCPU), 1e-3, 1e-3);
  EXPECT_EQ(equal, true);
  equal = out_B.allclose(outHabana_B.to(torch::kCPU), 1e-3, 1e-3);
  EXPECT_EQ(equal, true);
}

TEST(TestStream, TestEventblockandwait_1) {
  habana::HABANAGuardImpl device_guard;
  device_guard.getDevice();
  auto& device = habana::HPURegistrar::get_device();
  auto num_hpus = device.get_count_by_current_type();
  if (num_hpus == 0)
    return;

  auto t_dim_0 = 2000, t_dim_1 = 3000, t_dim_2 = 6000;
  if (is_simulator()) {
    t_dim_0 = 20;
    t_dim_1 = 30;
    t_dim_2 = 60;
  }

  c10::hpu::HPUStream compute1 = c10::hpu::getStreamFromPool();
  c10::hpu::HPUStream defaultStream = c10::hpu::getDefaultHPUStream();

  at::hpu::HPUEvent event1;
  torch::Tensor t1_cpu = torch::randn({t_dim_0, t_dim_1});
  torch::Tensor t2_cpu = torch::randn({t_dim_1, t_dim_2});
  torch::Tensor t1 = t1_cpu.to(torch::kHPU);
  torch::Tensor t2 = t2_cpu.to(torch::kHPU);
  torch::Tensor tres = torch::matmul(t1, t2);
  event1.record(defaultStream);
  c10::hpu::setCurrentHPUStream(compute1);
  torch::Tensor t3 = torch::matmul(t1, t2);
  c10::hpu::setCurrentHPUStream(defaultStream);
  event1.block(compute1);
  t1.add_(2);
  torch::Tensor tres_cpu = tres.to(torch::kCPU);
  torch::Tensor t3_cpu = t3.to(torch::kCPU);
  bool equal = tres_cpu.allclose(t3_cpu, 1e-3, 1e-3);
  EXPECT_EQ(equal, true);
}

TEST(TestStream, TestEventblockOnDifferentStream) {
  habana::HABANAGuardImpl device_guard;
  device_guard.getDevice();
  auto& device = habana::HPURegistrar::get_device();
  auto num_hpus = device.get_count_by_current_type();
  if (num_hpus == 0)
    return;

  c10::hpu::HPUStream compute1 = c10::hpu::getStreamFromPool();
  c10::hpu::HPUStream defaultStream = c10::hpu::getDefaultHPUStream();

  at::hpu::HPUEvent event1;
  torch::Tensor t1_cpu = torch::randn({200, 300});
  torch::Tensor t2_cpu = torch::randn({300, 600});
  torch::Tensor t1 = t1_cpu.to(torch::kHPU);
  torch::Tensor t2 = t2_cpu.to(torch::kHPU);
  torch::Tensor tres = torch::matmul(t1, t2);
  event1.record(compute1);
  torch::Tensor t3 = torch::matmul(t1, t2);
  event1.block(defaultStream);
  t1.add_(2);
  defaultStream.synchronize();
  torch::Tensor tres_cpu = tres.to(torch::kCPU);
  torch::Tensor t3_cpu = t3.to(torch::kCPU);
  bool equal = tres_cpu.allclose(t3_cpu, 1e-3, 1e-3);
  EXPECT_EQ(equal, true);
}

/// multiple events and using sync
TEST(TestStream, TestMultipleRecordEvent) {
  habana::HABANAGuardImpl device_guard;
  device_guard.getDevice();
  auto& device = habana::HPURegistrar::get_device();
  auto num_hpus = device.get_count_by_current_type();
  if (num_hpus == 0)
    return;

  c10::hpu::HPUStream compute1 = c10::hpu::getStreamFromPool();
  c10::hpu::HPUStream compute2 = c10::hpu::getStreamFromPool();

  at::hpu::HPUEvent event1;
  at::hpu::HPUEvent event2;
  at::hpu::HPUEvent event3;
  c10::hpu::setCurrentHPUStream(compute1);
  torch::Tensor tensor_A = torch::randn({200, 300});
  torch::Tensor tensor_B = torch::randn({200, 300});
  torch::Tensor tHabana_A = tensor_A.to(torch::kHPU);
  c10::hpu::setCurrentHPUStream(compute1);
  auto outHabana_A = torch::add(tHabana_A, 4.0);
  event1.record(compute1);
  event2.record(compute1);
  event3.record(compute1);
  event3.block(compute2);
  torch::Tensor tHabana_B = tensor_B.to(torch::kHPU);
  c10::hpu::setCurrentHPUStream(compute2);
  auto outHabana_B = torch::add(tHabana_B, 4.0);
  compute2.synchronize();
  auto out_A = torch::add(tensor_A, 4.0);
  auto out_B = torch::add(tensor_B, 4.0);
  bool equal = out_A.allclose(outHabana_A.to(torch::kCPU), 1e-3, 1e-3);
  EXPECT_EQ(equal, true);
  equal = out_B.allclose(outHabana_B.to(torch::kCPU), 1e-3, 1e-3);
  EXPECT_EQ(equal, true);
  // compute2.synchronize();
}

// get the event execution time
TEST(TestStream, TestTimerEvents) {
  habana::HABANAGuardImpl device_guard;
  device_guard.getDevice();
  auto& device = habana::HPURegistrar::get_device();
  auto num_hpus = device.get_count_by_current_type();
  if (num_hpus == 0)
    return;

  c10::hpu::HPUStream compute1 = c10::hpu::getStreamFromPool();

  // timer events
  at::hpu::HPUEvent start_event(1);
  at::hpu::HPUEvent stop_event(1);
  c10::hpu::setCurrentHPUStream(compute1);
  torch::Tensor tensor_A = torch::randn({200, 300});
  torch::Tensor tensor_B = torch::randn({200, 300});
  torch::Tensor tHabana_A = tensor_A.to(torch::kHPU);
  c10::hpu::setCurrentHPUStream(compute1);
  start_event.record(compute1);
  auto outHabana_A = torch::add(tHabana_A, 4.0);
  stop_event.record(compute1);
  compute1.synchronize();
  PT_TEST_DEBUG("elapsed time", start_event.elapsed_time(stop_event));
  auto out_A = torch::add(tensor_A, 4.0);
  bool equal = out_A.allclose(outHabana_A.to(torch::kCPU), 1e-3, 1e-3);
  EXPECT_EQ(equal, true);
}

// get the event execution time
TEST(TestStream, TestTimerEventsDifferentStream) {
  habana::HABANAGuardImpl device_guard;
  device_guard.getDevice();
  auto& device = habana::HPURegistrar::get_device();
  auto num_hpus = device.get_count_by_current_type();
  if (num_hpus == 0)
    return;

  c10::hpu::HPUStream compute1 = c10::hpu::getStreamFromPool();
  c10::hpu::HPUStream defaultStream = c10::hpu::getDefaultHPUStream();

  // timer events
  at::hpu::HPUEvent start_event(1);
  at::hpu::HPUEvent stop_event(1);
  torch::Tensor tensor_A = torch::randn({200, 300});
  torch::Tensor tensor_B = torch::randn({200, 300});
  torch::Tensor tHabana_A = tensor_A.to(torch::kHPU);
  torch::Tensor tHabana_B = tensor_B.to(torch::kHPU);
  start_event.record(compute1);
  auto outHabana_A = torch::add(tHabana_A, 4.0);
  c10::hpu::setCurrentHPUStream(compute1);
  auto outHabana_B = torch::add(tHabana_B, 4.0);
  stop_event.record(defaultStream);
  compute1.synchronize();
  defaultStream.synchronize();
  PT_TEST_DEBUG("elapsed time", start_event.elapsed_time(stop_event));
  auto out_A = torch::add(tensor_A, 4.0);
  bool equal = out_A.allclose(outHabana_A.to(torch::kCPU), 1e-3, 1e-3);
  EXPECT_EQ(equal, true);
}

// get the event empty ops
TEST(TestStream, TestEventsEmptyOPS) {
  habana::HABANAGuardImpl device_guard;
  device_guard.getDevice();
  auto& device = habana::HPURegistrar::get_device();
  auto num_hpus = device.get_count_by_current_type();
  if (num_hpus == 0)
    return;

  at::hpu::HPUEvent event1;
  at::hpu::HPUEvent event2;
  torch::Tensor tensor_A = torch::randn({200, 300});
  torch::Tensor tensor_B = torch::randn({200, 300});
  event1.record();
  event2.record();
  event1.synchronize();
}

// get the event empty ops
TEST(TestStream, TestEventsTimerEmptyOPS) {
  habana::HABANAGuardImpl device_guard;
  device_guard.getDevice();
  auto& device = habana::HPURegistrar::get_device();
  auto num_hpus = device.get_count_by_current_type();
  if (num_hpus == 0)
    return;

  at::hpu::HPUEvent event_start(1);
  at::hpu::HPUEvent event_end(1);
  torch::Tensor tensor_A = torch::randn({200, 300});
  torch::Tensor tensor_B = torch::randn({200, 300});
  event_start.record();
  event_end.record();
  PT_TEST_DEBUG("elapsed time", event_start.elapsed_time(event_end));
}

TEST(TestStream, TestEventsTimerWithDifferentStream) {
  habana::HABANAGuardImpl device_guard;
  device_guard.getDevice();
  auto& device = habana::HPURegistrar::get_device();
  auto num_hpus = device.get_count_by_current_type();
  if (num_hpus == 0)
    return;

  c10::hpu::HPUStream compute1 = c10::hpu::getStreamFromPool();
  c10::hpu::HPUStream defaultStream = c10::hpu::getDefaultHPUStream();
  at::hpu::HPUEvent event_start(1);
  at::hpu::HPUEvent event_end(1);
  event_start.record(compute1);
  event_end.record(defaultStream);
  PT_TEST_DEBUG("elapsed time", event_start.elapsed_time(event_end));
}

TEST(TestStream, TestEventWaitBlock_WAR) {
  habana::HABANAGuardImpl device_guard;
  device_guard.getDevice();
  auto& device = habana::HPURegistrar::get_device();
  auto num_hpus = device.get_count_by_current_type();
  if (num_hpus == 0)
    return;

  c10::hpu::HPUStream compute1 = c10::hpu::getStreamFromPool();
  c10::hpu::HPUStream default_s = c10::hpu::getDefaultHPUStream();
  at::hpu::HPUEvent event1;

  torch::Tensor tA_h = torch::zeros({4000, 2000}).to(torch::kHPU);
  torch::Tensor tB_h = torch::ones({4000, 2000}).to(torch::kHPU);
  torch::Tensor tC_h = torch::empty_like(tA_h);
  torch::Tensor tD_h = torch::empty_like(tA_h);

  c10::hpu::setCurrentHPUStream(compute1);
  tC_h = tA_h + tB_h;

  c10::hpu::setCurrentHPUStream(default_s);
  event1.record(compute1);
  event1.block(default_s);
  tD_h = tC_h.to(torch::dtype(torch::kBFloat16));

  torch::Tensor cpu_tensor = tD_h.cpu();
  PT_TEST_DEBUG("tensor::", cpu_tensor);
}

TEST(TestStream, TestEventWaitBlock_11) {
  habana::HABANAGuardImpl device_guard;
  device_guard.getDevice();
  auto& device = habana::HPURegistrar::get_device();
  auto num_hpus = device.get_count_by_current_type();
  if (num_hpus == 0)
    return;

  c10::hpu::HPUStream compute1 = c10::hpu::getStreamFromPool();
  c10::hpu::HPUStream default_s = c10::hpu::getDefaultHPUStream();
  at::hpu::HPUEvent event1;
  at::hpu::HPUEvent event2;

  torch::Tensor tA_h = torch::zeros({4, 2}).to(torch::kHPU);
  torch::Tensor tB_h = torch::ones({4, 2}).to(torch::kHPU);
  torch::Tensor tC_h = torch::empty_like(tA_h);
  torch::Tensor tD_h = torch::empty_like(tA_h);

  c10::hpu::setCurrentHPUStream(compute1);
  tC_h = tA_h + tB_h;

  event1.record(compute1);
  event1.block(default_s);
  c10::hpu::setCurrentHPUStream(default_s);
  tC_h = tC_h + 1;
  event2.record(default_s);
  event2.block(compute1);
  c10::hpu::setCurrentHPUStream(compute1);
  tD_h = tC_h.to(torch::dtype(torch::kBFloat16));

  torch::Tensor cpu_tensor = tD_h.cpu();
  PT_TEST_DEBUG("tensor::", cpu_tensor);
}

TEST(TestStream, HPUGuardEventSyncTest) {
  habana::HABANAGuardImpl guard;
  guard.getDevice();
  auto& device = habana::HPURegistrar::get_device();
  auto num_hpus = device.get_count_by_current_type();
  if (num_hpus == 0)
    return;

  const auto stream = c10::hpu::getStreamFromPool();

  void* event = nullptr;
  guard.record(&event, stream.unwrap(), 0, at::EventFlag::PYTORCH_DEFAULT);

  const auto wait_stream0 = c10::hpu::getStreamFromPool();
  const auto wait_stream1 = c10::hpu::getStreamFromPool();
  PT_TEST_DEBUG("query:", guard.queryEvent(event));
  guard.block(event, wait_stream0.unwrap());
  guard.block(event, wait_stream1.unwrap());

  wait_stream0.synchronize();
  PT_TEST_DEBUG("query:", guard.queryEvent(event));
  guard.destroyEvent(event, 0);
}

TEST(TestStream, TestWAR_defaultstream) {
  /*
  Case 1: WAR dependency
  x = op(param) <- Read param is NOT in output of op
  mark_step()
  load :
    // Wait for param to be read by op : No event for param is placed
    param.copy_(key) <- Write to param
  print(x.to('cpu'))
  */

  habana::HABANAGuardImpl device_guard;
  device_guard.getDevice();
  auto& device = habana::HPURegistrar::get_device();
  auto num_hpus = device.get_count_by_current_type();
  if (num_hpus == 0)
    return;

  torch::Tensor tensor_A = torch::randn({2000, 3000});
  torch::Tensor tensor_B = torch::randn({2000, 3000});
  torch::Tensor tensor_r = torch::relu(tensor_A);

  torch::Tensor tHabana_A = tensor_A.to(torch::kHPU);
  torch::Tensor tHabana_B = tensor_B.to(torch::kHPU);

  tensor_A.copy_(tensor_B);

  torch::Tensor tHabana_r = torch::relu(tHabana_A);
  HbLazyTensor::StepMarker({});

  tHabana_A.copy_(tHabana_B);
  torch::Tensor cpu_tensor = tHabana_A.cpu();
  EXPECT_EQ(tensor_A.allclose(cpu_tensor, 0, 0), true);

  tensor_r = torch::relu(tensor_A); // cpu
  tensor_A.copy_(tensor_B); // cpu

  tHabana_r = torch::relu(tHabana_A);
  HbLazyTensor::StepMarker({});

  tHabana_A.copy_(tHabana_B);

  cpu_tensor = tHabana_A.cpu();
  EXPECT_EQ(tensor_A.allclose(cpu_tensor, 0, 0), true);

  PT_TEST_DEBUG("tensor::", cpu_tensor[0][0]);
}

TEST(TestStream, TestWAR_multistream) {
  habana::HABANAGuardImpl device_guard;
  device_guard.getDevice();
  auto& device = habana::HPURegistrar::get_device();
  auto num_hpus = device.get_count_by_current_type();
  if (num_hpus == 0)
    return;

  c10::hpu::HPUStream default_s = c10::hpu::getDefaultHPUStream();
  c10::hpu::HPUStream compute1 = c10::hpu::getStreamFromPool();

  torch::Tensor tensor_A = torch::randn({2000, 3000});
  torch::Tensor tensor_B = torch::randn({2000, 3000});
  torch::Tensor tensor_r = torch::relu(tensor_A);

  torch::Tensor tHabana_A = tensor_A.to(torch::kHPU);
  torch::Tensor tHabana_B = tensor_B.to(torch::kHPU);

  tensor_A.copy_(tensor_B);
  torch::Tensor tHabana_r = torch::relu(tHabana_A);
  HbLazyTensor::StepMarker({});

  tHabana_A.copy_(tHabana_B);
  torch::Tensor cpu_tensor = tHabana_A.cpu();
  EXPECT_EQ(tensor_A.allclose(cpu_tensor, 0, 0), true);

  tensor_r = torch::relu(tensor_A); // cpu
  tensor_A.copy_(tensor_B); // cpu

  tHabana_r = torch::relu(tHabana_A);

  c10::hpu::setCurrentHPUStream(compute1);

  tHabana_A.copy_(tHabana_B);

  cpu_tensor = tHabana_A.cpu();
  EXPECT_EQ(tensor_A.allclose(cpu_tensor, 0, 0), true);

  PT_TEST_DEBUG("tensor::", cpu_tensor[0][0]);
}

TEST(TestStream, record_stream) {
  habana::HABANAGuardImpl device_guard;
  device_guard.getDevice();
  auto& device = habana::HPURegistrar::get_device().syn_device();
  void* ptr;
  c10::hpu::HPUStream default_s = c10::hpu::getDefaultHPUStream();
  device.get_device_memory().malloc(
      reinterpret_cast<void**>(&ptr), 104857600, default_s.stream());
  c10::hpu::HPUStream compute1 = c10::hpu::getStreamFromPool();
  device.get_device_memory().recordStream(ptr, compute1.stream());
  device.get_device_memory().free(ptr);
}

TEST(TestStream, ExternalTest) {
  habana::HABANAGuardImpl device_guard;
  device_guard.getDevice();

  synapse_helpers::hpuStream_t hpu_stream;
  hpu_stream = c10::hpu::getStreamFromPool();
  c10::hpu::HPUStream myStream = c10::hpu::getStreamByStreamPtr(hpu_stream, 0);

  c10::hpu::setCurrentHPUStream(myStream);
  c10::hpu::HPUStream curStream = c10::hpu::getCurrentHPUStream();
  ASSERT_EQ(curStream, myStream);
  ASSERT_EQ(curStream.stream(), hpu_stream);
}
