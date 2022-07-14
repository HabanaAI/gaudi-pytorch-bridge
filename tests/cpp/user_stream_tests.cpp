/******************************************************************************
 * Copyright (C) 2022 Habana Labs, Ltd. an Intel Company
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
#include "habana_kernels/eager_kernels_declarations.h"
#include "habana_kernels/lazy_kernels_declarations.h"
#include "habana_kernels/linear_kernels.h"
#include "habana_lazy/hpu_lazy_tensors.h"
#include "habana_lazy/ir_utils.h"
#include "habana_lazy_test_infra.h"
#include "synapse_helpers/env_flags.h"

#include <functional>
#include <future>
#include <thread>
#include <unordered_set>

#include "pytorch_helpers/habana_device/HPUGuardImpl.h"
#include "pytorch_helpers/habana_device/HPUStream.h"
#include "pytorch_helpers/habana_helpers/logging.h"
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
  auto& device = synapse_helpers::HPURegistrar::get_device();
  auto num_hpus = device.get_count_by_current_type();
  if (num_hpus == 0)
    return;
  int32_t device_id = -1;
  synapse_helpers::hpuStream_t hpu_stream;

  // Tests that copying works as expected and preserves the stream
  at::hpu::HPUStream copyStream = at::hpu::getStreamFromPool();
  {
    auto s = at::hpu::getStreamFromPool();
    device_id = s.device_index();
    hpu_stream = s.stream();

    copyStream = s;

    ASSERT_EQ_HPU(copyStream.device_index(), device_id);
    ASSERT_EQ_HPU(copyStream.stream(), hpu_stream);
  }

  ASSERT_EQ_HPU(copyStream.device_index(), device_id);
  ASSERT_EQ_HPU(copyStream.stream(), hpu_stream);

  // Tests that moving works as expected and preserves the stream
  at::hpu::HPUStream moveStream = at::hpu::getStreamFromPool();
  {
    auto s = at::hpu::getStreamFromPool();
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
  auto& device = synapse_helpers::HPURegistrar::get_device();
  auto num_hpus = device.get_count_by_current_type();
  if (num_hpus == 0)
    return;
  at::hpu::HPUStream myStream = at::hpu::getStreamFromPool();

  // Sets and gets
  at::hpu::setCurrentHPUStream(myStream);
  at::hpu::HPUStream curStream = at::hpu::getCurrentHPUStream();

  ASSERT_EQ_HPU(myStream, curStream);

  // Gets, sets, and gets default stream
  at::hpu::HPUStream defaultStream = at::hpu::getDefaultHPUStream();
  at::hpu::setCurrentHPUStream(defaultStream);
  curStream = at::hpu::getCurrentHPUStream();

  ASSERT_NE_HPU(defaultStream, myStream);
  ASSERT_EQ_HPU(curStream, defaultStream);
}

void thread_fun(at::optional<at::hpu::HPUStream>& cur_thread_stream) {
  auto new_stream = at::hpu::getStreamFromPool();
  at::hpu::HPUStream cur_stream = at::hpu::getCurrentHPUStream();
  at::hpu::HPUStream default_stream = at::hpu::getDefaultHPUStream();
  at::hpu::setCurrentHPUStream(new_stream);
  cur_thread_stream = {at::hpu::getCurrentHPUStream()};
  ASSERT_EQ_HPU(*cur_thread_stream, new_stream);
}

// Ensures streams are thread local
TEST(TestStream, DISABLED_MultithreadGetAndSetTest) {
  habana::HABANAGuardImpl device_guard;
  device_guard.getDevice();
  auto& device = synapse_helpers::HPURegistrar::get_device();
  auto num_hpus = device.get_count_by_current_type();
  if (num_hpus == 0)
    return;
  at::optional<at::hpu::HPUStream> s0, s1;

  std::thread t0{thread_fun, std::ref(s0)};
  std::thread t1{thread_fun, std::ref(s1)};
  t0.join();
  t1.join();

  at::hpu::HPUStream cur_stream = at::hpu::getCurrentHPUStream();
  at::hpu::HPUStream default_stream = at::hpu::getDefaultHPUStream();

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

// Streampool Round Robin
TEST(TestStream, StreamPoolTest) {
  habana::HABANAGuardImpl device_guard;
  device_guard.getDevice();
  auto& device = synapse_helpers::HPURegistrar::get_device();
  auto num_hpus = device.get_count_by_current_type();
  if (num_hpus == 0)
    return;
  std::vector<at::hpu::HPUStream> streams{};
  for (const auto i : c10::irange(200)) {
    (void)i;
    streams.emplace_back(at::hpu::getStreamFromPool());
  }

  std::unordered_set<synapse_helpers::hpuStream_t> stream_set{};
  bool hasDuplicates = false;
  for (const auto i : c10::irange(streams.size())) {
    synapse_helpers::hpuStream_t hpu_stream = streams[i];
    auto result_pair = stream_set.insert(hpu_stream);
    if (!result_pair.second)
      hasDuplicates = true;
  }

  ASSERT_TRUE(hasDuplicates);
}

TEST(TestStream, DISABLED_Use2StreamForadd) {
  habana::HABANAGuardImpl device_guard;
  device_guard.getDevice();
  auto& device = synapse_helpers::HPURegistrar::get_device();
  auto num_hpus = device.get_count_by_current_type();
  if (num_hpus == 0)
    return;

  at::hpu::HPUStream compute1 = at::hpu::getStreamFromPool();
  at::hpu::HPUStream compute2 = at::hpu::getStreamFromPool();

  /*default stream */
  torch::Tensor tensor_D = torch::randn({200, 300});
  torch::Tensor tHabana_D = tensor_D.to(torch::kHPU);
  auto outHabana_D = torch::add(tHabana_D, 4.0);

  at::hpu::setCurrentHPUStream(compute1);
  torch::Tensor tensor_A = torch::randn({200, 300});
  torch::Tensor tensor_B = torch::randn({200, 300});
  torch::Tensor tHabana_A = tensor_A.to(torch::kHPU);

  at::hpu::setCurrentHPUStream(compute1);
  auto outHabana_A = torch::add(tHabana_A, 4.0);

  torch::Tensor tHabana_B = tensor_B.to(torch::kHPU);
  at::hpu::setCurrentHPUStream(compute2);
  auto outHabana_B = torch::add(tHabana_B, 4.0);

  auto out_A = torch::add(tensor_A, 4.0);
  auto out_B = torch::add(tensor_B, 4.0);
  auto out_D = torch::add(tensor_D, 4.0);
  bool equal = out_A.allclose(outHabana_A.to(torch::kCPU), 0, 0);
  EXPECT_EQ(equal, true);
  equal = out_B.allclose(outHabana_B.to(torch::kCPU), 0, 0);
  EXPECT_EQ(equal, true);
  equal = out_D.allclose(outHabana_D.to(torch::kCPU), 0, 0);
  EXPECT_EQ(equal, true);
}

TEST(TestStream, ForceUseDefaultStream) {
  habana::HABANAGuardImpl device_guard;
  device_guard.getDevice();
  auto& device = synapse_helpers::HPURegistrar::get_device();
  auto num_hpus = device.get_count_by_current_type();
  if (num_hpus == 0)
    return;

  SET_ENV_FLAG_NEW(PT_HPU_FORCE_USE_DEFAULT_STREAM, true, 1);
  at::hpu::HPUStream compute1 = at::hpu::getStreamFromPool();
  at::hpu::HPUStream compute2 = at::hpu::getStreamFromPool();

  /*default stream */
  torch::Tensor tensor_D = torch::randn({200, 300});
  torch::Tensor tHabana_D = tensor_D.to(torch::kHPU);
  auto outHabana_D = torch::add(tHabana_D, 4.0);

  at::hpu::setCurrentHPUStream(compute1);
  torch::Tensor tensor_A = torch::randn({200, 300});
  torch::Tensor tensor_B = torch::randn({200, 300});
  torch::Tensor tHabana_A = tensor_A.to(torch::kHPU);

  at::hpu::setCurrentHPUStream(compute1);
  auto outHabana_A = torch::add(tHabana_A, 4.0);

  torch::Tensor tHabana_B = tensor_B.to(torch::kHPU);
  at::hpu::setCurrentHPUStream(compute2);
  auto outHabana_B = torch::add(tHabana_B, 4.0);

  auto out_A = torch::add(tensor_A, 4.0);
  auto out_B = torch::add(tensor_B, 4.0);
  auto out_D = torch::add(tensor_D, 4.0);
  bool equal = out_A.allclose(outHabana_A.to(torch::kCPU), 0, 0);
  EXPECT_EQ(equal, true);
  equal = out_B.allclose(outHabana_B.to(torch::kCPU), 0, 0);
  EXPECT_EQ(equal, true);
  equal = out_D.allclose(outHabana_D.to(torch::kCPU), 0, 0);
  EXPECT_EQ(equal, true);
  UNSET_ENV_FLAG_NEW(PT_HPU_FORCE_USE_DEFAULT_STREAM);
  std::cout << "unset flag" << GET_ENV_FLAG_NEW(PT_HPU_FORCE_USE_DEFAULT_STREAM)
            << std::endl;
}

void thread_fun_add(bool& result) {
  auto new_stream = at::hpu::getStreamFromPool();
  at::hpu::setCurrentHPUStream(new_stream);
  torch::Tensor tensor_A = torch::randn({200, 300});
  torch::Tensor tHabana_A = tensor_A.to(torch::kHPU);
  auto outHabana_A = torch::add(tHabana_A, 4.0);
  auto out_A = torch::add(tensor_A, 4.0);
  bool equal = out_A.allclose(outHabana_A.to(torch::kCPU), 0, 0);
  result = equal;
}

TEST(TestStream, DISABLED_MultithreadStreamAddOP) {
  habana::HABANAGuardImpl device_guard;
  device_guard.getDevice();
  auto& device = synapse_helpers::HPURegistrar::get_device();
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
  auto new_stream = at::hpu::getStreamFromPool();
  at::hpu::setCurrentHPUStream(new_stream);
  outtensor = torch::add(in_tensor, 4.0);
  HbLazyTensor::StepMarker({});
}

// Ensures streams are thread local
TEST(TestStream, DISABLED_MultithreadStreamKernelAdd) {
  habana::HABANAGuardImpl device_guard;
  device_guard.getDevice();
  auto& device = synapse_helpers::HPURegistrar::get_device();
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
  bool equal1 = out_A.allclose(outHabana_A.to(torch::kCPU), 0, 0);
  bool equal2 = out_B.allclose(outHabana_B.to(torch::kCPU), 0, 0);
  EXPECT_EQ(equal1, true);
  EXPECT_EQ(equal2, true);
}

TEST(TestStream, TestStreamQuery) {
  habana::HABANAGuardImpl device_guard;
  device_guard.getDevice();
  auto& device = synapse_helpers::HPURegistrar::get_device();
  auto num_hpus = device.get_count_by_current_type();
  if (num_hpus == 0)
    return;

  at::hpu::HPUStream compute1 = at::hpu::getStreamFromPool();
  at::hpu::HPUStream compute2 = at::hpu::getStreamFromPool();

  PT_TEST_DEBUG("stream query for stream1", compute1.query());
  at::hpu::setCurrentHPUStream(compute1);
  torch::Tensor tensor_A = torch::randn({200, 300});
  torch::Tensor tensor_B = torch::randn({200, 300});
  torch::Tensor tHabana_A = tensor_A.to(torch::kHPU);
  at::hpu::setCurrentHPUStream(compute1);
  auto outHabana_A = torch::add(tHabana_A, 4.0);
  PT_TEST_DEBUG("stream query for stream1", compute1.query());
  torch::Tensor tHabana_B = tensor_B.to(torch::kHPU);
  at::hpu::setCurrentHPUStream(compute2);
  auto outHabana_B = torch::add(tHabana_B, 4.0);
  PT_TEST_DEBUG("stream query for stream2", compute2.query());
  auto out_A = torch::add(tensor_A, 4.0);
  auto out_B = torch::add(tensor_B, 4.0);
  bool equal = out_A.allclose(outHabana_A.to(torch::kCPU), 0, 0);
  EXPECT_EQ(equal, true);
  equal = out_B.allclose(outHabana_B.to(torch::kCPU), 0, 0);
  EXPECT_EQ(equal, true);
  PT_TEST_DEBUG("stream query for stream1", compute1.query());
}

TEST(TestStream, TestStreamSynchronize) {
  habana::HABANAGuardImpl device_guard;
  device_guard.getDevice();
  auto& device = synapse_helpers::HPURegistrar::get_device();
  auto num_hpus = device.get_count_by_current_type();
  if (num_hpus == 0)
    return;

  at::hpu::HPUStream compute1 = at::hpu::getStreamFromPool();
  at::hpu::HPUStream compute2 = at::hpu::getStreamFromPool();

  at::hpu::setCurrentHPUStream(compute1);
  torch::Tensor tensor_A = torch::randn({200, 300});
  torch::Tensor tensor_B = torch::randn({200, 300});
  torch::Tensor tHabana_A = tensor_A.to(torch::kHPU);
  at::hpu::setCurrentHPUStream(compute1);
  auto outHabana_A = torch::add(tHabana_A, 4.0);
  compute1.synchronize();
  torch::Tensor tHabana_B = tensor_B.to(torch::kHPU);
  at::hpu::setCurrentHPUStream(compute2);
  auto outHabana_B = torch::add(tHabana_B, 4.0);
  compute2.synchronize();
  auto out_A = torch::add(tensor_A, 4.0);
  auto out_B = torch::add(tensor_B, 4.0);
  bool equal = out_A.allclose(outHabana_A.to(torch::kCPU), 0, 0);
  EXPECT_EQ(equal, true);
  equal = out_B.allclose(outHabana_B.to(torch::kCPU), 0, 0);
  EXPECT_EQ(equal, true);
  compute2.synchronize();
}
