/*******************************************************************************
 * Copyright (C) 2021-2023 Habana Labs, Ltd. an Intel Company
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

#include <absl/types/variant.h>
#include <gtest/gtest.h>
#include <synapse_api_types.h>
#include <synapse_common_types.h>
#include <torch/torch.h>
#include <algorithm>
#include <memory>
#include "backend/habana_device/HPUAllocator.h"
#include "backend/habana_device/HPUGuardImpl.h"
#include "backend/habana_device/hpu_cached_devices.h"
#include "backend/habana_device/tensor_builder.h"
#include "backend/helpers/tensor_utils.h"
#include "backend/synapse_helpers/habana_tensor.h"
#include "backend/synapse_helpers/synapse_error.h"
#include "habana_lazy/hpu_stage_submission.h"
#include "habana_lazy/lazy_executor.h"
#include "utils/check_device_type.h"

using namespace synapse_helpers;
class SynapseHelpersMemoryTest : public ::testing::Test {
  void SetUp() override {
    habana::HABANAGuardImpl device_guard;
    device_guard.getDevice();
    auto& device = habana::HPURegistrar::get_device().syn_device();
    // clear cache scalar tensors map
    setenv("PT_HPU_CLEAR_SCALAR_MAP_ON_MARKSTEP", "1", 1);
    habana_lazy::HbExecutionContext* context =
        habana_lazy::habana_lazy_executor.getDeviceExecutionContext(
            device.id());
    context->clear();
    // set up defragment, pool and 3gb pool for testing
    setenv("PT_HABANA_POOL_SIZE", "3", 1);
    setenv("PT_ENABLE_MEMORY_DEFRAGMENTATION", "true", 1);
    setenv("PT_HPU_POOL_STRATEGY", "5", 1);
    device.cleanup_workspace_buffer();
    device.get_device_memory().reset_pool();

    habana_lazy::StageSubmission::getInstance().resetCurrentAccumulatedOps();
  }

  void TearDown() override {
    // reset the pool when test is done to cleanup the pool
    unsetenv("PT_HABANA_POOL_SIZE");
    unsetenv("PT_HPU_POOL_STRATEGY");
    unsetenv("PT_ENABLE_MEMORY_DEFRAGMENTATION");
    unsetenv("PT_HPU_CLEAR_SCALAR_MAP_ON_MARKSTEP");
    auto& device = habana::HPURegistrar::get_device().syn_device();
    device.cleanup_workspace_buffer();
    device.get_device_memory().reset_pool();
  }
};

TEST_F(SynapseHelpersMemoryTest, degframentonOOM_1) {
  auto& device = habana::HPURegistrar::get_device().syn_device();
  // allocate workspace buffer 1gb
  device.get_workspace_buffer(1960834120);
  // allocate 5 varaibles of size 200 MB
  void* ptr1{nullptr};
  device.get_device_memory().malloc(&ptr1, 209715200);
  void* ptr2{nullptr};
  device.get_device_memory().malloc(&ptr2, 209715200);
  void* ptr3{nullptr};
  device.get_device_memory().malloc(&ptr3, 209715200);
  void* ptr4{nullptr};
  device.get_device_memory().malloc(&ptr4, 209715200);
  void* ptr5{nullptr};
  device.get_device_memory().malloc(&ptr5, 209715200);
  void* ptr6{nullptr};
  device.get_device_memory().malloc(&ptr6, 209715200);
  PT_TEST_DEBUG("getting device address");
  std::vector<device_ptr> address;
  address.push_back((uint64_t)ptr1);
  address.push_back((uint64_t)ptr2);
  address.push_back((uint64_t)ptr3);
  address.push_back((uint64_t)ptr4);
  address.push_back((uint64_t)ptr5);
  address.push_back((uint64_t)ptr6);
  device.lock_addresses(address);
  address.clear();
  PT_TEST_DEBUG("Done getting device address");
  // delete 2 & 4 varaible
  device.get_device_memory().free(ptr2);
  device.get_device_memory().free(ptr4);

  // allocate 400 MB memory this willl lead OOM and defragmentor will kick in
  // here
  void* ptr_400mb{nullptr};
  device.get_device_memory().malloc(&ptr_400mb, 419430400);
  address.push_back((uint64_t)ptr_400mb);
  device.lock_addresses(address);

  device.get_device_memory().free(ptr1);
  device.get_device_memory().free(ptr3);
  device.get_device_memory().free(ptr5);
  device.get_device_memory().free(ptr6);
  device.get_device_memory().free(ptr_400mb);
}

TEST_F(SynapseHelpersMemoryTest, degframentonOOM_2) {
  auto& device = habana::HPURegistrar::get_device().syn_device();
  // allocate workspace buffer 1gb
  device.get_workspace_buffer(1960834120);
  // allocate 5 varaibles of size 200 MB
  void* ptr1{nullptr};
  device.get_device_memory().malloc(&ptr1, 209715200);
  void* ptr2{nullptr};
  device.get_device_memory().malloc(&ptr2, 104857600);
  void* ptr3{nullptr};
  device.get_device_memory().malloc(&ptr3, 209715200);
  void* ptr4{nullptr};
  device.get_device_memory().malloc(&ptr4, 209715200);
  void* ptr5{nullptr};
  device.get_device_memory().malloc(&ptr5, 209715200);
  void* ptr6{nullptr};
  device.get_device_memory().malloc(&ptr6, 209715200);
  void* ptr7{nullptr};
  device.get_device_memory().malloc(&ptr7, 104857600);
  PT_TEST_DEBUG("getting device address");
  std::vector<device_ptr> address;
  address.push_back((uint64_t)ptr1);
  address.push_back((uint64_t)ptr2);
  address.push_back((uint64_t)ptr3);
  address.push_back((uint64_t)ptr4);
  address.push_back((uint64_t)ptr5);
  address.push_back((uint64_t)ptr6);
  address.push_back((uint64_t)ptr7);
  device.lock_addresses(address);
  address.clear();
  // delete 2 & 4 varaible
  device.get_device_memory().free(ptr2);
  device.get_device_memory().free(ptr6);
  device.get_device_memory().free(ptr7);

  // allocate 400 MB memory this willl lead OOM and defragmentor will kick in
  // here
  void* ptr_400mb{nullptr};
  device.get_device_memory().malloc(&ptr_400mb, 419430400);
  address.push_back((uint64_t)ptr_400mb);
  device.lock_addresses(address);

  device.get_device_memory().free(ptr1);
  device.get_device_memory().free(ptr3);
  device.get_device_memory().free(ptr5);
  device.get_device_memory().free(ptr4);
  device.get_device_memory().free(ptr_400mb);
}

TEST_F(SynapseHelpersMemoryTest, degframentonOOM_3) {
  auto& device = habana::HPURegistrar::get_device().syn_device();
  int num_mem_blk = 0;
  // Add required test pattern of required length.
  std::set<std::pair<int, std::vector<int>>> test_set{
      {7,
       {209715200,
        104857600,
        209715200,
        209715200,
        209715200,
        104857600,
        209715200}},
      {7,
       {209715200,
        104857600,
        209715200,
        209715200,
        209715200,
        209715200,
        104857600}}};

  // Add required patterns of pointers to create free blocks.
  std::vector<std::vector<int>> free_set{{1, 3, 5}, {1, 4, 6}};
  int k = -1;

  // allocate workspace buffer 1gb
  device.get_workspace_buffer(1960834120);
  // allocate 5 varaibles of size 200 MB
  for (auto mem_info : test_set) {
    num_mem_blk = mem_info.first;
    std::vector<int> blksize = mem_info.second;
    void* ptr[num_mem_blk];
    k++;
    for (int i = 0; i < num_mem_blk; i++) {
      ptr[i] = nullptr;
      device.get_device_memory().malloc(&ptr[i], blksize[i]);
    }

    PT_TEST_DEBUG("getting device address");
    std::vector<device_ptr> address;
    for (int i = 0; i < num_mem_blk; i++) {
      address.push_back((uint64_t)ptr[i]);
    }
    device.lock_addresses(address);
    address.clear();
    // delete 2 & 4 varaible
    std::vector<int> freeblks = free_set[k];

    device.get_device_memory().free(ptr[freeblks[0]]);
    device.get_device_memory().free(ptr[freeblks[1]]);
    device.get_device_memory().free(ptr[freeblks[2]]);

    // allocate 400 MB memory this willl lead OOM and defragmentor will kick in
    // here
    void* ptr_400mb{nullptr};
    device.get_device_memory().malloc(&ptr_400mb, 419430400);
    address.push_back((uint64_t)ptr_400mb);
    device.lock_addresses(address);

    for (int i = 0; i < num_mem_blk; i++) {
      if (i == freeblks[0] || i == freeblks[1] || i == freeblks[2])
        continue;

      device.get_device_memory().free(ptr[i]);
    }
    device.get_device_memory().free(ptr_400mb);
  }
}

TEST_F(SynapseHelpersMemoryTest, degframentonOOMWithWS) {
  auto& device = habana::HPURegistrar::get_device().syn_device();
  // allocate workspace buffer 1gb
  device.get_workspace_buffer(1960834120);
  // allocate 5 varaibles of size 200 MB
  void* ptr1{nullptr};
  device.get_device_memory().malloc(&ptr1, 209715200);
  void* ptr2{nullptr};
  device.get_device_memory().malloc(&ptr2, 209715200);
  void* ptr3{nullptr};
  device.get_device_memory().malloc(&ptr3, 209715200);
  void* ptr4{nullptr};
  device.get_device_memory().malloc(&ptr4, 209715200);
  void* ptr5{nullptr};
  device.get_device_memory().malloc(&ptr5, 209715200);
  void* ptr6{nullptr};
  device.get_device_memory().malloc(&ptr6, 209715200);
  PT_TEST_DEBUG("getting device address");
  std::vector<device_ptr> address;
  address.push_back((uint64_t)ptr1);
  address.push_back((uint64_t)ptr2);
  address.push_back((uint64_t)ptr3);
  address.push_back((uint64_t)ptr4);
  address.push_back((uint64_t)ptr5);
  address.push_back((uint64_t)ptr6);
  device.lock_addresses(address);
  address.clear();
  // delete ptr2 of size 200 MB
  device.get_device_memory().free(ptr2);

  // extend the ws by 200MB memory this willl lead OOM and defragmentor will
  // kick in here
  device.get_workspace_buffer(2170549320);

  PT_TEST_DEBUG("done with device memory extension")
  device.get_device_memory().free(ptr1);
  device.get_device_memory().free(ptr3);
  device.get_device_memory().free(ptr4);
  device.get_device_memory().free(ptr5);
  device.get_device_memory().free(ptr6);
}

// we can test when we have handling in smalalloc for equal to 256 size. so
// disabling it for now
TEST_F(SynapseHelpersMemoryTest, DISABLED_degframentonOOMWithSmallAlloc) {
  auto& device = habana::HPURegistrar::get_device().syn_device();
  // Fill up the entire space expect the small alloc region
  // allocate workspace buffer 1.06gb
  device.get_workspace_buffer(1143820277);
  // allocate 5 varaibles of size 200 MB
  void* ptr1{nullptr};
  device.get_device_memory().malloc(&ptr1, 209715200);
  void* ptr2{nullptr};
  device.get_device_memory().malloc(&ptr2, 209715200);
  void* ptr3{nullptr};
  device.get_device_memory().malloc(&ptr3, 209715200);
  void* ptr4{nullptr};
  device.get_device_memory().malloc(&ptr4, 209715200);
  void* ptr5{nullptr};
  device.get_device_memory().malloc(&ptr5, 209715200);
  void* ptr6{nullptr};
  device.get_device_memory().malloc(&ptr6, 209715200);
  PT_TEST_DEBUG("getting device address");
  std::vector<device_ptr> address;
  address.push_back((uint64_t)ptr1);
  address.push_back((uint64_t)ptr2);
  address.push_back((uint64_t)ptr3);
  address.push_back((uint64_t)ptr4);
  address.push_back((uint64_t)ptr5);
  address.push_back((uint64_t)ptr6);
  device.lock_addresses(address);
  address.clear();

  // allocate 16348 bytes of 128 bytes block
  std::map<int, void*> mem_ptr;
  for (int i = 0; i < 16384; i++) {
    void* ptr1{nullptr};
    device.get_device_memory().malloc(&ptr1, 128);
    mem_ptr[i] = ptr1;
    std::vector<device_ptr> address;
    address.push_back((uint64_t)ptr1);
    device.lock_addresses(address);
    address.clear();
  }

  // free few of the 128 bytes to create holes
  device.get_device_memory().free(mem_ptr[2]);
  device.get_device_memory().free(mem_ptr[12]);

  // allocate 256 bytes memory request(this willl lead OOM and defragmentor will
  // kick in here)
  void* ptr_256bytes{nullptr};
  device.get_device_memory().malloc(&ptr_256bytes, 128);
  address.push_back((uint64_t)ptr_256bytes);
  mem_ptr[12] = ptr_256bytes;
  device.lock_addresses(address);

  for (int i = 0; i < 16348; i++) {
    if (i != 2)
      device.get_device_memory().free(mem_ptr[i]);
  }
  device.get_device_memory().free(ptr1);
  device.get_device_memory().free(ptr2);
  device.get_device_memory().free(ptr3);
  device.get_device_memory().free(ptr4);
  device.get_device_memory().free(ptr5);
  device.get_device_memory().free(ptr6);
}

TEST_F(SynapseHelpersMemoryTest, degframentonOOMandVerify_1) {
  // Test should not be run on the simulator as verification part (data transfer
  // to/from device) takes too much time
  if (is_simulator()) {
    GTEST_SKIP();
  }
  auto& device = habana::HPURegistrar::get_device().syn_device();
  // allocate workspace buffer 1gb
  device.get_workspace_buffer(1960834120);
  // allocate 5 varaibles of size 200 MB and copy the src content
  void* dsts[6];
  void* srcs[6];
  for (int i = 0; i < 6; i++) {
    std::atomic<bool> copyDone{false};
    PT_TEST_DEBUG("copy data to device...");
    srcs[i] = (malloc(209715200));
    void* ptr{nullptr};
    device.get_device_memory().malloc(&ptr, 209715200);
    dsts[i] = ptr;
    memset(srcs[i], i + 1, sizeof(209715200));
    auto syn_error = device.copy_data_to_device(
        srcs[i],
        reinterpret_cast<synapse_helpers::device_ptr>(dsts[i]),
        reinterpret_cast<synapse_helpers::device_ptr>(dsts[i]),
        209715200,
        [&copyDone]() { copyDone = true; },
        false);
    TORCH_CHECK(syn_error.status == 0, syn_error.error);
    // wait for copy completion
    while (!copyDone) {
      std::this_thread::yield();
    }
  }

  // delete 2 & 4 varaible
  device.get_device_memory().free(dsts[2]);
  device.get_device_memory().free(dsts[4]);
  free(srcs[2]);
  free(srcs[4]);

  // allocate 400 MB memory this willl lead OOM and defragmentor will kick in
  // here
  void* ptr_400mb{nullptr};
  void* src_400mb = malloc(419430400);
  {
    device.get_device_memory().malloc(&ptr_400mb, 419430400);
    memset(src_400mb, 0xf, sizeof(419430400));
    std::atomic<bool> copyDone{false};
    auto syn_error = device.copy_data_to_device(
        src_400mb,
        reinterpret_cast<synapse_helpers::device_ptr>(ptr_400mb),
        reinterpret_cast<synapse_helpers::device_ptr>(ptr_400mb),
        419430400,
        [&copyDone]() { copyDone = true; },
        false);
    TORCH_CHECK(syn_error.status == 0, syn_error.error);
    // wait for copy completion
    while (!copyDone) {
      std::this_thread::yield();
    }
  }

  /*compare the moved data */
  void* src_hpu = malloc(209715200);
  for (int i = 0; i < 6; i++) {
    std::atomic<bool> copyDone{false};
    memset(src_hpu, 0, sizeof(209715200));
    // dont copy for delete ptr
    if (i == 2 || i == 4)
      continue;
    auto syn_error = device.copy_data_to_host(
        reinterpret_cast<synapse_helpers::device_ptr>(dsts[i]),
        src_hpu,
        reinterpret_cast<synapse_helpers::device_ptr>(dsts[i]),
        209715200,
        [&copyDone]() { copyDone = true; },
        false);
    TORCH_CHECK(syn_error.status == 0, syn_error.error);
    // wait for copy completion
    while (!copyDone) {
      std::this_thread::yield();
    }
    int n = std::memcmp(srcs[i], src_hpu, 209715200);
    EXPECT_TRUE((n == 0));
  }
  free(src_hpu);
  // compare the last 400mb copy
  src_hpu = malloc(419430400);
  {
    std::atomic<bool> copyDone{false};
    auto syn_error = device.copy_data_to_host(
        reinterpret_cast<synapse_helpers::device_ptr>(ptr_400mb),
        src_hpu,
        reinterpret_cast<synapse_helpers::device_ptr>(ptr_400mb),
        419430400,
        [&copyDone]() { copyDone = true; },
        false);
    TORCH_CHECK(syn_error.status == 0, syn_error.error);
    // wait for copy completion
    while (!copyDone) {
      std::this_thread::yield();
    }
  }
  int n = std::memcmp(src_400mb, src_hpu, 419430400);
  EXPECT_TRUE((n == 0));

  free(src_hpu);
  for (int i = 0; i < 6; i++) {
    // dont copy for delete ptr
    if (i == 2 || i == 4)
      continue;
    device.get_device_memory().free(dsts[i]);
    free(srcs[i]);
  }
  device.get_device_memory().free(ptr_400mb);
  free(src_400mb);
}

TEST_F(SynapseHelpersMemoryTest, degframentonOOMandVerify_2) {
  // Test should not be run on the simulator as verification part (data transfer
  // to/from device) takes too much time
  if (is_simulator()) {
    GTEST_SKIP();
  }
  auto& device = habana::HPURegistrar::get_device().syn_device();
  int num_mem_blk = 0;
  // Add required test pattern of required length.
  std::set<std::pair<int, std::vector<int>>> test_set{
      {7,
       {209715200,
        104857600,
        209715200,
        209715200,
        209715200,
        104857600,
        209715200}},
      {7,
       {209715200,
        104857600,
        209715200,
        209715200,
        209715200,
        209715200,
        104857600}}};

  // Add required patterns of pointers to create free blocks.
  std::vector<std::vector<int>> free_set{{1, 3, 5}, {1, 4, 6}};
  void* src_hpu = malloc(104857600);
  void* src1_hpu = malloc(209715200);
  int k = -1;

  // allocate workspace buffer 1gb
  device.get_workspace_buffer(1960834120);
  // allocate 5 varaibles of size 200 MB and copy the src content
  for (auto mem_info : test_set) {
    num_mem_blk = mem_info.first;
    std::vector<int> blksize = mem_info.second;
    void* dsts[num_mem_blk];
    void* srcs[num_mem_blk];
    synapse_error* err;
    k++;
    for (int i = 0; i < num_mem_blk; i++) {
      std::atomic<bool> copyDone{false};
      PT_TEST_DEBUG("copy data to device...");
      srcs[i] = malloc(blksize[i]);
      void* ptr{nullptr};
      device.get_device_memory().malloc(&ptr, blksize[i]);
      dsts[i] = ptr;
      memset(srcs[i], i + 1, sizeof(blksize[i]));
      auto syn_error = device.copy_data_to_device(
          srcs[i],
          reinterpret_cast<synapse_helpers::device_ptr>(dsts[i]),
          reinterpret_cast<synapse_helpers::device_ptr>(dsts[i]),
          blksize[i],
          [&copyDone]() { copyDone = true; },
          false);
      err = &syn_error;

      TORCH_CHECK(err->status == 0, err->error);
      // wait for copy completion
      while (!copyDone) {
        std::this_thread::yield();
      }
    }
    std::vector<int> freeblks = free_set[k];
    // delete x & y varaible and some j

    device.get_device_memory().free(dsts[freeblks[0]]);
    device.get_device_memory().free(dsts[freeblks[1]]);
    device.get_device_memory().free(dsts[freeblks[2]]);
    free(srcs[freeblks[0]]);
    free(srcs[freeblks[1]]);
    free(srcs[freeblks[2]]);

    // allocate 400 MB memory this willl lead OOM and defragmentor will kick
    // in here
    void* ptr_400mb{nullptr};
    void* src_400mb = malloc(419430400);
    {
      device.get_device_memory().malloc(&ptr_400mb, 419430400);
      memset(src_400mb, 0xf, sizeof(419430400));
      std::atomic<bool> copyDone{false};
      auto syn_error = device.copy_data_to_device(
          src_400mb,
          reinterpret_cast<synapse_helpers::device_ptr>(ptr_400mb),
          reinterpret_cast<synapse_helpers::device_ptr>(ptr_400mb),
          419430400,
          [&copyDone]() { copyDone = true; },
          false);
      err = &syn_error;
      TORCH_CHECK(err->status == 0, err->error);
      // wait for copy completion
      while (!copyDone) {
        std::this_thread::yield();
      }
    }

    /*compare the moved data */
    void* src_hpu = malloc(104857600);
    void* src1_hpu = malloc(209715200);

    for (int i = 0; i < num_mem_blk; i++) {
      std::atomic<bool> copyDone{false};
      memset(malloc(blksize[i]), 0, sizeof(blksize[i]));

      // dont copy for delete ptr
      if (i == freeblks[0] || i == freeblks[1] || i == freeblks[2])
        continue;

      auto syn_error = device.copy_data_to_host(
          reinterpret_cast<synapse_helpers::device_ptr>(dsts[i]),
          src1_hpu,
          reinterpret_cast<synapse_helpers::device_ptr>(dsts[i]),
          209715200,
          [&copyDone]() { copyDone = true; },
          false);
      err = &syn_error;
      TORCH_CHECK(err->status == 0, err->error);
      // wait for copy completion
      while (!copyDone) {
        std::this_thread::yield();
      }
      int n = std::memcmp(srcs[i], src1_hpu, 209715200);
      EXPECT_TRUE((n == 0));
    }
    free(src_hpu);
    free(src1_hpu);
    // compare the last 400mb copy
    src_hpu = malloc(419430400);
    {
      std::atomic<bool> copyDone{false};
      auto syn_error = device.copy_data_to_host(
          reinterpret_cast<synapse_helpers::device_ptr>(ptr_400mb),
          src_hpu,
          reinterpret_cast<synapse_helpers::device_ptr>(ptr_400mb),
          419430400,
          [&copyDone]() { copyDone = true; },
          false);
      err = &syn_error;
      TORCH_CHECK(err->status == 0, err->error);
      // wait for copy completion
      while (!copyDone) {
        std::this_thread::yield();
      }
    }
    int n = std::memcmp(src_400mb, src_hpu, 419430400);
    EXPECT_TRUE((n == 0));

    free(src_hpu);
    // dont copy for delete ptr
    for (int i = 0; i < num_mem_blk; i++) {
      std::atomic<bool> copyDone{false};
      if (i == freeblks[0] || i == freeblks[1] || i == freeblks[2])
        continue;

      device.get_device_memory().free(dsts[i]);
      free(srcs[i]);
    }
    device.get_device_memory().free(ptr_400mb);
    free(src_400mb);
  }
}

TEST_F(SynapseHelpersMemoryTest, GenTest) {
  auto& device = habana::HPURegistrar::get_device().syn_device();
  // allocate workspace buffer 1gb
  device.get_workspace_buffer(1960834120);
  int x = 0, y = 0;

  void* small[16384];
  std::vector<device_ptr> address;
  for (int j = 0; j < 16384; j++) {
    device.get_device_memory().malloc(&small[j], 128);
    address.push_back((uint64_t)small[j]);
  }
  device.lock_addresses(address);
  address.clear();
  int free_index1 = 22;
  int free_index2 = 100;
  int free_index3 = 16383;
  device.get_device_memory().free(small[22]);
  device.get_device_memory().free(small[100]);
  device.get_device_memory().free(small[16383]);

  void* ptr[6];
  for (int j = 0; j < 4; j++) {
    for (int i = 0; i < 6; i++) {
      device.get_device_memory().malloc(&ptr[i], 209715200);
    }
    PT_TEST_DEBUG("getting device address");
    std::vector<device_ptr> address;
    for (int i = 0; i < 6; i++) {
      address.push_back((uint64_t)ptr[i]);
    }
    device.lock_addresses(address);
    address.clear();
    // delete two non consecutive varaible
    x = j;
    y = x + 2;
    if (y > 5)
      break;
    device.get_device_memory().free(ptr[x]);
    device.get_device_memory().free(ptr[y]);

    // allocate 400 MB memory this willl lead OOM and defragmentor will kick in
    // here
    void* ptr_400mb{nullptr};
    device.get_device_memory().malloc(&ptr_400mb, 419430400);
    address.push_back((uint64_t)ptr_400mb);
    device.lock_addresses(address);
    for (int i = 0; i < 6; i++) {
      if (i == x || i == y)
        continue;
      device.get_device_memory().free(ptr[i]);
    }
    device.get_device_memory().free(ptr_400mb);
  }
  for (int j = 0; j < 16384; j++) {
    if (!(j == free_index1 || j == free_index2 || j == free_index3))
      device.get_device_memory().free(small[j]);
  }
}

TEST_F(SynapseHelpersMemoryTest, OOM_FreeMemInEndofsmallallocRegion) {
  auto& device = habana::HPURegistrar::get_device().syn_device();
  // allocate workspace buffer 1gb
  device.get_workspace_buffer(1960834120);
  std::vector<device_ptr> address;
  void* small[16384];
  for (int j = 0; j < 16384; j++) {
    device.get_device_memory().malloc(&small[j], 128);
    address.push_back((uint64_t)small[j]);
  }
  device.lock_addresses(address);
  address.clear();
  int free_start_index = 122;
  for (int k = 0; k < 48; k++) {
    int index = free_start_index + k;
    device.get_device_memory().free(small[index]);
  }
  free_start_index = 173;
  for (int k = 0; k < 40; k++) {
    int index = free_start_index + k;
    device.get_device_memory().free(small[index]);
  }

  free_start_index = 214;
  for (int k = 0; k <= 16169; k++) {
    int index = free_start_index + k;
    device.get_device_memory().free(small[index]);
  }
  // allocate 5 varaibles of size 200 MB
  void* ptr1{nullptr};
  device.get_device_memory().malloc(&ptr1, 209715200);
  void* ptr2{nullptr};
  device.get_device_memory().malloc(&ptr2, 104857600);
  void* ptr3{nullptr};
  device.get_device_memory().malloc(&ptr3, 209715200);
  void* ptr4{nullptr};
  device.get_device_memory().malloc(&ptr4, 209715200);
  void* ptr5{nullptr};
  device.get_device_memory().malloc(&ptr5, 209715200);
  void* ptr6{nullptr};
  device.get_device_memory().malloc(&ptr6, 209715200);
  void* ptr7{nullptr};
  device.get_device_memory().malloc(&ptr7, 104857600);
  PT_TEST_DEBUG("getting device address");
  address.push_back((uint64_t)ptr1);
  address.push_back((uint64_t)ptr2);
  address.push_back((uint64_t)ptr3);
  address.push_back((uint64_t)ptr4);
  address.push_back((uint64_t)ptr5);
  address.push_back((uint64_t)ptr6);
  address.push_back((uint64_t)ptr7);
  device.lock_addresses(address);
  address.clear();
  // delete 2 & 4 varaible
  device.get_device_memory().free(ptr1);
  device.get_device_memory().free(ptr2);
  device.get_device_memory().free(ptr7);

  // allocate 400 MB memory this willl lead OOM and defragmentor will kick in
  // here
  void* ptr_400mb{nullptr};
  device.get_device_memory().malloc(&ptr_400mb, 419430400);
  address.push_back((uint64_t)ptr_400mb);
  device.lock_addresses(address);

  device.get_device_memory().free(ptr6);
  device.get_device_memory().free(ptr3);
  device.get_device_memory().free(ptr5);
  device.get_device_memory().free(ptr4);
  device.get_device_memory().free(ptr_400mb);
}
