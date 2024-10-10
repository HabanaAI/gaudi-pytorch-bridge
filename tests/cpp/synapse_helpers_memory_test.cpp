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

#include <gtest/gtest.h>
#include <synapse_api_types.h>
#include <synapse_common_types.h>
#include <torch/torch.h>
#include <algorithm>
#include <memory>
#include "backend/habana_device/HPUGuardImpl.h"
#include "backend/habana_device/hpu_cached_devices.h"
#include "backend/habana_device/tensor_builder.h"
#include "backend/helpers/event_dispatcher.h"
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
        habana_lazy::get_device_lazy_execution_context(device.id());
    context->clear();
    // set up defragment, pool and 3gb pool for testing
    setenv("PT_HABANA_POOL_SIZE", "3", 1);
    setenv("PT_ENABLE_MEMORY_DEFRAGMENTATION", "true", 1);
    setenv("PT_HPU_POOL_STRATEGY", "5", 1);
    device.cleanup_workspace_buffer();
    device.get_device_memory().reset_pool();

    habana_lazy::StageSubmission::getInstance().resetCurrentAccumulatedOps();

    defragmentationCount = 0;
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
    habana_helpers::EventDispatcher::Instance().unsubscribe_all();
  }

 protected:
  std::atomic<int> defragmentationCount{0};
  habana_helpers::EventDispatcher::EventCallback
      defragmentationTriggeredCallback =
          [=](habana_helpers::EventDispatcher::EventTsType timestamp,
              const habana_helpers::EventDispatcher::EventParams& params) {
            for (const habana_helpers::EventDispatcher::EventParam& param :
                 params) {
              if (param.first == "success" && param.second == "1") {
                defragmentationCount++;
              }
            }
          };

  static constexpr size_t GB_1 = 1073741824;
  static constexpr size_t GB_2 = 2147483648;
  static constexpr size_t MB_100 = 104857600;
  static constexpr size_t MB_200 = 209715200;
  static constexpr size_t MB_400 = 419430400;

  bool compareDataFromDevice(
      void* src_local,
      synapse_helpers::device_ptr dst_device,
      size_t size,
      synapse_helpers::device& device) {
    std::atomic<bool> copyDone{false};
    auto temp_local = std::make_unique<char[]>(size);
    auto syn_error = device.copy_data_to_host(
        dst_device,
        temp_local.get(),
        dst_device,
        size,
        [&copyDone]() { copyDone = true; },
        false);
    TORCH_CHECK(syn_error.status == 0, syn_error.error);
    // wait for copy completion
    while (!copyDone) {
      std::this_thread::yield();
    }
    bool result = std::memcmp(src_local, temp_local.get(), size) == 0;
    return result;
  }

  void copyDataToDevice(
      void* src_local,
      synapse_helpers::device_ptr& dst_device,
      size_t size,
      synapse_helpers::device& device) {
    std::atomic<bool> copyDone{false};
    device.get_device_memory().malloc(
        reinterpret_cast<void**>(&dst_device), size);
    auto syn_error = device.copy_data_to_device(
        src_local,
        dst_device,
        dst_device,
        size,
        [&copyDone]() { copyDone = true; },
        false);
    TORCH_CHECK(syn_error.status == 0, syn_error.error);
    // wait for copy completion
    while (!copyDone) {
      std::this_thread::yield();
    }
  }
};

TEST_F(SynapseHelpersMemoryTest, degframentonOOM_one) {
  constexpr size_t CHUNKS_NUMBER = 5;

  auto& device = habana::HPURegistrar::get_device().syn_device();
  auto freeDeviceMemoryAtAdress =
      [&device](synapse_helpers::device_ptr address) {
        device.get_device_memory().free(reinterpret_cast<void*>(address));
      };

  std::vector<synapse_helpers::device_ptr> device_ptrs(CHUNKS_NUMBER);
  // allocate workspace buffer 2GB, so we leave 1GB for the remaining
  // allocations in the test
  device.get_workspace_buffer(GB_2);
  // allocate 5 chunks of size 200 MB each
  for (synapse_helpers::device_ptr& ptr : device_ptrs) {
    device.get_device_memory().malloc(reinterpret_cast<void**>(&ptr), MB_200);
  }
  device.lock_addresses(device_ptrs);

  // delete 2 & 4 chunk
  freeDeviceMemoryAtAdress(device_ptrs[1]);
  freeDeviceMemoryAtAdress(device_ptrs[3]);

  // set listener to detect defragmentation
  habana_helpers::EventDispatcher::Instance().subscribe(
      habana_helpers::EventDispatcher::Topic::MEMORY_DEFRAGMENTATION,
      defragmentationTriggeredCallback);
  // allocate 400 MB memory what will lead OOM and defragmentor will kick in
  // here
  synapse_helpers::device_ptr device_ptr_400mb;
  device.get_device_memory().malloc(
      reinterpret_cast<void**>(&device_ptr_400mb), MB_400);
  device.lock_addresses(device_ptr_400mb);

  freeDeviceMemoryAtAdress(device_ptrs[0]);
  freeDeviceMemoryAtAdress(device_ptrs[2]);
  freeDeviceMemoryAtAdress(device_ptrs[4]);
  freeDeviceMemoryAtAdress(device_ptr_400mb);

  EXPECT_EQ(defragmentationCount, 1);
}

TEST_F(SynapseHelpersMemoryTest, degframentonOOM_multiple) {
  auto& device = habana::HPURegistrar::get_device().syn_device();
  auto freeDeviceMemoryAtAdress =
      [&device](synapse_helpers::device_ptr address) {
        device.get_device_memory().free(reinterpret_cast<void*>(address));
      };

  // add required test patterns <indexes of blocks to free, sizes of blocks>
  std::set<std::pair<std::vector<int>, std::vector<int>>> test_set{
      {{1, 2, 4}, {MB_200, MB_100, MB_200, MB_200, MB_100, MB_200}},
      {{1, 3, 5}, {MB_200, MB_100, MB_200, MB_200, MB_200, MB_100}}};

  // allocate workspace buffer 2GB, so we leave 1GB for the remaining
  // allocations in the test
  device.get_workspace_buffer(GB_2);
  for (const auto& mem_info : test_set) {
    std::vector<int> chunks_to_free = mem_info.first;
    std::vector<int> chunks_sizes = mem_info.second;
    int number_of_blocks = mem_info.second.size();
    std::vector<synapse_helpers::device_ptr> device_ptrs(number_of_blocks);

    // allocate blocks with specified size in mem_info
    for (int i = 0; i < number_of_blocks; i++) {
      device.get_device_memory().malloc(
          reinterpret_cast<void**>(&(device_ptrs[i])), chunks_sizes[i]);
    }
    device.lock_addresses(device_ptrs);

    // delete blocks with indexes from chunks_sizes vector
    size_t memory_free = 0;
    for (const auto index : chunks_to_free) {
      freeDeviceMemoryAtAdress(device_ptrs[index]);
      memory_free += chunks_sizes[index];
    }
    // set listener to detect defragmentation
    habana_helpers::EventDispatcher::Instance().subscribe(
        habana_helpers::EventDispatcher::Topic::MEMORY_DEFRAGMENTATION,
        defragmentationTriggeredCallback);

    // allocate memoryFree memory this willl lead OOM and defragmentor will kick
    // in here
    synapse_helpers::device_ptr device_ptr_free_memory;
    device.get_device_memory().malloc(
        reinterpret_cast<void**>(&device_ptr_free_memory), memory_free);
    device.lock_addresses(device_ptr_free_memory);

    // unset the dealocation listener
    habana_helpers::EventDispatcher::Instance().unsubscribe_all();

    for (int i = 0; i < number_of_blocks; i++) {
      // If block was freed earlier, skip freeing it now
      if (std::find(chunks_to_free.begin(), chunks_to_free.end(), i) ==
          chunks_to_free.end()) {
        freeDeviceMemoryAtAdress(device_ptrs[i]);
      }
    }
    freeDeviceMemoryAtAdress(device_ptr_free_memory);
  }
  EXPECT_EQ(defragmentationCount, 2);
}

TEST_F(SynapseHelpersMemoryTest, degframentonOOMWithWS) {
  constexpr size_t CHUNKS_NUMBER = 5;

  auto& device = habana::HPURegistrar::get_device().syn_device();
  auto freeDeviceMemoryAtAdress =
      [&device](synapse_helpers::device_ptr address) {
        device.get_device_memory().free(reinterpret_cast<void*>(address));
      };

  std::vector<synapse_helpers::device_ptr> device_ptrs(CHUNKS_NUMBER);
  // allocate workspace buffer 1GB, so we leave 2GB for the remaining
  // allocations in the test
  device.get_workspace_buffer(GB_1);
  // allocate 5 chunks of size 200 MB each
  for (synapse_helpers::device_ptr& device_ptr : device_ptrs) {
    device.get_device_memory().malloc(
        reinterpret_cast<void**>(&device_ptr), MB_200);
  }
  device.lock_addresses(device_ptrs);

  // delete 2 chunk
  freeDeviceMemoryAtAdress(device_ptrs[1]);

  // Increase workspace buffer to 2.2GB what will trigger defragmentation
  habana_helpers::EventDispatcher::Instance().subscribe(
      habana_helpers::EventDispatcher::Topic::MEMORY_DEFRAGMENTATION,
      defragmentationTriggeredCallback);
  device.get_workspace_buffer(GB_2 + MB_200);

  freeDeviceMemoryAtAdress(device_ptrs[0]);
  freeDeviceMemoryAtAdress(device_ptrs[2]);
  freeDeviceMemoryAtAdress(device_ptrs[3]);
  freeDeviceMemoryAtAdress(device_ptrs[4]);

  EXPECT_EQ(defragmentationCount, 1);
}

// we can test when we have handling in smalalloc for equal to 256 size. so
// disabling it for now. While enabling the test case, add defragmentation event
// check as in other tests
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
  constexpr size_t CHUNKS_NUMBER = 6;

  auto& device = habana::HPURegistrar::get_device().syn_device();
  auto freeDeviceMemoryAtAdress =
      [&device](synapse_helpers::device_ptr address) {
        device.get_device_memory().free(reinterpret_cast<void*>(address));
      };

  // allocate workspace buffer 1.8GB
  device.get_workspace_buffer(GB_2 - MB_200);

  // allocate 6 chunks of size 200 MB each locally and copy the content to
  // device
  std::vector<synapse_helpers::device_ptr> device_ptrs(CHUNKS_NUMBER);
  std::array<std::vector<char>, CHUNKS_NUMBER> local_chunks;
  for (int i = 0; i < CHUNKS_NUMBER; i++) {
    local_chunks[i] = std::vector<char>(MB_200, i + 1);
    copyDataToDevice(local_chunks[i].data(), device_ptrs[i], MB_200, device);
  }
  // delete 3 & 5 chunk
  freeDeviceMemoryAtAdress(device_ptrs[2]);
  freeDeviceMemoryAtAdress(device_ptrs[4]);

  // set listener to detect defragmentation
  habana_helpers::EventDispatcher::Instance().subscribe(
      habana_helpers::EventDispatcher::Topic::MEMORY_DEFRAGMENTATION,
      defragmentationTriggeredCallback);

  // allocate 400 MB memory this willl lead OOM and defragmentor will kick in
  // here
  synapse_helpers::device_ptr device_ptr_400mb;
  std::vector<char> local_chunk_400mb = std::vector<char>(MB_400, 0xf);
  copyDataToDevice(local_chunk_400mb.data(), device_ptr_400mb, MB_400, device);

  // compare the moved data
  for (int i = 0; i < CHUNKS_NUMBER; i++) {
    if (i == 2 || i == 4)
      continue;
    EXPECT_TRUE(compareDataFromDevice(
        local_chunks[i].data(), device_ptrs[i], MB_200, device));
  }

  EXPECT_TRUE(compareDataFromDevice(
      local_chunk_400mb.data(),
      reinterpret_cast<synapse_helpers::device_ptr>(device_ptr_400mb),
      MB_400,
      device));

  // Free all memory
  for (int i = 0; i < CHUNKS_NUMBER; i++) {
    if (i == 2 || i == 4)
      continue;
    freeDeviceMemoryAtAdress(device_ptrs[i]);
  }
  freeDeviceMemoryAtAdress(device_ptr_400mb);

  EXPECT_EQ(defragmentationCount, 1);
}

TEST_F(SynapseHelpersMemoryTest, degframentonOOMandVerify_2) {
  // Test should not be run on the simulator as verification part (data transfer
  // to/from device) takes too much time
  if (is_simulator()) {
    GTEST_SKIP();
  }

  auto& device = habana::HPURegistrar::get_device().syn_device();
  auto freeDeviceMemoryAtAdress =
      [&device](synapse_helpers::device_ptr address) {
        device.get_device_memory().free(reinterpret_cast<void*>(address));
      };

  // Add required test pattern of required length.
  std::set<std::pair<std::vector<int>, std::vector<int>>> test_set{
      {{1, 3, 5}, {MB_200, MB_100, MB_200, MB_200, MB_200, MB_100, MB_200}},
      {{1, 4, 6}, {MB_200, MB_100, MB_200, MB_200, MB_200, MB_200, MB_100}}};

  // allocate workspace buffer 1.8GB
  device.get_workspace_buffer(GB_2 - MB_200);

  for (const auto& mem_info : test_set) {
    std::vector<int> chunks_to_free = mem_info.first;
    std::vector<int> chunks_sizes = mem_info.second;
    int number_of_blocks = mem_info.second.size();
    std::vector<synapse_helpers::device_ptr> device_ptrs(number_of_blocks);
    std::vector<std::vector<char>> local_chunks(number_of_blocks);

    // allocate blocks with specified size in mem_info and copy to the device
    for (int i = 0; i < number_of_blocks; i++) {
      local_chunks[i] = std::vector<char>(chunks_sizes[i], i + 1);
      copyDataToDevice(
          local_chunks[i].data(), device_ptrs[i], chunks_sizes[i], device);
    }

    unsigned long memoryFree = 0;
    for (const auto index : chunks_to_free) {
      freeDeviceMemoryAtAdress(device_ptrs[index]);
      memoryFree += chunks_sizes[index];
    }

    // set listener to detect defragmentation
    habana_helpers::EventDispatcher::Instance().subscribe(
        habana_helpers::EventDispatcher::Topic::MEMORY_DEFRAGMENTATION,
        defragmentationTriggeredCallback);

    synapse_helpers::device_ptr device_ptr_free_memory;
    std::vector<char> local_device_free_chunk =
        std::vector<char>(memoryFree, 0xf);

    // allocate free memory what willl lead OOM and defragmentor will kick
    // in here
    copyDataToDevice(
        local_device_free_chunk.data(),
        device_ptr_free_memory,
        memoryFree,
        device);
    habana_helpers::EventDispatcher::Instance().unsubscribe_all();

    // compare the moved data
    for (int i = 0; i < number_of_blocks; i++) {
      if (std::find(chunks_to_free.begin(), chunks_to_free.end(), i) ==
          chunks_to_free.end()) {
        EXPECT_TRUE(compareDataFromDevice(
            local_chunks[i].data(),
            reinterpret_cast<synapse_helpers::device_ptr>(device_ptrs[i]),
            MB_200,
            device));
      }
    }
    // compare the last 400mb copy
    EXPECT_TRUE(compareDataFromDevice(
        local_device_free_chunk.data(),
        reinterpret_cast<synapse_helpers::device_ptr>(device_ptr_free_memory),
        MB_400,
        device));

    // Free all memory
    for (int i = 0; i < number_of_blocks; i++) {
      if (std::find(chunks_to_free.begin(), chunks_to_free.end(), i) ==
          chunks_to_free.end()) {
        freeDeviceMemoryAtAdress(device_ptrs[i]);
      }
    }
    freeDeviceMemoryAtAdress(device_ptr_free_memory);
  }
  EXPECT_EQ(defragmentationCount, 2);
}

TEST_F(SynapseHelpersMemoryTest, GenTest) {
  constexpr size_t SMALL_CHUNKS_NUMBER = 16384;
  constexpr size_t SMALL_CHUNK_SIZE = 128;
  constexpr size_t FREE_CHUNK_INDEX_1 = 22;
  constexpr size_t FREE_CHUNK_INDEX_2 = 100;
  constexpr size_t FREE_CHUNK_INDEX_3 = 16383;
  constexpr size_t BIG_CHUNK_NUMBER = 6;

  std::vector<synapse_helpers::device_ptr> device_ptrs_small_chunks(
      SMALL_CHUNKS_NUMBER);

  auto& device = habana::HPURegistrar::get_device().syn_device();
  auto freeDeviceMemoryAtAdress =
      [&device](synapse_helpers::device_ptr address) {
        device.get_device_memory().free(reinterpret_cast<void*>(address));
      };
  // allocate workspace buffer 1.78GB, so the rest memory is used in the test
  device.get_workspace_buffer(
      GB_2 - MB_200 - SMALL_CHUNK_SIZE * SMALL_CHUNKS_NUMBER);

  // set listener to detect defragmentation
  habana_helpers::EventDispatcher::Instance().subscribe(
      habana_helpers::EventDispatcher::Topic::MEMORY_DEFRAGMENTATION,
      defragmentationTriggeredCallback);

  // allocate small chunks of memory
  for (int j = 0; j < SMALL_CHUNKS_NUMBER; j++) {
    device.get_device_memory().malloc(
        reinterpret_cast<void**>(&device_ptrs_small_chunks[j]),
        SMALL_CHUNK_SIZE);
  }
  device.lock_addresses(device_ptrs_small_chunks);

  freeDeviceMemoryAtAdress(device_ptrs_small_chunks[FREE_CHUNK_INDEX_1]);
  freeDeviceMemoryAtAdress(device_ptrs_small_chunks[FREE_CHUNK_INDEX_2]);
  freeDeviceMemoryAtAdress(device_ptrs_small_chunks[FREE_CHUNK_INDEX_3]);

  std::vector<synapse_helpers::device_ptr> device_ptrs(BIG_CHUNK_NUMBER);
  for (int j = 0; j < 3; j++) {
    // allocate large chunks of memory
    for (int i = 0; i < BIG_CHUNK_NUMBER; i++) {
      device.get_device_memory().malloc(
          reinterpret_cast<void**>(&device_ptrs[i]), MB_200);
    }
    device.lock_addresses(device_ptrs);
    // delete two non consecutive chunks
    int chunk_index_1 = j;
    int chunk_index_2 = chunk_index_1 + 2;
    freeDeviceMemoryAtAdress(device_ptrs[chunk_index_1]);
    freeDeviceMemoryAtAdress(device_ptrs[chunk_index_2]);

    // allocate 400 MB memory this willl lead OOM and defragmentor will kick in
    // here
    synapse_helpers::device_ptr device_ptr_400mb;
    device.get_device_memory().malloc(
        reinterpret_cast<void**>(&device_ptr_400mb), MB_400);
    device.lock_addresses(device_ptr_400mb);

    // free large chunks
    for (int i = 0; i < BIG_CHUNK_NUMBER; i++) {
      if (i == chunk_index_1 || i == chunk_index_2)
        continue;
      freeDeviceMemoryAtAdress(device_ptrs[i]);
    }
    freeDeviceMemoryAtAdress(device_ptr_400mb);
  }

  // free small chunks
  for (int j = 0; j < SMALL_CHUNKS_NUMBER; j++) {
    if (!(j == FREE_CHUNK_INDEX_1 || j == FREE_CHUNK_INDEX_2 ||
          j == FREE_CHUNK_INDEX_3))
      freeDeviceMemoryAtAdress(device_ptrs_small_chunks[j]);
  }
  EXPECT_EQ(defragmentationCount, 3);
}

TEST_F(SynapseHelpersMemoryTest, OOM_FreeMemInEndofsmallallocRegion) {
  constexpr size_t SMALL_CHUNKS_NUMBER = 16384;
  constexpr size_t SMALL_CHUNK_SIZE = 128;
  constexpr size_t ALLOCATED_SMALL_CHUNK_1 = 170;
  constexpr size_t ALLOCATED_SMALL_CHUNK_2 = 171;
  constexpr size_t ALLOCATED_SMALL_CHUNK_3 = 172;
  constexpr size_t ALLOCATED_SMALL_CHUNK_4 = 213;

  auto& device = habana::HPURegistrar::get_device().syn_device();
  auto freeDeviceMemoryAtAdress =
      [&device](synapse_helpers::device_ptr address) {
        device.get_device_memory().free(reinterpret_cast<void*>(address));
      };
  // allocate workspace buffer 1.98GB
  device.get_workspace_buffer(
      GB_2 - MB_200 - SMALL_CHUNKS_NUMBER * SMALL_CHUNK_SIZE);

  std::vector<device_ptr> device_ptrs_small_chunks(SMALL_CHUNKS_NUMBER);
  for (int j = 0; j < SMALL_CHUNKS_NUMBER; j++) {
    device.get_device_memory().malloc(
        reinterpret_cast<void**>(&device_ptrs_small_chunks[j]),
        SMALL_CHUNK_SIZE);
  }
  device.lock_addresses(device_ptrs_small_chunks);
  // Deallocate all small chunks except those with indexes:
  // 0 - 121, 170, 171, 172, 213
  constexpr size_t START_INDEX = 122;
  for (int index = START_INDEX; index < SMALL_CHUNKS_NUMBER; index++) {
    if (index == ALLOCATED_SMALL_CHUNK_1 || index == ALLOCATED_SMALL_CHUNK_2 ||
        index == ALLOCATED_SMALL_CHUNK_3 || index == ALLOCATED_SMALL_CHUNK_4) {
      continue;
    }
    freeDeviceMemoryAtAdress(device_ptrs_small_chunks[index]);
  }

  std::pair<std::vector<int>, std::vector<int>> mem_info{
      {0, 1, 6}, {MB_200, MB_100, MB_200, MB_200, MB_200, MB_200, MB_100}};
  std::vector<int> chunks_to_free = mem_info.first;
  std::vector<int> chunks_sizes = mem_info.second;

  int number_of_blocks = mem_info.second.size();
  std::vector<synapse_helpers::device_ptr> device_ptrs(number_of_blocks);
  // allocate blocks with specified size in mem_info
  for (int i = 0; i < number_of_blocks; i++) {
    device.get_device_memory().malloc(
        reinterpret_cast<void**>(&(device_ptrs[i])), chunks_sizes[i]);
  }

  device.lock_addresses(device_ptrs);
  // delete blocks with indexes from vector chunks_sizes
  unsigned long memoryFree = 0;
  for (const auto index : chunks_to_free) {
    freeDeviceMemoryAtAdress(device_ptrs[index]);
    memoryFree += chunks_sizes[index];
  }

  // set listener to detect defragmentation
  habana_helpers::EventDispatcher::Instance().subscribe(
      habana_helpers::EventDispatcher::Topic::MEMORY_DEFRAGMENTATION,
      defragmentationTriggeredCallback);

  synapse_helpers::device_ptr device_ptr_free_memory;
  // allocate memoryFree memory this willl lead OOM and defragmentor will kick
  // in here
  device.get_device_memory().malloc(
      reinterpret_cast<void**>(&device_ptr_free_memory), memoryFree);
  device.lock_addresses(device_ptr_free_memory);
  // free all memory
  for (int i = 0; i < number_of_blocks; i++) {
    // if block was freed earlier, skip freeing it now
    if (std::find(chunks_to_free.begin(), chunks_to_free.end(), i) ==
        chunks_to_free.end()) {
      freeDeviceMemoryAtAdress(device_ptrs[i]);
    }
  }
  freeDeviceMemoryAtAdress(device_ptr_free_memory);

  // free remaining small chunks
  for (int i = 0; i < START_INDEX; i++) {
    freeDeviceMemoryAtAdress(device_ptrs_small_chunks[i]);
  }

  freeDeviceMemoryAtAdress(device_ptrs_small_chunks[ALLOCATED_SMALL_CHUNK_1]);
  freeDeviceMemoryAtAdress(device_ptrs_small_chunks[ALLOCATED_SMALL_CHUNK_2]);
  freeDeviceMemoryAtAdress(device_ptrs_small_chunks[ALLOCATED_SMALL_CHUNK_3]);
  freeDeviceMemoryAtAdress(device_ptrs_small_chunks[ALLOCATED_SMALL_CHUNK_4]);

  habana_helpers::EventDispatcher::Instance().unsubscribe_all();

  EXPECT_EQ(defragmentationCount, 1);
}

TEST_F(SynapseHelpersMemoryTest, Verify_Reset_pool) {
  auto& device = habana::HPURegistrar::get_device().syn_device();
  size_t ws = device.get_workspace_size();
  // allocate workspace buffer 2GB, so we leave 1GB for the remaining
  // allocations in the test
  device.get_workspace_buffer(GB_2);
  size_t ws1 = device.get_workspace_size();
  EXPECT_EQ(ws1, GB_2);
}
