#include <absl/types/variant.h>
#include <gtest/gtest.h>
#include <synapse_api_types.h>
#include <synapse_common_types.h>
#include <torch/torch.h>
#include <algorithm>
#include <memory>
#include "habana_device/HPUAllocator.h"
#include "habana_device/HPUGuardImpl.h"
#include "habana_device/hpu_cached_devices.h"
#include "habana_device/tensor_builder.h"
#include "habana_helpers/tensor_utils.h"
#include "synapse_helpers/habana_tensor.h"
#include "synapse_helpers/synapse_error.h"

TEST(SynapseHelpersMemoryTest, degframentonOOM_1) {
  using namespace synapse_helpers;

  habana::HABANAGuardImpl device_guard;
  device_guard.getDevice();
  auto& device = synapse_helpers::HPURegistrar::get_device();

  if ((device.get_device_memory().get_pool_strategy() !=
       pool_allocator::startegy_coalesce_stringent) ||
      !device.IsMemorydefragmentationEnabled())
    return;
  device.cleanup_workspace_buffer();
  device.get_device_memory().reset_pool();

  // allocate workspace buffer 1gb
  device.get_workspace_buffer(1073741824);
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
  // reset the pool when test is done to cleanup the pool
  device.cleanup_workspace_buffer();
  device.get_device_memory().reset_pool();
}

TEST(SynapseHelpersMemoryTest, degframentonOOM_2) {
  using namespace synapse_helpers;

  habana::HABANAGuardImpl device_guard;
  device_guard.getDevice();
  auto& device = synapse_helpers::HPURegistrar::get_device();

  if ((device.get_device_memory().get_pool_strategy() !=
       pool_allocator::startegy_coalesce_stringent) ||
      !device.IsMemorydefragmentationEnabled())
    return;
  device.cleanup_workspace_buffer();
  device.get_device_memory().reset_pool();

  // allocate workspace buffer 1gb
  device.get_workspace_buffer(1073741824);
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
  // reset the pool when test is done to cleanup the pool
  device.cleanup_workspace_buffer();
  device.get_device_memory().reset_pool();
}

TEST(SynapseHelpersMemoryTest, degframentonOOMWithWS) {
  using namespace synapse_helpers;

  habana::HABANAGuardImpl device_guard;
  device_guard.getDevice();
  auto& device = synapse_helpers::HPURegistrar::get_device();

  if ((device.get_device_memory().get_pool_strategy() !=
       pool_allocator::startegy_coalesce_stringent) ||
      !device.IsMemorydefragmentationEnabled())
    return;
  device.cleanup_workspace_buffer();
  device.get_device_memory().reset_pool();

  // allocate workspace buffer 1gb
  device.get_workspace_buffer(1073741824);
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
  device.get_workspace_buffer(1178599424);

  device.get_device_memory().free(ptr1);
  device.get_device_memory().free(ptr3);
  device.get_device_memory().free(ptr4);
  device.get_device_memory().free(ptr5);
  device.get_device_memory().free(ptr6);

  // reset the pool when test is done
  PT_TEST_DEBUG("reset the pool when test is done");
  device.cleanup_workspace_buffer();
  device.get_device_memory().reset_pool();
}

// we can test when we have handling in smalalloc for equal to 256 size. so
// disabling it for now
TEST(SynapseHelpersMemoryTest, DISABLED_degframentonOOMWithSmallAlloc) {
  using namespace synapse_helpers;

  habana::HABANAGuardImpl device_guard;
  device_guard.getDevice();
  auto& device = synapse_helpers::HPURegistrar::get_device();

  if ((device.get_device_memory().get_pool_strategy() !=
       pool_allocator::startegy_coalesce_stringent) ||
      !device.IsMemorydefragmentationEnabled()) {
    return;
  }
  device.get_device_memory().reset_pool();
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

  // reset the pool when test is done
  PT_TEST_DEBUG("reset the pool when test is done");
  device.cleanup_workspace_buffer();
  device.get_device_memory().reset_pool();
}

TEST(SynapseHelpersMemoryTest, degframentonOOMandVerify_1) {
  using namespace synapse_helpers;

  habana::HABANAGuardImpl device_guard;
  device_guard.getDevice();
  auto& device = synapse_helpers::HPURegistrar::get_device();

  if ((device.get_device_memory().get_pool_strategy() !=
       pool_allocator::startegy_coalesce_stringent) ||
      !device.IsMemorydefragmentationEnabled()) {
    return;
  }
  device.get_device_memory().reset_pool();

  // allocate workspace buffer 1gb
  device.get_workspace_buffer(1073741824);
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
  // reset the pool when test is done
  PT_TEST_DEBUG("reset the pool when test is done");
  device.cleanup_workspace_buffer();
  device.get_device_memory().reset_pool();
}
