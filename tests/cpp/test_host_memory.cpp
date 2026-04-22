/**
 * Copyright (c) 2025 Intel Corporation
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

#include <fff.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <synapse_common_types.h>
#include <memory>
#include "backend/synapse_helpers/device_interface.h"
#include "backend/synapse_helpers/host_memory.h"
#include "synapse_api.h"

using namespace synapse_helpers;
using ::testing::_; // NOLINT (bugprone-reserved-identifier)
using ::testing::Return;

// Mock for device_interface
class MockDeviceInterface : public device_interface {
 public:
  MOCK_METHOD(synDeviceId, id, (), (const, override));
  MOCK_METHOD(bool, HostMemoryCacheEnabled, (), (const, override));
};

DEFINE_FFF_GLOBALS;

FAKE_VALUE_FUNC(
    synStatus,
    synHostMalloc,
    synDeviceId,
    size_t,
    unsigned int,
    void**);

FAKE_VALUE_FUNC(synStatus, synHostFree, synDeviceId, const void*, unsigned int);

namespace {
// Mock control variables
bool malloc_should_fail = false;
bool malloc_returns_oom_once = false;
std::unordered_map<void*, size_t> allocated_blocks;
synStatus malloc_return_code = synSuccess;
synStatus free_return_code = synSuccess;
int malloc_call_count = 0;
int free_call_count = 0;

// Custom handler for synHostMalloc
synStatus custom_synHostMalloc(
    synDeviceId device_id,
    size_t size,
    unsigned int flags,
    void** ptr) {
  malloc_call_count++;

  if (malloc_should_fail) {
    *ptr = nullptr;
    return synFail;
  }

  if (malloc_returns_oom_once && malloc_call_count == 1) {
    *ptr = nullptr;
    return synOutOfHostMemory;
  }

  void* mem = malloc(size);
  if (!mem) {
    return synOutOfHostMemory;
  }

  *ptr = mem;
  allocated_blocks[mem] = size;
  return malloc_return_code;
}

// Custom handler for synHostFree
synStatus custom_synHostFree(
    synDeviceId device_id,
    const void* ptr,
    unsigned int flags) {
  free_call_count++;

  if (ptr == nullptr) {
    return synSuccess;
  }

  auto it = allocated_blocks.find(const_cast<void*>(ptr));
  if (it != allocated_blocks.end()) {
    free(const_cast<void*>(ptr));
    allocated_blocks.erase(it);
    return free_return_code;
  }
  return synFail;
}
} // namespace

class HostMemoryTest : public ::testing::Test {
 protected:
  void SetUp() override {
    device = std::make_shared<::testing::NiceMock<MockDeviceInterface>>();
    ON_CALL(*device, id()).WillByDefault(Return(42));
    ON_CALL(*device, HostMemoryCacheEnabled()).WillByDefault(Return(true));
    RESET_FAKE(synHostMalloc);
    synHostMalloc_fake.custom_fake = custom_synHostMalloc;
    RESET_FAKE(synHostFree);
    synHostFree_fake.custom_fake = custom_synHostFree;

    malloc_should_fail = false;
    malloc_returns_oom_once = false;
    malloc_return_code = synSuccess;
    free_return_code = synSuccess;
    malloc_call_count = 0;
    free_call_count = 0;
    allocated_blocks.clear();
  }

  void TearDown() override {
    for (auto& pair : allocated_blocks) {
      free(pair.first);
    }
    allocated_blocks.clear();
  }

 private:
  std::shared_ptr<MockDeviceInterface> device;
};

TEST_F(HostMemoryTest, BasicMallocAndFree) {
  host_memory memory(*device);

  void* ptr = nullptr;
  EXPECT_EQ(memory.malloc(&ptr, 1024), synSuccess);
  EXPECT_NE(ptr, nullptr);
  EXPECT_TRUE(memory.is_host_memory(ptr));

  EXPECT_EQ(memory.free(ptr), synSuccess);
  EXPECT_FALSE(memory.is_host_memory(ptr));
}

TEST_F(HostMemoryTest, ZeroSizeAllocation) {
  host_memory memory(*device);

  void* ptr = nullptr;
  EXPECT_EQ(memory.malloc(&ptr, 0), synSuccess);
  EXPECT_EQ(ptr, nullptr);
}

TEST_F(HostMemoryTest, UncachedMallocAndFree) {
  host_memory memory(*device);

  void* ptr = nullptr;
  EXPECT_EQ(memory.uncached_malloc(&ptr, 1024), synSuccess);
  EXPECT_NE(ptr, nullptr);
  EXPECT_TRUE(memory.is_host_memory(ptr));

  EXPECT_EQ(memory.uncached_free(ptr), synSuccess);
  EXPECT_FALSE(memory.is_host_memory(ptr));
}

TEST_F(HostMemoryTest, MallocFailure) {
  host_memory memory(*device);

  malloc_should_fail = true;
  void* ptr = nullptr;
  EXPECT_NE(memory.malloc(&ptr, 1024), synSuccess);
  EXPECT_EQ(ptr, nullptr);
}

TEST_F(HostMemoryTest, UncachedMallocFailure) {
  host_memory memory(*device);

  malloc_should_fail = true;
  void* ptr = nullptr;
  EXPECT_NE(memory.uncached_malloc(&ptr, 1024), synSuccess);
  EXPECT_EQ(ptr, nullptr);
}

TEST_F(HostMemoryTest, CacheReuseForSameSize) {
  host_memory memory(*device);

  // Initial allocation
  void* ptr1 = nullptr;
  EXPECT_EQ(memory.malloc(&ptr1, 1024), synSuccess);
  EXPECT_NE(ptr1, nullptr);

  // Free it to cache
  EXPECT_EQ(memory.free(ptr1), synSuccess);

  // Should be 1 before second allocation
  int initial_malloc_count = malloc_call_count;

  // Second allocation of same size should use cache
  void* ptr2 = nullptr;
  EXPECT_EQ(memory.malloc(&ptr2, 1024), synSuccess);
  EXPECT_EQ(ptr1, ptr2); // Should get same memory address
  EXPECT_EQ(malloc_call_count, initial_malloc_count); // No new malloc calls

  EXPECT_EQ(memory.free(ptr2), synSuccess);
}

TEST_F(HostMemoryTest, CacheDisabled) {
  // Configure device to disable caching
  ON_CALL(*device, HostMemoryCacheEnabled()).WillByDefault(Return(false));

  host_memory memory(*device);

  // First allocation
  void* ptr1 = nullptr;
  EXPECT_EQ(memory.malloc(&ptr1, 1024), synSuccess);

  // Free it
  EXPECT_EQ(memory.free(ptr1), synSuccess);

  // First free should call synHostFree immediately
  EXPECT_EQ(free_call_count, 1);

  // Second allocation should be a new allocation
  void* ptr2 = nullptr;
  EXPECT_EQ(memory.malloc(&ptr2, 1024), synSuccess);

  // May be same address by chance due to malloc behavior, so don't check
  // addresses But should definitely call malloc again
  EXPECT_EQ(malloc_call_count, 2);

  EXPECT_EQ(memory.free(ptr2), synSuccess);
}

TEST_F(HostMemoryTest, DropCache) {
  host_memory memory(*device);

  // Allocate and free several blocks to put in cache
  std::vector<void*> ptrs;
  for (int i = 0; i < 5; i++) {
    void* ptr = nullptr;
    memory.malloc(&ptr, 1024ULL * (i + 1)); // Different sizes
    ptrs.push_back(ptr);
  }

  // Free them all (they go to cache)
  for (auto ptr : ptrs) {
    memory.free(ptr);
  }

  // At this point free_call_count should be 0 since we're caching
  EXPECT_EQ(free_call_count, 0);

  // Now drop the cache
  memory.dropCache();

  // All blocks should have been freed, but due to small buffers optimization
  // this may not resolve to actual synHostFree calls
  EXPECT_GT(free_call_count, 0); // At least some memory was freed
  EXPECT_LE(free_call_count, 5); // Can't free more than we allocated

  // Verify cache is actually empty by allocating again and ensuring new memory
  void* new_ptr = nullptr;
  malloc_call_count = 0;
  EXPECT_EQ(memory.malloc(&new_ptr, 1024), synSuccess);
  EXPECT_EQ(malloc_call_count, 1); // Should require a new allocation
  EXPECT_NE(new_ptr, nullptr);
  EXPECT_EQ(memory.free(new_ptr), synSuccess);
}

TEST_F(HostMemoryTest, OutOfMemoryWithCacheDrop) {
  host_memory memory(*device);

  constexpr size_t size_1mb = 1ULL * 1024 * 1024;
  constexpr size_t size_2mb = 2 * size_1mb;
  constexpr size_t size_3mb = 3 * size_1mb;

  // First allocation into cache. 2M to omit small buffer optimization.
  void* ptr1 = nullptr;
  EXPECT_EQ(memory.malloc(&ptr1, size_2mb), synSuccess);
  EXPECT_EQ(memory.free(ptr1), synSuccess);

  // Reset counters
  malloc_call_count = 0;
  free_call_count = 0;
  malloc_returns_oom_once = true;

  // This should fail first time (OOM), drop cache, then succeed on retry
  void* ptr2 = nullptr;
  EXPECT_EQ(memory.malloc(&ptr2, size_3mb), synSuccess);

  // Should have called malloc twice (first OOM, then success)
  EXPECT_EQ(malloc_call_count, 2);

  // Should have dropped cache (called free)
  EXPECT_GT(free_call_count, 0);

  EXPECT_EQ(memory.free(ptr2), synSuccess);
}

TEST_F(HostMemoryTest, IsHostMemory) {
  host_memory memory(*device);

  // Nullptr check
  EXPECT_FALSE(memory.is_host_memory(nullptr));

  // Allocated memory check
  void* ptr = nullptr;
  EXPECT_EQ(memory.malloc(&ptr, 1024), synSuccess);
  EXPECT_TRUE(memory.is_host_memory(ptr));

  // External pointer check
  void* external_ptr = malloc(1024);
  EXPECT_FALSE(memory.is_host_memory(external_ptr));
  free(external_ptr);

  // Freed memory check
  EXPECT_EQ(memory.free(ptr), synSuccess);
  EXPECT_FALSE(memory.is_host_memory(ptr));
}

TEST_F(HostMemoryTest, MultipleSizes) {
  host_memory memory(*device);

  // Allocate multiple blocks of different sizes
  void* ptr1 = nullptr;
  void* ptr2 = nullptr;

  EXPECT_EQ(memory.malloc(&ptr1, 1024), synSuccess);
  EXPECT_EQ(memory.malloc(&ptr2, 2048), synSuccess);

  EXPECT_NE(ptr1, nullptr);
  EXPECT_NE(ptr2, nullptr);
  EXPECT_NE(ptr1, ptr2);

  EXPECT_EQ(memory.free(ptr1), synSuccess);
  EXPECT_EQ(memory.free(ptr2), synSuccess);
}

// Test finding the smallest block that can hold allocation
TEST_F(HostMemoryTest, SmallestBlockReuse) {
  host_memory memory(*device);

  // Allocate and free blocks of different sizes to populate cache
  void* large_ptr = nullptr;
  void* medium_ptr = nullptr;
  void* small_ptr = nullptr;

  EXPECT_EQ(memory.malloc(&large_ptr, 4096), synSuccess);
  EXPECT_EQ(memory.malloc(&medium_ptr, 2048), synSuccess);
  EXPECT_EQ(memory.malloc(&small_ptr, 1024), synSuccess);

  EXPECT_EQ(memory.free(large_ptr), synSuccess);
  EXPECT_EQ(memory.free(medium_ptr), synSuccess);
  EXPECT_EQ(memory.free(small_ptr), synSuccess);

  // Reset counter
  malloc_call_count = 0;

  // Allocate a medium size - should get the medium block
  void* new_ptr = nullptr;
  EXPECT_EQ(memory.malloc(&new_ptr, 2048), synSuccess);
  EXPECT_EQ(new_ptr, medium_ptr);
  EXPECT_EQ(malloc_call_count, 0); // No new malloc calls

  EXPECT_EQ(memory.free(new_ptr), synSuccess);
}
