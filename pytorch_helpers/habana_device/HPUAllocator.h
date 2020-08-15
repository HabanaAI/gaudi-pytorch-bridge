/******************************************************************************
 * Copyright (C) 2020 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */
#pragma once
#include <ATen/ATen.h>
#include <c10/core/Allocator.h>
#include <synapse_api_types.h>
#include <synapse_helpers/device.h>
#include <synapse_helpers/habana_tensor.h>
#include "habana_helpers/logging.h"
#include "pool_allocator/PoolAllocator.h"
#include "pool_allocator/CoalescedPoolAllocator.h"

typedef bool (*pgmDropCachedRecipe) (size_t &recipe_count);

namespace at {
namespace habana {

at::Allocator* getHABANADeviceAllocator();

class HPUAllocator : public synapse_helpers::device_allocator {
 public:
  HPUAllocator(synDeviceId);

  void  reset() override;
  void  release() override;
  void* alloc(size_t num_bytes) override;
  void  free(void* ptr) override;

 private:
  synDeviceId device_id{synapse_helpers::device::INVALID_ID};
};

class HPUDeviceAllocator final : public at::Allocator {
 public:
  HPUDeviceAllocator();
  ~HPUDeviceAllocator();
  static pool_allocator::PoolStrategyType get_pooling_strategy() {
    static const string poolEnvValue = "PT_HPU_POOL_STRATEGY";
    const char* poolValue = getenv(poolEnvValue.c_str());
    if (poolValue) {
      if (strncmp(poolValue, "1", 1) == 0) {
        PT_DEVICE_DEBUG("Bump pooling Enabled");
        return pool_allocator::strategy_bump;
      } else if (strncmp(poolValue, "2", 1) == 0) {
        PT_DEVICE_DEBUG("Dyanmic pooling Enabled");
        return pool_allocator::strategy_dynamic;
      } else if (strncmp(poolValue, "3", 1) == 0) {
        PT_DEVICE_DEBUG("static pooling with coalescing Enabled");
        return pool_allocator::startegy_static_coalesce;
      } else if (strncmp(poolValue, "0", 1) == 0) {
        PT_DEVICE_DEBUG("pooling Disabled");
        return pool_allocator::strategy_none;
      }
    }
    PT_DEVICE_DEBUG("default pooling strategy set");
    return pool_allocator::strategy_none;
  }

  static uint64_t get_pool_size() {
    uint64_t poolSize = DEFAULT_POOL_SIZE;
    static const string poolEnvValue = "PT_HPU_POOL_SIZE";
    const char* poolValue = getenv(poolEnvValue.c_str());
    if (poolValue) {
      poolSize = atoi(getenv("ENV_POOL_SIZE"));
      poolSize = poolSize*1024*1024*1024;
      if (poolSize == 0) {
        PT_DEVICE_DEBUG("Pool size not specified, setting default");
        poolSize = DEFAULT_POOL_SIZE;
      }
    }
    PT_DEVICE_DEBUG("Pool size requested for :: ", poolSize);
    return poolSize;
  }

  static void create_pool(synDeviceId deviceID,  uint64_t poolSize);
  static void delete_pool();
  static pool_allocator::SubAllocator *suballoc;

  static void* mem_pool;
  static pool_allocator::PoolStrategyType poolingType;
  static uint64_t poolSize;

  at::DataPtr allocate(size_t size) const override;
  at::DeleterFnPtr raw_deleter() const override;

  void* allocate_impl(size_t size, synStatus &status) const;
  static void deleter(void *ptr);

  // user must manually set active device before calling allocator functions
  static synDeviceId allocator_active_device_id;
  static pgmDropCachedRecipe drop_cached_recipe_cb;
};

} // namespace habana
} // namespace at
