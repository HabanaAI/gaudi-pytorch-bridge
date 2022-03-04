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
#include "habana_helpers/logging.h"

typedef bool (*pgmDropCachedRecipe)(size_t& recipe_count);

namespace habana {

at::Allocator* getHABANADeviceAllocator();

class HPUAllocator : public synapse_helpers::device_allocator {
 public:
  HPUAllocator(synDeviceId);

  void reset() override;
  void release() override;
  void* alloc(size_t num_bytes) override;
  void free(void* ptr) override;
  pgmDropCachedRecipe drop_cached_recipe_cb;

 private:
  synDeviceId device_id{synapse_helpers::device::INVALID_ID};
};

class HPUDeviceAllocator final : public at::Allocator {
 public:
  HPUDeviceAllocator();
  ~HPUDeviceAllocator();

  at::DataPtr allocate(size_t size) const override;
  at::DeleterFnPtr raw_deleter() const override;

  void* allocate_impl(size_t size, synStatus& status) const;
  static void deleter(void* ptr);

  // user must manually set active device before calling allocator functions
  static synDeviceId allocator_active_device_id;
  static pgmDropCachedRecipe drop_cached_recipe_cb;

  // At the time of destruction, enture that the stream manager is not in
  // the middle of releasing tensors
  void flush_stream_events() const;
  static void print_memory_stats(const char* msg);
  static void memstat_devmem_start_collect(
      const char* msg,
      bool show_leaked_callstacks);
  static void memstat_devmem_stop_collect(const char* msg);
};

} // namespace habana
