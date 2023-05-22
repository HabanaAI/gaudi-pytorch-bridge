/*******************************************************************************
 * Copyright (C) 2020-2023 Habana Labs, Ltd. an Intel Company
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
#pragma once
#include <ATen/ATen.h>
#include <c10/core/Allocator.h>
#include <synapse_api_types.h>
#include "backend/synapse_helpers/device.h"
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

/** Device memory allocator for pytorch.
 * Note that static singleton instance of the allocator is registered in
 * torch. This means that lifetime of the allocator is until static
 * finalizers, which is after the synapse device has been already disposed.
 * For this reason ~HPUDeviceAllocator cannot reliably refer to HPURegistrar
 * resources. Conversely, destruction of the HPUDevice to park allocator in
 * a proper state.
 */
class HPUDeviceAllocator final : public at::Allocator {
 public:
  HPUDeviceAllocator();

  at::DataPtr allocate(size_t size) const override;
  at::DeleterFnPtr raw_deleter() const override;

  void* allocate_impl(size_t size, synStatus& status) const;
  static void deleter(void* ptr);

  // user must manually set active device before calling allocator functions
  static synDeviceId allocator_active_device_id;
  static pgmDropCachedRecipe drop_cached_recipe_cb;

  static void print_memory_stats(const char* msg);
  static void memstat_devmem_start_collect(
      const char* msg,
      bool show_leaked_callstacks);
  static void memstat_devmem_stop_collect(const char* msg);
  static void dump_memory_reporter();
};

} // namespace habana
