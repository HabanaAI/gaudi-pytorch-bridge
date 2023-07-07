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
#include "backend/backend_meta.h"
#include "backend/habana_device/HPUStream.h"
#include "backend/synapse_helpers/device.h"
#include "habana_helpers/logging.h"

#include <map>

namespace habana {

using StorageExtraMetaMap = std::map<int64_t, habana::StorageExtraMeta>;

// This struct is passed to at:DataPtr during HPUAllocator::allocate() in case
// of >0 bytes allocated. It can be retrieved by at::DataPtr::get_context().
struct HPUAllocationContext {
  void* data_ptr; // raw data address
  size_t num_bytes; // number of bytes allocated
  // In case of contiguous views with different storage offsets, we can have
  // different permutations for each view. In particular, we can use big buffer
  // for storing all the gradients results with offsets, that can be easily used
  // for reduction-like operations (reduction can be done in one shot).
  StorageExtraMetaMap meta_map;
  // This field is used, when accessing StorageExtraMeta for Tensor of the same
  // size (or bigger) as the allocated one (num_bytes >= tensor.nbytes()).
  StorageExtraMeta base_meta;
};

at::Allocator* getHABANADeviceAllocator();

class HPUDeviceAllocator final : public at::Allocator {
 public:
  HPUDeviceAllocator();
  ~HPUDeviceAllocator();

  at::DataPtr allocate(size_t size) const override;
  at::DeleterFnPtr raw_deleter() const override;
  static void recordStream(const at::DataPtr& ptr, c10::hpu::HPUStream stream);

  void* allocate_impl(size_t size, synStatus& status) const;
  static void deleter(void* ptr);

  // user must manually set active device before calling allocator functions
  static synDeviceId allocator_active_device_id;

  // At the time of destruction, enture that the stream manager is not in
  // the middle of releasing tensors
  void flush_stream_events() const;
  static void print_memory_stats(const char* msg);
  static void memstat_devmem_start_collect(
      const char* msg,
      bool show_leaked_callstacks);
  static void memstat_devmem_stop_collect(const char* msg);
  static void dump_memory_reporter();
};

} // namespace habana
