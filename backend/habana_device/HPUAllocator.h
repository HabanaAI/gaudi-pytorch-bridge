/**
* Copyright (c) 2021-2024 Intel Corporation
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
#pragma once
#include <ATen/ATen.h>
#include <c10/core/Allocator.h>
#include <synapse_api_types.h>
#include "backend/backend_meta.h"
#include "backend/habana_device/HPUStream.h"
#include "backend/synapse_helpers/device.h"
#include "habana_helpers/logging.h"
#include "pytorch_helpers/habana_helpers/pt_version_check.h"

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

  at::DeleterFnPtr raw_deleter() const override;
  static void recordStream(const at::DataPtr& ptr, c10::hpu::HPUStream stream);

  void* allocate_impl(size_t size, synStatus& status) const;
  static void deleter(void* ptr);

  // user must manually set active device before calling allocator functions
  static synDeviceId allocator_active_device_id;

  static void print_memory_stats(const char* msg);
  static void memstat_devmem_start_collect(
      const char* msg,
      bool show_leaked_callstacks);
  static void memstat_devmem_stop_collect(const char* msg);
  static void dump_memory_reporter();
  at::DataPtr allocate(size_t size) override;
  void copy_data(void* dest, const void* src, std::size_t count) const override;
};

} // namespace habana
