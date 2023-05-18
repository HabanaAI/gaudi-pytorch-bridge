/*******************************************************************************
 * Copyright (C) 2023 Habana Labs, Ltd. an Intel Company
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
#include "backend/habana_device/hpu_cached_devices.h"
#include <mutex>
#include "backend/synapse_helpers/session.h"
#include "habana_helpers/logging.h"

namespace synapse_helpers {

std::unique_ptr<HPURegistrar> HPURegistrar::instance_{nullptr};
HPURegistrar* HPURegistrar::raw_instance_{nullptr};
bool HPURegistrar::finalized_{false};

std::once_flag HPURegistrar::initialize_once_flag_{};

void HPURegistrar::create_instance() {
  if (finalized_) {
    PT_BRIDGE_WARN(
        "Ignoring attempt to initialize HPURegistrar after it has been finalized");
    return;
  }

  // create session to force a call to synInitialize.
  // This ensures that static objects inside synapse (OSAL) are initialized
  // before the registrar, thus will be deleted after the registrar and devices
  // are gone.
  synapse_helpers::session::get_or_create();
  // HPURegistrar should not outlive synapse, so register static destructor
  static CallFinally destroy{[]() {
    PT_BRIDGE_DEBUG("static finalization");
    finalize_instance();
  }};
  instance_.reset(new HPURegistrar());
  raw_instance_ = instance_.get();
}

void HPURegistrar::finalize_instance() {
  if (HPURegistrar::instance_) {
    PT_BRIDGE_BEGIN;
    HPURegistrar::instance_.reset(nullptr);
    raw_instance_ = nullptr;
  }
  finalized_ = true;
}

const std::thread::id synapse_helpers::HPURegistrar::main_thread_id_ =
    std::this_thread::get_id();

HPURegistrar::~HPURegistrar() {
  if (is_initialized()) {
    auto& device = get_active_device();
    // Cleanup the device
    device.cleanup();

    // Theoretically another refernce can be kept elsewhere.
    if (acquired_device_.use_count() != 1) {
      TORCH_WARN(
          "when deleting HPURegistar, device is kept alive by another references ",
          acquired_device_.use_count());
    }

    // Run late-cleanup test hook if armed.
    if (test_inject_late_cleanup_) {
      test_inject_late_cleanup_();
      test_inject_late_cleanup_ = nullptr;
    }

    acquired_device_ =
        nullptr; // destroy the device. while deleting it is still accessible
                 // by get_device() via active_device_, albeit with a warning.
                 // Notably Tensor deallocations due to streams being flushed
                 // are expected to still work.
    active_device_ = nullptr; // stop resolving get_device()
  }

  habana::HPUDeviceAllocator::allocator_active_device_id = -1;
  habana::PinnedMemoryAllocator::allocator_active_device_id = -1;
}

bool HPURegistrar::is_closing() {
  return active_device_ && !acquired_device_;
}

device& HPURegistrar::get_or_create_device() {
  PT_BRIDGE_BEGIN;
  if (is_initialized()) {
    return get_active_device();
  }

  auto allocatorVar =
      [](synDeviceId id) -> std::unique_ptr<synapse_helpers::device_allocator> {
    return std::make_unique<habana::HPUAllocator>(id);
  };
  // Create the synapse_helpers::device, which will create the OSAL object.
  auto device_ptr_or_error = synapse_helpers::device::get_or_create(
      synapse_helpers::device::get_supported_devices(), allocatorVar);

  if (absl::holds_alternative<synapse_helpers::synapse_error>(
          device_ptr_or_error)) {
    auto error = absl::get<synapse_helpers::synapse_error>(device_ptr_or_error);
    TORCH_HABANA_CHECK(error.status, error.error);
  } else {
    auto device_ptr = absl::get<std::shared_ptr<synapse_helpers::device>>(
        device_ptr_or_error);
    acquired_device_ = std::move(device_ptr);
    active_device_ = acquired_device_.get();
    PT_BRIDGE_DEBUG("Created hpu device ", acquired_device_.get());
  }
  habana::HPUDeviceAllocator::allocator_active_device_id =
      acquired_device_->id();
  habana::PinnedMemoryAllocator::allocator_active_device_id =
      acquired_device_->id();

  TORCH_CHECK(
      habana::HPUDeviceAllocator::allocator_active_device_id == 0,
      "habana active device: ",
      habana::HPUDeviceAllocator::allocator_active_device_id,
      " != 0");
  return *acquired_device_;
}
} // namespace synapse_helpers
