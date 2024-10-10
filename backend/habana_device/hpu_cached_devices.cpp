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

namespace habana {

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
  static std::shared_ptr<synapse_helpers::session> session{
      synapse_helpers::get_value(synapse_helpers::session::get_or_create())};

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

const std::thread::id HPURegistrar::main_thread_id_ =
    std::this_thread::get_id();

HPURegistrar::HPURegistrar()
    : acquired_device_{nullptr, &HPURegistrar::device_deleter} {
  PT_BRIDGE_DEBUG("Creating hpu registrar ", acquired_device_.get());
}

HPURegistrar::~HPURegistrar() {
  PT_BRIDGE_DEBUG("Releasing hpu registrar ", acquired_device_.get());
}

void HPURegistrar::device_deleter(HPUDevice* device) {
  if (device != nullptr) {
    get_hpu_registrar().device_deleter_internal(device);
  }
}

void HPURegistrar::device_deleter_internal(HPUDevice* device) {
  PT_BRIDGE_DEBUG("Releasing hpu device ", acquired_device_.get());
  // Cleanup the device
  device->cleanup();

  // Run late-cleanup test hook if armed.
  if (test_inject_late_cleanup_) {
    test_inject_late_cleanup_();
    test_inject_late_cleanup_ = nullptr;
  }

  // while deleting, the device is still accessible
  // by get_device() via active_device_, albeit with a warning.
  // Notably Tensor deallocations due to streams being flushed
  // are expected to still work.
  delete device;

  // stop resolving get_device()
  active_device_ = nullptr;

  habana::HPUDeviceAllocator::allocator_active_device_id = -1;
  habana::PinnedMemoryAllocator::allocator_active_device_id = -1;
}

HPUDevice& HPURegistrar::get_or_create_device() {
  PT_BRIDGE_BEGIN;
  if (is_initialized()) {
    return get_active_device();
  }

  acquired_device_ =
      device_holder(new HPUDevice(), HPURegistrar::device_deleter);
  active_device_ = acquired_device_.get();
  PT_BRIDGE_DEBUG("Created hpu device ", acquired_device_.get());
  habana::HPUDeviceAllocator::allocator_active_device_id = active_device_->id();
  habana::PinnedMemoryAllocator::allocator_active_device_id =
      active_device_->id();

  TORCH_CHECK(
      habana::HPUDeviceAllocator::allocator_active_device_id == 0,
      "habana active device: ",
      habana::HPUDeviceAllocator::allocator_active_device_id,
      " != 0");
  return *acquired_device_;
}
} // namespace habana
