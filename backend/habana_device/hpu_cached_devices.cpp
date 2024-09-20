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
std::once_flag HPURegistrar::initialize_once_flag_{};

void HPURegistrar::create_instance() {
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
}

void HPURegistrar::finalize_instance() {
  if (HPURegistrar::instance_) {
    PT_BRIDGE_BEGIN;
    HPURegistrar::instance_.reset(nullptr);
  }
}

const std::thread::id HPURegistrar::main_thread_id_ =
    std::this_thread::get_id();

const std::thread::id& HPURegistrar::get_main_thread_id() {
  return HPURegistrar::main_thread_id_;
}

HPURegistrar::HPURegistrar() {
  PT_BRIDGE_DEBUG("Creating hpu registrar ");
}

HPURegistrar::~HPURegistrar() {
  PT_BRIDGE_DEBUG("Releasing hpu registrar ");
}

} // namespace habana
