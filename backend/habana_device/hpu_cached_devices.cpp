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
#include "backend/habana_device/hpu_cached_devices.h"
#include <Python.h>
#include <pybind11/pybind11.h>
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
  if (Py_IsInitialized() != 0) {
    // If the interpreter is initialized we anticipate to be running in a python
    // session. In such case HPURegistrar must be deleted even earlier when it
    // is still possible to access the interpreter and remove python resources
    // (e.g. Tensors).
    // Lastly there are tensors kept alive by the python interpreter that are
    // disposed in Py_FinalizeEx/_PyModule_Clear. This is already after the
    // device got deleted.
    PT_BRIDGE_DEBUG(
        "python session: registering removal of HPURegistrar on python ataxit");
    pybind11::gil_scoped_acquire gil;
    auto atexit = pybind11::module_::import("atexit");
    atexit.attr("register")(pybind11::cpp_function([]() {
      PT_BRIDGE_DEBUG("python atexit cleanup");
      finalize_instance();
    }));
  }
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
