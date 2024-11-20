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

#include <pybind11/chrono.h>
#include <pybind11/functional.h>
#include <pybind11/stl.h>

#include "backend/helpers/event_dispatcher.h"

namespace py = pybind11;

void cleanup_callback() {
  py::gil_scoped_release nogil{};
  habana_helpers::EventDispatcher::Instance().unsubscribe_all();
};

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  py::class_<habana_helpers::EventDispatcher>(m, "EventDispatcher")
      .def(
          "instance",
          &habana_helpers::EventDispatcher::Instance,
          py::return_value_policy::reference)
      .def("subscribe", &habana_helpers::EventDispatcher::subscribe)
      .def(
          "unsubscribe",
          [](habana_helpers::EventDispatcher& instance,
             const std::shared_ptr<habana_helpers::EventDispatcherHandle>&
                 handle) { instance.unsubscribe(handle); })
      .def("publish", &habana_helpers::EventDispatcher::publish);

  py::class_<
      habana_helpers::EventDispatcherHandle,
      std::shared_ptr<habana_helpers::EventDispatcherHandle>>(
      m, "EventDispatcherHandle");

  pybind11::enum_<habana_helpers::EventDispatcher::Topic>(m, "EventId")
      .value(
          "GRAPH_COMPILATION",
          habana_helpers::EventDispatcher::Topic::GRAPH_COMPILE)
      .value(
          "CPU_FALLBACK", habana_helpers::EventDispatcher::Topic::CPU_FALLBACK)
      .value("CACHE_HIT", habana_helpers::EventDispatcher::Topic::CACHE_HIT)
      .value("CACHE_MISS", habana_helpers::EventDispatcher::Topic::CACHE_MISS)
      .value(
          "MEMORY_DEFRAGMENTATION",
          habana_helpers::EventDispatcher::Topic::MEMORY_DEFRAGMENTATION)
      .value("MARK_STEP", habana_helpers::EventDispatcher::Topic::MARK_STEP)
      .value(
          "PROCESS_EXIT", habana_helpers::EventDispatcher::Topic::PROCESS_EXIT)
      .value(
          "DEVICE_ACQUIRED",
          habana_helpers::EventDispatcher::Topic::DEVICE_ACQUIRED)
      .value(
          "CUSTOM_EVENT", habana_helpers::EventDispatcher::Topic::CUSTOM_EVENT);
  py::module::import("atexit").attr("register")(
      py::cpp_function{cleanup_callback});
  m.doc() =
      "Exposes API for subscribing and publishing events from Habana Pytorch plugin.";
};
