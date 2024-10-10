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
