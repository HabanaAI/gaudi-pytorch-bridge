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
      .value("MARK_STEP", habana_helpers::EventDispatcher::Topic::MARK_STEP)
      .value(
          "PROCESS_EXIT", habana_helpers::EventDispatcher::Topic::PROCESS_EXIT)
      .value(
          "CUSTOM_EVENT", habana_helpers::EventDispatcher::Topic::CUSTOM_EVENT);
  m.add_object("_cleanup", py::capsule(cleanup_callback));

  m.doc() =
      "Exposes API for subscribing and publishing events from Habana Pytorch plugin.";
}
