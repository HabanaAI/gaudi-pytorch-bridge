/**
 * Copyright (c) 2024-2025 Intel Corporation
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
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <torch/csrc/Dtype.h>
#include "slrg_functions.h"
#include "utils/shared_structures.h"
namespace py = pybind11;
using OperatorReportPair = std::pair<slrg::OperatorDescriptor, slrg::Report>;
using OperatorReportVector = std::vector<OperatorReportPair>;
PYBIND11_MAKE_OPAQUE(OperatorReportPair)
PYBIND11_MAKE_OPAQUE(OperatorReportVector)
PYBIND11_MAKE_OPAQUE(slrg::FinalReport)
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "Shared Layer Report Generator";
  py::class_<slrg::Report>(m, "Report")
      .def(py::init<>())
      .def(
          "__getitem__",
          [](slrg::Report& report, py::object py_dtype) -> bool& {
            at::ScalarType precision_type =
                reinterpret_cast<THPDtype*>(py_dtype.ptr())->scalar_type;
            return report[precision_type];
          },
          py::return_value_policy::reference_internal)
      .def(
          "__setitem__",
          [](slrg::Report& report, py::object py_dtype, bool value) {
            at::ScalarType precision_type =
                reinterpret_cast<THPDtype*>(py_dtype.ptr())->scalar_type;
            report[precision_type] = value;
          });
  py::class_<slrg::OperatorDescriptor>(m, "OperatorDescriptor")
      .def(py::init<>())
      .def(py::init<std::string, std::string, std::string>())
      .def_readwrite("op_name", &slrg::OperatorDescriptor::op_name)
      .def_readwrite("overload", &slrg::OperatorDescriptor::overload)
      .def_readwrite("op_namespace", &slrg::OperatorDescriptor::op_namespace);
  py::class_<OperatorReportPair>(m, "OperatorReportPair")
      .def(py::init<>())
      .def(py::init<const slrg::OperatorDescriptor&, const slrg::Report&>())
      .def_readwrite("first", &OperatorReportPair::first)
      .def_readwrite("second", &OperatorReportPair::second);

  py::bind_vector<OperatorReportVector>(m, "OperatorReportVector");
  py::bind_map<slrg::FinalReport>(m, "FinalReport");
  m.def(
      "run_report_gen",
      &slrg::run_report_gen,
      "Runs the report generator and returns its result as FinalReport");
}
