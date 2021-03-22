/******************************************************************************
 * Copyright (C) 2020 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */

#include <init.h>
#include <pybind11/chrono.h>
#include "ProcessGroupHCL.hpp"
namespace py = pybind11;

template <typename T>
using shared_ptr_class_ = py::class_<T, std::shared_ptr<T>>;

TORCH_HCL_CPP_API void torch_hcl_python_init(pybind11::module& m) {
  py::object module = py::module::import("torch.distributed");
  py::object register_backend = module.attr("Backend").attr("register_backend");

  register_backend(
      "hcl",
      py::cpp_function(
          &c10d::ProcessGroupHCL::createProcessGroupHCL,
          py::arg("store"),
          py::arg("rank"),
          py::arg("size"),
          py::arg("timeout") = std::chrono::milliseconds(40 * 1000)));

  auto processGroup = module.attr("ProcessGroup");
  auto processGroupHCL = shared_ptr_class_<::c10d::ProcessGroupHCL>(
      module, "ProcessGroupHCL", processGroup);

  processGroupHCL.def(
      py::init([](const std::shared_ptr<::c10d::Store>& store,
                  int rank,
                  int size,
                  std::chrono::milliseconds timeout) {
        return std::make_shared<::c10d::ProcessGroupHCL>(
            store, rank, size, timeout);
      }),
      py::arg("store"),
      py::arg("rank"),
      py::arg("size"),
      py::arg("timeout") = std::chrono::milliseconds(10 * 1000));
}
