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
#include <torch/csrc/Exceptions.h>
#include <torch/csrc/python_headers.h>
#include "pytorch_helpers/habana_helpers/pt_version_check.h"

#include <array>
#include <cstdarg>
#include <exception>
#include <utility>

#include <fmt/format.h>
#include <torch/csrc/THP.h>

#include <c10/util/StringUtil.h>

namespace torch {

void processErrorMsgInplace(std::string& str) {
  // Translate Aten types to their respective pytorch ones
  constexpr std::array<std::pair<c10::string_view, c10::string_view>, 64>
      changes{{
          {"Variable[SparseHPUByteType]", "torch.hpu.sparse.ByteTensor"},
          {"Variable[SparseHPUCharType]", "torch.hpu.sparse.CharTensor"},
          {"Variable[SparseHPUDoubleType]", "torch.hpu.sparse.DoubleTensor"},
          {"Variable[SparseHPUFloatType]", "torch.hpu.sparse.FloatTensor"},
          {"Variable[SparseHPUIntType]", "torch.hpu.sparse.IntTensor"},
          {"Variable[SparseHPULongType]", "torch.hpu.sparse.LongTensor"},
          {"Variable[SparseHPUShortType]", "torch.hpu.sparse.ShortTensor"},
          {"Variable[SparseHPUHalfType]", "torch.hpu.sparse.HalfTensor"},
          {"Variable[SparseCPUByteType]", "torch.sparse.ByteTensor"},
          {"Variable[SparseCPUCharType]", "torch.sparse.CharTensor"},
          {"Variable[SparseCPUDoubleType]", "torch.sparse.DoubleTensor"},
          {"Variable[SparseCPUFloatType]", "torch.sparse.FloatTensor"},
          {"Variable[SparseCPUIntType]", "torch.sparse.IntTensor"},
          {"Variable[SparseCPULongType]", "torch.sparse.LongTensor"},
          {"Variable[SparseCPUShortType]", "torch.sparse.ShortTensor"},
          {"Variable[SparseCPUHalfType]", "torch.sparse.HalfTensor"},
          {"Variable[HPUByteType]", "torch.hpu.ByteTensor"},
          {"Variable[HPUCharType]", "torch.hpu.CharTensor"},
          {"Variable[HPUDoubleType]", "torch.hpu.DoubleTensor"},
          {"Variable[HPUFloatType]", "torch.hpu.FloatTensor"},
          {"Variable[HPUIntType]", "torch.hpu.IntTensor"},
          {"Variable[HPULongType]", "torch.hpu.LongTensor"},
          {"Variable[HPUShortType]", "torch.hpu.ShortTensor"},
          {"Variable[HPUHalfType]", "torch.hpu.HalfTensor"},
          {"Variable[CPUByteType]", "torch.ByteTensor"},
          {"Variable[CPUCharType]", "torch.CharTensor"},
          {"Variable[CPUDoubleType]", "torch.DoubleTensor"},
          {"Variable[CPUFloatType]", "torch.FloatTensor"},
          {"Variable[CPUIntType]", "torch.IntTensor"},
          {"Variable[CPULongType]", "torch.LongTensor"},
          {"Variable[CPUShortType]", "torch.ShortTensor"},
          {"Variable[CPUHalfType]", "torch.HalfTensor"},
          {"SparseHPUByteType", "torch.hpu.sparse.ByteTensor"},
          {"SparseHPUCharType", "torch.hpu.sparse.CharTensor"},
          {"SparseHPUDoubleType", "torch.hpu.sparse.DoubleTensor"},
          {"SparseHPUFloatType", "torch.hpu.sparse.FloatTensor"},
          {"SparseHPUIntType", "torch.hpu.sparse.IntTensor"},
          {"SparseHPULongType", "torch.hpu.sparse.LongTensor"},
          {"SparseHPUShortType", "torch.hpu.sparse.ShortTensor"},
          {"SparseHPUHalfType", "torch.hpu.sparse.HalfTensor"},
          {"SparseCPUByteType", "torch.sparse.ByteTensor"},
          {"SparseCPUCharType", "torch.sparse.CharTensor"},
          {"SparseCPUDoubleType", "torch.sparse.DoubleTensor"},
          {"SparseCPUFloatType", "torch.sparse.FloatTensor"},
          {"SparseCPUIntType", "torch.sparse.IntTensor"},
          {"SparseCPULongType", "torch.sparse.LongTensor"},
          {"SparseCPUShortType", "torch.sparse.ShortTensor"},
          {"SparseCPUHalfType", "torch.sparse.HalfTensor"},
          {"HPUByteType", "torch.hpu.ByteTensor"},
          {"HPUCharType", "torch.hpu.CharTensor"},
          {"HPUDoubleType", "torch.hpu.DoubleTensor"},
          {"HPUFloatType", "torch.hpu.FloatTensor"},
          {"HPUIntType", "torch.hpu.IntTensor"},
          {"HPULongType", "torch.hpu.LongTensor"},
          {"HPUShortType", "torch.hpu.ShortTensor"},
          {"HPUHalfType", "torch.hpu.HalfTensor"},
          {"CPUByteType", "torch.ByteTensor"},
          {"CPUCharType", "torch.CharTensor"},
          {"CPUDoubleType", "torch.DoubleTensor"},
          {"CPUFloatType", "torch.FloatTensor"},
          {"CPUIntType", "torch.IntTensor"},
          {"CPULongType", "torch.LongTensor"},
          {"CPUShortType", "torch.ShortTensor"},
          {"CPUHalfType", "torch.HalfTensor"},
      }};

  // Avoid doing any work if no types need translated
  if (str.find("Type") == str.npos) {
    return;
  }
  for (const auto& it : changes) {
    c10::ReplaceAll(str, it.first, it.second);
  }
}

void PyWarningHandler::InternalHandler::process(const c10::Warning& warning) {
  warning_buffer_.push_back(warning);
}

PyWarningHandler::PyWarningHandler() noexcept(true)
    : prev_handler_(c10::WarningUtils::get_warning_handler()),
      in_exception_(false) {
  c10::WarningUtils::set_warning_handler(&internal_handler_);
}

// Get the Python warning type for a warning
PyObject* map_warning_to_python_type(const c10::Warning& warning) {
  struct Visitor {
    PyObject* operator()(const c10::UserWarning&) const {
      return PyExc_UserWarning;
    }
    PyObject* operator()(const c10::DeprecationWarning&) const {
      return PyExc_DeprecationWarning;
    }
  };
  return std::visit(Visitor(), warning.type());
}

/// See NOTE [ Conversion Cpp Python Warning ] for noexcept justification
/// NOLINTNEXTLINE(bugprone-exception-escape)
PyWarningHandler::~PyWarningHandler() noexcept(false) {
  c10::WarningUtils::set_warning_handler(prev_handler_);
  auto& warning_buffer = internal_handler_.warning_buffer_;

  if (!warning_buffer.empty()) {
    PyObject *type = nullptr, *value = nullptr, *traceback = nullptr;
    pybind11::gil_scoped_acquire gil;
    auto result = 0;
    if (in_exception_) {
      // This (combined with PyErr_Restore below) also works when no python
      // error has been set yet
      PyErr_Fetch(&type, &value, &traceback);
    }
    for (const auto& warning : warning_buffer) {
      auto source_location = warning.source_location();
      auto msg = warning.msg();
      processErrorMsgInplace(msg);
      if (source_location.file == nullptr) {
        result =
            PyErr_WarnEx(map_warning_to_python_type(warning), msg.c_str(), 1);
      } else if (warning.verbatim()) {
        // Sets the source location from the warning
        // Note: PyErr_WarnExplicit will disregard Python's warning filter
        // and always appear. This is in contrast to PyErr_WarnEx,
        // which respects the warning filter.
        result = PyErr_WarnExplicit(
            /*category=*/map_warning_to_python_type(warning),
            /*message=*/msg.c_str(),
            /*filename=*/source_location.file,
            /*lineno=*/static_cast<int>(source_location.line),
            /*module=*/nullptr,
            /*registry=*/nullptr);
      } else {
        // Lets Python set the source location and puts the C++ warning
        // location into the message.
        auto buf = fmt::format(
            "{} (Triggered internally at {}:{}.)",
            msg,
            source_location.file,
            source_location.line);
        result =
            PyErr_WarnEx(map_warning_to_python_type(warning), buf.c_str(), 1);
      }
      if (result < 0) {
        if (in_exception_) {
          // PyErr_Print prints the traceback to sys.stderr and
          // clears the error indicator
          PyErr_Print();
        } else {
          break;
        }
      }
    }
    warning_buffer.clear();
    if ((result < 0) && (!in_exception_)) {
      /// A warning raised an error, we need to force the parent
      /// function to return an error code.
      throw python_error();
    }
    if (in_exception_) {
      PyErr_Restore(type, value, traceback);
    }
  }
}

} // namespace torch
