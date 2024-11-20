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

#include <pybind11/pybind11.h>
#include <torch/csrc/Device.h>
#include <torch/csrc/THP.h>
#include <torch/csrc/utils/pycfunction_helpers.h>

#include "Module.h"
#include "backend/habana_device/HPUDevice.h"
#include "backend/habana_device/HPUStream.h"

PyObject* THP_HPU_Module_getCurrentStream_wrap(
    [[maybe_unused]] PyObject* self,
    PyObject* device_index) {
  HANDLE_TH_ERRORS
  TORCH_CHECK(
      THPUtils_checkLong(device_index), "invalid argument to getCurrentStream");
  auto c10_device_index = THPUtils_unpackDeviceIndex(device_index);
  auto stream = c10::hpu::getCurrentHPUStream(c10_device_index);

  PyObject* output_tuple = PyTuple_New(3);
  PyTuple_SetItem(
      output_tuple, 0, THPUtils_packInt64(static_cast<int64_t>(stream.id())));
  PyTuple_SetItem(
      output_tuple, 1, THPUtils_packDeviceIndex(stream.device_index()));
  PyTuple_SetItem(
      output_tuple,
      2,
      THPUtils_packInt64(static_cast<int64_t>(stream.device_type())));
  return output_tuple;
  END_HANDLE_TH_ERRORS
}

PyObject* THP_HPU_Module_getCurrentStream_raw(
    [[maybe_unused]] PyObject* self,
    PyObject* device_index) {
  HANDLE_TH_ERRORS
  TORCH_CHECK(
      THPUtils_checkLong(device_index), "invalid argument to getCurrentStream");
  auto c10_device_index = THPUtils_unpackDeviceIndex(device_index);
  return THPUtils_packInt64(
      c10::hpu::getCurrentHPUStream(c10_device_index).stream());
  END_HANDLE_TH_ERRORS
}

PyObject* THP_HPU_Module_getDefaultStream_wrap(
    [[maybe_unused]] PyObject* self,
    PyObject* device_index) {
  HANDLE_TH_ERRORS
  TORCH_CHECK(
      THPUtils_checkLong(device_index), "invalid argument to getDefaultStream");
  auto c10_device_index = THPUtils_unpackDeviceIndex(device_index);
  auto stream = c10::hpu::getDefaultHPUStream(c10_device_index);
  PyObject* output_tuple = PyTuple_New(3);
  PyTuple_SetItem(
      output_tuple, 0, THPUtils_packInt64(static_cast<int64_t>(stream.id())));
  PyTuple_SetItem(
      output_tuple, 1, THPUtils_packDeviceIndex(stream.device_index()));
  PyTuple_SetItem(
      output_tuple,
      2,
      THPUtils_packInt64(static_cast<int64_t>(stream.device_type())));
  return output_tuple;
  END_HANDLE_TH_ERRORS
}

PyObject* THP_HPU_Module_getStreamInfo_wrap(
    [[maybe_unused]] PyObject* self,
    PyObject* args,
    PyObject* kwargs) {
  HANDLE_TH_ERRORS
  int64_t stream_id = 0;
  int64_t device_index = 0;
  int64_t device_type = 0;
  // NOLINTNEXTLINE(modernize-avoid-c-arrays,cppcoreguidelines-avoid-c-arrays)
  constexpr const char* kwlist[] = {
      "stream_id", "device_index", "device_type", nullptr};

  if (!PyArg_ParseTupleAndKeywords(
          args,
          kwargs,
          "|LLL",
          // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
          const_cast<char**>(kwlist),
          &stream_id,
          &device_index,
          &device_type)) {
  }

  auto stream = c10::hpu::HPUStream::unpack3(
      stream_id,
      static_cast<c10::DeviceIndex>(device_index),
      static_cast<c10::DeviceType>(device_type));

  TORCH_CHECK(
      stream.device_index() ==
          (int64_t)habana::HPUDeviceContext::get_device().id(),
      "getStreamInfo invalid device_index");
  PyObject* output_tuple = PyTuple_New(2);
  PyTuple_SetItem(output_tuple, 0, THPDevice_New(stream.device()));
  PyTuple_SetItem(
      output_tuple, 1, THPUtils_packInt64(static_cast<int64_t>(stream.id())));
  return output_tuple;
  END_HANDLE_TH_ERRORS
}
PyObject* THP_HPU_Module_setStream_wrap(
    [[maybe_unused]] PyObject* self,
    PyObject* args,
    PyObject* kwargs) {
  HANDLE_TH_ERRORS
  int64_t stream_id = 0;
  int64_t device_index = 0;
  int64_t device_type = 0;

  // NOLINTNEXTLINE(modernize-avoid-c-arrays,cppcoreguidelines-avoid-c-arrays)
  constexpr const char* kwlist[] = {
      "stream_id", "device_index", "device_type", nullptr};
  if (!PyArg_ParseTupleAndKeywords(
          args,
          kwargs,
          "|LLL",
          // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
          const_cast<char**>(kwlist),
          &stream_id,
          &device_index,
          &device_type)) {
  }

  auto stream = c10::hpu::HPUStream::unpack3(
      stream_id,
      static_cast<c10::DeviceIndex>(device_index),
      static_cast<c10::DeviceType>(device_type));

  TORCH_CHECK(
      stream.device_index() ==
          (int64_t)habana::HPUDeviceContext::get_device().id(),
      "setStream invalid device_index");

  c10::hpu::setCurrentHPUStream(stream);
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// NOLINTNEXTLINE(*-c-arrays*, *-global-variables)
static struct PyMethodDef _THP_HPU_Module_methods[] = {
    {"_hpu_getCurrentStream",
     THP_HPU_Module_getCurrentStream_wrap,
     METH_O,
     nullptr},
    {"_hpu_getCurrentRawStream",
     THP_HPU_Module_getCurrentStream_raw,
     METH_O,
     nullptr},
    {"_hpu_getDefaultStream",
     THP_HPU_Module_getDefaultStream_wrap,
     METH_O,
     nullptr},
    {"_hpu_getStreamInfo",
     castPyCFunctionWithKeywords(THP_HPU_Module_getStreamInfo_wrap),
     METH_VARARGS | METH_KEYWORDS,
     nullptr},
    {"_hpu_setStream",
     castPyCFunctionWithKeywords(THP_HPU_Module_setStream_wrap),
     METH_VARARGS | METH_KEYWORDS,
     nullptr},
};

PyMethodDef* THP_HPU_Module_methods() {
  return _THP_HPU_Module_methods;
}
