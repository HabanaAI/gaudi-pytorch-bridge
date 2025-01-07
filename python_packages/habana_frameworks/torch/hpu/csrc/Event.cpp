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
#include <torch/csrc/cuda/Module.h>
#include <torch/csrc/utils/pybind.h>
#include <torch/csrc/utils/pycfunction_helpers.h>
#include <torch/csrc/utils/python_arg_parser.h>
#include "backend/synapse_helpers/device_types.h"
// #include <c10/cuda/CUDAGuard.h>

// #include <cuda_runtime_api.h>
#include <structmember.h>

#include "Event.h"
#include "Stream.h"

PyObject* THP_HPU_EventClass = nullptr;

static PyObject* THP_HPU_Event_pynew(
    PyTypeObject* type,
    PyObject* args,
    PyObject* kwargs) {
  HANDLE_TH_ERRORS
  unsigned char enable_timing = 0;

  // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
  constexpr const char* kwlist[] = {"enable_timing", nullptr};
  if (!PyArg_ParseTupleAndKeywords(
          args, kwargs, "|b", const_cast<char**>(kwlist), &enable_timing)) {
    return nullptr;
  }

  THPObjectPtr ptr(type->tp_alloc(type, 0));
  if (!ptr) {
    return nullptr;
  }

  THP_HPU_Event* self = (THP_HPU_Event*)ptr.get();
  unsigned int flags = (enable_timing ? 1 : 0);

  new (&self->hpu_event) at::hpu::HPUEvent(flags);

  return (PyObject*)ptr.release();
  END_HANDLE_TH_ERRORS
}

static void THP_HPU_Event_dealloc(THP_HPU_Event* self) {
  self->hpu_event.~HPUEvent();
  Py_TYPE(self)->tp_free((PyObject*)self);
}

static PyObject* THP_HPU_Event_get_hpu_event(
    THP_HPU_Event* self,
    [[maybe_unused]] void* unused) {
  HANDLE_TH_ERRORS
  return THPUtils_packInt64(self->hpu_event.event());
  END_HANDLE_TH_ERRORS
}

static PyObject* THP_HPU_Event_get_device(
    THP_HPU_Event* self,
    [[maybe_unused]] void* unused) {
  HANDLE_TH_ERRORS
  at::optional<at::Device> device = self->hpu_event.device();
  if (!device) {
    Py_RETURN_NONE;
  }
  return THPDevice_New(device.value());
  END_HANDLE_TH_ERRORS
}

static PyObject* THP_HPU_Event_record(PyObject* _self, PyObject* _stream) {
  HANDLE_TH_ERRORS
  auto self = (THP_HPU_Event*)_self;
  auto stream = (THP_HPU_Stream*)_stream;
  self->hpu_event.record(stream->hpu_stream);
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static PyObject* THP_HPU_Event_wait(PyObject* _self, PyObject* _stream) {
  HANDLE_TH_ERRORS {
    auto self = (THP_HPU_Event*)_self;
    auto stream = (THP_HPU_Stream*)_stream;
    pybind11::gil_scoped_release no_gil{};
    self->hpu_event.block(stream->hpu_stream);
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static PyObject* THP_HPU_Event_query(
    PyObject* _self,
    [[maybe_unused]] PyObject* noargs) {
  HANDLE_TH_ERRORS
  auto self = (THP_HPU_Event*)_self;
  return PyBool_FromLong(self->hpu_event.query());
  END_HANDLE_TH_ERRORS
}

static PyObject* THP_HPU_Event_elapsed_time(PyObject* _self, PyObject* _other) {
  HANDLE_TH_ERRORS
  auto self = (THP_HPU_Event*)_self;
  auto other = (THP_HPU_Event*)_other;
  return PyFloat_FromDouble(self->hpu_event.elapsed_time(other->hpu_event));
  END_HANDLE_TH_ERRORS
}

static PyObject* THP_HPU_Event_synchronize(
    PyObject* _self,
    [[maybe_unused]] PyObject* noargs) {
  HANDLE_TH_ERRORS {
    auto self = (THP_HPU_Event*)_self;
    pybind11::gil_scoped_release no_gil{};
    self->hpu_event.synchronize();
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,
// cppcoreguidelines-avoid-non-const-global-variables, modernize-avoid-c-arrays)
static struct PyGetSetDef THP_HPU_Event_properties[] = {
    {"device", (getter)THP_HPU_Event_get_device, nullptr, nullptr, nullptr},
    {"hpu_event",
     (getter)THP_HPU_Event_get_hpu_event,
     nullptr,
     nullptr,
     nullptr},
    {nullptr, nullptr, nullptr, nullptr, nullptr}};

// NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,
// cppcoreguidelines-avoid-non-const-global-variables, modernize-avoid-c-arrays)
static PyMethodDef THP_HPU_Event_methods[] = {
    {(char*)"record", THP_HPU_Event_record, METH_O, nullptr},
    {(char*)"wait", THP_HPU_Event_wait, METH_O, nullptr},
    {(char*)"query", THP_HPU_Event_query, METH_NOARGS, nullptr},
    {(char*)"elapsed_time", THP_HPU_Event_elapsed_time, METH_O, nullptr},
    {(char*)"synchronize", THP_HPU_Event_synchronize, METH_NOARGS, nullptr},
    {nullptr, nullptr, 0, nullptr}};

PyTypeObject THP_HPU_EventType = {
#if PY_VERSION_HEX >= 0x03000000
    PyVarObject_HEAD_INIT(NULL, 0)
#else
    PyObject_HEAD_INIT(NULL) 0, /* ob_size */
#endif
        "habana_frameworks.torch._hpu_C._HpuEventBase", /* tp_name */
    sizeof(THP_HPU_Event), /* tp_basicsize */
    0, /* tp_itemsize */
    (destructor)THP_HPU_Event_dealloc, /* tp_dealloc */
    0, /* tp_vectorcall_offset */
    nullptr, /* tp_getattr */
    nullptr, /* tp_setattr */
    nullptr, /* tp_reserved */
    nullptr, /* tp_repr */
    nullptr, /* tp_as_number */
    nullptr, /* tp_as_sequence */
    nullptr, /* tp_as_mapping */
    nullptr, /* tp_hash  */
    nullptr, /* tp_call */
    nullptr, /* tp_str */
    nullptr, /* tp_getattro */
    nullptr, /* tp_setattro */
    nullptr, /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, /* tp_flags */
    nullptr, /* tp_doc */
    nullptr, /* tp_traverse */
    nullptr, /* tp_clear */
    nullptr, /* tp_richcompare */
    0, /* tp_weaklistoffset */
    nullptr, /* tp_iter */
    nullptr, /* tp_iternext */
    THP_HPU_Event_methods, /* tp_methods */
    nullptr, /* tp_members */
    THP_HPU_Event_properties, /* tp_getset */
    nullptr, /* tp_base */
    nullptr, /* tp_dict */
    nullptr, /* tp_descr_get */
    nullptr, /* tp_descr_set */
    0, /* tp_dictoffset */
    nullptr, /* tp_init */
    nullptr, /* tp_alloc */
    THP_HPU_Event_pynew, /* tp_new */
    0, /* tp_free */
    0, /* tp_is_gc */
    0, /* tp_bases */
    0, /* tp_mro */
    0, /* tp_cache */
    0, /* tp_subclasses */
    0, /* tp_weaklist */
#if PY_VERSION_HEX >= 0x02030000
    0, /* tp_del */
#endif
#if PY_VERSION_HEX >= 0x02060000
    0, /* tp_version_tag */
#endif
#if PY_VERSION_HEX >= 0x03040000
    0, /* tp_finalize */
#endif
#ifdef COUNT_ALLOCS
    0, /* tp_allocs */
    0, /* tp_frees */
    0, /* tp_maxalloc */
#if PY_VERSION_HEX >= 0x02050000
    0, /* tp_prev */
#endif
    0, /* tp_next */
#endif
    nullptr, /* tp_vectorcall */
#if PY_VERSION_HEX >= 0x030c0000
    0, /* tp_watched */
#endif
#if PY_VERSION_HEX < 0x03090000
    nullptr, /* int (*tp_print)(PyObject *, FILE *, int); */
#endif
};

void THP_HPU_Event_init(PyObject* module) {
  THP_HPU_EventClass = (PyObject*)&THP_HPU_EventType;
  if (PyType_Ready(&THP_HPU_EventType) < 0) {
    throw python_error();
  }
  Py_INCREF(&THP_HPU_EventType);
  if (PyModule_AddObject(
          module, "_HpuEventBase", (PyObject*)&THP_HPU_EventType) < 0) {
    throw python_error();
  }
}
