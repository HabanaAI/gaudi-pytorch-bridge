/*******************************************************************************
 * Copyright (C) 2022-2024 Habana Labs, Ltd. an Intel Company
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

#include <pybind11/pybind11.h>
#include <torch/csrc/Device.h>
#include <torch/csrc/THP.h>
#include <torch/csrc/cuda/Module.h>
#include <torch/csrc/utils/pybind.h>
#include <torch/csrc/utils/python_numbers.h>

#include <structmember.h>

#include "Stream.h"
#include "backend/habana_device/hpu_cached_devices.h"

// namespace hpu {

PyObject* THP_HPU_StreamClass = nullptr;

static PyObject* THP_HPU_Stream_pynew(
    PyTypeObject* type,
    PyObject* args,
    PyObject* kwargs) {
  HANDLE_TH_ERRORS
  int priority = 0;
  int64_t stream_id = 0;
  int64_t device_index = 0;
  int64_t device_type = 0;
  uint64_t stream_ptr = 0;
  bool is_default_stream = false;

  // NOLINTNEXTLINE(modernize-avoid-c-arrays,cppcoreguidelines-avoid-c-arrays)
  constexpr const char* kwlist[] = {
      "priority",
      "stream_id",
      "device_index",
      "device_type",
      "stream_ptr",
      "is_default_stream",
      nullptr};
  if (!PyArg_ParseTupleAndKeywords(
          args,
          kwargs,
          "|iLLLKb",
          const_cast<char**>(kwlist),
          &priority,
          &stream_id,
          &device_index,
          &device_type,
          &stream_ptr,
          &is_default_stream)) {
    return nullptr;
  }

  THPObjectPtr ptr(type->tp_alloc(type, 0));
  if (!ptr) {
    return nullptr;
  }

  if (stream_ptr) {
    TORCH_CHECK(
        priority == 0, "Priority was explicitly set for a external stream")
  }

  const auto current_device = habana::HPURegistrar::get_device().id();

  auto stream = is_default_stream ? c10::hpu::getDefaultHPUStream(device_index)
                                  : (stream_id || device_index)
          ? c10::hpu::HPUStream::unpack3(
                stream_id,
                device_index,
                static_cast<c10::DeviceType>(device_type))
          : stream_ptr
              ? at::hpu::getStreamByStreamPtr(
                    reinterpret_cast<synapse_helpers::hpuStream_t>(stream_ptr),
                    current_device)
              : c10::hpu::getStreamFromPool((priority < 0), device_index);

  THP_HPU_Stream* self = (THP_HPU_Stream*)ptr.get();
  self->stream_id = static_cast<int64_t>(stream.id());
  self->device_index = static_cast<int64_t>(stream.device_index());
  self->device_type = static_cast<int64_t>(stream.device_type());
  new (&self->hpu_stream) c10::hpu::HPUStream(stream);

  return (PyObject*)ptr.release();
  END_HANDLE_TH_ERRORS
}

static void THP_HPU_Stream_dealloc(THP_HPU_Stream* self) {
  self->hpu_stream.~HPUStream();
  Py_TYPE(self)->tp_free((PyObject*)self);
}

[[maybe_unused]] static PyObject* THP_HPU_Stream_get_device(
    THP_HPU_Stream* self,
    [[maybe_unused]] void* unused) {
  HANDLE_TH_ERRORS
  return THPDevice_New(self->hpu_stream.device());
  END_HANDLE_TH_ERRORS
}

static PyObject* THP_HPU_Stream_get_hpu_stream(
    THP_HPU_Stream* self,
    [[maybe_unused]] void* unused) {
  HANDLE_TH_ERRORS

  return THPUtils_packInt64(self->hpu_stream.stream());
  // return PyLong_FromVoidPtr(self->hpu_stream.stream());
  END_HANDLE_TH_ERRORS
}

static PyObject* THP_HPU_Stream_get_priority(
    THP_HPU_Stream* self,
    [[maybe_unused]] void* unused) {
  HANDLE_TH_ERRORS
  return THPUtils_packInt64(self->hpu_stream.priority());
  END_HANDLE_TH_ERRORS
}

static PyObject* THP_HPU_Stream_priority_range(
    [[maybe_unused]] PyObject* _unused,
    [[maybe_unused]] PyObject* noargs) {
  HANDLE_TH_ERRORS
  auto [least_priority, greatest_priority] =
      c10::hpu::HPUStream::priority_range();
  return Py_BuildValue("(ii)", least_priority, greatest_priority);
  END_HANDLE_TH_ERRORS
}

static PyObject* THP_HPU_Stream_query(
    PyObject* _self,
    [[maybe_unused]] PyObject* noargs) {
  HANDLE_TH_ERRORS

  auto self = (THP_HPU_Stream*)_self;
  return PyBool_FromLong(self->hpu_stream.query());
  END_HANDLE_TH_ERRORS
}

static PyObject* THP_HPU_Stream_synchronize(
    PyObject* _self,
    [[maybe_unused]] PyObject* noargs) {
  HANDLE_TH_ERRORS {
    pybind11::gil_scoped_release no_gil;
    auto self = (THP_HPU_Stream*)_self;
    self->hpu_stream.synchronize();
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static PyObject* THP_HPU_Stream_eq(PyObject* _self, PyObject* _other) {
  HANDLE_TH_ERRORS
  auto self = (THP_HPU_Stream*)_self;
  auto other = (THP_HPU_Stream*)_other;
  return PyBool_FromLong(self->hpu_stream == other->hpu_stream);
  END_HANDLE_TH_ERRORS
}

// NOLINTNEXTLINE(modernize-avoid-c-arrays,
// cppcoreguidelines-avoid-non-const-global-variables,
// cppcoreguidelines-avoid-c-arrays)
static struct PyMemberDef THP_HPU_Stream_members[] = {
    {nullptr, 0, 0, 0, nullptr}};

// NOLINTNEXTLINE(modernize-avoid-c-arrays,
// cppcoreguidelines-avoid-non-const-global-variables,
// cppcoreguidelines-avoid-c-arrays)
static struct PyGetSetDef THP_HPU_Stream_properties[] = {
    {"hpu_stream",
     (getter)THP_HPU_Stream_get_hpu_stream,
     nullptr,
     nullptr,
     nullptr},
    {"priority",
     (getter)THP_HPU_Stream_get_priority,
     nullptr,
     nullptr,
     nullptr},
    {nullptr, nullptr, nullptr, nullptr, nullptr}};

// NOLINTNEXTLINE(modernize-avoid-c-arrays,
// cppcoreguidelines-avoid-non-const-global-variables,
// cppcoreguidelines-avoid-c-arrays)
static PyMethodDef THP_HPU_Stream_methods[] = {
    {(char*)"query", THP_HPU_Stream_query, METH_NOARGS, nullptr},
    {(char*)"synchronize", THP_HPU_Stream_synchronize, METH_NOARGS, nullptr},
    {(char*)"priority_range",
     THP_HPU_Stream_priority_range,
     METH_STATIC | METH_NOARGS,
     nullptr},
    {(char*)"__eq__", THP_HPU_Stream_eq, METH_O, nullptr},
    {nullptr, nullptr, 0, nullptr}};

PyTypeObject THP_HPU_StreamType = {
#if PY_VERSION_HEX >= 0x03000000
    PyVarObject_HEAD_INIT(NULL, 0)
#else
    PyObject_HEAD_INIT(NULL) 0, /* ob_size */
#endif
        "habana_frameworks.torch._hpu_C._HPUStreamBase", /* tp_name */
    sizeof(THP_HPU_Stream), /* tp_basicsize */
    0, /* tp_itemsize */
    (destructor)THP_HPU_Stream_dealloc, /* tp_dealloc */
    0, /* tp_vectorcall_offset  */
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
    THP_HPU_Stream_methods, /* tp_methods */
    THP_HPU_Stream_members, /* tp_members */
    THP_HPU_Stream_properties, /* tp_getset */
    nullptr, /* tp_base */
    nullptr, /* tp_dict */
    nullptr, /* tp_descr_get */
    nullptr, /* tp_descr_set */
    0, /* tp_dictoffset */
    nullptr, /* tp_init */
    nullptr, /* tp_alloc */
    THP_HPU_Stream_pynew, /* tp_new */
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

void THP_HPU_Stream_init(PyObject* module) {
  Py_INCREF(THPStreamClass);
  THP_HPU_StreamType.tp_base = THPStreamClass;
  THP_HPU_StreamClass = (PyObject*)&THP_HPU_StreamType;
  if (PyType_Ready(&THP_HPU_StreamType) < 0) {
    throw python_error();
  }
  Py_INCREF(&THP_HPU_StreamType);
  if (PyModule_AddObject(
          module, "_HpuStreamBase", (PyObject*)&THP_HPU_StreamType) < 0) {
    throw python_error();
  }
}

// } // namespace hpu