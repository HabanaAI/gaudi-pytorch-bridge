/**
 * Copyright (c) 2025 Intel Corporation
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

#include <Python.h>
#include <dlfcn.h>
#include <frameobject.h> // Python header required for Pythons up to 3.10
#include <iostream>
#include <memory>
#include <ostream>
#include <string>

namespace {

template <typename... Args>
void ShimAssert(bool condition, Args&&... args) {
  if (!condition) {
    std::cerr << "ERROR: ";
    ((std::cerr << args << " "), ...);
    std::cerr << std::endl;
    std::abort();
  }
}

class DLHandle {
 public:
  DLHandle(const char* lib_name, int flags) : handle_{dlopen(lib_name, flags)} {
    ShimAssert(
        handle_ != nullptr,
        "Failed to load",
        lib_name != nullptr ? lib_name : "the main program's symbols",
        " due to the following error: ",
        dlerror());
  }

  DLHandle(void* pseudo_handle) : handle_{pseudo_handle} {
    ShimAssert(
        pseudo_handle == RTLD_DEFAULT || pseudo_handle == RTLD_NEXT,
        "Invalid pseudo handle provided to DLHandle: ",
        pseudo_handle);
  }

  void* get() const {
    return handle_.get();
  }

 private:
  struct DLCloseDeleter {
    void operator()(void* handle) const {
      static_assert(RTLD_DEFAULT == nullptr);
      if (handle != RTLD_DEFAULT && handle != RTLD_NEXT) {
        dlclose(handle);
      }
    }
  };

  std::unique_ptr<void, DLCloseDeleter> handle_;
};

void* CheckedDlsym(const DLHandle& lib, const char* name) {
  auto address = dlsym(lib.get(), name);
  ShimAssert(
      address != nullptr,
      "Failed to load symbol: ",
      name,
      "due to the following error: ",
      dlerror());
  return address;
}

struct TypedLoad {
  TypedLoad(const DLHandle& lib, const char* name) : lib_{lib}, name_{name} {}

  template <typename T>
  operator T() {
    return reinterpret_cast<T>(CheckedDlsym(lib_, name_));
  }

  const DLHandle& lib_;
  const char* name_;
};

} // namespace

class Proxy {
 private:
  // must be declared before the wrapped symbols
  DLHandle embed_{determinePythonEmbedHandle()};

 public:
  static const Proxy& instance() {
    static Proxy instance;
    return instance;
  }

// GCC gives us a parentheses warning when we change to (name)
// NOLINTNEXTLINE(bugprone-macro-parentheses)
#define WRAP_SYMBOL(name) decltype(&::name) name = TypedLoad(embed_, #name)
  WRAP_SYMBOL(PyBaseObject_Type);
  WRAP_SYMBOL(PyBool_Type);
  WRAP_SYMBOL(PyBuffer_Release);
  WRAP_SYMBOL(PyByteArray_AsString);
  WRAP_SYMBOL(PyByteArray_Size);
  WRAP_SYMBOL(PyByteArray_Type);
  WRAP_SYMBOL(PyBytes_AsString);
  WRAP_SYMBOL(PyBytes_AsStringAndSize);
  WRAP_SYMBOL(PyBytes_FromString);
  WRAP_SYMBOL(PyBytes_Size);
  WRAP_SYMBOL(PyCallable_Check);
  WRAP_SYMBOL(PyCFunction_Type);
  WRAP_SYMBOL(PyCMethod_New);
  WRAP_SYMBOL(PyCapsule_GetContext);
  WRAP_SYMBOL(PyCapsule_GetName);
  WRAP_SYMBOL(PyCapsule_GetPointer);
  WRAP_SYMBOL(PyCapsule_New);
  WRAP_SYMBOL(PyCapsule_SetContext);
  WRAP_SYMBOL(PyCapsule_SetPointer);
  WRAP_SYMBOL(PyCapsule_Type);
  WRAP_SYMBOL(PyDict_Copy);
  WRAP_SYMBOL(PyDict_DelItemString);
  WRAP_SYMBOL(PyDict_GetItemWithError);
  WRAP_SYMBOL(PyDict_New);
  WRAP_SYMBOL(PyDict_Next);
  WRAP_SYMBOL(PyDict_Size);
  WRAP_SYMBOL(PyDict_Type);
  WRAP_SYMBOL(PyErr_Clear);
  WRAP_SYMBOL(PyErr_Fetch);
  WRAP_SYMBOL(PyErr_Format);
  WRAP_SYMBOL(PyErr_NormalizeException);
  WRAP_SYMBOL(PyErr_Occurred);
  WRAP_SYMBOL(PyErr_Restore);
  WRAP_SYMBOL(PyErr_SetString);
  WRAP_SYMBOL(PyErr_WarnEx);
  WRAP_SYMBOL(PyErr_WriteUnraisable);
  WRAP_SYMBOL(PyEval_AcquireThread);
  WRAP_SYMBOL(PyEval_GetBuiltins);
  WRAP_SYMBOL(PyEval_RestoreThread);
  WRAP_SYMBOL(PyEval_SaveThread);
  WRAP_SYMBOL(PyExc_BufferError);
  WRAP_SYMBOL(PyExc_FutureWarning);
  WRAP_SYMBOL(PyExc_ImportError);
  WRAP_SYMBOL(PyExc_IndexError);
  WRAP_SYMBOL(PyExc_KeyError);
  WRAP_SYMBOL(PyExc_MemoryError);
  WRAP_SYMBOL(PyExc_OverflowError);
  WRAP_SYMBOL(PyExc_RuntimeError);
  WRAP_SYMBOL(PyExc_StopIteration);
  WRAP_SYMBOL(PyExc_SystemError);
  WRAP_SYMBOL(PyExc_TypeError);
  WRAP_SYMBOL(PyExc_ValueError);
  WRAP_SYMBOL(PyException_SetCause);
  WRAP_SYMBOL(PyException_SetContext);
  WRAP_SYMBOL(PyException_SetTraceback);
  WRAP_SYMBOL(PyFloat_Type);
  WRAP_SYMBOL(PyFrame_GetBack);
  WRAP_SYMBOL(PyFrame_GetCode);
  WRAP_SYMBOL(PyFrame_GetLineNumber);
  WRAP_SYMBOL(PyGILState_Check);
  WRAP_SYMBOL(PyGILState_Ensure);
  WRAP_SYMBOL(PyGILState_GetThisThreadState);
  WRAP_SYMBOL(PyGILState_Release);
  WRAP_SYMBOL(PyImport_ImportModule);
  WRAP_SYMBOL(PyIndex_Check);
  WRAP_SYMBOL(PyInstanceMethod_New);
  WRAP_SYMBOL(PyInstanceMethod_Type);
  WRAP_SYMBOL(PyInterpreterState_Get);
  WRAP_SYMBOL(PyInterpreterState_GetDict);
  WRAP_SYMBOL(PyIter_Check);
  WRAP_SYMBOL(PyIter_Next);
  WRAP_SYMBOL(PyList_GetItem);
  WRAP_SYMBOL(PyList_New);
  WRAP_SYMBOL(PyList_Size);
  WRAP_SYMBOL(PyLong_AsLong);
  WRAP_SYMBOL(PyLong_FromSize_t);
  WRAP_SYMBOL(PyLong_Type);
  WRAP_SYMBOL(PyMethod_Type);
  WRAP_SYMBOL(PyModule_Type);
  WRAP_SYMBOL(PyMem_Calloc);
  WRAP_SYMBOL(PyMem_Free);
  WRAP_SYMBOL(PyModule_AddObject);
  WRAP_SYMBOL(PyModule_Create2);
  WRAP_SYMBOL(PyNumber_Check);
  WRAP_SYMBOL(PyNumber_Long);
  WRAP_SYMBOL(PyObject_CallFunctionObjArgs);
  WRAP_SYMBOL(PyObject_CallObject);
  WRAP_SYMBOL(PyObject_ClearWeakRefs);
  WRAP_SYMBOL(PyObject_GC_UnTrack);
  WRAP_SYMBOL(PyObject_GenericGetDict);
  WRAP_SYMBOL(PyObject_GenericSetDict);
  WRAP_SYMBOL(PyObject_GetAttr);
  WRAP_SYMBOL(PyObject_GetAttrString);
  WRAP_SYMBOL(PyObject_GetIter);
  WRAP_SYMBOL(PyObject_HasAttrString);
  WRAP_SYMBOL(PyObject_IsInstance);
  WRAP_SYMBOL(PyObject_LengthHint);
  WRAP_SYMBOL(PyObject_Malloc);
  WRAP_SYMBOL(PyObject_Repr);
  WRAP_SYMBOL(PyObject_SetAttr);
  WRAP_SYMBOL(PyObject_SetAttrString);
  WRAP_SYMBOL(PyObject_SetItem);
  WRAP_SYMBOL(PyObject_Str);
  WRAP_SYMBOL(PyProperty_Type);
  WRAP_SYMBOL(PySequence_Tuple);
  WRAP_SYMBOL(PySlice_AdjustIndices);
  WRAP_SYMBOL(PySlice_Type);
  WRAP_SYMBOL(PySlice_Unpack);
  WRAP_SYMBOL(PyThreadState_Clear);
  WRAP_SYMBOL(PyThreadState_DeleteCurrent);
  WRAP_SYMBOL(PyThreadState_Get);
  WRAP_SYMBOL(PyThreadState_New);
  WRAP_SYMBOL(PyThread_tss_alloc);
  WRAP_SYMBOL(PyThread_tss_create);
  WRAP_SYMBOL(PyThread_tss_get);
  WRAP_SYMBOL(PyThread_tss_set);
  WRAP_SYMBOL(PyTuple_GetItem);
  WRAP_SYMBOL(PyTuple_New);
  WRAP_SYMBOL(PyTuple_SetItem);
  WRAP_SYMBOL(PyTuple_Size);
  WRAP_SYMBOL(PyType_IsSubtype);
  WRAP_SYMBOL(PyType_Ready);
  WRAP_SYMBOL(PyType_Type);
  WRAP_SYMBOL(PyUnicode_AsEncodedString);
  WRAP_SYMBOL(PyUnicode_AsUTF8AndSize);
  WRAP_SYMBOL(PyUnicode_AsUTF8String);
  WRAP_SYMBOL(PyUnicode_DecodeUTF8);
  WRAP_SYMBOL(PyUnicode_FromFormat);
  WRAP_SYMBOL(PyUnicode_FromString);
  WRAP_SYMBOL(PyWeakref_NewRef);
  WRAP_SYMBOL(Py_GetVersion);
  WRAP_SYMBOL(Py_IsInitialized);
  WRAP_SYMBOL(_PyObject_GetDictPtr);
  WRAP_SYMBOL(_PyThreadState_UncheckedGet);
  WRAP_SYMBOL(_PyType_Lookup);
  WRAP_SYMBOL(_Py_Dealloc);
  WRAP_SYMBOL(_Py_FalseStruct);
  WRAP_SYMBOL(_Py_NoneStruct);
  WRAP_SYMBOL(_Py_NotImplementedStruct);
  WRAP_SYMBOL(_Py_TrueStruct);
#undef WRAP_SYMBOL

 private:
  Proxy() = default;

  static DLHandle determinePythonEmbedHandle() {
    // We have to load Python.Embed symbols from either the current process or
    // the Python library, depending on how we were loaded.
    //
    // If we were loaded by the Python interpreter, there should be a Py_Main
    // available in the current process. Otherwise, we must load the
    // Python library ourselves and dlsym with it.
    //
    // Note that we can't shim Py_Main for this approach to work.
    if (dlsym(RTLD_DEFAULT, "Py_Main") != nullptr && dlerror() == nullptr) {
      return DLHandle{RTLD_DEFAULT};
    } else {
      return DLHandle{namePythonLib().c_str(), RTLD_NOW | RTLD_GLOBAL};
    }
  }

  static std::string namePythonLib() {
    auto major = std::to_string(PY_MAJOR_VERSION);
    auto minor = std::to_string(PY_MINOR_VERSION);
    return "libpython" + major + "." + minor + ".so";
  }
};

extern "C" {

// Wrappers are sorted alphabetically by name.
// Objects are listed after functions - at the bottom.

void PyBuffer_Release(Py_buffer* view) {
  Proxy::instance().PyBuffer_Release(view);
}

char* PyByteArray_AsString(PyObject* obj) {
  return Proxy::instance().PyByteArray_AsString(obj);
}

Py_ssize_t PyByteArray_Size(PyObject* obj) {
  return Proxy::instance().PyByteArray_Size(obj);
}

char* PyBytes_AsString(PyObject* obj) {
  return Proxy::instance().PyBytes_AsString(obj);
}

int PyBytes_AsStringAndSize(PyObject* obj, char** buffer, Py_ssize_t* length) {
  return Proxy::instance().PyBytes_AsStringAndSize(obj, buffer, length);
}

PyObject* PyBytes_FromString(const char* str) {
  return Proxy::instance().PyBytes_FromString(str);
}

Py_ssize_t PyBytes_Size(PyObject* obj) {
  return Proxy::instance().PyBytes_Size(obj);
}

int PyCallable_Check(PyObject* obj) {
  return Proxy::instance().PyCallable_Check(obj);
}

PyObject* PyCMethod_New(
    PyMethodDef* ml,
    PyObject* self,
    PyObject* module,
    PyTypeObject* cls) {
  return Proxy::instance().PyCMethod_New(ml, self, module, cls);
}

void* PyCapsule_GetContext(PyObject* capsule) {
  return Proxy::instance().PyCapsule_GetContext(capsule);
}

void* PyCapsule_GetPointer(PyObject* capsule, const char* name) {
  return Proxy::instance().PyCapsule_GetPointer(capsule, name);
}

PyObject* PyCapsule_New(
    void* pointer,
    const char* name,
    PyCapsule_Destructor destructor) {
  return Proxy::instance().PyCapsule_New(pointer, name, destructor);
}

int PyCapsule_SetContext(PyObject* capsule, void* context) {
  return Proxy::instance().PyCapsule_SetContext(capsule, context);
}

int PyCapsule_SetPointer(PyObject* capsule, void* pointer) {
  return Proxy::instance().PyCapsule_SetPointer(capsule, pointer);
}

PyObject* PyDict_Copy(PyObject* dict) {
  return Proxy::instance().PyDict_Copy(dict);
}

int PyDict_DelItemString(PyObject* dict, const char* key) {
  return Proxy::instance().PyDict_DelItemString(dict, key);
}

PyObject* PyDict_GetItemWithError(PyObject* dict, PyObject* key) {
  return Proxy::instance().PyDict_GetItemWithError(dict, key);
}

PyObject* PyDict_New() {
  return Proxy::instance().PyDict_New();
}

int PyDict_Next(
    PyObject* dict,
    Py_ssize_t* ppos,
    PyObject** pkey,
    PyObject** pvalue) {
  return Proxy::instance().PyDict_Next(dict, ppos, pkey, pvalue);
}

Py_ssize_t PyDict_Size(PyObject* dict) {
  return Proxy::instance().PyDict_Size(dict);
}

void PyErr_Clear() {
  Proxy::instance().PyErr_Clear();
}

void PyErr_Fetch(PyObject** ptype, PyObject** pvalue, PyObject** ptraceback) {
  Proxy::instance().PyErr_Fetch(ptype, pvalue, ptraceback);
}

PyObject* PyErr_Format(PyObject* exception, const char* format, ...) {
  va_list args;
  va_start(args, format);
  PyObject* result = Proxy::instance().PyErr_Format(exception, format, args);
  va_end(args);
  return result;
}

void PyErr_NormalizeException(PyObject** exc, PyObject** val, PyObject** tb) {
  Proxy::instance().PyErr_NormalizeException(exc, val, tb);
}

PyObject* PyErr_Occurred() {
  return Proxy::instance().PyErr_Occurred();
}

void PyErr_Restore(PyObject* type, PyObject* value, PyObject* traceback) {
  Proxy::instance().PyErr_Restore(type, value, traceback);
}

void PyErr_SetString(PyObject* exception, const char* string) {
  Proxy::instance().PyErr_SetString(exception, string);
}

int PyErr_WarnEx(
    PyObject* category,
    const char* message,
    Py_ssize_t stack_level) {
  return Proxy::instance().PyErr_WarnEx(category, message, stack_level);
}

void PyErr_WriteUnraisable(PyObject* obj) {
  Proxy::instance().PyErr_WriteUnraisable(obj);
}

void PyEval_AcquireThread(PyThreadState* tstate) {
  Proxy::instance().PyEval_AcquireThread(tstate);
}

PyObject* PyEval_GetBuiltins() {
  return Proxy::instance().PyEval_GetBuiltins();
}

void PyEval_RestoreThread(PyThreadState* tstate) {
  Proxy::instance().PyEval_RestoreThread(tstate);
}

PyThreadState* PyEval_SaveThread() {
  return Proxy::instance().PyEval_SaveThread();
}

void PyException_SetCause(PyObject* exception, PyObject* cause) {
  Proxy::instance().PyException_SetCause(exception, cause);
}

void PyException_SetContext(PyObject* exception, PyObject* context) {
  Proxy::instance().PyException_SetContext(exception, context);
}

int PyException_SetTraceback(PyObject* exception, PyObject* traceback) {
  return Proxy::instance().PyException_SetTraceback(exception, traceback);
}

PyFrameObject* PyFrame_GetBack(PyFrameObject* frame) {
  return Proxy::instance().PyFrame_GetBack(frame);
}

PyCodeObject* PyFrame_GetCode(PyFrameObject* frame) {
  return Proxy::instance().PyFrame_GetCode(frame);
}

int PyFrame_GetLineNumber(PyFrameObject* frame) {
  return Proxy::instance().PyFrame_GetLineNumber(frame);
}

int PyGILState_Check() {
  return Proxy::instance().PyGILState_Check();
}

PyGILState_STATE PyGILState_Ensure() {
  return Proxy::instance().PyGILState_Ensure();
}

PyThreadState* PyGILState_GetThisThreadState() {
  return Proxy::instance().PyGILState_GetThisThreadState();
}

void PyGILState_Release(PyGILState_STATE state) {
  Proxy::instance().PyGILState_Release(state);
}

PyObject* PyImport_ImportModule(const char* name) {
  return Proxy::instance().PyImport_ImportModule(name);
}

int PyIndex_Check(PyObject* obj) {
  return Proxy::instance().PyIndex_Check(obj);
}

PyObject* PyInstanceMethod_New(PyObject* func) {
  return Proxy::instance().PyInstanceMethod_New(func);
}

PyInterpreterState* PyInterpreterState_Get() {
  return Proxy::instance().PyInterpreterState_Get();
}

PyObject* PyInterpreterState_GetDict(PyInterpreterState* interp) {
  return Proxy::instance().PyInterpreterState_GetDict(interp);
}

int PyIter_Check(PyObject* obj) {
  return Proxy::instance().PyIter_Check(obj);
}

PyObject* PyIter_Next(PyObject* obj) {
  return Proxy::instance().PyIter_Next(obj);
}

PyObject* PyList_GetItem(PyObject* list, Py_ssize_t index) {
  return Proxy::instance().PyList_GetItem(list, index);
}

PyObject* PyList_New(Py_ssize_t size) {
  return Proxy::instance().PyList_New(size);
}

Py_ssize_t PyList_Size(PyObject* list) {
  return Proxy::instance().PyList_Size(list);
}

long PyLong_AsLong(PyObject* obj) {
  return Proxy::instance().PyLong_AsLong(obj);
}

PyObject* PyLong_FromSize_t(size_t size) {
  return Proxy::instance().PyLong_FromSize_t(size);
}

void* PyMem_Calloc(size_t nelem, size_t elsize) {
  return Proxy::instance().PyMem_Calloc(nelem, elsize);
}

void PyMem_Free(void* ptr) {
  Proxy::instance().PyMem_Free(ptr);
}

int PyModule_AddObject(PyObject* mod, const char* str, PyObject* value) {
  return Proxy::instance().PyModule_AddObject(mod, str, value);
}

PyObject* PyModule_Create2(PyModuleDef* mod, int apiver) {
  return Proxy::instance().PyModule_Create2(mod, apiver);
}

int PyNumber_Check(PyObject* obj) {
  return Proxy::instance().PyNumber_Check(obj);
}

PyObject* PyNumber_Long(PyObject* obj) {
  return Proxy::instance().PyNumber_Long(obj);
}

PyObject* PyObject_CallFunctionObjArgs(PyObject* callable, ...) {
  va_list args;
  va_start(args, callable);
  PyObject* result =
      Proxy::instance().PyObject_CallFunctionObjArgs(callable, args);
  va_end(args);
  return result;
}

PyObject* PyObject_CallObject(PyObject* callable, PyObject* args) {
  return Proxy::instance().PyObject_CallObject(callable, args);
}

void PyObject_ClearWeakRefs(PyObject* obj) {
  Proxy::instance().PyObject_ClearWeakRefs(obj);
}

void PyObject_GC_UnTrack(void* obj) {
  Proxy::instance().PyObject_GC_UnTrack(obj);
}

PyObject* PyObject_GenericGetDict(PyObject* obj, void* context) {
  return Proxy::instance().PyObject_GenericGetDict(obj, context);
}

int PyObject_GenericSetDict(PyObject* obj, PyObject* dict, void* context) {
  return Proxy::instance().PyObject_GenericSetDict(obj, dict, context);
}

PyObject* PyObject_GetAttr(PyObject* obj, PyObject* obj2) {
  return Proxy::instance().PyObject_GetAttr(obj, obj2);
}

PyObject* PyObject_GetAttrString(PyObject* obj, const char* attr_name) {
  return Proxy::instance().PyObject_GetAttrString(obj, attr_name);
}

PyObject* PyObject_GetIter(PyObject* obj) {
  return Proxy::instance().PyObject_GetIter(obj);
}

int PyObject_HasAttrString(PyObject* obj, const char* attr_name) {
  return Proxy::instance().PyObject_HasAttrString(obj, attr_name);
}

int PyObject_IsInstance(PyObject* inst, PyObject* cls) {
  return Proxy::instance().PyObject_IsInstance(inst, cls);
}

Py_ssize_t PyObject_LengthHint(PyObject* obj, Py_ssize_t size) {
  return Proxy::instance().PyObject_LengthHint(obj, size);
}

void* PyObject_Malloc(size_t size) {
  return Proxy::instance().PyObject_Malloc(size);
}

PyObject* PyObject_Repr(PyObject* obj) {
  return Proxy::instance().PyObject_Repr(obj);
}

int PyObject_SetAttr(PyObject* obj, PyObject* attr, PyObject* value) {
  return Proxy::instance().PyObject_SetAttr(obj, attr, value);
}

int PyObject_SetAttrString(
    PyObject* obj,
    const char* attr_name,
    PyObject* value) {
  return Proxy::instance().PyObject_SetAttrString(obj, attr_name, value);
}

int PyObject_SetItem(PyObject* obj, PyObject* key, PyObject* value) {
  return Proxy::instance().PyObject_SetItem(obj, key, value);
}

PyObject* PyObject_Str(PyObject* obj) {
  return Proxy::instance().PyObject_Str(obj);
}

PyObject* PySequence_Tuple(PyObject* obj) {
  return Proxy::instance().PySequence_Tuple(obj);
}

Py_ssize_t PySlice_AdjustIndices(
    Py_ssize_t length,
    Py_ssize_t* start,
    Py_ssize_t* stop,
    Py_ssize_t step) {
  return Proxy::instance().PySlice_AdjustIndices(length, start, stop, step);
}

int PySlice_Unpack(
    PyObject* slice,
    Py_ssize_t* start,
    Py_ssize_t* stop,
    Py_ssize_t* step) {
  return Proxy::instance().PySlice_Unpack(slice, start, stop, step);
}

void PyThreadState_Clear(PyThreadState* tstate) {
  Proxy::instance().PyThreadState_Clear(tstate);
}

void PyThreadState_DeleteCurrent() {
  Proxy::instance().PyThreadState_DeleteCurrent();
}

PyThreadState* PyThreadState_Get() {
  return Proxy::instance().PyThreadState_Get();
}

PyThreadState* PyThreadState_New(PyInterpreterState* interp) {
  return Proxy::instance().PyThreadState_New(interp);
}

Py_tss_t* PyThread_tss_alloc() {
  return Proxy::instance().PyThread_tss_alloc();
}

int PyThread_tss_create(Py_tss_t* key) {
  return Proxy::instance().PyThread_tss_create(key);
}

void* PyThread_tss_get(Py_tss_t* key) {
  return Proxy::instance().PyThread_tss_get(key);
}

int PyThread_tss_set(Py_tss_t* key, void* value) {
  return Proxy::instance().PyThread_tss_set(key, value);
}

PyObject* PyTuple_GetItem(PyObject* tuple, Py_ssize_t pos) {
  return Proxy::instance().PyTuple_GetItem(tuple, pos);
}

PyObject* PyTuple_New(Py_ssize_t size) {
  return Proxy::instance().PyTuple_New(size);
}

int PyTuple_SetItem(PyObject* tuple, Py_ssize_t pos, PyObject* item) {
  return Proxy::instance().PyTuple_SetItem(tuple, pos, item);
}

Py_ssize_t PyTuple_Size(PyObject* tuple) {
  return Proxy::instance().PyTuple_Size(tuple);
}

int PyType_IsSubtype(PyTypeObject* a, PyTypeObject* b) {
  return Proxy::instance().PyType_IsSubtype(a, b);
}

int PyType_Ready(PyTypeObject* type) {
  return Proxy::instance().PyType_Ready(type);
}

PyObject* PyUnicode_AsEncodedString(
    PyObject* unicode,
    const char* encoding,
    const char* errors) {
  return Proxy::instance().PyUnicode_AsEncodedString(unicode, encoding, errors);
}

PyObject* PyUnicode_AsUTF8String(PyObject* unicode) {
  return Proxy::instance().PyUnicode_AsUTF8String(unicode);
}

PyObject* PyUnicode_DecodeUTF8(
    const char* string,
    Py_ssize_t length,
    const char* errors) {
  return Proxy::instance().PyUnicode_DecodeUTF8(string, length, errors);
}

PyObject* PyUnicode_FromFormat(const char* format, ...) {
  return Proxy::instance().PyUnicode_FromFormat(format);
}

PyObject* PyUnicode_FromString(const char* str) {
  return Proxy::instance().PyUnicode_FromString(str);
}

PyObject* PyWeakref_NewRef(PyObject* obj, PyObject* callback) {
  return Proxy::instance().PyWeakref_NewRef(obj, callback);
}

const char* Py_GetVersion() {
  return Proxy::instance().Py_GetVersion();
}

int Py_IsInitialized() {
  return Proxy::instance().Py_IsInitialized();
}

PyObject** _PyObject_GetDictPtr(PyObject* obj) {
  return Proxy::instance()._PyObject_GetDictPtr(obj);
}

PyThreadState* _PyThreadState_UncheckedGet() {
  return Proxy::instance()._PyThreadState_UncheckedGet();
}

PyObject* _PyType_Lookup(PyTypeObject* type, PyObject* name) {
  return Proxy::instance()._PyType_Lookup(type, name);
}

void _Py_Dealloc(PyObject* obj) {
  Proxy::instance()._Py_Dealloc(obj);
}

const char* PyCapsule_GetName(PyObject* capsule) {
  return Proxy::instance().PyCapsule_GetName(capsule);
}

const char* PyUnicode_AsUTF8AndSize(PyObject* unicode, Py_ssize_t* size) {
  return Proxy::instance().PyUnicode_AsUTF8AndSize(unicode, size);
}

// Globals
PyLongObject _Py_FalseStruct = *Proxy::instance()._Py_FalseStruct;
PyObject _Py_NoneStruct = *Proxy::instance()._Py_NoneStruct;
PyObject _Py_NotImplementedStruct = *Proxy::instance()._Py_NotImplementedStruct;
PyLongObject _Py_TrueStruct = *Proxy::instance()._Py_TrueStruct;
PyTypeObject PyBaseObject_Type = *Proxy::instance().PyBaseObject_Type;
PyTypeObject PyBool_Type = *Proxy::instance().PyBool_Type;
PyTypeObject PyByteArray_Type = *Proxy::instance().PyByteArray_Type;
PyTypeObject PyCFunction_Type = *Proxy::instance().PyCFunction_Type;
PyTypeObject PyCapsule_Type = *Proxy::instance().PyCapsule_Type;
PyTypeObject PyDict_Type = *Proxy::instance().PyDict_Type;
PyObject* PyExc_BufferError = *Proxy::instance().PyExc_BufferError;
PyObject* PyExc_FutureWarning = *Proxy::instance().PyExc_FutureWarning;
PyObject* PyExc_ImportError = *Proxy::instance().PyExc_ImportError;
PyObject* PyExc_IndexError = *Proxy::instance().PyExc_IndexError;
PyObject* PyExc_KeyError = *Proxy::instance().PyExc_KeyError;
PyObject* PyExc_MemoryError = *Proxy::instance().PyExc_MemoryError;
PyObject* PyExc_OverflowError = *Proxy::instance().PyExc_OverflowError;
PyObject* PyExc_RuntimeError = *Proxy::instance().PyExc_RuntimeError;
PyObject* PyExc_StopIteration = *Proxy::instance().PyExc_StopIteration;
PyObject* PyExc_SystemError = *Proxy::instance().PyExc_SystemError;
PyObject* PyExc_TypeError = *Proxy::instance().PyExc_TypeError;
PyObject* PyExc_ValueError = *Proxy::instance().PyExc_ValueError;
PyTypeObject PyFloat_Type = *Proxy::instance().PyFloat_Type;
PyTypeObject PyInstanceMethod_Type = *Proxy::instance().PyInstanceMethod_Type;
PyTypeObject PyLong_Type = *Proxy::instance().PyLong_Type;
PyTypeObject PyMethod_Type = *Proxy::instance().PyMethod_Type;
PyTypeObject PyModule_Type = *Proxy::instance().PyModule_Type;
PyTypeObject PyProperty_Type = *Proxy::instance().PyProperty_Type;
PyTypeObject PySlice_Type = *Proxy::instance().PySlice_Type;
PyTypeObject PyType_Type = *Proxy::instance().PyType_Type;

} // extern "C"
