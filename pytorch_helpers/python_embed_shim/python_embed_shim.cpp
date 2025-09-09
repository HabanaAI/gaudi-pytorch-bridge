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

//  NOLINTBEGIN(cert-dcl37-c)
//  NOLINTBEGIN(cert-dcl51-cpp)
extern "C" {
int _PyArg_ParseTuple_SizeT(PyObject* args, const char* format, ...);
#if PY_VERSION_HEX >= 0x030b0000 // Python 3.11+
PyThreadState* _PyThreadState_GetCurrent();
#endif
}
//  NOLINTEND(cert-dcl51-cpp)
//  NOLINTEND(cert-dcl37-c)

namespace {

template <typename... Args>
void ShimAssert(bool condition, const Args&... args) {
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
//  NOLINTBEGIN(cert-dcl37-c)
//  NOLINTBEGIN(cert-dcl51-cpp)
//  NOLINTNEXTLINE(bugprone-macro-parentheses)
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
  WRAP_SYMBOL(PyBytes_FromStringAndSize);
  WRAP_SYMBOL(PyBytes_Size);
  WRAP_SYMBOL(PyBytes_Type);
  WRAP_SYMBOL(PyCallable_Check);
  WRAP_SYMBOL(PyCFunction_Type);
  WRAP_SYMBOL(PyCMethod_New);
  WRAP_SYMBOL(PyCapsule_GetContext);
  WRAP_SYMBOL(PyCapsule_GetName);
  WRAP_SYMBOL(PyCapsule_GetPointer);
  WRAP_SYMBOL(PyCapsule_Import);
  WRAP_SYMBOL(PyCapsule_IsValid);
  WRAP_SYMBOL(PyCapsule_New);
  WRAP_SYMBOL(PyCapsule_SetContext);
  WRAP_SYMBOL(PyCapsule_SetName);
  WRAP_SYMBOL(PyCapsule_SetPointer);
  WRAP_SYMBOL(PyCapsule_Type);
  WRAP_SYMBOL(PyCell_Type);
  WRAP_SYMBOL(PyCode_Addr2Line);
#if PY_VERSION_HEX >= 0x030b0000 // Python 3.11+
  WRAP_SYMBOL(PyCode_GetVarnames);
#endif
  WRAP_SYMBOL(PyCode_Type);
  WRAP_SYMBOL(PyComplex_AsCComplex);
  WRAP_SYMBOL(PyComplex_FromCComplex);
  WRAP_SYMBOL(PyComplex_FromDoubles);
  WRAP_SYMBOL(PyComplex_ImagAsDouble);
  WRAP_SYMBOL(PyComplex_RealAsDouble);
  WRAP_SYMBOL(PyComplex_Type);
#if PY_VERSION_HEX >= 0x030c0000 // Python 3.12+
  WRAP_SYMBOL(PyDict_AddWatcher);
#endif
  WRAP_SYMBOL(PyDict_Clear);
  WRAP_SYMBOL(PyDict_Contains);
  WRAP_SYMBOL(PyDict_Copy);
  WRAP_SYMBOL(PyDict_DelItem);
  WRAP_SYMBOL(PyDict_DelItemString);
  WRAP_SYMBOL(PyDict_GetItem);
  WRAP_SYMBOL(PyDict_GetItemString);
  WRAP_SYMBOL(PyDict_GetItemWithError);
  WRAP_SYMBOL(PyDict_Items);
  WRAP_SYMBOL(PyDict_Merge);
  WRAP_SYMBOL(PyDict_New);
  WRAP_SYMBOL(PyDict_Next);
  WRAP_SYMBOL(PyDict_SetDefault);
  WRAP_SYMBOL(PyDict_SetItem);
  WRAP_SYMBOL(PyDict_SetItemString);
  WRAP_SYMBOL(PyDict_Size);
  WRAP_SYMBOL(PyDict_Type);
  WRAP_SYMBOL(PyDict_Values);
#if PY_VERSION_HEX >= 0x030c0000 // Python 3.12+
  WRAP_SYMBOL(PyDict_Watch);
#endif
  WRAP_SYMBOL(PyErr_Clear);
  WRAP_SYMBOL(PyErr_ExceptionMatches);
  WRAP_SYMBOL(PyErr_Fetch);
  WRAP_SYMBOL(PyErr_Format);
  WRAP_SYMBOL(PyErr_GivenExceptionMatches);
  WRAP_SYMBOL(PyErr_NewException);
  WRAP_SYMBOL(PyErr_NewExceptionWithDoc);
  WRAP_SYMBOL(PyErr_NoMemory);
  WRAP_SYMBOL(PyErr_NormalizeException);
  WRAP_SYMBOL(PyErr_Occurred);
  WRAP_SYMBOL(PyErr_Print);
  WRAP_SYMBOL(PyErr_Restore);
  WRAP_SYMBOL(PyErr_SetNone);
  WRAP_SYMBOL(PyErr_SetObject);
  WRAP_SYMBOL(PyErr_SetString);
  WRAP_SYMBOL(PyErr_WarnEx);
  WRAP_SYMBOL(PyErr_WarnExplicit);
  WRAP_SYMBOL(PyErr_WriteUnraisable);
  WRAP_SYMBOL(PyEval_AcquireThread);
  WRAP_SYMBOL(PyEval_GetBuiltins);
  WRAP_SYMBOL(PyEval_GetFrame);
  WRAP_SYMBOL(PyEval_GetLocals);
  WRAP_SYMBOL(PyEval_RestoreThread);
  WRAP_SYMBOL(PyEval_SaveThread);
  WRAP_SYMBOL(PyEval_SetProfile);
  WRAP_SYMBOL(PyExc_AssertionError);
  WRAP_SYMBOL(PyExc_AttributeError);
  WRAP_SYMBOL(PyExc_BufferError);
  WRAP_SYMBOL(PyExc_DeprecationWarning);
  WRAP_SYMBOL(PyExc_Exception);
  WRAP_SYMBOL(PyExc_FutureWarning);
  WRAP_SYMBOL(PyExc_ImportError);
  WRAP_SYMBOL(PyExc_IndexError);
  WRAP_SYMBOL(PyExc_KeyError);
  WRAP_SYMBOL(PyExc_MemoryError);
  WRAP_SYMBOL(PyExc_ModuleNotFoundError);
  WRAP_SYMBOL(PyExc_NotImplementedError);
  WRAP_SYMBOL(PyExc_OverflowError);
  WRAP_SYMBOL(PyExc_RuntimeError);
  WRAP_SYMBOL(PyExc_StopIteration);
  WRAP_SYMBOL(PyExc_SyntaxError);
  WRAP_SYMBOL(PyExc_SystemError);
  WRAP_SYMBOL(PyExc_TypeError);
  WRAP_SYMBOL(PyExc_UserWarning);
  WRAP_SYMBOL(PyExc_ValueError);
  WRAP_SYMBOL(PyException_SetCause);
  WRAP_SYMBOL(PyException_SetContext);
  WRAP_SYMBOL(PyException_SetTraceback);
  WRAP_SYMBOL(PyFloat_AsDouble);
  WRAP_SYMBOL(PyFloat_FromDouble);
  WRAP_SYMBOL(PyFloat_Type);
  WRAP_SYMBOL(PyFrame_GetBack);
  WRAP_SYMBOL(PyFrame_GetCode);
#if PY_VERSION_HEX >= 0x030b0000 // Python 3.11+
  WRAP_SYMBOL(PyFrame_GetGlobals);
  WRAP_SYMBOL(PyFrame_GetLasti);
  WRAP_SYMBOL(PyFrame_GetLocals);
#endif
  WRAP_SYMBOL(PyFrame_GetLineNumber);
  WRAP_SYMBOL(PyFrame_Type);
  WRAP_SYMBOL(PyFrozenSet_Type);
  WRAP_SYMBOL(PyFunction_GetDefaults);
  WRAP_SYMBOL(PyFunction_GetKwDefaults);
  WRAP_SYMBOL(PyFunction_Type);
  WRAP_SYMBOL(PyGILState_Check);
  WRAP_SYMBOL(PyGILState_Ensure);
  WRAP_SYMBOL(PyGILState_GetThisThreadState);
  WRAP_SYMBOL(PyGILState_Release);
  WRAP_SYMBOL(PyImport_AddModule);
  WRAP_SYMBOL(PyImport_ImportModule);
  WRAP_SYMBOL(PyIndex_Check);
  WRAP_SYMBOL(PyInstanceMethod_New);
  WRAP_SYMBOL(PyInstanceMethod_Type);
  WRAP_SYMBOL(PyInterpreterState_Get);
  WRAP_SYMBOL(PyInterpreterState_GetDict);
  WRAP_SYMBOL(PyInterpreterState_ThreadHead);
  WRAP_SYMBOL(PyIter_Check);
  WRAP_SYMBOL(PyIter_Next);
  WRAP_SYMBOL(PyList_Append);
  WRAP_SYMBOL(PyList_AsTuple);
  WRAP_SYMBOL(PyList_GetItem);
  WRAP_SYMBOL(PyList_New);
  WRAP_SYMBOL(PyList_SetItem);
  WRAP_SYMBOL(PyList_Size);
  WRAP_SYMBOL(PyList_Type);
  WRAP_SYMBOL(PyLong_AsDouble);
  WRAP_SYMBOL(PyLong_AsLong);
  WRAP_SYMBOL(PyLong_AsLongAndOverflow);
  WRAP_SYMBOL(PyLong_AsLongLong);
  WRAP_SYMBOL(PyLong_AsLongLongAndOverflow);
  WRAP_SYMBOL(PyLong_AsSize_t);
  WRAP_SYMBOL(PyLong_AsSsize_t);
  WRAP_SYMBOL(PyLong_AsUnsignedLong);
  WRAP_SYMBOL(PyLong_AsUnsignedLongLong);
  WRAP_SYMBOL(PyLong_AsVoidPtr);
  WRAP_SYMBOL(PyLong_FromDouble);
  WRAP_SYMBOL(PyLong_FromLong);
  WRAP_SYMBOL(PyLong_FromLongLong);
  WRAP_SYMBOL(PyLong_FromSize_t);
  WRAP_SYMBOL(PyLong_FromSsize_t);
  WRAP_SYMBOL(PyLong_FromUnsignedLong);
  WRAP_SYMBOL(PyLong_FromUnsignedLongLong);
  WRAP_SYMBOL(PyLong_FromVoidPtr);
  WRAP_SYMBOL(PyLong_Type);
  WRAP_SYMBOL(PyMapping_Keys);
  WRAP_SYMBOL(PyMem_Calloc);
  WRAP_SYMBOL(PyMem_Free);
  WRAP_SYMBOL(PyMemoryView_FromMemory);
  WRAP_SYMBOL(PyMemoryView_FromObject);
  WRAP_SYMBOL(PyMemoryView_Type);
  WRAP_SYMBOL(PyMethod_Type);
  WRAP_SYMBOL(PyModule_AddFunctions);
  WRAP_SYMBOL(PyModule_AddObject);
  WRAP_SYMBOL(PyModule_AddType);
  WRAP_SYMBOL(PyModule_Create2);
  WRAP_SYMBOL(PyModule_GetName);
  WRAP_SYMBOL(PyModule_GetState);
  WRAP_SYMBOL(PyModule_New);
  WRAP_SYMBOL(PyModule_Type);
  WRAP_SYMBOL(PyNumber_And);
  WRAP_SYMBOL(PyNumber_Check);
  WRAP_SYMBOL(PyNumber_Float);
  WRAP_SYMBOL(PyNumber_Index);
  WRAP_SYMBOL(PyNumber_Invert);
  WRAP_SYMBOL(PyNumber_Long);
  WRAP_SYMBOL(PyNumber_Or);
  WRAP_SYMBOL(PyNumber_Xor);
  WRAP_SYMBOL(PyObject_AsFileDescriptor);
  WRAP_SYMBOL(PyObject_Call);
  WRAP_SYMBOL(PyObject_CallFunction);
  WRAP_SYMBOL(PyObject_CallFunctionObjArgs);
  WRAP_SYMBOL(PyObject_CallMethod);
  WRAP_SYMBOL(PyObject_CallMethodObjArgs);
  WRAP_SYMBOL(PyObject_CallNoArgs);
  WRAP_SYMBOL(PyObject_CallObject);
  // Python < 3.10 (PyObject_CallOneArg is inline in 3.10+)
#if PY_VERSION_HEX < 0x030a0000
  WRAP_SYMBOL(PyObject_CallOneArg);
#endif
  WRAP_SYMBOL(PyObject_CheckBuffer);
  WRAP_SYMBOL(PyObject_ClearWeakRefs);
  WRAP_SYMBOL(PyObject_GC_Del);
  WRAP_SYMBOL(PyObject_GC_IsTracked);
  WRAP_SYMBOL(PyObject_GC_UnTrack);
  WRAP_SYMBOL(PyObject_GenericGetAttr);
  WRAP_SYMBOL(PyObject_GenericGetDict);
  WRAP_SYMBOL(PyObject_GenericSetDict);
  WRAP_SYMBOL(PyObject_GetAttr);
  WRAP_SYMBOL(PyObject_GetAttrString);
  WRAP_SYMBOL(PyObject_GetBuffer);
  WRAP_SYMBOL(PyObject_GetIter);
  WRAP_SYMBOL(PyObject_GetItem);
  WRAP_SYMBOL(PyObject_HasAttr);
  WRAP_SYMBOL(PyObject_HasAttrString);
  WRAP_SYMBOL(PyObject_IsInstance);
  WRAP_SYMBOL(PyObject_IsSubclass);
  WRAP_SYMBOL(PyObject_IsTrue);
  WRAP_SYMBOL(PyObject_LengthHint);
  WRAP_SYMBOL(PyObject_Malloc);
  WRAP_SYMBOL(PyObject_Repr);
  WRAP_SYMBOL(PyObject_RichCompareBool);
  WRAP_SYMBOL(PyObject_SelfIter);
  WRAP_SYMBOL(PyObject_SetAttr);
  WRAP_SYMBOL(PyObject_SetAttrString);
  WRAP_SYMBOL(PyObject_SetItem);
  WRAP_SYMBOL(PyObject_Size);
  WRAP_SYMBOL(PyObject_Str);
  WRAP_SYMBOL(PyObject_Type);
  WRAP_SYMBOL(PyObject_CallFinalizerFromDealloc);
  WRAP_SYMBOL(PyObject_GC_Track);
  WRAP_SYMBOL(PyObject_GET_WEAKREFS_LISTPTR);
  WRAP_SYMBOL(PyObject_GetArenaAllocator);
  WRAP_SYMBOL(PyProperty_Type);
  WRAP_SYMBOL(PySequence_Check);
  WRAP_SYMBOL(PySequence_Fast);
  WRAP_SYMBOL(PySequence_GetItem);
  WRAP_SYMBOL(PySequence_List);
  WRAP_SYMBOL(PySequence_Size);
  WRAP_SYMBOL(PySequence_Tuple);
  WRAP_SYMBOL(PySet_Add);
  WRAP_SYMBOL(PySet_Contains);
  WRAP_SYMBOL(PySet_New);
  WRAP_SYMBOL(PySet_Size);
  WRAP_SYMBOL(PySet_Type);
  WRAP_SYMBOL(PySlice_AdjustIndices);
  WRAP_SYMBOL(PySlice_New);
  WRAP_SYMBOL(PySlice_Type);
  WRAP_SYMBOL(PySlice_Unpack);
  WRAP_SYMBOL(PyStaticMethod_New);
  WRAP_SYMBOL(PyStaticMethod_Type);
  WRAP_SYMBOL(PyStructSequence_InitType);
  WRAP_SYMBOL(PyStructSequence_New);
  WRAP_SYMBOL(PyThreadState_Clear);
  WRAP_SYMBOL(PyThreadState_DeleteCurrent);
  WRAP_SYMBOL(PyThreadState_Get);
  WRAP_SYMBOL(PyThreadState_GetFrame);
  WRAP_SYMBOL(PyThreadState_New);
  WRAP_SYMBOL(PyThreadState_Next);
  WRAP_SYMBOL(PyThreadState_Swap);
  WRAP_SYMBOL(PyThread_tss_alloc);
  WRAP_SYMBOL(PyThread_tss_create);
  WRAP_SYMBOL(PyThread_tss_get);
  WRAP_SYMBOL(PyThread_tss_set);
  WRAP_SYMBOL(PyTuple_GetItem);
  WRAP_SYMBOL(PyTuple_GetSlice);
  WRAP_SYMBOL(PyTuple_New);
  WRAP_SYMBOL(PyTuple_Pack);
  WRAP_SYMBOL(PyTuple_SetItem);
  WRAP_SYMBOL(PyTuple_Size);
  WRAP_SYMBOL(PyTuple_Type);
  WRAP_SYMBOL(PyType_FromSpec);
  WRAP_SYMBOL(PyType_GenericAlloc);
  WRAP_SYMBOL(PyType_GenericNew);
  WRAP_SYMBOL(PyType_IsSubtype);
  WRAP_SYMBOL(PyType_Ready);
  WRAP_SYMBOL(PyType_Type);
  WRAP_SYMBOL(PyUnicode_AsEncodedString);
  WRAP_SYMBOL(PyUnicode_AsUTF8);
  WRAP_SYMBOL(PyUnicode_AsUTF8AndSize);
  WRAP_SYMBOL(PyUnicode_AsUTF8String);
  WRAP_SYMBOL(PyUnicode_DecodeUTF8);
  WRAP_SYMBOL(PyUnicode_FromFormat);
  WRAP_SYMBOL(PyUnicode_FromKindAndData);
  WRAP_SYMBOL(PyUnicode_FromString);
  WRAP_SYMBOL(PyUnicode_FromStringAndSize);
  WRAP_SYMBOL(PyUnicode_InternFromString);
  WRAP_SYMBOL(PyUnicode_InternInPlace);
  WRAP_SYMBOL(PyUnicode_Join);
  WRAP_SYMBOL(PyUnicode_Type);
  WRAP_SYMBOL(PyWeakref_GetObject);
  WRAP_SYMBOL(PyWeakref_NewRef);
  WRAP_SYMBOL(Py_BuildValue);
  WRAP_SYMBOL(Py_GetVersion);
  WRAP_SYMBOL(Py_IsInitialized);
  WRAP_SYMBOL(PyArg_ParseTuple);
  WRAP_SYMBOL(PyArg_ParseTupleAndKeywords);
  WRAP_SYMBOL(_PyArg_ParseTuple_SizeT);
  WRAP_SYMBOL(PyBool_FromLong);
  WRAP_SYMBOL(_PyEval_EvalFrameDefault);
  WRAP_SYMBOL(_PyEval_SliceIndex);
  WRAP_SYMBOL(_PyInterpreterState_GetEvalFrameFunc);
  WRAP_SYMBOL(_PyInterpreterState_SetEvalFrameFunc);
  WRAP_SYMBOL(_PyObject_GC_New);
  WRAP_SYMBOL(_PyObject_GC_NewVar);
  WRAP_SYMBOL(_PyUnicode_IsAlpha);
  WRAP_SYMBOL(_PyUnicode_IsDecimalDigit);
  WRAP_SYMBOL(_PyUnicode_IsDigit);
  WRAP_SYMBOL(_PyUnicode_IsNumeric);
  WRAP_SYMBOL(_PyWeakref_CallableProxyType);
  WRAP_SYMBOL(_PyWeakref_ClearRef);
  WRAP_SYMBOL(_PyWeakref_ProxyType);
  WRAP_SYMBOL(_PyWeakref_RefType);
  WRAP_SYMBOL(_Py_BuildValue_SizeT);
  WRAP_SYMBOL(_PyObject_GetDictPtr);
#if PY_VERSION_HEX >= 0x030b0000 // Python 3.11+
  WRAP_SYMBOL(_PyThreadState_GetCurrent);
#endif
  WRAP_SYMBOL(_PyThreadState_UncheckedGet);
  WRAP_SYMBOL(_PyType_Lookup);
  WRAP_SYMBOL(_Py_Dealloc);
  WRAP_SYMBOL(_Py_EllipsisObject);
  WRAP_SYMBOL(_Py_FalseStruct);
  WRAP_SYMBOL(_Py_NewReference);
  WRAP_SYMBOL(_Py_NoneStruct);
  WRAP_SYMBOL(_Py_NotImplementedStruct);
  WRAP_SYMBOL(_Py_TrueStruct);
#if PY_VERSION_HEX >= 0x030c0000 // Python 3.12+
  WRAP_SYMBOL(PyUnstable_Code_GetExtra);
  WRAP_SYMBOL(PyUnstable_Code_SetExtra);
  WRAP_SYMBOL(PyUnstable_Eval_RequestCodeExtraIndex);
#endif
#undef WRAP_SYMBOL
  //  NOLINTEND(cert-dcl51-cpp)
  //  NOLINTEND(cert-dcl37-c)

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

int PyArg_ParseTuple(PyObject* args, const char* format, ...) {
  va_list vargs;
  va_start(vargs, format);
  int result = Proxy::instance().PyArg_ParseTuple(args, format, vargs);
  va_end(vargs);
  return result;
}

int PyArg_ParseTupleAndKeywords(
    PyObject* args,
    PyObject* kw,
    const char* format,
    char** keywords,
    ...) {
  va_list vargs;
  va_start(vargs, keywords);
  int result = Proxy::instance().PyArg_ParseTupleAndKeywords(
      args, kw, format, keywords, vargs);
  va_end(vargs);
  return result;
}

int _PyArg_ParseTuple_SizeT(PyObject* args, const char* format, ...) {
  va_list vargs;
  va_start(vargs, format);
  int result = Proxy::instance().PyArg_ParseTuple(args, format, vargs);
  va_end(vargs);
  return result;
}

PyObject* PyBool_FromLong(long value) {
  return Proxy::instance().PyBool_FromLong(value);
}

PyObject* Py_BuildValue(const char* format, ...) {
  va_list vargs;
  va_start(vargs, format);
  PyObject* result = Proxy::instance().Py_BuildValue(format, vargs);
  va_end(vargs);
  return result;
}

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

PyObject* PyBytes_FromStringAndSize(const char* str, Py_ssize_t size) {
  return Proxy::instance().PyBytes_FromStringAndSize(str, size);
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

const char* PyCapsule_GetName(PyObject* capsule) {
  return Proxy::instance().PyCapsule_GetName(capsule);
}

void* PyCapsule_GetPointer(PyObject* capsule, const char* name) {
  return Proxy::instance().PyCapsule_GetPointer(capsule, name);
}

void* PyCapsule_Import(const char* name, int no_block) {
  return Proxy::instance().PyCapsule_Import(name, no_block);
}

int PyCapsule_IsValid(PyObject* capsule, const char* name) {
  return Proxy::instance().PyCapsule_IsValid(capsule, name);
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

int PyCapsule_SetName(PyObject* capsule, const char* name) {
  return Proxy::instance().PyCapsule_SetName(capsule, name);
}

int PyCode_Addr2Line(PyCodeObject* co, int byte_offset) {
  return Proxy::instance().PyCode_Addr2Line(co, byte_offset);
}

#if PY_VERSION_HEX >= 0x030b0000 // Python 3.11+
PyObject* PyCode_GetVarnames(PyCodeObject* co) {
  return Proxy::instance().PyCode_GetVarnames(co);
}
#endif

Py_complex PyComplex_AsCComplex(PyObject* obj) {
  return Proxy::instance().PyComplex_AsCComplex(obj);
}

PyObject* PyComplex_FromCComplex(Py_complex cval) {
  return Proxy::instance().PyComplex_FromCComplex(cval);
}

PyObject* PyComplex_FromDoubles(double real, double imag) {
  return Proxy::instance().PyComplex_FromDoubles(real, imag);
}

double PyComplex_ImagAsDouble(PyObject* obj) {
  return Proxy::instance().PyComplex_ImagAsDouble(obj);
}

double PyComplex_RealAsDouble(PyObject* obj) {
  return Proxy::instance().PyComplex_RealAsDouble(obj);
}

void _Py_Dealloc(PyObject* obj) {
  Proxy::instance()._Py_Dealloc(obj);
}

#if PY_VERSION_HEX >= 0x030c0000 // Python 3.12+
int PyDict_AddWatcher(PyDict_WatchCallback callback) {
  return Proxy::instance().PyDict_AddWatcher(callback);
}
#endif

void PyDict_Clear(PyObject* dict) {
  Proxy::instance().PyDict_Clear(dict);
}

int PyDict_Contains(PyObject* dict, PyObject* key) {
  return Proxy::instance().PyDict_Contains(dict, key);
}

PyObject* PyDict_Copy(PyObject* dict) {
  return Proxy::instance().PyDict_Copy(dict);
}

int PyDict_DelItem(PyObject* dict, PyObject* key) {
  return Proxy::instance().PyDict_DelItem(dict, key);
}

int PyDict_DelItemString(PyObject* dict, const char* key) {
  return Proxy::instance().PyDict_DelItemString(dict, key);
}

PyObject* PyDict_GetItem(PyObject* dict, PyObject* key) {
  return Proxy::instance().PyDict_GetItem(dict, key);
}

PyObject* PyDict_GetItemString(PyObject* dict, const char* key) {
  return Proxy::instance().PyDict_GetItemString(dict, key);
}

PyObject* PyDict_GetItemWithError(PyObject* dict, PyObject* key) {
  return Proxy::instance().PyDict_GetItemWithError(dict, key);
}

PyObject* PyDict_Items(PyObject* dict) {
  return Proxy::instance().PyDict_Items(dict);
}

int PyDict_Merge(PyObject* a, PyObject* b, int override) {
  return Proxy::instance().PyDict_Merge(a, b, override);
}

PyObject* PyDict_SetDefault(
    PyObject* dict,
    PyObject* key,
    PyObject* defaultobj) {
  return Proxy::instance().PyDict_SetDefault(dict, key, defaultobj);
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

int PyDict_SetItem(PyObject* dict, PyObject* key, PyObject* value) {
  return Proxy::instance().PyDict_SetItem(dict, key, value);
}

int PyDict_SetItemString(PyObject* dict, const char* key, PyObject* value) {
  return Proxy::instance().PyDict_SetItemString(dict, key, value);
}

Py_ssize_t PyDict_Size(PyObject* dict) {
  return Proxy::instance().PyDict_Size(dict);
}

PyObject* PyDict_Values(PyObject* dict) {
  return Proxy::instance().PyDict_Values(dict);
}

#if PY_VERSION_HEX >= 0x030c0000 // Python 3.12+
int PyDict_Watch(int watcher_id, PyObject* dict) {
  return Proxy::instance().PyDict_Watch(watcher_id, dict);
}
#endif

void PyErr_Clear() {
  Proxy::instance().PyErr_Clear();
}

int PyErr_ExceptionMatches(PyObject* exc) {
  return Proxy::instance().PyErr_ExceptionMatches(exc);
}

int PyErr_GivenExceptionMatches(PyObject* err, PyObject* exc) {
  return Proxy::instance().PyErr_GivenExceptionMatches(err, exc);
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

PyObject* PyErr_NewException(const char* name, PyObject* base, PyObject* dict) {
  return Proxy::instance().PyErr_NewException(name, base, dict);
}

PyObject* PyErr_NewExceptionWithDoc(
    const char* name,
    const char* doc,
    PyObject* base,
    PyObject* dict) {
  return Proxy::instance().PyErr_NewExceptionWithDoc(name, doc, base, dict);
}

PyObject* PyErr_NoMemory() {
  return Proxy::instance().PyErr_NoMemory();
}

void PyErr_NormalizeException(PyObject** exc, PyObject** val, PyObject** tb) {
  Proxy::instance().PyErr_NormalizeException(exc, val, tb);
}

PyObject* PyErr_Occurred() {
  return Proxy::instance().PyErr_Occurred();
}

void PyErr_Print() {
  Proxy::instance().PyErr_Print();
}

void PyErr_Restore(PyObject* type, PyObject* value, PyObject* traceback) {
  Proxy::instance().PyErr_Restore(type, value, traceback);
}

void PyErr_SetNone(PyObject* exception) {
  Proxy::instance().PyErr_SetNone(exception);
}

void PyErr_SetObject(PyObject* exception, PyObject* value) {
  Proxy::instance().PyErr_SetObject(exception, value);
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

int PyErr_WarnExplicit(
    PyObject* category,
    const char* message,
    const char* filename,
    int lineno,
    const char* module,
    PyObject* registry) {
  return Proxy::instance().PyErr_WarnExplicit(
      category, message, filename, lineno, module, registry);
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

PyFrameObject* PyEval_GetFrame() {
  return Proxy::instance().PyEval_GetFrame();
}

PyObject* PyEval_GetLocals() {
  return Proxy::instance().PyEval_GetLocals();
}

void PyEval_RestoreThread(PyThreadState* tstate) {
  Proxy::instance().PyEval_RestoreThread(tstate);
}

PyThreadState* PyEval_SaveThread() {
  return Proxy::instance().PyEval_SaveThread();
}

void PyEval_SetProfile(Py_tracefunc func, PyObject* obj) {
  Proxy::instance().PyEval_SetProfile(func, obj);
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

double PyFloat_AsDouble(PyObject* obj) {
  return Proxy::instance().PyFloat_AsDouble(obj);
}

PyObject* PyFloat_FromDouble(double v) {
  return Proxy::instance().PyFloat_FromDouble(v);
}

PyFrameObject* PyFrame_GetBack(PyFrameObject* frame) {
  return Proxy::instance().PyFrame_GetBack(frame);
}

PyCodeObject* PyFrame_GetCode(PyFrameObject* frame) {
  return Proxy::instance().PyFrame_GetCode(frame);
}

#if PY_VERSION_HEX >= 0x030b0000 // Python 3.11+
PyObject* PyFrame_GetGlobals(PyFrameObject* frame) {
  return Proxy::instance().PyFrame_GetGlobals(frame);
}

int PyFrame_GetLasti(PyFrameObject* frame) {
  return Proxy::instance().PyFrame_GetLasti(frame);
}

PyObject* PyFrame_GetLocals(PyFrameObject* frame) {
  return Proxy::instance().PyFrame_GetLocals(frame);
}
#endif

int PyFrame_GetLineNumber(PyFrameObject* frame) {
  return Proxy::instance().PyFrame_GetLineNumber(frame);
}

const char* Py_GetVersion() {
  return Proxy::instance().Py_GetVersion();
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

PyObject* PyImport_AddModule(const char* name) {
  return Proxy::instance().PyImport_AddModule(name);
}

PyObject* PyImport_ImportModule(const char* name) {
  return Proxy::instance().PyImport_ImportModule(name);
}

int PyIndex_Check(PyObject* obj) {
  return Proxy::instance().PyIndex_Check(obj);
}

int Py_IsInitialized() {
  return Proxy::instance().Py_IsInitialized();
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

PyThreadState* PyInterpreterState_ThreadHead(PyInterpreterState* interp) {
  return Proxy::instance().PyInterpreterState_ThreadHead(interp);
}

int PyIter_Check(PyObject* obj) {
  return Proxy::instance().PyIter_Check(obj);
}

PyObject* PyIter_Next(PyObject* obj) {
  return Proxy::instance().PyIter_Next(obj);
}

int PyList_Append(PyObject* list, PyObject* item) {
  return Proxy::instance().PyList_Append(list, item);
}

PyObject* PyList_AsTuple(PyObject* list) {
  return Proxy::instance().PyList_AsTuple(list);
}

PyObject* PyList_GetItem(PyObject* list, Py_ssize_t index) {
  return Proxy::instance().PyList_GetItem(list, index);
}

PyObject* PyList_New(Py_ssize_t size) {
  return Proxy::instance().PyList_New(size);
}

int PyList_SetItem(PyObject* list, Py_ssize_t index, PyObject* item) {
  return Proxy::instance().PyList_SetItem(list, index, item);
}

Py_ssize_t PyList_Size(PyObject* list) {
  return Proxy::instance().PyList_Size(list);
}

double PyLong_AsDouble(PyObject* obj) {
  return Proxy::instance().PyLong_AsDouble(obj);
}

long PyLong_AsLong(PyObject* obj) {
  return Proxy::instance().PyLong_AsLong(obj);
}

long PyLong_AsLongAndOverflow(PyObject* obj, int* overflow) {
  return Proxy::instance().PyLong_AsLongAndOverflow(obj, overflow);
}

long long PyLong_AsLongLong(PyObject* obj) {
  return Proxy::instance().PyLong_AsLongLong(obj);
}

long long PyLong_AsLongLongAndOverflow(PyObject* obj, int* overflow) {
  return Proxy::instance().PyLong_AsLongLongAndOverflow(obj, overflow);
}

size_t PyLong_AsSize_t(PyObject* obj) {
  return Proxy::instance().PyLong_AsSize_t(obj);
}

Py_ssize_t PyLong_AsSsize_t(PyObject* obj) {
  return Proxy::instance().PyLong_AsSsize_t(obj);
}

unsigned long PyLong_AsUnsignedLong(PyObject* obj) {
  return Proxy::instance().PyLong_AsUnsignedLong(obj);
}

unsigned long long PyLong_AsUnsignedLongLong(PyObject* obj) {
  return Proxy::instance().PyLong_AsUnsignedLongLong(obj);
}

void* PyLong_AsVoidPtr(PyObject* obj) {
  return Proxy::instance().PyLong_AsVoidPtr(obj);
}

PyObject* PyLong_FromDouble(double v) {
  return Proxy::instance().PyLong_FromDouble(v);
}

PyObject* PyLong_FromLong(long v) {
  return Proxy::instance().PyLong_FromLong(v);
}

PyObject* PyLong_FromLongLong(long long v) {
  return Proxy::instance().PyLong_FromLongLong(v);
}

PyObject* PyLong_FromSize_t(size_t size) {
  return Proxy::instance().PyLong_FromSize_t(size);
}

PyObject* PyLong_FromSsize_t(Py_ssize_t v) {
  return Proxy::instance().PyLong_FromSsize_t(v);
}

PyObject* PyLong_FromUnsignedLong(unsigned long v) {
  return Proxy::instance().PyLong_FromUnsignedLong(v);
}

PyObject* PyLong_FromUnsignedLongLong(unsigned long long v) {
  return Proxy::instance().PyLong_FromUnsignedLongLong(v);
}

PyObject* PyLong_FromVoidPtr(void* p) {
  return Proxy::instance().PyLong_FromVoidPtr(p);
}

PyObject* PyMapping_Keys(PyObject* obj) {
  return Proxy::instance().PyMapping_Keys(obj);
}

void* PyMem_Calloc(size_t nelem, size_t elsize) {
  return Proxy::instance().PyMem_Calloc(nelem, elsize);
}

void PyMem_Free(void* ptr) {
  Proxy::instance().PyMem_Free(ptr);
}

PyObject* PyMemoryView_FromMemory(char* mem, Py_ssize_t size, int flags) {
  return Proxy::instance().PyMemoryView_FromMemory(mem, size, flags);
}

PyObject* PyMemoryView_FromObject(PyObject* obj) {
  return Proxy::instance().PyMemoryView_FromObject(obj);
}

int PyModule_AddObject(PyObject* mod, const char* str, PyObject* value) {
  return Proxy::instance().PyModule_AddObject(mod, str, value);
}

int PyModule_AddFunctions(PyObject* module, PyMethodDef* functions) {
  return Proxy::instance().PyModule_AddFunctions(module, functions);
}
int PyModule_AddType(PyObject* module, PyTypeObject* type) {
  return Proxy::instance().PyModule_AddType(module, type);
}
PyObject* PyModule_Create2(PyModuleDef* mod, int apiver) {
  return Proxy::instance().PyModule_Create2(mod, apiver);
}

const char* PyModule_GetName(PyObject* module) {
  return Proxy::instance().PyModule_GetName(module);
}

void* PyModule_GetState(PyObject* module) {
  return Proxy::instance().PyModule_GetState(module);
}

PyObject* PyModule_New(const char* name) {
  return Proxy::instance().PyModule_New(name);
}

PyObject* PyNumber_And(PyObject* obj1, PyObject* obj2) {
  return Proxy::instance().PyNumber_And(obj1, obj2);
}

int PyNumber_Check(PyObject* obj) {
  return Proxy::instance().PyNumber_Check(obj);
}

PyObject* PyNumber_Float(PyObject* obj) {
  return Proxy::instance().PyNumber_Float(obj);
}

PyObject* PyNumber_Index(PyObject* obj) {
  return Proxy::instance().PyNumber_Index(obj);
}

PyObject* PyNumber_Invert(PyObject* obj) {
  return Proxy::instance().PyNumber_Invert(obj);
}

PyObject* PyNumber_Long(PyObject* obj) {
  return Proxy::instance().PyNumber_Long(obj);
}

PyObject* PyNumber_Or(PyObject* obj1, PyObject* obj2) {
  return Proxy::instance().PyNumber_Or(obj1, obj2);
}

PyObject* PyNumber_Xor(PyObject* obj1, PyObject* obj2) {
  return Proxy::instance().PyNumber_Xor(obj1, obj2);
}

int PyObject_AsFileDescriptor(PyObject* obj) {
  return Proxy::instance().PyObject_AsFileDescriptor(obj);
}

PyObject* PyObject_Call(PyObject* callable, PyObject* args, PyObject* kwargs) {
  return Proxy::instance().PyObject_Call(callable, args, kwargs);
}

int PyObject_CallFinalizerFromDealloc(PyObject* obj) {
  return Proxy::instance().PyObject_CallFinalizerFromDealloc(obj);
}

PyObject* PyObject_CallFunctionObjArgs(PyObject* callable, ...) {
  va_list args;
  va_start(args, callable);
  PyObject* result =
      Proxy::instance().PyObject_CallFunctionObjArgs(callable, args);
  va_end(args);
  return result;
}

PyObject* PyObject_CallFunction(PyObject* callable, const char* format, ...) {
  va_list args;
  va_start(args, format);
  PyObject* result =
      Proxy::instance().PyObject_CallFunction(callable, format, args);
  va_end(args);
  return result;
}

PyObject* PyObject_CallMethod(
    PyObject* obj,
    const char* name,
    const char* format,
    ...) {
  va_list args;
  va_start(args, format);
  PyObject* result =
      Proxy::instance().PyObject_CallMethod(obj, name, format, args);
  va_end(args);
  return result;
}

PyObject* PyObject_CallMethodObjArgs(PyObject* obj, PyObject* name, ...) {
  va_list args;
  va_start(args, name);
  PyObject* result =
      Proxy::instance().PyObject_CallMethodObjArgs(obj, name, args);
  va_end(args);
  return result;
}

PyObject* PyObject_CallNoArgs(PyObject* callable) {
  return Proxy::instance().PyObject_CallNoArgs(callable);
}

PyObject* PyObject_CallObject(PyObject* callable, PyObject* args) {
  return Proxy::instance().PyObject_CallObject(callable, args);
}

// Python < 3.10 (PyObject_CallOneArg is inline in 3.10+)
#if PY_VERSION_HEX < 0x030a0000
PyObject* PyObject_CallOneArg(PyObject* callable, PyObject* arg) {
  return Proxy::instance().PyObject_CallOneArg(callable, arg);
}
#endif

int PyObject_CheckBuffer(PyObject* obj) {
  return Proxy::instance().PyObject_CheckBuffer(obj);
}

void PyObject_ClearWeakRefs(PyObject* obj) {
  Proxy::instance().PyObject_ClearWeakRefs(obj);
}

void PyObject_GC_Del(void* obj) {
  Proxy::instance().PyObject_GC_Del(obj);
}

int PyObject_GC_IsTracked(PyObject* obj) {
  return Proxy::instance().PyObject_GC_IsTracked(obj);
}

void PyObject_GC_Track(void* obj) {
  Proxy::instance().PyObject_GC_Track(obj);
}

void PyObject_GC_UnTrack(void* obj) {
  Proxy::instance().PyObject_GC_UnTrack(obj);
}

PyObject* PyObject_GenericGetAttr(PyObject* obj, PyObject* name) {
  return Proxy::instance().PyObject_GenericGetAttr(obj, name);
}

PyObject* PyObject_GenericGetDict(PyObject* obj, void* context) {
  return Proxy::instance().PyObject_GenericGetDict(obj, context);
}

int PyObject_GenericSetDict(PyObject* obj, PyObject* dict, void* context) {
  return Proxy::instance().PyObject_GenericSetDict(obj, dict, context);
}

void PyObject_GetArenaAllocator(PyObjectArenaAllocator* allocator) {
  Proxy::instance().PyObject_GetArenaAllocator(allocator);
}

PyObject* PyObject_GetAttr(PyObject* obj, PyObject* obj2) {
  return Proxy::instance().PyObject_GetAttr(obj, obj2);
}

PyObject* PyObject_GetAttrString(PyObject* obj, const char* attr_name) {
  return Proxy::instance().PyObject_GetAttrString(obj, attr_name);
}

int PyObject_GetBuffer(PyObject* exporter, Py_buffer* view, int flags) {
  return Proxy::instance().PyObject_GetBuffer(exporter, view, flags);
}

PyObject* PyObject_GetItem(PyObject* obj, PyObject* key) {
  return Proxy::instance().PyObject_GetItem(obj, key);
}

PyObject* PyObject_GetIter(PyObject* obj) {
  return Proxy::instance().PyObject_GetIter(obj);
}

PyObject** PyObject_GET_WEAKREFS_LISTPTR(PyObject* obj) {
  return Proxy::instance().PyObject_GET_WEAKREFS_LISTPTR(obj);
}

int PyObject_HasAttr(PyObject* obj, PyObject* attr_name) {
  return Proxy::instance().PyObject_HasAttr(obj, attr_name);
}

int PyObject_HasAttrString(PyObject* obj, const char* attr_name) {
  return Proxy::instance().PyObject_HasAttrString(obj, attr_name);
}

int PyObject_IsInstance(PyObject* inst, PyObject* cls) {
  return Proxy::instance().PyObject_IsInstance(inst, cls);
}

int PyObject_IsSubclass(PyObject* derived, PyObject* cls) {
  return Proxy::instance().PyObject_IsSubclass(derived, cls);
}

int PyObject_IsTrue(PyObject* obj) {
  return Proxy::instance().PyObject_IsTrue(obj);
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

int PyObject_RichCompareBool(PyObject* obj1, PyObject* obj2, int opid) {
  return Proxy::instance().PyObject_RichCompareBool(obj1, obj2, opid);
}

PyObject* PyObject_SelfIter(PyObject* obj) {
  return Proxy::instance().PyObject_SelfIter(obj);
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

Py_ssize_t PyObject_Size(PyObject* obj) {
  return Proxy::instance().PyObject_Size(obj);
}

PyObject* PyObject_Str(PyObject* obj) {
  return Proxy::instance().PyObject_Str(obj);
}

PyObject* PyObject_Type(PyObject* obj) {
  return Proxy::instance().PyObject_Type(obj);
}

PyObject** _PyObject_GetDictPtr(PyObject* obj) {
  return Proxy::instance()._PyObject_GetDictPtr(obj);
}

#if PY_VERSION_HEX >= 0x030b0000 // Python 3.11+
PyThreadState* _PyThreadState_GetCurrent() {
  return Proxy::instance()._PyThreadState_GetCurrent();
}
#endif
PyThreadState* _PyThreadState_UncheckedGet() {
  return Proxy::instance()._PyThreadState_UncheckedGet();
}

PyObject* _PyType_Lookup(PyTypeObject* type, PyObject* name) {
  return Proxy::instance()._PyType_Lookup(type, name);
}

int PySequence_Check(PyObject* obj) {
  return Proxy::instance().PySequence_Check(obj);
}

PyObject* PySequence_Fast(PyObject* obj, const char* msg) {
  return Proxy::instance().PySequence_Fast(obj, msg);
}

PyObject* PySequence_GetItem(PyObject* obj, Py_ssize_t index) {
  return Proxy::instance().PySequence_GetItem(obj, index);
}

PyObject* PySequence_List(PyObject* obj) {
  return Proxy::instance().PySequence_List(obj);
}

Py_ssize_t PySequence_Size(PyObject* obj) {
  return Proxy::instance().PySequence_Size(obj);
}

PyObject* PySequence_Tuple(PyObject* obj) {
  return Proxy::instance().PySequence_Tuple(obj);
}

int PySet_Add(PyObject* set, PyObject* key) {
  return Proxy::instance().PySet_Add(set, key);
}

int PySet_Contains(PyObject* set, PyObject* key) {
  return Proxy::instance().PySet_Contains(set, key);
}

PyObject* PySet_New(PyObject* iterable) {
  return Proxy::instance().PySet_New(iterable);
}

Py_ssize_t PySet_Size(PyObject* set) {
  return Proxy::instance().PySet_Size(set);
}

Py_ssize_t PySlice_AdjustIndices(
    Py_ssize_t length,
    Py_ssize_t* start,
    Py_ssize_t* stop,
    Py_ssize_t step) {
  return Proxy::instance().PySlice_AdjustIndices(length, start, stop, step);
}

PyObject* PySlice_New(PyObject* start, PyObject* stop, PyObject* step) {
  return Proxy::instance().PySlice_New(start, stop, step);
}

int PySlice_Unpack(
    PyObject* slice,
    Py_ssize_t* start,
    Py_ssize_t* stop,
    Py_ssize_t* step) {
  return Proxy::instance().PySlice_Unpack(slice, start, stop, step);
}

PyObject* PyStaticMethod_New(PyObject* func) {
  return Proxy::instance().PyStaticMethod_New(func);
}

void PyStructSequence_InitType(
    PyTypeObject* type,
    PyStructSequence_Desc* desc) {
  Proxy::instance().PyStructSequence_InitType(type, desc);
}

PyObject* PyStructSequence_New(PyTypeObject* type) {
  return Proxy::instance().PyStructSequence_New(type);
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

PyFrameObject* PyThreadState_GetFrame(PyThreadState* tstate) {
  return Proxy::instance().PyThreadState_GetFrame(tstate);
}

PyThreadState* PyThreadState_New(PyInterpreterState* interp) {
  return Proxy::instance().PyThreadState_New(interp);
}

PyThreadState* PyThreadState_Next(PyThreadState* tstate) {
  return Proxy::instance().PyThreadState_Next(tstate);
}

PyThreadState* PyThreadState_Swap(PyThreadState* tstate) {
  return Proxy::instance().PyThreadState_Swap(tstate);
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

PyObject* PyTuple_GetSlice(PyObject* tuple, Py_ssize_t low, Py_ssize_t high) {
  return Proxy::instance().PyTuple_GetSlice(tuple, low, high);
}

PyObject* PyTuple_New(Py_ssize_t size) {
  return Proxy::instance().PyTuple_New(size);
}

PyObject* PyTuple_Pack(Py_ssize_t n, ...) {
  va_list vargs;
  va_start(vargs, n);
  PyObject* result = Proxy::instance().PyTuple_Pack(n, vargs);
  va_end(vargs);
  return result;
}

int PyTuple_SetItem(PyObject* tuple, Py_ssize_t pos, PyObject* item) {
  return Proxy::instance().PyTuple_SetItem(tuple, pos, item);
}

Py_ssize_t PyTuple_Size(PyObject* tuple) {
  return Proxy::instance().PyTuple_Size(tuple);
}

PyObject* PyType_FromSpec(PyType_Spec* spec) {
  return Proxy::instance().PyType_FromSpec(spec);
}

PyObject* PyType_GenericAlloc(PyTypeObject* type, Py_ssize_t nitems) {
  return Proxy::instance().PyType_GenericAlloc(type, nitems);
}

PyObject* PyType_GenericNew(
    PyTypeObject* type,
    PyObject* args,
    PyObject* kwds) {
  return Proxy::instance().PyType_GenericNew(type, args, kwds);
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

const char* PyUnicode_AsUTF8(PyObject* unicode) {
  return Proxy::instance().PyUnicode_AsUTF8(unicode);
}

const char* PyUnicode_AsUTF8AndSize(PyObject* unicode, Py_ssize_t* size) {
  return Proxy::instance().PyUnicode_AsUTF8AndSize(unicode, size);
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

PyObject* PyUnicode_FromKindAndData(
    int kind,
    const void* buffer,
    Py_ssize_t size) {
  return Proxy::instance().PyUnicode_FromKindAndData(kind, buffer, size);
}

PyObject* PyUnicode_FromString(const char* str) {
  return Proxy::instance().PyUnicode_FromString(str);
}

PyObject* PyUnicode_FromStringAndSize(const char* str, Py_ssize_t size) {
  return Proxy::instance().PyUnicode_FromStringAndSize(str, size);
}

PyObject* PyUnicode_InternFromString(const char* str) {
  return Proxy::instance().PyUnicode_InternFromString(str);
}

void PyUnicode_InternInPlace(PyObject** string) {
  Proxy::instance().PyUnicode_InternInPlace(string);
}

PyObject* PyUnicode_Join(PyObject* separator, PyObject* seq) {
  return Proxy::instance().PyUnicode_Join(separator, seq);
}

#if PY_VERSION_HEX >= 0x030c0000 // Python 3.12+
int PyUnstable_Code_GetExtra(PyObject* code, Py_ssize_t index, void** extra) {
  return Proxy::instance().PyUnstable_Code_GetExtra(code, index, extra);
}

int PyUnstable_Code_SetExtra(PyObject* code, Py_ssize_t index, void* extra) {
  return Proxy::instance().PyUnstable_Code_SetExtra(code, index, extra);
}

Py_ssize_t PyUnstable_Eval_RequestCodeExtraIndex(freefunc free) {
  return Proxy::instance().PyUnstable_Eval_RequestCodeExtraIndex(free);
}
#endif

void _PyWeakref_ClearRef(PyWeakReference* ref) {
  Proxy::instance()._PyWeakref_ClearRef(ref);
}

PyObject* PyWeakref_GetObject(PyObject* ref) {
  return Proxy::instance().PyWeakref_GetObject(ref);
}

PyObject* PyWeakref_NewRef(PyObject* obj, PyObject* callback) {
  return Proxy::instance().PyWeakref_NewRef(obj, callback);
}

// Globals
PyObject _Py_EllipsisObject = *Proxy::instance()._Py_EllipsisObject;
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
PyObject* PyExc_AssertionError = *Proxy::instance().PyExc_AssertionError;
PyObject* PyExc_AttributeError = *Proxy::instance().PyExc_AttributeError;
PyObject* PyExc_BufferError = *Proxy::instance().PyExc_BufferError;
PyObject* PyExc_DeprecationWarning =
    *Proxy::instance().PyExc_DeprecationWarning;
PyObject* PyExc_Exception = *Proxy::instance().PyExc_Exception;
PyObject* PyExc_FutureWarning = *Proxy::instance().PyExc_FutureWarning;
PyObject* PyExc_ImportError = *Proxy::instance().PyExc_ImportError;
PyObject* PyExc_IndexError = *Proxy::instance().PyExc_IndexError;
PyObject* PyExc_KeyError = *Proxy::instance().PyExc_KeyError;
PyObject* PyExc_MemoryError = *Proxy::instance().PyExc_MemoryError;
PyObject* PyExc_ModuleNotFoundError =
    *Proxy::instance().PyExc_ModuleNotFoundError;
PyObject* PyExc_NotImplementedError =
    *Proxy::instance().PyExc_NotImplementedError;
PyObject* PyExc_OverflowError = *Proxy::instance().PyExc_OverflowError;
PyObject* PyExc_RuntimeError = *Proxy::instance().PyExc_RuntimeError;
PyObject* PyExc_StopIteration = *Proxy::instance().PyExc_StopIteration;
PyObject* PyExc_SyntaxError = *Proxy::instance().PyExc_SyntaxError;
PyObject* PyExc_SystemError = *Proxy::instance().PyExc_SystemError;
PyObject* PyExc_TypeError = *Proxy::instance().PyExc_TypeError;
PyObject* PyExc_UserWarning = *Proxy::instance().PyExc_UserWarning;
PyObject* PyExc_ValueError = *Proxy::instance().PyExc_ValueError;
PyTypeObject PyBytes_Type = *Proxy::instance().PyBytes_Type;
PyTypeObject PyCell_Type = *Proxy::instance().PyCell_Type;
PyTypeObject PyCode_Type = *Proxy::instance().PyCode_Type;
PyTypeObject PyComplex_Type = *Proxy::instance().PyComplex_Type;
PyTypeObject PyFloat_Type = *Proxy::instance().PyFloat_Type;
PyTypeObject PyFrame_Type = *Proxy::instance().PyFrame_Type;
PyTypeObject PyFrozenSet_Type = *Proxy::instance().PyFrozenSet_Type;
PyTypeObject PyFunction_Type = *Proxy::instance().PyFunction_Type;
PyTypeObject PyInstanceMethod_Type = *Proxy::instance().PyInstanceMethod_Type;
PyTypeObject PyList_Type = *Proxy::instance().PyList_Type;
PyTypeObject PyLong_Type = *Proxy::instance().PyLong_Type;
PyTypeObject PyMemoryView_Type = *Proxy::instance().PyMemoryView_Type;
PyTypeObject PyMethod_Type = *Proxy::instance().PyMethod_Type;
PyTypeObject PyModule_Type = *Proxy::instance().PyModule_Type;
PyTypeObject PyProperty_Type = *Proxy::instance().PyProperty_Type;
PyTypeObject PySlice_Type = *Proxy::instance().PySlice_Type;
PyTypeObject PyType_Type = *Proxy::instance().PyType_Type;
PyTypeObject PySet_Type = *Proxy::instance().PySet_Type;
PyTypeObject PyStaticMethod_Type = *Proxy::instance().PyStaticMethod_Type;
PyTypeObject PyTuple_Type = *Proxy::instance().PyTuple_Type;
PyTypeObject PyUnicode_Type = *Proxy::instance().PyUnicode_Type;
PyTypeObject _PyWeakref_CallableProxyType =
    *Proxy::instance()._PyWeakref_CallableProxyType;
PyTypeObject _PyWeakref_ProxyType = *Proxy::instance()._PyWeakref_ProxyType;
PyTypeObject _PyWeakref_RefType = *Proxy::instance()._PyWeakref_RefType;

} // extern "C"
