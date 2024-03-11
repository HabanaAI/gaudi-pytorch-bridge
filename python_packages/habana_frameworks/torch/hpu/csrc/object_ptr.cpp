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
#include <c10/macros/Macros.h>
#include <torch/csrc/utils/object_ptr.h>

#include <torch/csrc/python_headers.h>

template <>
void THPPointer<PyObject>::free() {
  if (ptr && C10_LIKELY(Py_IsInitialized()))
    Py_DECREF(ptr);
}

template class THPPointer<PyObject>;

template <>
void THPPointer<PyCodeObject>::free() {
  if (ptr && C10_LIKELY(Py_IsInitialized()))
    Py_DECREF(ptr);
}

template class THPPointer<PyCodeObject>;

template <>
void THPPointer<PyFrameObject>::free() {
  if (ptr && C10_LIKELY(Py_IsInitialized()))
    Py_DECREF(ptr);
}

template class THPPointer<PyFrameObject>;
