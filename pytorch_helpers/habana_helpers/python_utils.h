/******************************************************************************
 * Copyright (C) 2021 Habana Labs, Ltd. an Intel Company
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
#pragma once

#include "Python.h"

namespace habana_helpers {
// RAII structs to acquire/release Python's global interpreter lock (GIL)
// Releases the GIL on construction, if this thread already have the GIL
// acquired
struct AutoNoGIL {
  AutoNoGIL() {
    if (PyGILState_Check() && PyGILState_GetThisThreadState()) {
      save_state = PyEval_SaveThread();
    }
  }
  void Acquire() {
    if (save_state) {
      PyEval_RestoreThread(save_state);
    }
    save_state = nullptr;
  }
  ~AutoNoGIL() {
    Acquire();
  }
  PyThreadState* save_state = nullptr;
};
} // namespace habana_helpers
