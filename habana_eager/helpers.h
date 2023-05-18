/*******************************************************************************
 * Copyright (C) 2020-2023 Habana Labs, Ltd. an Intel Company
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
#include <Python.h>
namespace habana {
namespace eager {

struct gil_scoped_release_if_held {
  gil_scoped_release_if_held() {
    if (PyGILState_Check() && PyGILState_GetThisThreadState()) {
      save_state = PyEval_SaveThread();
    }
  }
  ~gil_scoped_release_if_held() {
    if (save_state) {
      PyEval_RestoreThread(save_state);
    }
  }
  PyThreadState* save_state = nullptr;
};

} // namespace eager
} // namespace habana
