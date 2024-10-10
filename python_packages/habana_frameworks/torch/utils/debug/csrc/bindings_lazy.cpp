/*******************************************************************************
 * Copyright (C) 2022-2023 Habana Labs, Ltd. an Intel Company
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
#include <torch/extension.h>
#include "backend/kernel/hpu_habana_launch_op_pt.h"
#include "habana_lazy/view_utils.h"
#include "pytorch_helpers/habana_helpers/kernels_accumulation.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("bridge_cleanup", []() {
    try {
      habana_lazy::AccThread::Get().SyncAccThreadPool();
    } catch (const c10::Error& e) {
    }
    habana::HabanaLaunchOpPT::cleanUp();
  });
  m.def("get_tensor_info", [](const at::Tensor& t) -> pybind11::object {
    auto base_tensor = habana_lazy::HbLazyTensorViews::get_base_tensor(t);
    if (not base_tensor.has_storage()) {
      return pybind11::none();
    }

    auto data_ptr = (std::uintptr_t)base_tensor.storage().data();
    auto size = base_tensor.storage().nbytes();
    return pybind11::make_tuple(data_ptr, size);
  });
}