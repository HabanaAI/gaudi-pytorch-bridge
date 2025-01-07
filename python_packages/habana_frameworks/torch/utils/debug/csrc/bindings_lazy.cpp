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
    habana::HabanaLaunchOpUtils::cleanUp();
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