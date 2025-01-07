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

#include "habana_eager/graph_dynamic.h"
#include "backend/kernel/hpu_habana_launch_op_pt.h"

#include "habana_helpers/logging.h"

namespace habana {
namespace graph {

int64_t GetSymintValue(torch::jit::Stack& original_stack, uint64_t index) {
  int64_t value;
  HABANA_ASSERT(original_stack[index].isScalar() == 1);
  value = original_stack[index].toScalar().toLong();
  return value;
}

std::string GetDynamicTensorName(
    const std::string& prefix,
    synTensorType type) {
  std::string t_name;

  switch (type) {
    case SHAPE_TENSOR:
      t_name = prefix + "_ST";
      break;
    case HOST_TO_DEVICE_TENSOR:
      t_name = prefix + "_H2D";
      break;
    default:
      break;
  }
  return t_name;
}

template <typename T>
std::vector<T> GetH2DTensorHostData(at::Tensor& tensor) {
  std::vector<T> host_data;
  auto tmeta = get_tensor_extra_meta(tensor);
  size_t data_size = tensor.sizes()[0];
  PT_EAGER_DEBUG("Read H2D data of size :", data_size);
  if (tmeta->get_tensor_type() == HOST_TO_DEVICE_TENSOR) {
    void* host_ptr = tmeta->get_host_ptr();
    T* h2d_data = static_cast<T*>(host_ptr);
    for (size_t i = 0; i < data_size; i++) {
      host_data.push_back(static_cast<T>(*h2d_data++));
    }
  }

  PT_EAGER_DEBUG("H2D data read from host buffer:", host_data);
  return host_data;
}

template std::vector<int32_t> GetH2DTensorHostData(at::Tensor&);
template std::vector<uint64_t> GetH2DTensorHostData(at::Tensor&);
template std::vector<uint32_t> GetH2DTensorHostData(at::Tensor&);
} // namespace graph
} // namespace habana
