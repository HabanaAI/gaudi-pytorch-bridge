/*******************************************************************************
 * Copyright (C) 2023 Habana Labs, Ltd. an Intel Company
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

#include "habana_eager/graph_dynamic.h"
#include "backend/kernel/hpu_habana_launch_op_pt.h"

#include "habana_helpers/logging.h"

namespace habana {
namespace graph {

int64_t GetSymintValue(torch::jit::Stack& original_stack, uint64_t index) {
  int64_t value;
  HABANA_ASSERT(original_stack[index].isScalar() == 1);
  value = original_stack[index].toScalar().toInt();
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
  }
  return t_name;
}

at::Tensor createDynamicTensor(
    const std::vector<int64_t>& size,
    synTensorType type) {
  // TODO need to get a proper type and memory format for tensor creation
  at::Tensor tensor = at::empty(size, c10::ScalarType::Float).to("hpu");
  auto tmeta = get_tensor_extra_meta(tensor);
  tmeta->set_tensor_type(type);
  PT_EAGER_DEBUG(
      "Created dynamic tensor of type:", type, ", size:", tensor.sizes());
  return tensor;
}

template <typename T>
std::vector<T> GetH2DTensorHostData(at::Tensor& tensor) {
  std::vector<T> host_data;
  auto tmeta = get_tensor_extra_meta(tensor);
  size_t data_size = tensor.sizes()[0];
  PT_EAGER_DEBUG("Read H2D data of size :", data_size);
  if (tmeta->get_tensor_type() == HOST_TO_DEVICE_TENSOR) {
    habana::HostDataType h2d_dt_type = tmeta->get_host_dt_type();
    void* host_ptr = tmeta->get_host_ptr();
    if (h2d_dt_type == habana::HostDataType::INT32_T) {
      T* h2d_data = static_cast<T*>(host_ptr);
      for (size_t i = 0; i < data_size; i++) {
        host_data.push_back(static_cast<T>(*h2d_data++));
      }
    } else if (h2d_dt_type == habana::HostDataType::UINT64_T) {
      PT_EAGER_DEBUG("H2D tensor type not supported!!");
    }
  }

  PT_EAGER_DEBUG("H2D data read from host buffer:", host_data);
  return host_data;
}

template std::vector<int32_t> habana::graph::GetH2DTensorHostData(at::Tensor&);

} // namespace graph
} // namespace habana
