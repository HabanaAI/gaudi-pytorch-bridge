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

#include "collective_kernel_info.h"
#include "habana_serialization/deserializers.h"
#include "habana_serialization/serializers.h"
#include "tensor_info.h"

namespace {
template <typename T>
std::vector<int64_t> ptr_array_indices(
    const std::vector<std::shared_ptr<T>>& elements,
    const std::vector<std::shared_ptr<T>>& src_array) {
  std::vector<int64_t> indices;
  indices.reserve(elements.size());
  for (const auto& e : elements) {
    if (e.get() == nullptr) {
      indices.push_back(-1);
      continue;
    }
    auto iter = std::find(src_array.begin(), src_array.end(), e);
    TORCH_CHECK(iter != src_array.end(), "Failed to find element in src_array");
    indices.push_back(std::distance(src_array.begin(), iter));
  }
  return indices;
}

template <typename T>
std::vector<std::shared_ptr<T>> indices_array_to_ptr_array(
    const std::vector<int64_t>& indices,
    const std::vector<std::shared_ptr<T>>& src_array) {
  std::vector<std::shared_ptr<T>> ptr_array;
  ptr_array.reserve(indices.size());
  for (const auto idx : indices) {
    if (idx == -1) {
      ptr_array.push_back(nullptr);
      continue;
    }

    TORCH_CHECK(
        idx <= (int64_t)src_array.size(),
        "idx ",
        idx,
        " out of range. src array size = ",
        src_array.size());
    ptr_array.push_back(src_array.at(idx));
  }
  return ptr_array;
}
} // namespace

namespace habana_helpers {

size_t CollectiveKernelInfos::Info::Size() const {
  size_t size = sizeof(*this);
  size += input_tensor_infos.size() *
      sizeof(decltype(input_tensor_infos)::value_type);
  size += output_tensor_infos.size() *
      sizeof(decltype(output_tensor_infos)::value_type);
  return size;
}

void CollectiveKernelInfos::Launch(
    bool async,
    synapse_helpers::event_done_callback cleanup_callback) const {
  HABANA_ASSERT(
      infos_.empty() || GET_ENV_FLAG_NEW(PT_HPU_ENABLE_LAZY_COLLECTIVES))
  for (const auto& kernel_info : infos_) {
    HABANA_ASSERT(kernel_info.kernel);
    habana::CollectiveOperator& collective = *kernel_info.kernel;
    PT_BRIDGE_DEBUG("Running collective op ", collective.GetGuid());
    collective.RunCollective(
        kernel_info.input_tensor_infos, async, cleanup_callback);
  }
}
void CollectiveKernelInfos::ClearAllPtAndSynTensors() {
  for (const auto& kernel_info : infos_) {
    HABANA_ASSERT(kernel_info.kernel);
    kernel_info.kernel->clear_all_pt_and_syn_tensors();
  }
}

void CollectiveKernelInfos::Serialize(
    std::ostream& os,
    const std::vector<PtTensorInfoShared>& dtensorinfos) const {
  using namespace serialization;
  serialize(os, infos_.size());
  for (const auto& collective_kernel : infos_) {
    auto input_indices =
        ptr_array_indices(collective_kernel.input_tensor_infos, dtensorinfos);
    serialize(os, input_indices);

    auto output_indices =
        ptr_array_indices(collective_kernel.output_tensor_infos, dtensorinfos);
    serialize(os, output_indices);

    serialize(os, collective_kernel.kernel->GetGuid());
    serialize(os, collective_kernel.kernel->GetDeviceId());
    serialize(os, collective_kernel.kernel->GetScalarType());
    collective_kernel.kernel->Serialize(os);
  }
}

void CollectiveKernelInfos::Deserialize(
    std::istream& is,
    const std::vector<PtTensorInfoShared>& dtensorinfos) {
  infos_.clear();

  using namespace serialization;

  size_t num_collective_kernels = 0;
  deserialize(is, num_collective_kernels);
  for (size_t i = 0; i < num_collective_kernels; i++) {
    habana_helpers::CollectiveKernelInfos::Info kernel_info;

    std::vector<int64_t> input_indices;
    deserialize(is, input_indices);
    kernel_info.input_tensor_infos =
        indices_array_to_ptr_array(input_indices, dtensorinfos);

    std::vector<int64_t> output_indices;
    deserialize(is, output_indices);
    kernel_info.output_tensor_infos =
        indices_array_to_ptr_array(output_indices, dtensorinfos);

    std::string guid;
    int device_id;
    c10::ScalarType scalar_type;
    deserialize(is, guid);
    deserialize(is, device_id);
    deserialize(is, scalar_type);
    c10::OperatorName op_name(guid, "");
    habana::HabanaOperatorPtr habana_kernel =
        habana::KernelRegistry().get(device_id, op_name, scalar_type);
    auto collective_kernel =
        std::dynamic_pointer_cast<habana::CollectiveOperator>(habana_kernel);
    TORCH_CHECK(
        collective_kernel,
        "Failed to find collective kernel for ",
        guid,
        "during recipe load from disk");

    collective_kernel->Deserialize(is);
    kernel_info.kernel = collective_kernel;
    infos_.push_back(std::move(kernel_info));
  }
}

} // namespace habana_helpers
