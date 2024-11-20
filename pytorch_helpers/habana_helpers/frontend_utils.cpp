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

#include "habana_helpers/frontend_utils.h"
#include <ATen/core/ivalue.h>
#include "backend/create_pt_tensor.h"
#include "backend/habana_device/PinnedMemoryAllocator.h"
#include "backend/habana_operator.h"
#include "backend/helpers/graph.h"
#include "backend/synapse_helpers/env_flags.h"
#include "habana_helpers/dtype_helpers.h"
#include "habana_helpers/logging.h"
#include "habana_helpers/python_utils.h"
#include "habana_kernels/kernel_utils.h"
#include "habana_lazy/aten_lazy_bridge.h"
#include "habana_lazy/lazy_executor.h"
#include "habana_lazy/permute_tensors.h"

/*************************************************************************
 * @brief This helper function casts a long tensor to int (on CPU)
 ************************************************************************/
at::Tensor habana_helpers::cast_tensor_to_integer(
    const at::Tensor& long_tensor) {
  // TODO Remove this cast on CPU when int64_t->int32 cast available on
  // HPU

  auto int_tensor = std::make_unique<at::Tensor>();
  if (!habana_lazy::isDeviceInLoweringMode()) {
    // if not in lowering mode just return a tensor storageless wrapper as a
    // placeholder to avoid dma in case we need backend end tensor in future we
    // can replace createpttensor with empty_hpu_lazy
    *int_tensor = habana::createPTTensor(
        long_tensor,
        long_tensor.sizes(),
        long_tensor.options().dtype(c10::ScalarType::Int),
        long_tensor.suggest_memory_format(),
        long_tensor.scalar_type(),
        false);
  } else {
    if (long_tensor.scalar_type() == c10::ScalarType::Long) {
      *int_tensor = long_tensor.to("cpu")
                        .to(c10::ScalarType::Int)
                        .to(long_tensor.device(), c10::attr::non_blocking);
    } else {
      *int_tensor = long_tensor;
    }
  }

  return *int_tensor;
}

at::Tensor habana_helpers::downcast_to_int_if_needed(const at::Tensor& in) {
  return habana_helpers::is_downcast_to_int_needed(in.scalar_type())
      ? habana_helpers::cast_tensor_to_integer(in)
      : in;
}

at::Tensor habana_helpers::cast_tensor_to_long(const at::Tensor& int_tensor) {
  // TODO Remove this cast on CPU when int32->int64_t cast available on
  // HPU
  auto long_tensor = std::make_unique<at::Tensor>();
  if (!habana_lazy::isDeviceInLoweringMode() &&
      GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) != 0) {
    // if not in lowering mode just return a tensor storageless wrapper as a
    // placeholder to avoid dma
    *long_tensor = habana::createPTTensor(
        int_tensor,
        int_tensor.sizes(),
        int_tensor.options().dtype(c10::ScalarType::Long),
        int_tensor.suggest_memory_format(),
        int_tensor.scalar_type(),
        false);
  } else {
    if (int_tensor.scalar_type() == c10::ScalarType::Int) {
      *long_tensor = int_tensor.to("cpu")
                         .to(c10::ScalarType::Long)
                         .to(int_tensor.device(), c10::attr::non_blocking);
    } else {
      *long_tensor = int_tensor;
    }
  }

  return *long_tensor;
}

/******************************************************************************
 * @brief helper function for copying data from device to host
 * @param[in] src - source tensor in device
 * @param[in] size - transfer data size in bytes
 * @param[out] dst_ptr - destination memory address in cpu
 *****************************************************************************/
void habana_helpers::copy_scalar_to_host(
    const at::Tensor& src,
    void* dst_ptr,
    uint32_t size,
    c10::hpu::HPUStream hpu_stream) {
  std::atomic<bool> copyDone{false};
  bool is_pinned = habana::PinnedMemoryAllocator_is_pinned(src.data_ptr());

  habana::HPUDeviceContext::copy_data_to_host(
      reinterpret_cast<synapse_helpers::device_ptr>(src.data_ptr()),
      dst_ptr,
      reinterpret_cast<synapse_helpers::device_ptr>(
          src.storage().data_ptr().get()),
      size,
      [&copyDone]() { copyDone = true; },
      is_pinned,
      hpu_stream);

  // Release GIL if going to wait. This thread might already acquired GIL and
  // the second thread will be waiting
  {
    habana_helpers::AutoNoGIL gil_release;
    // wait for copy completion
    while (!copyDone) {
      std::this_thread::yield();
    }
  }
}
c10::Scalar habana_helpers::_local_scalar_dense_internal(
    const at::Tensor& self) {
  c10::Scalar r;
  // Note:
  // 1. This macro expands to more types than HPU supports,
  //   but that should not be an issue issue.
  // 2. Pytorch uses this function to check a specific emement of a tensor
  //   eg. embedding_bag validates the first value offsets to be 0 using this
  //   function
  // 3. A TORCH_CHECK is added to ensure that the size at source
  //   matches with the destination.

  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(
      at::ScalarType::Bool,
      at::ScalarType::BFloat16,
      at::ScalarType::Half,
      self.scalar_type(),
      "_local_scalar_dense",
      [&] {
        scalar_t val;
        TORCH_CHECK(
            elementSize(self.scalar_type()) == sizeof(val),
            " source and destination size mismatch");
        habana_helpers::copy_scalar_to_host(
            self, &val, sizeof(val), c10::hpu::getCurrentHPUStream());
        r = c10::Scalar(val);
      });
  return r;
}

/*************************************************************************
 * @brief Generic helper function to cast tensors on HPU
 ************************************************************************/
at::Tensor habana_helpers::hpu_cast_tensor(
    const at::Tensor& Input,
    caffe2::TypeMeta type) {
  PT_KERNEL_BEGIN;

  // At times we get 0-D tensor which cannot be handled by Synapse. Convert it
  // 1-D tensor before proceeding further.
  if (Input.dim() == 0) {
    SET_SIZE_STRIDE_1D(Input);
  }

  // Determine cast node_type to use based on src & dst dtypes
  std::pair<c10::ScalarType, c10::ScalarType> type_key{
      Input.scalar_type(), at::typeMetaToScalarType(type)};
  auto node_type{direct_cast_guid(type_key)};
  HABANA_ASSERT(
      node_type.has_value() &&
      "Unsupported Cast operation requested in hpu_cast_tensor()");

  int device_id = Input.device().index();
  auto& device = habana::HPUDeviceContext::get_device(device_id);
  CastOperator Op(device_id, node_type.value());
  std::vector<c10::IValue> stack = {
      c10::IValue(Input), c10::IValue(at::typeMetaToScalarType(type))};
  std::vector<at::Tensor> pt_inputs{Input};

  size_t key = Op.GetRecipeKey(node_type.value(), stack);
  if (device.get_recipe_handle_cache().isCached(key)) {
    PT_KERNEL_DEBUG("Cache hit key:", key);
    auto Output = at::empty(
        Input.sizes(),
        Input.options().dtype(type),
        Input.suggest_memory_format());
    Op.SetPTInputs(pt_inputs);
    std::vector<at::Tensor> v{Output};
    Op.SetPTOutputs(v);
    Op.Execute(key);
  } else {
    PT_KERNEL_DEBUG("key:", key);
    // Create Graph
    auto graph = habana_helpers::create_graph(device_id, node_type.value());
    // Allocate synapse inputs
    Op.AllocateSynapseInputs(graph, pt_inputs, true);
    habana::OutputMetaDataVector output_metadata(1);
    output_metadata.at(0).persistent = true;
    Op.AllocateAndAddSynapseNode(graph, stack, output_metadata);
    // compile and execute the graph
    Op.Compile(graph);
  }

  PT_KERNEL_END;
  return Op.GetOutputs()[0];
}
