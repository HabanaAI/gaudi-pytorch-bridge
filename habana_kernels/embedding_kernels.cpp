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
#include <ATen/InferSize.h>
#include <perf_lib_layer_params.h>
#include <synapse_api.h>
#include <torch/script.h>

#include "backend/create_pt_tensor.h"
#include "backend/habana_device/hpu_cached_devices.h"
#include "backend/helpers/create_tensor.h"
#include "backend/helpers/graph.h"
#include "backend/helpers/tensor_utils.h"
#include "backend/kernel/hpu_shape_inference.h"
#include "habana_helpers/frontend_utils.h"
#include "habana_helpers/logging.h"
#include "habana_helpers/logging_pt.h"
#include "habana_kernels/basic_kernels.h"
#include "habana_kernels/embedding_kernels.h"
#include "habana_kernels/index_kernels.h"
#include "habana_kernels/tensor_shape_kernels.h"
#include "habana_kernels/topk_kernels.h"
#include "habana_lazy/aten_lazy_bridge.h"
#include "habana_lazy/tensor_impl.h"
#include "kernel_utils.h"

using namespace torch;
using namespace habana;

std::vector<int64_t> PadOperator::compute_output_shape(
    const at::Tensor& self,
    c10::IntArrayRef pad) {
  auto ndim = self.dim();
  auto lpad = pad.size() / 2;

  TORCH_CHECK(
      pad.size() % 2 == 0,
      "Length of pad must be even but instead it equals ",
      pad.size());

  TORCH_CHECK(
      ndim >= (int64_t)lpad,
      "Length of pad should be no more than twice the number of "
      "dimensions of the input. Pad length is ",
      pad.size(),
      "while the input has ",
      ndim,
      "dimensions.");

  auto shape = self.sizes().vec();

  for (unsigned int i = 0; i < lpad; i++) {
    auto pad_start = pad[2 * i];
    auto pad_end = pad[2 * i + 1];
    shape[ndim - i - 1] += (pad_start + pad_end);
    TORCH_CHECK(
        shape[ndim - i - 1] > 0,
        "The input size ",
        self.sizes()[i],
        ", plus negative padding ",
        pad_start,
        " and ",
        pad_end,
        " resulted in a invalid output size, "
        "Check dimension ",
        i,
        " of your input.");
  }
  return shape;
}

std::vector<int64_t> PadOperator::compute_output_shape_ds(
    const at::Tensor& self,
    c10::IntArrayRef pad_before,
    c10::IntArrayRef pad_after) {
  auto ndim = self.dim();

  auto shape = self.sizes().vec();

  for (unsigned int i = 0; i < ndim; i++) {
    auto pad_start = pad_before[MAX_DIMENSIONS_NUM - i - 1];
    auto pad_end = pad_after[MAX_DIMENSIONS_NUM - i - 1];
    shape[ndim - i - 1] += (pad_start + pad_end);
    TORCH_CHECK(
        shape[ndim - i - 1] > 0,
        "The input size ",
        self.sizes()[i],
        ", plus negative padding ",
        pad_start,
        " and ",
        pad_end,
        " resulted in a invalid output size, "
        "Check dimension ",
        i,
        " of your input.");
  }
  return shape;
}

void PadOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    const OutputMetaDataVector& output_metadata) {
  TORCH_CHECK(
      inputs.size() == 3,
      "Incorrect size of inputs expected for PadOperator Operator");
  TORCH_CHECK(
      inputs[0].isTensor(),
      "Input arg1 expected to be Tensor for PadOperator Operator");
  TORCH_CHECK(
      inputs[1].isIntList(),
      "Input arg2 expected to be IntList for PadOperator Operator");
  TORCH_CHECK(
      inputs[2].isScalar(),
      "Input arg3 expected to be Scalar for PadOperator Operator");

  auto self = inputs[0].toTensor();
  auto pad = inputs[1].toIntVector();

  std::vector<int64_t> shape;
  shape = compute_output_shape(self, pad);

  auto ndim = self.dim();
  auto lpad = pad.size() / 2;

  ns_PadKernelEx::Params param;
  param.mode = PadMode_t::PAD_MODE_CONSTANT;
  if (c10::isIntegralType(self.scalar_type(), false)) {
    param.value.i = inputs[2].toScalar().to<decltype(param.value.i)>();
  } else {
    param.value.f = inputs[2].toScalar().to<float>();
  }
  memset(param.pads, 0, sizeof(param.pads));
  for (unsigned int i = 0; i < lpad; i++) {
    param.pads[i] = pad[2 * i];
    param.pads[i + ndim] = pad[2 * i + 1];
  }

  at::Tensor output;
  if (!graph.is_dry_run() &&
      output_metadata.at(0).allocated_tensor.has_value()) {
    output = output_metadata.at(0).allocated_tensor.value();
  } else {
    output = at::empty(shape, self.options());
  }
  AllocateSynapseOutput(graph, output, output_metadata.at(0));
  AddNodeToSynapseGraph(graph, &param, sizeof(param));
}

InferOutputMetaRetType PadOperatorHT::InferOutputMeta(
    torch::jit::Stack& inputs) {
  std::vector<int64_t> shape;
  auto input = inputs[0].toTensor();
  shape = inputs[2].toTensor().sizes().vec();

  InferOutputMetaRetType out;
  out.AddOutputTensor(TensorMetaData(
      shape,
      HabanaOperator::CalculateStrides(shape, input.suggest_memory_format()),
      input.scalar_type(),
      input.suggest_memory_format()));
  return out;
}
void PadOperatorHT::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    const OutputMetaDataVector& output_metadata) {
  TORCH_CHECK(
      inputs.size() == 4,
      "Incorrect size of inputs expected for PadOperatorHT Operator");
  TORCH_CHECK(
      inputs[0].isTensor(),
      "Input arg1 expected to be Tensor for PadOperatorHT Operator");
  TORCH_CHECK(
      inputs[1].isTensor(),
      "Input arg2 expected to be of type Tensor for PadOperatorHT operator");
  TORCH_CHECK(
      inputs[2].isTensor(),
      "Input arg3 expected to be of type Tensor for PadOperatorHT operator");
  TORCH_CHECK(
      inputs[3].isScalar(),
      "Input arg4 expected to be of type Scalar for PadOperatorHT operator");

  std::vector<int64_t> shape;
  auto self = inputs[0].toTensor();

  TORCH_CHECK(p_context_->syn_inputs_[1].ref().is_host_to_device_tensor());
  shape = inputs[2].toTensor().sizes().vec();
  at::Tensor host_tensor = inputs[1].toTensor();
  auto tmeta{get_tensor_extra_meta(host_tensor)};
  auto output_shape = inputs[2].toTensor().sizes().vec();
  auto input_shape = self.sizes().vec();
  TORCH_CHECK(
      tmeta->get_host_dt_type() == habana::HostDataType::UINT32_T,
      "Incorrect datatype of HOST");
  if (habana::ShapeInference::GetCurrentPass() ==
      habana::ShapeInfo::InferencePass::MIN_SHAPE) {
    auto ndim = self.dim();
    auto in_data = self.sizes().vec();
    std::vector<uint32_t> data(MAX_DIMENSIONS_NUM * 2, 0);
    for (unsigned int i = 0; i < ndim; i++) {
      // order of dims is reversed in H2D tensor
      data[ndim - i - 1] = output_shape[i] - input_shape[i];
    }
    tmeta->set_min<uint32_t>(data);
  } else if (
      habana::ShapeInference::GetCurrentPass() ==
      habana::ShapeInfo::InferencePass::MAX_SHAPE) {
    auto ndim = self.dim();
    auto in_data = self.sizes().vec();
    std::vector<uint32_t> data(MAX_DIMENSIONS_NUM * 2, 0);
    for (unsigned int i = 0; i < ndim; i++) {
      // order of dims is reversed in H2D tensor
      data[ndim - i - 1] = output_shape[i] - input_shape[i];
    }
    tmeta->set_max<uint32_t>(data);
  }

  ns_PadKernelEx::Params param;
  param.mode = PadMode_t::PAD_MODE_CONSTANT;
  if (c10::isIntegralType(self.scalar_type(), false)) {
    param.value.i = inputs[3].toScalar().to<decltype(param.value.i)>();
  } else {
    param.value.f = inputs[3].toScalar().to<float>();
  }
  // pads value shall be picked from H2D tensor, set this to 0's to be safe
  memset(param.pads, 0, sizeof(param.pads));
  auto output = at::empty(shape, self.options());
  // throw away shape tensor before adding synapse node
  p_context_->syn_inputs_.pop_back();
  AllocateSynapseOutput(graph, output, output_metadata.at(0));
  AddNodeToSynapseGraph(graph, &param, sizeof(param));
}

void EmbeddingBagSumOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    const OutputMetaDataVector& output_metadata) {
  TORCH_CHECK(
      inputs.size() == 5,
      "Incorrect size of inputs expected for EmbeddingBagSumOperator operator");
  TORCH_CHECK(inputs[0].isTensor(), "Input arg1 type expected to be tensor");
  TORCH_CHECK(inputs[1].isTensor(), "Input arg2 type expected to be tensor");
  TORCH_CHECK(inputs[2].isTensor(), "Input arg3 type expected to be tensor");
  TORCH_CHECK(inputs[3].isTensor(), "Input arg4 type expected to be tensor");
  TORCH_CHECK(inputs[4].isInt(), "Input arg5 type expected to be tensor");

  auto input = inputs[0].toTensor();
  auto indices = inputs[1].toTensor();
  auto offsets = inputs[2].toTensor();
  auto valid_count = inputs[3].toTensor();

  TORCH_CHECK(indices.dim() <= 1, "index tensor cannot be more than 1D")
  TORCH_CHECK(offsets.dim() <= 1, "offsets tensor cannot be more than 1D")
  TORCH_CHECK(input.dim() == 2, "Input tensor should be 2D")
  TORCH_CHECK(valid_count.dim() == 1, "valid count tensor should be 1D")

  auto kernel_mode = inputs[4].toInt();
  auto output_size_dim0 =
      kernel_mode ? (offsets.sizes()[0] - 1) : indices.sizes()[0];
  auto output = habana::createPTTensor(
      input,
      {output_size_dim0, input.size(1)},
      input.options(),
      input.suggest_memory_format(), // TBD: not reqd?
      output_metadata.at(0).persistent);
  if (kernel_mode == 0) {
    auto guid = get_guid_with_precision(
        "gather_with_valid_count_2d", input.scalar_type());
    SetGuid(guid);
    p_context_->syn_inputs_.erase(p_context_->syn_inputs_.begin() + 2);
    p_context_->pt_inputs_.erase(p_context_->pt_inputs_.begin() + 2);
  } else if (kernel_mode == 2) {
    auto guid = get_guid_with_precision(
        "embedding_bag_sum_small_lengths_2d_fwd", input.scalar_type());
    SetGuid(guid);
  }
  AllocateSynapseOutput(graph, output, output_metadata.at(0));
  AddNodeToSynapseGraph(graph, nullptr, 0);
}

void EmbeddingBagSumBwdKernelModeOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    const OutputMetaDataVector& output_metadata) {
  static_cast<void>(output_metadata);
  HABANA_ASSERT(inputs.size() == 6);

  HABANA_ASSERT(inputs[0].isTensor());
  HABANA_ASSERT(inputs[1].isTensor());
  HABANA_ASSERT(inputs[2].isTensor());
  HABANA_ASSERT(inputs[3].isTensor());
  HABANA_ASSERT(inputs[4].isTensor());
  HABANA_ASSERT(inputs[5].isInt());

  auto out = inputs[0].toTensor();
  auto input = inputs[1].toTensor();
  auto indices_bwd = inputs[2].toTensor();
  auto offsets_bwd = inputs[3].toTensor();
  auto valid_count_bwd = inputs[4].toTensor();
  auto kernel_mode = inputs[5].toInt();

  HABANA_ASSERT(out.dim() == 2);
  HABANA_ASSERT(input.dim() == 2);
  HABANA_ASSERT(indices_bwd.dim() == 1);
  HABANA_ASSERT(offsets_bwd.dim() == 1);
  HABANA_ASSERT(valid_count_bwd.numel() == 2);
  HABANA_ASSERT((kernel_mode >= 0) && (kernel_mode < 3));

  p_context_->syn_outputs_.emplace_back(
      habana_helpers::duplicate_tensor_in_memory_section(
          p_context_->syn_inputs_[0], graph, output_metadata.at(0).external));
  p_context_->syn_inputs_.erase(p_context_->syn_inputs_.begin());

  p_context_->pt_outputs_.emplace_back(p_context_->pt_inputs_[0]);
  p_context_->pt_inputs_.erase(p_context_->pt_inputs_.begin());

  AddNodeToSynapseGraph(graph, nullptr, 0);
}

static auto& EmbeddingKernelsKernelRegistry =
    habana::KernelRegistry()
        .add("hpu::constant_pad_nd_lazy", KERNEL_FN(PadOperator))
        .add("hpu::constant_pad_nd_ht", KERNEL_FN(PadOperatorHT))
        .add("hpu::embedding_bag_sum", KERNEL_FN(EmbeddingBagSumOperator))
        .add(
            "hpu::embedding_bag_sum_bwd_out",
            KERNEL_FN(EmbeddingBagSumBwdKernelModeOperator));
