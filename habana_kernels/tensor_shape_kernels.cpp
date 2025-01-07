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
#include <ATen/ExpandUtils.h>
#include <ATen/InferSize.h>
#include <ATen/native/TypeProperties.h>
#include <synapse_api.h>
#include <torch/script.h>

#include "backend/create_pt_tensor.h"
#include "backend/habana_device/hpu_cached_devices.h"
#include "backend/helpers/create_tensor.h"
#include "backend/helpers/graph.h"
#include "backend/helpers/tensor_utils.h"
#include "habana_helpers/dtype_helpers.h"
#include "habana_helpers/frontend_utils.h"
#include "habana_helpers/logging.h"
#include "habana_helpers/logging_pt.h"
#include "habana_kernels/index_kernels.h"
#include "habana_kernels/kernel_utils.h"
#include "habana_kernels/resize.h"
#include "habana_kernels/tensor_shape_kernels.h"
#include "habana_lazy/hlexec.h"

using namespace torch;
using namespace habana;

std::vector<int64_t> CatOperator::compute_output_shape(
    const at::TensorList tensors,
    int64_t dim_) {
  int64_t dim = at::maybe_wrap_dim(
      dim_,
      tensors[0].dim(),
      /*wrap_scalar=*/true);

  auto in_tensor_count = tensors.size();
  auto first_tensor = tensors[0];
  auto out_size = first_tensor.sizes().vec();
  out_size[dim] = 0;
  for (unsigned i = 0; i < in_tensor_count; i++) {
    out_size[dim] += tensors[i].sizes()[dim];
  }
  return out_size;
}

auto CatOperator::CreateParamsAndAddToContext(int64_t axis) {
  synConcatenateParams params;
  params.axis = axis;
  p_context_->params_.emplace<synConcatenateParams>(params);
  p_context_->params_size_ = sizeof(params);

  return params;
}

void CatOperator::validate_cat_tensor_dim_sizes(
    const std::vector<std::vector<int64_t>>* tensors,
    int64_t dim) {
  unsigned i = 0;
  auto tensor_count = tensors->size();
  auto tempT_i = 0;
  for (i = 1; i < tensor_count; i++) {
    // check whether sizes along dimensions match except for cat dimension.
    unsigned j = 0;
    auto sz1 = tensors->at(i);
    auto sz2 = tensors->at(tempT_i);
    for (j = 0; j < tensors->at(i).size(); j++) {
      if (j != dim && (sz1[j] - sz2[j]) != 0) {
        TORCH_CHECK(
            ((sz1[j] - sz2[j]) == 0),
            "Sizes of tensors along one of the non-cat dimensions don't match");
      }
    }
    tempT_i = i;
  }
}

Tensor CatOperator::CheckAllocateOutput(
    Stack& inputs,
    const OutputMetaData& output_metadata,
    bool is_dry_run) {
  TORCH_CHECK(
      inputs.size() == 2 || inputs.size() == 3,
      "Incorrect size of inputs expected for cat operator");

  TORCH_CHECK(
      inputs[0].isTensorList(), "Input arg2 type expected to be tensor list");
  TORCH_CHECK(inputs[1].isInt(), "Input arg3 type expected to be int");

  auto tensors = inputs[0].toTensorList();
  auto dim_ = inputs[1].toInt();

  auto first_tensor = tensors.get(0);
  int64_t dim =
      at::maybe_wrap_dim(dim_, first_tensor.dim(), /*wrap_scalar=*/true);
  TORCH_CHECK(
      dim < first_tensor.ndimension(),
      "Cat dimension specified exceeds tensors dimensions");

  auto output_dtype =
      at::native::result_type(static_cast<ITensorListRef>(tensors));

  std::vector<std::vector<int64_t>> tensors_size;
  auto tensor_count = tensors.size();
  for (unsigned i = 0; i < tensor_count; i++)
    tensors_size.emplace_back(tensors.get(i).sizes().vec());

  validate_cat_tensor_dim_sizes(&tensors_size, dim);

  if (!is_dry_run && output_metadata.allocated_tensor.has_value()) {
    return output_metadata.allocated_tensor.value();
  }

  if (dim != dim_) {
    inputs[1] = IValue(dim);
  }

  std::vector<int64_t> out_size;
  if (inputs.size() == 2) {
    auto tensor_count = tensors.size();
    // out tensor size should match along all dimensions for input tensors
    // except along the dim in which to cat
    out_size = first_tensor.sizes().vec();
    out_size[dim] = 0;
    for (unsigned i = 0; i < tensor_count; i++) {
      out_size[dim] += tensors.get(i).sizes()[dim];
    }
  } else { // shape tensor being used
    out_size = inputs[2].toTensor().sizes().vec();
  }

  return habana::createPTTensor(
      first_tensor,
      out_size,
      first_tensor.options().dtype(output_dtype),
      first_tensor.suggest_memory_format(),
      output_metadata.persistent);
}

InferOutputMetaRetType CatOperator::InferOutputMeta(torch::jit::Stack& inputs) {
  auto out_tensor = CheckAllocateOutput(inputs, OutputMetaData(), false);
  InferOutputMetaRetType out;
  out.AddOutputTensor(TensorMetaData(
      out_tensor.sizes().vec(),
      HabanaOperator::CalculateStrides(
          out_tensor.sizes().vec(), out_tensor.suggest_memory_format()),
      out_tensor.scalar_type(),
      out_tensor.suggest_memory_format()));
  return out;
}

void CatOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    Stack& inputs,
    const OutputMetaDataVector& output_metadata) {
  auto out =
      CheckAllocateOutput(inputs, output_metadata.at(0), graph.is_dry_run());
  inputs.emplace_back(out);
  auto dim = inputs[1].toInt();
  auto kernel_dim = (out.ndimension() - dim) - 1;
  auto tensors = inputs[0].toTensorList();
  auto output_dtype = out.scalar_type();
  for (unsigned i = 0; i < tensors.size(); i++) {
    const auto& tensor = tensors.get(i);
    if (habana_helpers::getInternalDtype(tensor.scalar_type()) !=
        habana_helpers::getInternalDtype(output_dtype)) {
      auto cast = make_operator<CastOperator>(p_context_->device_id_, "");
      cast->SetSynapseInput(GetSynInputs()[i]);
      at::Stack stack{tensor, output_dtype};
      OutputMetaData md;
      md.dtype = output_dtype;
      cast->AllocateAndAddSynapseNode(graph, stack, {md});
      p_context_->syn_input_orig_.emplace_back(
          std::move(p_context_->syn_inputs_[i]));
      p_context_->syn_inputs_[i] = std::move(cast->GetSynOutputs()[0]);
    }
  }

  auto params = CreateParamsAndAddToContext(kernel_dim);
  AllocateSynapseOutput(graph, out, output_metadata.at(0));
  AddNodeToSynapseGraph(graph, &params, sizeof(params));

  // Revert input stack
  inputs.pop_back();
}

/****************************************************************************
 * @brief Kernel implementation for N-D out = torch.transpose(self,dim0,dim1)
 * @param self - input
 * @param dim0 - first dimension to swap
 * @param dim0 - second dimension to swap
 ***************************************************************************/
TransposeOperator::TransposeOperator(int device_id, c10::ScalarType scalarType)
    : HabanaOperator("transpose") {
  static_cast<void>(scalarType);
  this->CreateSynContext(device_id);
  this->setNoComputeFlag();
}

std::tuple<std::vector<int64_t>, std::vector<int64_t>> TransposeOperator::
    compute_output_shape(const at::Tensor& self, int dim0_, int dim1_) {
  int64_t dim0 = at::maybe_wrap_dim(dim0_, self.dim(), /*wrap_scalar=*/true);
  int64_t dim1 = at::maybe_wrap_dim(dim1_, self.dim(), /*wrap_scalar=*/true);
  TORCH_CHECK(
      (dim0 < self.dim()) && (dim1 < self.dim()),
      "Specified dims are beyond tensor dims");

  auto self_sizes = self.sizes().vec();
  auto self_strides = self.strides().vec();
  std::swap(self_sizes[dim0], self_sizes[dim1]);
  // Recalculate the strides to account for transpose size changes
  // In effect, keep the tensor contiguous.
  habana_helpers::recalc_strides(self_strides, self_sizes);
  return std::make_tuple(self_sizes, self_strides);
}

InferOutputMetaRetType TransposeOperator::InferOutputMeta(
    torch::jit::Stack& inputs) {
  Tensor self = inputs[0].toTensor();
  auto dim0_ = inputs[1].toInt();
  auto dim1_ = inputs[2].toInt();

  std::vector<int64_t> self_sizes, self_strides;
  std::tie(self_sizes, self_strides) =
      TransposeOperator::compute_output_shape(self, dim0_, dim1_);

  InferOutputMetaRetType out;
  out.AddOutputTensor(TensorMetaData(
      self_sizes,
      self_strides,
      self.scalar_type(),
      self.suggest_memory_format()));
  return out;
}

void TransposeOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    Stack& inputs,
    const OutputMetaDataVector& output_metadata) {
  TORCH_CHECK(
      inputs.size() == 3,
      "Incorrect size of input arguments for Transpose Operator");
  TORCH_CHECK(
      inputs[0].isTensor(),
      "Input arg 1 for transpose op needs to be tensor type");
  TORCH_CHECK(
      inputs[1].isInt(),
      "Input arg 2 for transpose op needs to be of Int type");
  TORCH_CHECK(
      inputs[2].isInt(),
      "Input arg 3 for transpose op needs to be of Int type");
  Tensor self = inputs[0].toTensor();
  auto dim0_ = inputs[1].toInt();
  auto dim1_ = inputs[2].toInt();

  int64_t dim0 = at::maybe_wrap_dim(dim0_, self.dim(), /*wrap_scalar=*/true);
  int64_t dim1 = at::maybe_wrap_dim(dim1_, self.dim(), /*wrap_scalar=*/true);

  std::vector<int64_t> self_sizes, self_strides;
  std::tie(self_sizes, self_strides) =
      TransposeOperator::compute_output_shape(self, dim0_, dim1_);
  auto out = habana::createPTTensor(
      self,
      self_sizes,
      self_strides,
      self.options(),
      self.suggest_memory_format(),
      output_metadata.at(0).persistent);
  synTransposeParamsNDims params;
  params.tensorDim = self.dim();
  int i;
  for (i = 0; i < HABANA_DIM_MAX; i++) {
    params.permutation[i] = static_cast<TransposePermutationDim>(i);
  }
  std::swap(
      params.permutation[self.dim() - 1 - dim0],
      params.permutation[self.dim() - 1 - dim1]);

  p_context_->params_.emplace<synTransposeParamsNDims>(params);
  p_context_->params_size_ = sizeof(params);

  AllocateSynapseOutput(graph, out, output_metadata.at(0));
  AddNodeToSynapseGraph(graph, &params, sizeof(params));
}

namespace habana {
SharedMetaDataVector TransposeSharedMeta(
    const at::Stack& stack,
    habana_helpers::HabanaExecutionMode) {
  const auto& self = stack.at(0).toTensor();
  const auto selfDim = self.dim();
  const auto selfDtype = self.scalar_type();

  SharedMetaData transposeSharedMeta("transpose");
  transposeSharedMeta.inputs_data.emplace_back(selfDim, selfDtype);
  transposeSharedMeta.outputs_data.emplace_back(selfDim, selfDtype);

  return {transposeSharedMeta};
}
} // namespace habana

/*************************************************************************
 * @brief Kernel implementation for torch.Tensor.permute(dims)
 * @param self - input on which permute needs to be applied
 * @param dims_ - permute dims array
 ************************************************************************/
PermuteOperator::PermuteOperator(int device_id, c10::ScalarType scalarType)
    : HabanaOperator("transpose") {
  static_cast<void>(scalarType);
  this->CreateSynContext(device_id);
  this->setNoComputeFlag();
}

std::tuple<std::vector<int64_t>, std::vector<int64_t>> PermuteOperator::
    compute_output_shape(
        const at::Tensor& in,
        const std::vector<int64_t>& dims) {
  TORCH_CHECK(
      dims.size() == static_cast<size_t>(in.dim()),
      "Number of dims in tensor don't match in permute");
  auto self_sizes = in.sizes().vec();
  // calculate new sizes and strides after permute for out tensor
  auto new_sizes = in.sizes().vec();
  auto new_strides = in.strides().vec();
  new_sizes[new_sizes.size() - 1] = self_sizes[dims[new_sizes.size() - 1]];
  new_strides[new_sizes.size() - 1] = 1;
  for (int i = new_sizes.size() - 2; i >= 0; i--) {
    new_sizes[i] = self_sizes[dims[i]];
    new_strides[i] = new_strides[i + 1] * new_sizes[i + 1];
  }
  return std::make_tuple(new_sizes, new_strides);
}

InferOutputMetaRetType PermuteOperator::InferOutputMeta(
    torch::jit::Stack& inputs) {
  Tensor self = inputs[0].toTensor();
  const auto dims = inputs[1].toIntVector();
  std::vector<int64_t> new_sizes, new_strides;
  std::tie(new_sizes, new_strides) =
      PermuteOperator::compute_output_shape(self, dims);

  InferOutputMetaRetType out;
  out.AddOutputTensor(TensorMetaData(
      new_sizes,
      new_strides,
      self.scalar_type(),
      self.suggest_memory_format()));
  return out;
}

void PermuteOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    Stack& inputs,
    const OutputMetaDataVector& output_metadata) {
  TORCH_CHECK(
      inputs.size() == 2,
      "Incorrect size of input arguments for Permute Operator");
  TORCH_CHECK(
      inputs[0].isTensor(),
      "Input arg 1 for permute op needs to be tensor type");
  TORCH_CHECK(
      inputs[1].isIntList(),
      "Input arg 2 for permute op needs to be of Int List type");
  Tensor self = inputs[0].toTensor();
  const auto dims = inputs[1].toIntVector();

  TORCH_CHECK(
      dims.size() == static_cast<size_t>(self.dim()),
      "Number of dims in tensor don't match in permute");
  TORCH_CHECK(
      (self.dim() <= HABANA_DIM_MAX),
      "Number of tensor dims larger then allowed max limit");

  std::vector<int64_t> new_sizes, new_strides;
  std::tie(new_sizes, new_strides) =
      PermuteOperator::compute_output_shape(self, dims);

  auto& mdata = output_metadata.at(0);
  auto output = !graph.is_dry_run() && mdata.allocated_tensor.has_value()
      ? mdata.allocated_tensor.value()
      : habana::createPTTensor(
            self,
            new_sizes,
            new_strides,
            self.options(),
            self.suggest_memory_format(),
            mdata.persistent);

  synTransposeParamsNDims params;
  params.tensorDim = self.dim();
  // params.permute has to be populated in a reverse order for HPU FCD-LCD order
  for (int i = 0; i < self.dim(); i++) {
    params.permutation[i] = static_cast<TransposePermutationDim>(
        self.dim() - dims[dims.size() - i - 1] - 1);
  }
  for (int i = self.dim(); i < HABANA_DIM_MAX; i++) {
    params.permutation[i] = static_cast<TransposePermutationDim>(i);
  }

  p_context_->params_.emplace<synTransposeParamsNDims>(params);
  p_context_->params_size_ = sizeof(params);
  AllocateSynapseOutput(graph, output, mdata);
  AddNodeToSynapseGraph(graph, &params, sizeof(params));
}

void PermuteCLOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    Stack& inputs,
    const OutputMetaDataVector& output_metadata) {
  PermuteOperator::AllocateAndAddSynapseNode(graph, inputs, output_metadata);

  if (!habana_lazy::exec::OptPassCfg::GetInstance()->IsEnabledPermutePass()) {
    auto& output = p_context_->pt_outputs_[0];
    auto sizes = output.sizes().vec();
    auto strides = output.strides().vec();
    std::vector<int> out_pos = {
        LayoutFormatDims::N,
        LayoutFormatDims::W,
        LayoutFormatDims::C,
        LayoutFormatDims::H};
    std::vector<long int> swapped_sizes = {
        sizes[out_pos[0]],
        sizes[out_pos[1]],
        sizes[out_pos[2]],
        sizes[out_pos[3]]};
    std::vector<long int> swapped_strides = {
        strides[out_pos[0]],
        strides[out_pos[1]],
        strides[out_pos[2]],
        strides[out_pos[3]]};
    output.unsafeGetTensorImpl()->set_sizes_and_strides(
        swapped_sizes, swapped_strides);
  }
}

InferOutputMetaRetType ReshapeOperator::InferOutputMeta(
    torch::jit::Stack& inputs) {
  std::vector<int64_t> inferred_size;
  Tensor self = inputs[0].toTensor();
  /*
   * if we have already created shape tensor at the frontend, then
   * we dont need the below processing at all.
   */
  if (inputs[1].isIntList()) {
    auto shape = inputs[1].toIntList();
    auto shape_vector = shape.vec();
    auto input_shape = IntArrayRef(shape_vector.data(), shape_vector.size());
    inferred_size = habana_helpers::infer_size(input_shape, self.numel());
  } else {
    auto shapeTensor = inputs[1].toTensor();
    inferred_size = shapeTensor.sizes().vec();
    if (shapeTensor.sizes().empty() && inputs.size() == 3) {
      auto shape = inputs[2].toIntList();
      auto shape_vector = shape.vec();
      auto input_shape = IntArrayRef(shape_vector.data(), shape_vector.size());
      inferred_size = habana_helpers::infer_size(input_shape, self.numel());
    }
  }

  auto memory_format = self.suggest_memory_format();
  if (inferred_size.size() < 4) {
    memory_format = at::MemoryFormat::Contiguous;
  }

  InferOutputMetaRetType out;
  auto tensor_meta_data = TensorMetaData(
      inferred_size,
      HabanaOperator::CalculateStrides(inferred_size, memory_format),
      self.scalar_type(),
      memory_format);
  out.AddOutputTensor(tensor_meta_data);

  if (inputs[1].isIntList()) {
    out.AddShapeTensor(tensor_meta_data);
  }

  return out;
}

/*************************************************************************
 * @brief Kernel implementation for torch.Tensor.reshape
 * @param self - input on which reshape needs to be applied
 * @param shape - reshape  shape array
 ************************************************************************/
void ReshapeOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    Stack& inputs,
    const OutputMetaDataVector& output_metadata) {
  TORCH_CHECK(
      inputs.size() == 2 || inputs.size() == 3,
      "Incorrect size of input arguments for Reshape Operator");
  std::vector<int64_t> inferred_size;
  Tensor self = inputs[0].toTensor();
  /*
   * if we have already created shape tensor at the frontend, then
   * we dont need the below processing at all.
   */
  if (inputs[1].isIntList()) {
    auto shape = inputs[1].toIntList();
    auto shape_vector = shape.vec();
    auto input_shape = IntArrayRef(shape_vector.data(), shape_vector.size());
    inferred_size = habana_helpers::infer_size(input_shape, self.numel());
  } else {
    TORCH_CHECK(p_context_->syn_inputs_.back().ref().is_shape_tensor());
    inferred_size = p_context_->syn_inputs_.back().ref().pt_shape();
  }

  auto memory_format = self.suggest_memory_format();
  if (inferred_size.size() < 4) {
    memory_format = at::MemoryFormat::Contiguous;
  }

  at::Tensor output;
  if (!graph.is_dry_run() &&
      output_metadata.at(0).allocated_tensor.has_value()) {
    output = output_metadata.at(0).allocated_tensor.value();
  } else {
    output = habana::createPTTensor(
        self,
        inferred_size,
        self.options(),
        memory_format,
        output_metadata.at(0).persistent);
  }

  TORCH_CHECK(
      self.numel() == output.numel(),
      "Reshape doesnt support change in number of elements: ",
      self.sizes(),
      " Size of output: ",
      output.sizes());
  p_context_->params_size_ = 0;

  if (inputs[1].isIntList()) {
    // Allocate Shape tensor
    if (graph.is_dynamic_graph()) {
      AllocateSynapseShapeTensor(graph, output);
    }
  }

  AllocateSynapseOutput(graph, output, output_metadata.at(0));
  AddNodeToSynapseGraph(graph, NULL, 0);
}

InferOutputMetaRetType ViewOperator::InferOutputMeta(
    torch::jit::Stack& inputs) {
  auto self = inputs[0].toTensor();
  if (inputs[1].isIntList()) {
    auto dims = inputs[1].toIntVector();

    // Reshape Operator doesnt support -1 argument, remove it if present
    auto inferred_dims = habana_helpers::infer_size(dims, self.numel());
    // remove start_dim & end_dim. we have already used these to compute shape
    inputs.pop_back();
    // insert computed shape into inputs stack before calling reshape
    inputs.push_back(IValue(inferred_dims));
  }

  return ReshapeOperator::InferOutputMeta(inputs);
}

bool ViewOperator::STMeta(
    habana_helpers::IShapeList& inputs,
    habana_helpers::IShapeList& outputs) {
  std::vector<int64_t> out_shape = outputs[0].getTensorShape();
  static_cast<void>(inputs);
  PT_BRIDGE_DEBUG("ViewOperatorSTMeta output shape ", out_shape);
  habana_helpers::UpdateSTShapeInfo(out_shape);

  return true;
}

void ViewOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    Stack& inputs,
    const OutputMetaDataVector& output_metadata) {
  TORCH_CHECK(
      inputs.size() == 2 || inputs.size() == 3,
      "Incorrect size of input arguments for View Operator");
  TORCH_CHECK(
      inputs[0].isTensor(), "Input arg 1 for View op needs to be tensor type");
  TORCH_CHECK(
      inputs[1].isIntList() || inputs[1].isTensor(),
      "Input arg 2 for View op needs to be either Int List or Shape Tensor");

  auto self = inputs[0].toTensor();
  if (inputs[1].isIntList()) {
    auto dims = inputs[1].toIntVector();

    // Reshape Operator doesnt support -1 argument, remove it if present
    auto inferred_dims = habana_helpers::infer_size(dims, self.numel());
    // remove start_dim & end_dim. we have already used these to compute shape
    inputs.pop_back();
    // insert computed shape into inputs stack before calling reshape
    inputs.push_back(IValue(inferred_dims));
  } else {
    TORCH_CHECK(p_context_->syn_inputs_.back().ref().is_shape_tensor());
  }

  ReshapeOperator::AllocateAndAddSynapseNode(graph, inputs, output_metadata);
}

InferOutputMetaRetType BroadcastOperator::InferOutputMeta(
    torch::jit::Stack& inputs) {
  auto self = inputs[0].toTensor();

  std::vector<int64_t> expandedSizes;
  std::vector<int64_t> expandedStrides;
  InferOutputMetaRetType out;
  if (inputs[1].isIntList()) {
    auto size = inputs[1].toIntList();
    std::tie(expandedSizes, expandedStrides) = at::inferExpandGeometry(
        self.sizes(), self.strides(), IntArrayRef(size.vec()));

    habana_helpers::recalc_strides(expandedStrides, expandedSizes);
    out.AddShapeTensor(TensorMetaData(
        expandedSizes,
        expandedStrides,
        self.scalar_type(),
        self.suggest_memory_format()));
  } else {
    auto expand_shape = inputs[1].toTensor();
    expandedSizes = expand_shape.sizes().vec();
    expandedStrides = HabanaOperator::CalculateStrides(
        expandedSizes, self.suggest_memory_format());
  }

  out.AddOutputTensor(TensorMetaData(
      expandedSizes,
      expandedStrides,
      self.scalar_type(),
      self.suggest_memory_format()));
  return out;
}

void BroadcastOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    Stack& inputs,
    const OutputMetaDataVector& output_metadata) {
  TORCH_CHECK(
      inputs.size() == 3,
      "Incorrect size of input arguments for Broadcast Operator");
  TORCH_CHECK(
      inputs[1].isIntList() || inputs[1].isTensor(),
      "Input 1 can be either int list or shape tensor");
  auto self = inputs[0].toTensor();
  auto implicit = inputs[2].toBool();
  at::Tensor result;
  // [expand implicit]
  // The implicit flag is set to true for any expand calls inserted by broadcast
  // operators in ExpandUtils.h This flag is recorded by the tracer to
  // distinguish between expands inserted by broadcasts and those explicitly
  // requested by the user, because it is legal to remove implicit expands
  // from the graph, but not legal to remove the explicit ones.
  // implicit is not used in this kernel.

  if (inputs[1].isIntList()) {
    auto size = inputs[1].toIntList();
    auto sizeI = IntArrayRef(size.vec());
    TORCH_CHECK(
        sizeI.size() >= (size_t)self.dim(),
        "expand(",
        self.toString(),
        "{",
        self.sizes(),
        "}, size=",
        sizeI,
        "): the number of sizes provided (",
        sizeI.size(),
        ") ",
        "must be greater or equal to the number of dimensions in the tensor (",
        self.dim(),
        ")",
        "implicit = ",
        implicit);

    std::vector<int64_t> expandedSizes;
    std::vector<int64_t> expandedStrides;
    std::tie(expandedSizes, expandedStrides) = at::inferExpandGeometry(
        self.sizes(), self.strides(), IntArrayRef(size.vec()));

    // expandedStrides will be set to 0 by inferExpandGeometry.
    // Since we give back a contiguous tensor, we will set strides
    // to proper values.
    habana_helpers::recalc_strides(expandedStrides, expandedSizes);

    result = habana::createPTTensor(
        self,
        expandedSizes,
        expandedStrides,
        self.options(),
        self.suggest_memory_format(),
        output_metadata.at(0).persistent);
    // Allocate Shape tensor
    if (graph.is_dynamic_graph()) {
      AllocateSynapseShapeTensor(graph, result);
    }
  } else {
    TORCH_CHECK(p_context_->syn_inputs_.back().ref().is_shape_tensor());
    auto expand_shape = p_context_->syn_inputs_.back().ref().pt_shape();
    // This call is to check compatibility of shapes for broadcast and fail in
    // bridge if required (instead of failing at GC). Also required for
    // switching policy correctly in DS shape inference passes.
    at::inferExpandGeometry(
        self.sizes(), self.strides(), IntArrayRef(expand_shape));
    result = habana::createPTTensor(
        self,
        expand_shape,
        self.options(),
        self.suggest_memory_format(),
        output_metadata.at(0).persistent);
  }

  AllocateSynapseOutput(graph, result, output_metadata.at(0));
  AddNodeToSynapseGraph(graph, nullptr, 0);
}

static const auto& TensorShapeKernelsKernelRegistry =
    habana::KernelRegistry()
        .add("aten::permute", KERNEL_FN_GLOBAL(PermuteOperator))
        .add("hpu::permute_cl", KERNEL_FN_GLOBAL(PermuteCLOperator))
        .add("hpu::permute_weight", KERNEL_FN_GLOBAL(PermuteOperator))
        .add("hpu::permuted_weight_restride", KERNEL_FN_GLOBAL(PermuteOperator))
        .add("aten::t", KERNEL_FN_GLOBAL(TOperator))
        .add("aten::transpose.int", KERNEL_FN_GLOBAL(TransposeOperator))
        .add("aten::reshape", KERNEL_FN_GLOBAL(ReshapeOperator))
        .add("hpu::expand", KERNEL_FN_GLOBAL(BroadcastOperator))
        .add("hpu::expand_ds", KERNEL_FN_GLOBAL(BroadcastOperator))
        .add("aten::view", KERNEL_FN_GLOBAL(ViewOperator))
        .add("hpu::view", KERNEL_FN_GLOBAL(ViewOperator))
        .add("hpu::view_neg", KERNEL_FN_GLOBAL(ViewOperator))
        .add("hpu::reshape", KERNEL_FN_GLOBAL(ViewOperator))
        .add("aten::_unsafe_view", KERNEL_FN_GLOBAL(ViewOperator));
