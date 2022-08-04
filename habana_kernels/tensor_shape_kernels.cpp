/******************************************************************************
 * Copyright (C) 2020 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */
#include <ATen/ExpandUtils.h>
#include <ATen/InferSize.h>
#include <synapse_api.h>
#include <torch/script.h>

#include "habana_device/HPUCheck.h"
#include "habana_device/hpu_cached_devices.h"
#include "habana_helpers/graph.h"
#include "habana_helpers/logging.h"
#include "habana_helpers/tensor_utils.h"
#include "habana_kernels/index_kernels.h"
#include "habana_kernels/kernel_utils.h"
#include "habana_kernels/resize.h"
#include "habana_kernels/simple_generic_kernel.h"
#include "habana_kernels/tensor_shape_kernels.h"
#include "habana_lazy/hlexec.h"

using namespace torch;
using namespace habana;

void CatOutOperator::validate_tensor_dim_sizes(
    c10::List<at::Tensor> tensors,
    int64_t dim) {
  unsigned i = 0;
  // tensors[0] is out tensor
  auto tensor_count = tensors.size();
  auto tempT = tensors.get(0);
  for (i = 1; i < tensor_count; i++) {
    // check whether sizes along dimensions match except for cat dimension.
    unsigned j = 0;
    auto sz1 = tensors.get(i).sizes().vec();
    auto sz2 = tempT.sizes().vec();
    for (j = 0; j < tensors.get(i).dim(); j++) {
      if (j != dim) {
        if ((sz1[j] - sz2[j]) != 0)
          TORCH_CHECK(
              ((sz1[j] - sz2[j]) == 0),
              "Sizes of tensors along one of the non-cat dimensions don't match");
      }
    }
    tempT = tensors[i];
  }
}

Tensor CatOperator::CheckAllocateOutput(
    Stack& inputs,
    const OutputMetaData& output_metadata) {
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
  CatOutOperator::validate_tensor_dim_sizes(tensors, dim);
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

  auto out = habana_helpers::createPTTensor(
      first_tensor,
      out_size,
      first_tensor.options(),
      first_tensor.suggest_memory_format(),
      first_tensor.scalar_type(),
      output_metadata.persistent);

  return out;
}

void CatOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    Stack& inputs,
    const OutputMetaDataVector& output_metadata) {
  auto out = CheckAllocateOutput(inputs, output_metadata.at(0));
  inputs.emplace_back(out);
  auto dim = inputs[1].toInt();
  auto kernel_dim = (out.ndimension() - dim) - 1;

  p_context_->params_.emplace<int64_t>(kernel_dim);
  p_context_->params_size_ = sizeof(kernel_dim);
  AllocateSynapseOutput(graph, out, output_metadata.at(0));
  AddNodeToSynapseGraph(graph, &kernel_dim, sizeof(kernel_dim));
}

/*************************************************************************
 * @brief Kernel implementation for torch.cat(tensors, dim)
 * @param tensors - tensor list/tuple of inputs
 * @param dim - dimension along which to concatenate the tensors
 ************************************************************************/
Tensor cat_hpu(const TensorList in_tensors, int64_t dim_ = 0) {
  PT_KERNEL_BEGIN;

  size_t device_id = in_tensors[0].device().index();
  auto& device = synapse_helpers::HPURegistrar::get_device(device_id);
  at::ScalarType scalar_type = in_tensors[0].scalar_type();
  std::string node_type = "concat";
  std::vector<at::Tensor> pt_inputs;
  // Create operator
  at::ScalarType mod_scalar_type = in_tensors[0].scalar_type();

  // Assign Tensor Inputs to the Operator
  std::vector<c10::IValue> stack;
  std::vector<at::Tensor> tensors;
  for (unsigned i = 0; i < in_tensors.size(); i++) {
    if (in_tensors[i].scalar_type() == c10::ScalarType::Long) {
      tensors.push_back(habana_helpers::cast_tensor_to_integer(in_tensors[i]));
      pt_inputs.push_back(tensors[i]);
      mod_scalar_type = tensors[i].scalar_type();
    } else {
      tensors.push_back(in_tensors[i]);
      pt_inputs.push_back(in_tensors[i]);
    }
  }

  // Handle duplicate tensors. GC runtime expects each input to a Synapse graph
  // to be unique, therefore check if we have same tensor(s) given as input more
  // than once, replace duplicated tensor with its clone. Note this a eager mode
  // only solution where performance is not a concern, in graph mode this will
  // be handled as part of lowering of JIT graph to synapse graph.
  std::vector<void*> tensor_dptr;
  for (unsigned i = 0; i < tensors.size(); i++) {
    auto iter = std::find(
        tensor_dptr.begin(), tensor_dptr.end(), tensors[i].data_ptr());
    if (iter != tensor_dptr.end()) {
      auto clone = tensors[i].clone();
      tensors[i] = clone;
      pt_inputs[i] = clone;
    }
    tensor_dptr.push_back(tensors[i].data_ptr());
  }

  TensorList out_tensorlist(tensors);

  CatOperator Op(device_id, mod_scalar_type);
  // Push tensorlist as it is
  stack.push_back(IValue(out_tensorlist));
  stack.push_back(IValue(dim_));
  size_t key = Op.GetRecipeKey(node_type, stack);
  if (device.get_recipe_handle_cache().isCached(key)) {
    OutputMetaData md;
    md.persistent = true;
    auto out = Op.CheckAllocateOutput(stack, md);
    Op.Execute(key, pt_inputs, out);
  } else {
    // Build Params for the graph
    // AllocateAndAddSynapseNode() should be given all inputs in
    // same order as in the schema function signature
    OutputMetaDataVector output_metadata(1);
    output_metadata.at(0).persistent = true;
    // compile and execute the graph
    Op.CreateGraphAndCompile(key, pt_inputs, stack, output_metadata, true);
  }

  std::vector<at::Tensor> out = Op.GetOutputs();
  TORCH_CHECK(out.size() == 1, "Incorrect size of outputs");

  Tensor cast_out, temp;
  if (scalar_type == c10::ScalarType::Long) {
    cast_out = habana_helpers::cast_tensor_to_long(out.at(0));
  } else {
    cast_out = out.at(0);
  }
  PT_KERNEL_END;
  return cast_out;
}

int64_t CatOutOperator::CheckAllocateOutput(Stack& inputs) {
  TORCH_CHECK(
      inputs.size() == 3,
      "Incorrect size of inputs expected for catout operator");
  TORCH_CHECK(
      inputs[0].isTensorList(), "Input arg1 type expected to be tensor list");
  TORCH_CHECK(inputs[1].isInt(), "Input arg2 type expected to be int");
  TORCH_CHECK(inputs[2].isTensor(), "Input arg3 type expected to be tensor");

  auto tensors = inputs[0].toTensorList();
  auto dim_ = inputs[1].toInt();

  int64_t dim = at::maybe_wrap_dim(
      dim_,
      tensors.get(0).dim(),
      /*wrap_scalar=*/true);

  validate_tensor_dim_sizes(tensors, dim);

  return dim;
}

std::vector<int64_t> CatOutOperator::compute_output_shape(
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

OutputShapeInfRetType CatOutOperator::ComputeOutputShape(
    torch::jit::Stack& inputs) {
  auto tensors = inputs[0].toTensorVector();
  auto dim_ = inputs[1].toInt();

  // Convert "c10::List<at::Tensor>" to "at::TensorList"
  auto out_shape = CatOutOperator::compute_output_shape(tensors, dim_);

  auto metaData = TensorMetaData(
      out_shape,
      HabanaOperator::CalculateStrides(
          out_shape, tensors[0].suggest_memory_format()),
      tensors[0].scalar_type(),
      tensors[0].suggest_memory_format());
  OutputShapeInfRetType out_dup;
  out_dup.AddOutputTensor(metaData);
  out_dup.AddDupTensor(metaData);
  return out_dup;
}

void CatOutOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    Stack& inputs,
    const OutputMetaDataVector& output_metadata) {
  static_cast<void>(output_metadata);
  auto dim = CheckAllocateOutput(inputs);
  auto out = inputs[2].toTensor();
  auto kernel_dim = (out.ndimension() - dim) - 1;

  p_context_->params_.emplace<int64_t>(kernel_dim);
  p_context_->params_size_ = sizeof(kernel_dim);

  int64_t numTensors = p_context_->syn_inputs_.size();
  std::vector<synTensor> syn_inputs;

  for (int i = 0; i < (numTensors - 1); i++) {
    synapse_helpers::tensor& arg_syn_tensor = p_context_->syn_inputs_[i];
    syn_inputs.push_back(arg_syn_tensor.get());
  }

  p_context_->syn_outputs_.emplace_back(
      habana_helpers::duplicate_tensor_in_memory_section(
          p_context_->syn_inputs_[numTensors - 1],
          graph,
          output_metadata.at(0).external));
  p_context_->pt_outputs_.emplace_back(out);
  p_context_->syn_inputs_.erase(p_context_->syn_inputs_.cend());

  synapse_helpers::tensor& output_syn_tensor = p_context_->syn_outputs_[0];
  std::vector<synTensor> syn_outputs{output_syn_tensor.get()};

  graph.add_node(
      std::move(syn_inputs),
      std::move(syn_outputs),
      &kernel_dim,
      sizeof(kernel_dim),
      guid_);
}

void CatOutOperator::SetPTOutput(const Tensor& out) {
  HabanaOperator::SetPTOutput(out);
}

void CatOutOperator::SetPTOutput(torch::jit::Stack& inputs) {
  CheckAllocateOutput(inputs);
  auto out = inputs[2].toTensor();
  HabanaOperator::SetPTOutput(out);
}

/*************************************************************************
 * @brief Kernel implementation for torch.cat(tensors, dim, out=result)
 * @param result - result of concatenate
 * @param tensors - tensor list/tuple of inputs
 * @param dim - dimension along which to concatenate the tensors
 ************************************************************************/
Tensor& cat_hpu_out(
    Tensor& result,
    const TensorList tensors,
    int64_t dim_ = 0) {
  PT_KERNEL_BEGIN;
  size_t device_id = tensors[0].device().index();
  auto& device = synapse_helpers::HPURegistrar::get_device(device_id);
  at::ScalarType scalar_type = tensors[0].scalar_type();
  std::string node_type = "concat";

  // Create operator
  CatOutOperator Op(device_id, scalar_type);

  std::vector<at::Tensor> pt_inputs;
  std::vector<c10::IValue> stack;
  for (unsigned i = 0; i < tensors.size(); i++) {
    pt_inputs.push_back(tensors[i]);
  }
  // Tensorlist should be pushed as it is
  stack.push_back(IValue(tensors));
  stack.push_back(IValue(dim_));

  auto out_size = CatOutOperator::compute_output_shape(tensors, dim_);
  if (result.numel() == 0 && result.sizes().vec() != out_size) {
    auto tht_result = result.unsafeGetTensorImpl();
    THHTensor_resizeNd(tht_result, out_size.size(), out_size.data(), nullptr);
  } else if (result.sizes().vec() != out_size) {
    HABANA_ASSERT(
        false && "result size is not matching with expected output size");
  }
  pt_inputs.push_back(result);
  stack.push_back(IValue(result));

  /*Cache generation requires unique parameter distinctions which are not
   * guaranteed by tensors alone for ops like cat/cat.out because their guids
   * are same. cat and cat.out have same guid "concat". In habanaqa tests the
   * "cat" tests with 3 inputs generate same cache-key as "cat.out" test with 2
   * inputs + 1 out tensor. This causes cat.out to use same cached recipe as
   * cat. So, we set outOp=true for cache key generation.
   */
  size_t key =
      Op.GetRecipeKey(node_type, stack, /*inPlaceOp*/ false, /*outOp*/ true);
  if (device.get_recipe_handle_cache().isCached(key)) {
    Op.Execute(key, pt_inputs, stack);
  } else {
    // Build Params for the graph
    // AllocateAndAddSynapseNode() should be given all inputs in same order
    // as in the schema function signature
    OutputMetaDataVector output_metadata(1);
    output_metadata.at(0).persistent = true;
    // compile and execute the graph
    Op.CreateGraphAndCompile(key, pt_inputs, stack, output_metadata, true);
  }
  std::vector<at::Tensor> out = Op.GetOutputs();
  TORCH_CHECK(out.size() == 1, "Incorrect size of outputs");
  PT_KERNEL_END;
  return result;
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

OutputShapeInfRetType TransposeOperator::ComputeOutputShape(
    torch::jit::Stack& inputs) {
  Tensor self = inputs[0].toTensor();
  auto dim0_ = inputs[1].toInt();
  auto dim1_ = inputs[2].toInt();

  std::vector<int64_t> self_sizes, self_strides;
  std::tie(self_sizes, self_strides) =
      TransposeOperator::compute_output_shape(self, dim0_, dim1_);

  OutputShapeInfRetType out;
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
  auto out = habana_helpers::createPTTensor(
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

Tensor transpose_hpu(const Tensor& self, int64_t dim0_, int64_t dim1_) {
  PT_KERNEL_BEGIN;
  size_t device_id = self.device().index();
  auto& device = synapse_helpers::HPURegistrar::get_device(device_id);
  at::ScalarType scalar_type = self.scalar_type();
  std::string node_type =
      "transpose_fwd_" + habana_helpers::name_suffix_from_type(scalar_type);
  // Create operator
  TransposeOperator Op(device_id, scalar_type);
  // Build Params for the graph
  std::vector<c10::IValue> stack = {IValue(self), IValue(dim0_), IValue(dim1_)};
  std::vector<at::Tensor> pt_inputs{self};
  size_t key = Op.GetRecipeKey(node_type, stack);

  if (device.get_recipe_handle_cache().isCached(key)) {
    // handle negative dimensions (backward indexing) in pytorch
    int64_t dim0 = at::maybe_wrap_dim(dim0_, self.dim(), /*wrap_scalar=*/true);
    int64_t dim1 = at::maybe_wrap_dim(dim1_, self.dim(), /*wrap_scalar=*/true);
    auto self_sizes = self.sizes().vec();
    auto self_strides = self.strides().vec();
    std::swap(self_sizes[dim0], self_sizes[dim1]);
    // Recalculate the strides to account for transpose size changes
    // In effect, keep the tensor contiguous.
    habana_helpers::recalc_strides(self_strides, self_sizes);
    auto output = at::empty_strided(self_sizes, self_strides, self.options());
    Op.Execute(key, pt_inputs, output);
  } else {
    OutputMetaDataVector output_metadata(1);
    output_metadata.at(0).persistent = true;
    // compile and execute the graph
    Op.CreateGraphAndCompile(key, pt_inputs, stack, output_metadata, true);
  }
  std::vector<at::Tensor> out = Op.GetOutputs();
  TORCH_CHECK(out.size() == 1, "Incorrect size of outputs");
  PT_KERNEL_END;
  return out.at(0);
}

/*************************************************************************
 * @brief Kernel implementation for 2D torch.t(self,dim0,dim1)
 * @param self - input
 * @param dim0 - first dimension to swap
 * @param dim0 - second dimension to swap
 ************************************************************************/
Tensor t_hpu(const Tensor& self) { // t() is defined only for dims <= 2
  PT_KERNEL_BEGIN;
  if ((1 == self.dim())) {
    Tensor out = self;
    PT_KERNEL_END;
    return out;
  }
  auto ret = transpose_hpu(self, 0, 1);
  PT_KERNEL_END;
  return ret;
}

inline bool is_hpu_supported_transpose_type(const c10::ScalarType pt_type) {
  switch (pt_type) {
    case c10::ScalarType::Float:
    case c10::ScalarType::BFloat16:
    case c10::ScalarType::Int:
    case c10::ScalarType::Byte:
    case c10::ScalarType::Char:
    case c10::ScalarType::Short:
    case c10::ScalarType::Bool:
      return true;
    default:
      return false;
  }
}

/*************************************************************************
 * @brief Kernel implementation for torch.Tensor.permute(dims)
 * @param self - input on which permute needs to be applied
 * @param dims_ - permute dims array
 ************************************************************************/
PermuteOperator::PermuteOperator(int device_id, c10::ScalarType scalarType)
    : HabanaOperator("transpose") {
  static_cast<void>(scalarType);
  this->CreateSynContext(device_id);
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

OutputShapeInfRetType PermuteOperator::ComputeOutputShape(
    torch::jit::Stack& inputs) {
  Tensor self = inputs[0].toTensor();
  const auto dims = inputs[1].toIntVector();
  std::vector<int64_t> new_sizes, new_strides;
  std::tie(new_sizes, new_strides) =
      PermuteOperator::compute_output_shape(self, dims);

  OutputShapeInfRetType out;
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
      (self.dim() <= HABANA_DIM_MAX) &&
          is_hpu_supported_transpose_type(self.scalar_type()),
      "Unsupported permute operation on Habana device");

  std::vector<int64_t> new_sizes, new_strides;
  std::tie(new_sizes, new_strides) =
      PermuteOperator::compute_output_shape(self, dims);

  auto output = habana_helpers::createPTTensor(
      self,
      new_sizes,
      new_strides,
      self.options(),
      self.suggest_memory_format(),
      output_metadata.at(0).persistent);

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
  AllocateSynapseOutput(graph, output, output_metadata.at(0));
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

Tensor permute_hpu(const Tensor& self, IntArrayRef dims_) {
  PT_KERNEL_BEGIN;
  TORCH_CHECK(
      dims_.size() == static_cast<size_t>(self.dim()),
      "Number of dims in tensor don't match in permute");

  auto permute = [&] {
    size_t device_id = self.device().index();
    auto& device = synapse_helpers::HPURegistrar::get_device(device_id);
    at::ScalarType scalar_type = self.scalar_type();
    std::string node_type =
        "transpose_fwd_" + habana_helpers::name_suffix_from_type(scalar_type);
    // Create the operator
    PermuteOperator Op(device_id, scalar_type);
    // Build Params for the graph
    std::vector<c10::IValue> stack = {IValue(self), IValue(dims_)};
    std::vector<at::Tensor> pt_inputs{self};
    size_t key = Op.GetRecipeKey(node_type, stack);

    if (device.get_recipe_handle_cache().isCached(key)) {
      auto self_sizes = self.sizes().vec();
      // calculate new sizes and strides after permute for out tensor
      auto new_sizes = self.sizes().vec();
      auto new_strides = self.strides().vec();
      new_sizes[new_sizes.size() - 1] = self_sizes[dims_[new_sizes.size() - 1]];
      new_strides[new_sizes.size() - 1] = 1;
      for (int i = new_strides.size() - 2; i >= 0; i--) {
        new_sizes[i] = self_sizes[dims_[i]];
        new_strides[i] = new_strides[i + 1] * new_sizes[i + 1];
      }
      auto output = at::empty_strided(new_sizes, new_strides, self.options());
      Op.Execute(key, pt_inputs, output);
    } else {
      habana::OutputMetaDataVector output_metadata(1);
      output_metadata.at(0).persistent = true;
      // compile and execute the graph
      Op.CreateGraphAndCompile(key, pt_inputs, stack, output_metadata, true);
    }

    std::vector<at::Tensor> out = Op.GetOutputs();
    TORCH_CHECK(out.size() == 1, "Incorrect size of outputs");
    PT_KERNEL_END;
    return out.at(0);
  };

  if ((self.dim() <= 5) &&
      is_hpu_supported_transpose_type(self.scalar_type())) {
    return permute();
  }

  // HPU won't support permute for larger num of dims - do it on CPU
  auto ret =
      self.to(DeviceType::CPU).permute(dims_).contiguous().to(self.device());
  PT_KERNEL_END;
  return ret;
}

OutputShapeInfRetType ReshapeOperator::ComputeOutputShape(
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
  }

  auto memory_format = self.suggest_memory_format();
  if (inferred_size.size() < 4) {
    memory_format = at::MemoryFormat::Contiguous;
  }

  OutputShapeInfRetType out;
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
      inputs.size() == 2,
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

  auto output = habana_helpers::createPTTensor(
      self,
      inferred_size,
      self.options(),
      memory_format,
      output_metadata.at(0).persistent);

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

void FlattenOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    Stack& inputs,
    const OutputMetaDataVector& output_metadata) {
  TORCH_CHECK(
      inputs.size() == 3,
      "Incorrect size of input arguments for Flatten Operator");
  TORCH_CHECK(
      inputs[0].isTensor(),
      "Input arg 1 for Flatten op needs to be tensor type");
  TORCH_CHECK(
      inputs[1].isInt(), "Input arg 2 for Flatten op needs to be Int type");
  TORCH_CHECK(
      inputs[1].isInt(), "Input arg 3 for Flatten op needs to be Int type");

  auto self = inputs[0].toTensor();
  auto start_dim = inputs[1].toInt();
  auto end_dim = inputs[2].toInt();

  start_dim = at::maybe_wrap_dim(start_dim, self.dim());
  end_dim = at::maybe_wrap_dim(end_dim, self.dim());
  TORCH_CHECK(
      start_dim <= end_dim,
      "flatten() has invalid args: start_dim cannot come after end_dim");
  std::vector<int64_t> shape;

  if (self.dim() != 0) {
    // We don't want to infer_size on the entire shape, because that can give us
    // an extra degree of freedom we don't want; for example, consider shape [0,
    // 1, 3, 0], with start_dim=1, end_dim=2. It's clear we want result shape
    // [0, 3, 0] but passing [0, -1, 0] to infer_size means the -1 can take on
    // any value and satisfy the constraints.
    auto slice_numel = multiply_integers(
        self.sizes().slice(start_dim, end_dim - start_dim + 1));
    shape.reserve(self.dim() - end_dim + start_dim);
    for (int64_t i = 0; i < start_dim; i++) {
      shape.push_back(self.size(i));
    }
    shape.push_back(slice_numel);
    for (int64_t i = end_dim + 1; i < self.dim(); i++) {
      shape.push_back(self.size(i));
    }
  } else { // handle 0-d tensor
    shape.push_back(1);
  }

  // remove start_dim & end_dim. we have already used these to compute shape
  inputs.pop_back();
  inputs.pop_back();
  // insert computed shape into inputs stack before calling reshape
  inputs.push_back(IValue(shape));

  ReshapeOperator::AllocateAndAddSynapseNode(graph, inputs, output_metadata);
}

OutputShapeInfRetType ViewOperator::ComputeOutputShape(
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

  return ReshapeOperator::ComputeOutputShape(inputs);
}

void ViewOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    Stack& inputs,
    const OutputMetaDataVector& output_metadata) {
  TORCH_CHECK(
      inputs.size() == 2,
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

OutputShapeInfRetType BroadcastOperator::ComputeOutputShape(
    torch::jit::Stack& inputs) {
  auto self = inputs[0].toTensor();

  std::vector<int64_t> expandedSizes;
  std::vector<int64_t> expandedStrides;
  OutputShapeInfRetType out;
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

    result = habana_helpers::createPTTensor(
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
    result = habana_helpers::createPTTensor(
        self,
        expand_shape,
        self.options(),
        self.suggest_memory_format(),
        output_metadata.at(0).persistent);
  }

  AllocateSynapseOutput(graph, result, output_metadata.at(0));
  AddNodeToSynapseGraph(graph, nullptr, 0);
}

/*************************************************************************
 * @brief Kernel implementation for torch.Tensor.expand(*sizes)
 * @param self - input that needs to be expanded to a larger size.
 * @param dims_ - expanded dim sizes
 * NOTE: Tensor can be also expanded to a larger number of dimensions, and the
 * new ones will be appended at the front. For the new dimensions, the size
 * cannot be set to -1. We are using expand for braodcast op implementation
 * and we differ from the PyTorch expand that says "does not allocate new
 * memory, but only creates a new view on the existing tensor where a dimension
 * of size one is expanded to a larger size by setting the stride to 0. "
 ************************************************************************/
Tensor expand_hpu(const Tensor& in_self, IntArrayRef size, bool implicit) {
  PT_KERNEL_BEGIN;

  auto scalar_type = in_self.scalar_type();
  std::string node_type = "broadcast";

  size_t device_id = in_self.device().index();
  Tensor self;
  if (in_self.scalar_type() == c10::ScalarType::Long) {
    self = habana_helpers::cast_tensor_to_integer(in_self);
  } else {
    self = in_self;
  }
  BroadcastOperator Op(device_id, scalar_type);
  // Create Graph
  auto graph = habana_helpers::create_graph(device_id, node_type);

  // Convert index tensor from 0D to 1D if required
  if (self.dim() == 0) {
    self.unsafeGetTensorImpl()->set_sizes_and_strides({1}, {1});
  }

  // Return early for trivial case
  if (self.sizes().equals(size)) {
    PT_KERNEL_END;
    return self;
  }

  // Assign Inputs to the Operator
  std::vector<at::Tensor> pt_inputs{self};
  Op.AllocateSynapseInputs(graph, pt_inputs, true);

  // Build Params for the graph
  std::vector<c10::IValue> stack = {
      IValue(self), IValue(size), IValue(implicit)};
  OutputMetaDataVector output_metadata(1);
  output_metadata.at(0).persistent = true;
  Op.AllocateAndAddSynapseNode(graph, stack, output_metadata);

  // compile and execute the graph
  Op.Compile(graph);

  std::vector<at::Tensor> out = Op.GetOutputs();
  TORCH_CHECK(out.size() == 1, "Incorrect size of outputs");

  Tensor cast_out;
  if (in_self.scalar_type() == c10::ScalarType::Long) {
    cast_out = habana_helpers::cast_tensor_to_long(out.at(0));
  } else {
    cast_out = out.at(0);
  }
  PT_KERNEL_END;
  return cast_out;
}

void SplitWithSizeOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    Stack& inputs,
    const OutputMetaDataVector& output_metadata) {
  TORCH_CHECK(
      inputs.size() == 3,
      "Incorrect size of input arguments for SplitWithSizes Operator");
  auto self = inputs[0].toTensor();
  auto split_sizes = inputs[1].toIntList();
  HABANA_ASSERT(output_metadata.size() == split_sizes.size());
  auto dim = inputs[2].toInt();

  TORCH_CHECK(self.dim() != 0, "split expects at least a 1-dimensional tensor");
  int64_t dim_size = self.size(dim);
  int64_t num_splits = split_sizes.size();
  std::vector<Tensor> splits(num_splits);
  int64_t start_idx = 0;
  int64_t i;

  for (i = 0; i < num_splits; ++i) {
    auto length = split_sizes.get(i);
    TORCH_CHECK(
        length >= 0,
        "split_with_sizes expects split_sizes have only non-negative ",
        "entries, but got split_sizes=",
        split_sizes.vec());

    auto narrowOp = make_operator<NarrowOperator>(
        self.device().index(), self.scalar_type());
    narrowOp->SetSynapseInput(p_context_->syn_inputs_[0]);
    torch::jit::Stack stack = {
        IValue(self), IValue(dim), IValue(start_idx), IValue(length)};
    narrowOp->AllocateAndAddSynapseNode(graph, stack, {output_metadata.at(i)});
    p_context_->syn_outputs_.emplace_back(
        std::move(narrowOp->GetSynOutputs()[0]));
    p_context_->pt_outputs_.emplace_back(std::move(narrowOp->GetOutputs()[0]));

    start_idx += length;
  }

  TORCH_CHECK(
      start_idx == dim_size,
      "split_with_sizes expects split_sizes to sum exactly to ",
      dim_size,
      " (input tensor's size at dimension ",
      dim,
      "), ",
      "but got split_sizes=",
      split_sizes.vec());
}

std::vector<std::vector<int64_t>> SplitWithSizeOperator::compute_output_shape(
    const Tensor& self,
    IntArrayRef split_sizes,
    int64_t dim) {
  int64_t num_splits = split_sizes.size();
  int64_t start_idx = 0;
  int64_t i = 0;
  std::vector<std::vector<int64_t>> shapes;
  for (i = 0; i < num_splits; ++i) {
    auto length = split_sizes[i];
    auto end = start_idx + length;
    int64_t step = 1;

    auto size =
        SliceOperator::compute_output_shape(self, dim, start_idx, end, step);
    shapes.push_back(size);
    start_idx += length;
  }
  return shapes;
}
void SplitWithSizeOperator::SetPTOutputs(torch::jit::Stack& inputs) {
  auto self = inputs[0].toTensor();
  auto split_sizes = inputs[1].toIntList();
  auto dim = inputs[2].toInt();

  std::vector<std::vector<int64_t>> shapes =
      SplitWithSizeOperator::compute_output_shape(self, split_sizes.vec(), dim);
  int64_t i = 0;
  std::vector<Tensor> splits(split_sizes.size());
  for (const auto& shape : shapes) {
    splits[i++] = habana_helpers::createPTTensor(
        self, shape, self.options(), self.suggest_memory_format(), true);
  }

  HabanaOperator::SetPTOutputs(splits);
}

/**
 * @brief This function implements torch.split_with_size()
 * @param self - [fp32/bf16] Input tensor
 * @param split_sizes - [Int[]] List of sizes to be used for split along given
 * dim
 * @param dim - [Int] dim along which tensor is to be split
 */
std::vector<Tensor> split_with_sizes_hpu(
    const Tensor& self,
    IntArrayRef split_sizes,
    int64_t dim) {
  PT_KERNEL_BEGIN;

  at::ScalarType scalar_type = self.scalar_type();
  std::string node_type = "split_with_sizes";
  size_t device_id = self.device().index();
  auto& device = synapse_helpers::HPURegistrar::get_device(device_id);

  SplitWithSizeOperator Op(device_id, scalar_type);
  std::vector<at::Tensor> pt_inputs{self};
  std::vector<c10::IValue> stack = {
      IValue(self), IValue(split_sizes), IValue(dim)};

  size_t key = Op.GetRecipeKey(node_type, stack);
  if (device.get_recipe_handle_cache().isCached(key)) {
    Op.Execute(key, pt_inputs, stack);
  } else {
    habana::OutputMetaDataVector output_metadata(split_sizes.size());
    for (auto& md : output_metadata) {
      md.persistent = true;
    }
    Op.CreateGraphAndCompile(key, pt_inputs, stack, output_metadata, true);
  }
  std::vector<Tensor> out = Op.GetOutputs();

  PT_KERNEL_END;
  return out;
}

static auto& KernelRegistry =
    habana::KernelRegistry()
        .add("aten::cat", KERNEL_FN_GLOBAL(CatOperator))
        .add("hpu::cat", KERNEL_FN_GLOBAL(CatOperator))
        .add("aten::cat.out", KERNEL_FN_GLOBAL(CatOutOperator))
        .add("aten::permute", KERNEL_FN_GLOBAL(PermuteOperator))
        .add("hpu::permute", KERNEL_FN_GLOBAL(PermuteOperator))
        .add("hpu::permute_cl", KERNEL_FN_GLOBAL(PermuteCLOperator))
        .add("hpu::permute_weight", KERNEL_FN_GLOBAL(PermuteOperator))
        .add("hpu::permuted_weight_restride", KERNEL_FN_GLOBAL(PermuteOperator))
        .add("aten::t", KERNEL_FN_GLOBAL(TOperator))
        .add("aten::transpose.int", KERNEL_FN_GLOBAL(TransposeOperator))
        .add("aten::reshape", KERNEL_FN_GLOBAL(ReshapeOperator))
        .add("aten::flatten", KERNEL_FN_GLOBAL(FlattenOperator))
        .add("aten::expand", KERNEL_FN_GLOBAL(BroadcastOperator))
        .add("hpu::expand", KERNEL_FN_GLOBAL(BroadcastOperator))
        .add("aten::view", KERNEL_FN_GLOBAL(ViewOperator))
        .add("hpu::view", KERNEL_FN_GLOBAL(ViewOperator))
        .add("hpu::reshape", KERNEL_FN_GLOBAL(ViewOperator))
        .add("aten::split_with_sizes", KERNEL_FN_GLOBAL(SplitWithSizeOperator));
