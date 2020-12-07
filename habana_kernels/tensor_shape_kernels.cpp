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

using namespace torch;

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

Tensor CatOperator::CheckAllocateOutput(Stack& inputs) {
  TORCH_CHECK(
      inputs.size() == 2,
      "Incorrect size of inputs expected for matmul operator");

  TORCH_CHECK(
      inputs[0].isTensorList(), "Input arg2 type expected to be tensor list");
  TORCH_CHECK(inputs[1].isInt(), "Input arg3 type expected to be int");

  auto tensors = inputs[0].toTensorList();
  auto dim_ = inputs[1].toInt();
  auto tensor_count = tensors.size();

  auto first_tensor = tensors.get(0);
  int64_t dim =
      at::maybe_wrap_dim(dim_, first_tensor.dim(), /*wrap_scalar=*/true);
  TORCH_CHECK(
      dim < first_tensor.ndimension(),
      "Cat dimension specified exceeds tensors dimensions");

  // out tensor size should match along all dimensions for input tensors except
  // along the dim in which to cat
  auto out_size = first_tensor.sizes().vec();
  out_size[dim] = 0;
  for (unsigned i = 0; i < tensor_count; i++) {
    out_size[dim] += tensors.get(i).sizes()[dim];
  }
  return std::move(at::empty(
      out_size, first_tensor.options(), first_tensor.suggest_memory_format()));
}

void CatOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    Stack& inputs,
    bool is_output_persistent) {
  auto out = CheckAllocateOutput(inputs);
  inputs.insert(inputs.begin(), IValue(out));
  // inputs pos : 0 = out, 1,2,3,... = cat inputs, "tensor_count"th elem = dim
  CatOutOperator::AllocateAndAddSynapseNode(
      graph, inputs, is_output_persistent);
}

void CatOperator::SetPTOutput(torch::jit::Stack& inputs) {
  auto out = CheckAllocateOutput(inputs);
  inputs.insert(inputs.begin(), IValue(out));
  HabanaOperator::SetPTOutput(out);
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

  TensorList out_tensorlist(tensors);

  CatOperator Op(device_id, mod_scalar_type);
  // Push tensorlist as it is
  stack.push_back(IValue(out_tensorlist));
  stack.push_back(IValue(dim_));
  size_t key = Op.GetRecipeKey(node_type, stack);
  if (device.get_recipe_handle_cache().isCached(key)) {
    PT_KERNEL_DEBUG("Cache hit key:", key);
    Op.SetPTInputs(pt_inputs);
    Op.SetPTOutput(stack);
    Op.Execute(key);
  } else {
    PT_KERNEL_DEBUG("Key:", key);
    // Create Graph
    auto graph = habana_helpers::create_graph(device_id, node_type);
    Op.AllocateSynapseInputs(graph, pt_inputs, true);
    // Build Params for the graph
    // AllocateAndAddSynapseNode() should be given all inputs in
    // same order as in the schema function signature
    Op.AllocateAndAddSynapseNode(graph, stack, true);

    // compile and execute the graph
    Op.Compile(graph);
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
      "Incorrect size of inputs expected for matmul operator");

  TORCH_CHECK(inputs[0].isTensor(), "Input arg1 type expected to be tensor");
  TORCH_CHECK(
      inputs[1].isTensorList(), "Input arg2 type expected to be tensor list");
  TORCH_CHECK(inputs[2].isInt(), "Input arg3 type expected to be int");

  auto out = inputs[0].toTensor();
  auto tensors = inputs[1].toTensorList();
  auto dim_ = inputs[2].toInt();

  int64_t dim = at::maybe_wrap_dim(
      dim_,
      tensors.get(0).dim(),
      /*wrap_scalar=*/true);

  auto in_tensor_count = tensors.size(); // num input tensors

  auto first_tensor = tensors.get(0);
  auto out_size = first_tensor.sizes().vec();
  out_size[dim] = 0;
  for (unsigned i = 0; i < in_tensor_count; i++) {
    out_size[dim] += tensors.get(i).sizes()[dim];
  }
  validate_tensor_dim_sizes(tensors, dim);

  if (out.defined()) {
    TORCH_CHECK(
        first_tensor.options().type_equal(out.options()),
        "output values must be of same type as input");
    auto tht_result = out.unsafeGetTensorImpl();
    THHTensor_resizeNd(
        tht_result, first_tensor.dim(), out_size.data(), nullptr);
  } else {
    out = at::empty(
        out_size, first_tensor.options(), first_tensor.suggest_memory_format());
  }

  // insert allocated output tensor back
  inputs.erase(inputs.cbegin());
  inputs.emplace(inputs.cbegin(), out);

  return dim;
}

void CatOutOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    Stack& inputs,
    bool is_output_persistent) {
  auto dim = CheckAllocateOutput(inputs);
  auto out = inputs[0].toTensor();
  auto kernel_dim = (out.ndimension() - dim) - 1;
  p_context_->params_.emplace<int64_t>(kernel_dim);
  p_context_->params_size_ = sizeof(kernel_dim);
  AllocateSynapseOutput(graph, out, is_output_persistent);
  AddNodeToSynapseGraph(graph, &kernel_dim, sizeof(kernel_dim));
}

void CatOutOperator::SetPTOutput(torch::jit::Stack& inputs) {
  CheckAllocateOutput(inputs);
  auto out = inputs[0].toTensor();
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
  stack.push_back(IValue(result));
  for (unsigned i = 0; i < tensors.size(); i++) {
    pt_inputs.push_back(tensors[i]);
  }
  // Tensorlist should be pushed as it is
  stack.push_back(IValue(tensors));
  stack.push_back(IValue(dim_));

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
    PT_KERNEL_DEBUG("Cache hit key:", key);
    Op.SetPTInputs(pt_inputs);
    Op.SetPTOutput(stack);
    Op.Execute(key);
  } else {
    PT_KERNEL_DEBUG("Key:", key);
    // Create Graph
    auto graph = habana_helpers::create_graph(device_id, node_type);
    Op.AllocateSynapseInputs(graph, pt_inputs, true);
    // Build Params for the graph
    // AllocateAndAddSynapseNode() should be given all inputs in same order
    // as in the schema function signature
    Op.AllocateAndAddSynapseNode(graph, stack, true);
    // compile and execute the graph
    Op.Compile(graph);
  }
  std::vector<at::Tensor> out = Op.GetOutputs();
  TORCH_CHECK(out.size() == 1, "Incorrect size of outputs");
  PT_KERNEL_END;
  return out.at(0);
}

inline void recalc_strides(
    std::vector<int64_t>& self_strides,
    const std::vector<int64_t>& self_sizes) {
  int k;
  self_strides[self_strides.size() - 1] = 1;
  for (k = self_strides.size() - 2; k >= 0; k--) {
    self_strides[k] = self_strides[k + 1] * self_sizes[k + 1];
  }
  return;
}

/****************************************************************************
 * @brief Kernel implementation for N-D out = torch.transpose(self,dim0,dim1)
 * @param self - input
 * @param dim0 - first dimension to swap
 * @param dim0 - second dimension to swap
 ***************************************************************************/
TransposeOperator::TransposeOperator(int device_id, c10::ScalarType scalarType)
    : HabanaOperator("transpose") {
  this->CreateSynContext(device_id);
}

void TransposeOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    Stack& inputs,
    bool is_output_persistent) {
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
  // handle negative dimensions (backward indexing) in pytorch
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
  recalc_strides(self_strides, self_sizes);
  auto out = habana_helpers::createPTTensor(
      self,
      self_sizes,
      self_strides,
      self.options(),
      self.suggest_memory_format(),
      is_output_persistent);
  synTransposeParams params;
  params.tensorDim = self.dim();
  int i;
  for (i = 0; i < MAX_DIMENSIONS_NUM; i++) {
    params.permutation[i] = static_cast<TransposePermutationDim>(i);
  }
  std::swap(
      params.permutation[self.dim() - 1 - dim0],
      params.permutation[self.dim() - 1 - dim1]);

  p_context_->params_.emplace<synTransposeParams>(params);
  p_context_->params_size_ = sizeof(params);
  AllocateSynapseOutput(graph, out, is_output_persistent);
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
    PT_KERNEL_DEBUG("Cache hit key:", key);
    // handle negative dimensions (backward indexing) in pytorch
    int64_t dim0 = at::maybe_wrap_dim(dim0_, self.dim(), /*wrap_scalar=*/true);
    int64_t dim1 = at::maybe_wrap_dim(dim1_, self.dim(), /*wrap_scalar=*/true);
    auto self_sizes = self.sizes().vec();
    auto self_strides = self.strides().vec();
    std::swap(self_sizes[dim0], self_sizes[dim1]);
    // Recalculate the strides to account for transpose size changes
    // In effect, keep the tensor contiguous.
    recalc_strides(self_strides, self_sizes);
    auto output = at::empty_strided(self_sizes, self_strides, self.options());
    Op.SetPTInputs(pt_inputs);
    Op.SetPTOutput(output);
    Op.Execute(key);
  } else {
    PT_KERNEL_DEBUG("key:", key);
    //
    // Create Graph
    auto graph = habana_helpers::create_graph(device_id, node_type);

    // Assign Inputs to the Operator
    Op.AllocateSynapseInputs(graph, pt_inputs, true);

    Op.AllocateAndAddSynapseNode(graph, stack, true);

    // compile and execute the graph
    Op.Compile(graph);
  }
  std::vector<at::Tensor> out = Op.GetOutputs();
  TORCH_CHECK(out.size() == 1, "Incorrect size of outputs");
  PT_KERNEL_END;
  return out.at(0);
}

/*******************************************************************************
 * @brief Kernel implementation for N-D inplace torch.transpose_(self,dim0,dim1)
 * @param self - input as well as output
 * @param dim0 - first dimension to swap
 * @param dim0 - second dimension to swap
 *******************************************************************************/
Tensor& transpose_hpu_(Tensor& self, int64_t dim0_, int64_t dim1_) {
  PT_KERNEL_BEGIN;
  /*NOTE: The normal inplace op implementation approach to through a duplicate
   * synapse tensor for input won't work as synapse backend does block
   * transposes - so if your matrix is AB CD then C will overwrite B before B is
   * written or vice-versa. So, we do the following (inefficient way, but helps
   * to support the functionality). tempTensor = transpose_outofplace(inTensor)
   * Reshape inTensor to transposed sizes for required dims.
   * Use synapse memcpy guid to do a transfer data from tempTensor to inTensor
   * Return inTensor back to PyTorch frontend
   */
  auto tempT = transpose_hpu(self, dim0_, dim1_);

  std::vector<at::Tensor> pt_inputs;
  std::vector<at::Tensor> pt_outputs;
  pt_inputs.push_back(tempT);

  // handle negative dimensions (backward indexing) in pytorch
  int64_t dim0 = at::maybe_wrap_dim(dim0_, self.dim(), /*wrap_scalar=*/true);
  int64_t dim1 = at::maybe_wrap_dim(dim1_, self.dim(), /*wrap_scalar=*/true);

  auto self_sizes = self.sizes().vec();
  auto self_strides = self.strides().vec();
  std::swap(self_sizes[dim0], self_sizes[dim1]);
  // Recalculate the strides to account for transpose size changes
  // In effect, keep the tensor contiguous.
  recalc_strides(self_strides, self_sizes);
  auto tht_result = self.unsafeGetTensorImpl();
  THHTensor_resizeNd(
      tht_result, self.dim(), self_sizes.data(), self_strides.data());
  pt_outputs.push_back(self);

  synapse_simple_generic_kernel(
      pt_outputs, pt_inputs, "memcpy", nullptr, 0, SynapsePassType::NO_PASS);

  PT_KERNEL_END;
  return self;
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

/*************************************************************************
 * @brief Kernel implementation for 2D inplace torch.t_(self,dim0,dim1)
 * @param self - input as well as output
 * @param dim0 - first dimension to swap
 * @param dim0 - second dimension to swap
 ************************************************************************/
Tensor& t_hpu_(Tensor& self) { // t_() is defined only for dims <= 2
  PT_KERNEL_BEGIN;
  if (1 == self.dim()) {
    PT_KERNEL_END;
    return self;
  }
  self = transpose_hpu_(self, 0, 1);
  PT_KERNEL_END;

  return self;
}

inline bool is_hpu_supported_transpose_type(const c10::ScalarType pt_type) {
  switch (pt_type) {
    case c10::ScalarType::Float:
    case c10::ScalarType::BFloat16:
    case c10::ScalarType::Int:
    case c10::ScalarType::Byte:
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
  this->CreateSynContext(device_id);
}

void PermuteOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    Stack& inputs,
    bool is_output_persistent) {
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
  const auto dims = inputs[1].toIntList();

  TORCH_CHECK(
      dims.size() == static_cast<size_t>(self.dim()),
      "Number of dims in tensor don't match in permute");
  TORCH_CHECK(
      (self.dim() <= 4) && is_hpu_supported_transpose_type(self.scalar_type()),
      "Unsupported permute operation on Habana device");

  auto self_sizes = self.sizes().vec();
  // calculate new sizes and strides after permute for out tensor
  auto new_sizes = self.sizes().vec();
  auto new_strides = self.strides().vec();
  new_sizes[new_sizes.size() - 1] = self_sizes[dims[new_sizes.size() - 1]];
  new_strides[new_sizes.size() - 1] = 1;
  for (int i = new_sizes.size() - 2; i >= 0; i--) {
    new_sizes[i] = self_sizes[dims[i]];
    new_strides[i] = new_strides[i + 1] * new_sizes[i + 1];
  }

  auto output = habana_helpers::createPTTensor(
      self,
      new_sizes,
      new_strides,
      self.options(),
      self.suggest_memory_format(),
      is_output_persistent);
  synTransposeParams params;
  params.tensorDim = self.dim();
  // params.permute has to be populated in a reverse order for HPU FCD-LCD order
  for (int i = 0; i < self.dim(); i++) {
    params.permutation[i] = static_cast<TransposePermutationDim>(
        self.dim() - dims[dims.size() - i - 1] - 1);
  }
  for (int i = self.dim(); i < MAX_DIMENSIONS_NUM; i++) {
    params.permutation[i] = static_cast<TransposePermutationDim>(i);
  }

  p_context_->params_.emplace<synTransposeParams>(params);
  p_context_->params_size_ = sizeof(params);
  AllocateSynapseOutput(graph, output, is_output_persistent);
  AddNodeToSynapseGraph(graph, &params, sizeof(params));
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
      PT_KERNEL_DEBUG("Cache hit key:", key);
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
      Op.SetPTInputs(pt_inputs);
      Op.SetPTOutput(output);
      Op.Execute(key);
    } else {
      PT_KERNEL_DEBUG("key:", key);

      //
      // Create Graph
      auto graph = habana_helpers::create_graph(device_id, node_type);

      // Assign Inputs to the Operator
      Op.AllocateSynapseInputs(graph, pt_inputs, true);

      Op.AllocateAndAddSynapseNode(graph, stack, true /*is_output_persistent*/);

      // compile and execute the graph
      Op.Compile(graph);
    }

    std::vector<at::Tensor> out = Op.GetOutputs();
    TORCH_CHECK(out.size() == 1, "Incorrect size of outputs");
    PT_KERNEL_END;
    return out.at(0);
  };

  if ((self.dim() <= 4) &&
      is_hpu_supported_transpose_type(self.scalar_type())) {
    return permute();
  }

  // HPU won't support permute for larger num of dims - do it on CPU
  auto ret =
      self.to(DeviceType::CPU).permute(dims_).contiguous().to(self.device());
  PT_KERNEL_END;
  return ret;
}

/*************************************************************************
 * @brief Kernel implementation for torch.Tensor.reshape
 * @param self - input on which reshape needs to be applied
 * @param shape - reshape  shape array
 ************************************************************************/
void ReshapeOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    Stack& inputs,
    bool is_output_persistent) {
  TORCH_CHECK(
      inputs.size() == 2,
      "Incorrect size of input arguments for Reshape Operator");
  Tensor self = inputs[0].toTensor();
  TORCH_CHECK(
      self.is_contiguous(),
      "Right now Reshape is only supported for contiguous Tensor.");

  auto shape = inputs[1].toIntList();
  auto shape_vector = shape.vec();
  auto input_shape = IntArrayRef(shape_vector.data(), shape_vector.size());
  auto inferred_size = at::infer_size(input_shape, self.numel());
  auto output = habana_helpers::createPTTensor(
      self,
      inferred_size,
      self.options(),
      self.suggest_memory_format(),
      is_output_persistent);
  TORCH_CHECK(
      self.numel() == output.numel(),
      "Reshape doesnt support change in number of elements: ",
      self.sizes(),
      " Size of output: ",
      output.sizes());
  p_context_->params_size_ = 0;
  AllocateSynapseOutput(graph, output, is_output_persistent);
  AddNodeToSynapseGraph(graph, NULL, 0);
}

void FlattenOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    Stack& inputs,
    bool is_output_persistent) {
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
    auto slice_numel =
        prod_intlist(self.sizes().slice(start_dim, end_dim - start_dim + 1));
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

  ReshapeOperator::AllocateAndAddSynapseNode(
      graph, inputs, is_output_persistent);
}

void ViewOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    Stack& inputs,
    bool is_output_persistent) {
  TORCH_CHECK(
      inputs.size() == 2,
      "Incorrect size of input arguments for View Operator");
  TORCH_CHECK(
      inputs[0].isTensor(), "Input arg 1 for View op needs to be tensor type");
  TORCH_CHECK(
      inputs[1].isIntList(), "Input arg 2 for View op needs to be Int List");

  auto self = inputs[0].toTensor();
  auto dims = inputs[1].toIntVector();
  ;

  // Reshape Operator doesnt support -1 argument, remove it if present
  auto inferred_dims = at::infer_size(dims, self.numel());
  // remove start_dim & end_dim. we have already used these to compute shape
  inputs.pop_back();
  // insert computed shape into inputs stack before calling reshape
  inputs.push_back(IValue(inferred_dims));

  ReshapeOperator::AllocateAndAddSynapseNode(
      graph, inputs, is_output_persistent);
}

void BroadcastOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    Stack& inputs,
    bool is_output_persistent) {
  TORCH_CHECK(
      inputs.size() == 3,
      "Incorrect size of input arguments for Broadcast Operator");
  auto self = inputs[0].toTensor();
  auto size = inputs[1].toIntList();
  auto implicit = inputs[2].toBool();

  // [expand implicit]
  // The implicit flag is set to true for any expand calls inserted by broadcast
  // operators in ExpandUtils.h This flag is recorded by the tracer to
  // distinguish between expands inserted by broadcasts and those explicitly
  // requested by the user, because it is legal to remove implicit expands
  // from the graph, but not legal to remove the explicit ones.
  // implicit is not used in this kernel.
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
  recalc_strides(expandedStrides, expandedSizes);
  Tensor result;
  // remove if part causing issue, if broadcast is used as intermediate node
  // let gc handle the optimizatin if sizes equal
  {
    result = habana_helpers::createPTTensor(
        self,
        expandedSizes,
        expandedStrides,
        self.options(),
        self.suggest_memory_format(),
        is_output_persistent);
    auto expanded_self_view_sizes =
        std::vector<int64_t>(expandedSizes.size(), 1);
    for (unsigned i = 0; i < self.dim(); i++) {
      expanded_self_view_sizes[expandedSizes.size() - self.dim() + i] =
          self.sizes()[i];
    }

    // Add Reshape node to graph
    ReshapeOperator reshape_op(self.device().index(), self.scalar_type());
    auto& syn_in =
        reshape_op.SetSynapseInput(std::move(p_context_->syn_inputs_[0]));
    torch::jit::Stack stack = {
        c10::IValue(self), c10::IValue(expanded_self_view_sizes)};
    reshape_op.AllocateAndAddSynapseNode(graph, stack, false);
    p_context_->syn_inputs_[0] = std::move(syn_in);

    // Add broadcast node to graph
    AllocateSynapseOutput(graph, result, is_output_persistent);
    synapse_helpers::tensor& syn_in_broadcast = reshape_op.GetSynOutputs()[0];
    std::vector<synTensor> syn_inputs{syn_in_broadcast.get()};
    synapse_helpers::tensor& syn_out_broadcast = p_context_->syn_outputs_[0];
    std::vector<synTensor> syn_outputs{syn_out_broadcast.get()};
    std::string guid_ = "broadcast";
    graph.add_node(
        std::move(syn_inputs),
        std::move(syn_outputs),
        nullptr,
        0,
        std::move(guid_));
  }
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
  Op.AllocateAndAddSynapseNode(graph, stack, true);

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
    bool is_output_persistent) {
  TORCH_CHECK(
      inputs.size() == 3,
      "Incorrect size of input arguments for SplitWithSizes Operator");
  auto self = inputs[0].toTensor();
  auto split_sizes = inputs[1].toIntList();
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

    NarrowOperator narrowOp(self.device().index(), self.scalar_type());
    auto& syn_in =
        narrowOp.SetSynapseInput(std::move(p_context_->syn_inputs_[0]));
    torch::jit::Stack stack = {
        IValue(self), IValue(dim), IValue(start_idx), IValue(length)};
    narrowOp.AllocateAndAddSynapseNode(graph, stack, is_output_persistent);
    p_context_->syn_inputs_[0] = std::move(syn_in);
    p_context_->syn_outputs_.emplace_back(
        std::move(narrowOp.GetSynOutputs()[0]));
    p_context_->pt_outputs_.emplace_back(std::move(narrowOp.GetOutputs()[0]));

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

void SplitWithSizeOperator::SetPTOutputs(torch::jit::Stack& inputs) {
  auto self = inputs[0].toTensor();
  auto split_sizes = inputs[1].toIntList();
  auto dim = inputs[2].toInt();

  int64_t num_splits = split_sizes.size();
  std::vector<Tensor> splits(num_splits);
  int64_t start_idx = 0;
  int64_t i = 0;

  for (i = 0; i < num_splits; ++i) {
    auto length = split_sizes.get(i);
    auto end = start_idx + length;
    int64_t step = 1;

    SliceOperator slice_op(self.device().index(), self.scalar_type());
    splits[i] =
        slice_op.AllocateOutputTensor(self, dim, start_idx, end, step, true);

    start_idx += length;
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
    PT_KERNEL_DEBUG("Cache hit key:", key);
    Op.SetPTInputs(pt_inputs);
    Op.SetPTOutputs(stack);
    Op.Execute(key);
  } else {
    PT_KERNEL_DEBUG("key:", key);
    auto graph = habana_helpers::create_graph(device_id, node_type);
    Op.AllocateSynapseInputs(graph, pt_inputs, true);
    Op.AllocateAndAddSynapseNode(graph, stack, true);
    Op.Compile(graph);
  }
  std::vector<Tensor> out = Op.GetOutputs();

  PT_KERNEL_END;
  return out;
}

static auto& KernelRegistry =
    ::habana::KernelRegistry()
        .add(
            "aten::cat",
            [](const int device_id, c10::ScalarType node_type) {
              return std::make_shared<CatOperator>(device_id, node_type);
            })
        .add(
            "aten::permute",
            [](const int device_id, c10::ScalarType node_type) {
              return std::make_shared<PermuteOperator>(device_id, node_type);
            })
        .add(
            "aten::t",
            [](const int device_id, c10::ScalarType node_type) {
              return std::make_shared<TOperator>(device_id, node_type);
            })
        .add(
            "aten::transpose",
            [](const int device_id, c10::ScalarType node_type) {
              return std::make_shared<TransposeOperator>(device_id, node_type);
            })
        .add(
            "aten::reshape",
            [](const int device_id, c10::ScalarType node_type) {
              return std::make_shared<ReshapeOperator>(device_id, node_type);
            })
        .add(
            "aten::flatten",
            [](const int device_id, c10::ScalarType node_type) {
              return std::make_shared<FlattenOperator>(device_id, node_type);
            })
        .add(
            "aten::expand",
            [](const int device_id, c10::ScalarType node_type) {
              return std::make_shared<BroadcastOperator>(device_id, node_type);
            })
        .add("aten::view", [](const int device_id, c10::ScalarType node_type) {
          return std::make_shared<ViewOperator>(device_id, node_type);
        });
