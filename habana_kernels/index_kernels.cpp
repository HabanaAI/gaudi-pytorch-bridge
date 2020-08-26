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
#include <ATen/WrapDimUtils.h>
#include <perf_lib_layer_params.h>
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

/*************************************************************************
 * @brief This helper function makes the size of index tensor to be same as
 * value tensor, with broadcast of indices (within index tensor)
 ************************************************************************/
static Tensor make_index_same_size_as_value(
    const Tensor& index,
    const Tensor& value,
    int64_t dim) {
  // We use "View" + "Broadcast" so that index tensor becomes same shape as
  // value tensor with indices repeated in right pattern. This 2-step approach
  // is required because "Scatter" TPC kernel does not support Broadcast for
  // index tensor.

  // Expand 1D index tensor to same number of dimensions as value tensor
  auto expanded_sizes = std::vector<int64_t>(value.ndimension(), 1);
  expanded_sizes[dim] = index.sizes()[0];
  auto index_expanded = index.view(expanded_sizes);

  // Broadcast index tensor to same shape as value tensor
  auto index_broadcast = at::empty(DimVector(value.sizes()), index.options());
  std::vector<at::Tensor> pt_inputs{index_expanded};
  std::vector<at::Tensor> pt_outputs{index_broadcast};
  synapse_simple_generic_kernel(
      pt_outputs, pt_inputs, "broadcast", nullptr, 0, SynapsePassType::NO_PASS);

  return index_broadcast;
}

Tensor GatherOperator::AllocateOutput(torch::jit::Stack& inputs) {
  auto self = inputs[0].toTensor();
  auto dim_ = inputs[1].toInt();
  auto index = inputs[2].toTensor();

  auto dim = at::maybe_wrap_dim(dim_, self.dim(), /*wrap_scalar=*/true);

  auto shape = DimVector(self.sizes());
  shape.erase(shape.begin() + dim);
  shape.insert(shape.begin() + dim, index.numel());
  auto output = at::empty(shape, self.options(), self.suggest_memory_format());
  return output;
}

void GatherOperator::SetPTOutputs(torch::jit::Stack& inputs) {
  auto output = AllocateOutput(inputs);
  HabanaOperator::SetPTOutput(output);
}

void GatherOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    Stack& inputs,
    bool is_output_persistent) {
  TORCH_CHECK(
      inputs.size() == 4,
      "Incorrect size of input expected for Gather operator");
  TORCH_CHECK(
      inputs[0].isTensor(),
      "Input type expected to be Tensor for Gather operator");
  TORCH_CHECK(
      inputs[1].isInt(), "Input type expected to be Int for Gather operator");
  TORCH_CHECK(
      inputs[2].isTensor(),
      "Input type expected to be Tensor for Gather operator");
  TORCH_CHECK(
      inputs[3].isBool(), "Input type expected to be Bool for Gather operator");

  auto self = inputs[0].toTensor();
  auto dim_ = inputs[1].toInt();
  auto index = inputs[2].toTensor();
  auto sparse_grad = inputs[3].toBool();

  TORCH_CHECK(sparse_grad == false, "spare_grad is not supported")
  if (index.dim() == 0) {
    index.unsafeGetTensorImpl()->set_sizes_and_strides({1}, {1});
  }

  auto dim = at::maybe_wrap_dim(dim_, self.dim(), /*wrap_scalar=*/true);

  auto output = AllocateOutput(inputs);

  ns_GatherKernel::Params params;
  params.axis = self.dim() - dim - 1;

  p_context_->params_.emplace<ns_GatherKernel::Params>(params);
  p_context_->params_size_ = sizeof(params);

  AllocateSynapseOutput(graph, output, is_output_persistent);
  AddNodeToSynapseGraph(graph, &params, sizeof(params));
}

/*************************************************************************
 * @brief Kernel implementation for torch.gather
 * @param self - Input tensor 1-4D bf16/fp32
 * @param dim - dimension along which to index
 * @param index - Tensor used to index into self
 * @param sparse_grad - Boolean to indicate if sparse grad is supported
 ************************************************************************/
Tensor gather_src_hpu(
    const Tensor& self,
    int64_t dim_,
    const Tensor& index,
    bool sparse_grad) {
  PT_KERNEL_BEGIN;

  auto index_int = habana_helpers::cast_tensor_to_integer(index);

  size_t device_id = self.device().index();
  auto& device = synapse_helpers::HPURegistrar::get_device(device_id);
  at::ScalarType scalar_type = self.scalar_type();
  std::string node_type =
      "gather_fwd_" + habana_helpers::name_suffix_from_type(scalar_type);

  // create the operator
  GatherOperator Op(device_id, scalar_type);
  std::vector<c10::IValue> stack = {
      IValue(self), IValue(dim_), IValue(index_int), IValue(sparse_grad)};
  size_t key = Op.GetRecipeKey(node_type, stack);
  /*
    std::vector<at::Tensor> pt_inputs{self, index_int};
    std::vector<at::Tensor> pt_outputs{output};*/
  // Assign Inputs to the Operator
  // std::vector<const at::Tensor*> pt_inputs{&self, &index_int};
  std::vector<at::Tensor> pt_inputs{self, index_int};

  if (device.get_recipe_handle_cache().isCached(key)) {
    PT_KERNEL_DEBUG("Cache hit key:", key);
    Op.SetPTInputs(pt_inputs);
    Op.SetPTOutputs(stack);
    Op.Execute(key);
  } else {
    PT_KERNEL_DEBUG("Key:", key);
    // create graph
    auto graph = habana_helpers::create_graph(device_id, node_type);
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

/*************************************************************************
 * @brief Kernel implementation for scatter_.src(Tensor(a!) self, int dim,
 *Tensor index, Tensor src) -> Tensor(a!)
 * @param self - Input tensor 1-4D bf16/fp32
 * @param dim - dimension along which to index
 * @param index - Tensor used to index into self
 * @param src -Tensor with values to be updated (of same type as self)
 ************************************************************************/
Tensor& scatter_inplace_src_hpu(
    Tensor& self,
    int64_t dim_,
    const Tensor& index,
    const Tensor& src) {
  PT_KERNEL_BEGIN;
  auto dim = at::maybe_wrap_dim(dim_, self.dim(), /*wrap_scalar=*/true);

  ns_ScatterKernel::Params params;
  params.axis = self.dim() - dim - 1;

  std::vector<at::Tensor> pt_inputs{self, index, src};

  synapse_simple_generic_inplace_kernel(
      pt_inputs,
      "scatter",
      &params,
      sizeof(params),
      SynapsePassType::FORWARD_PASS);

  PT_KERNEL_END;
  return self;
}

/*************************************************************************
 * @brief Kernel implementation for index_add(dim, index, tensor) → Tensor
 * @param self - Input tensor 1-4D bf16/fp32
 * @param dim - dimension along which to index
 * @param indices - Tensor used to index into self
 * @param source -Tensor with values to be updated (of same type as self)
 ************************************************************************/
Tensor& index_add_hpu_(
    Tensor& self,
    int64_t dim_,
    const Tensor& indices,
    const Tensor& source) {
  PT_KERNEL_BEGIN;

  TORCH_CHECK(indices.dim() <= 1, "index tensor cannot be more than 1D")
  // Convert index tensor from 0D to 1D if required
  if (indices.dim() == 0) {
    indices.unsafeGetTensorImpl()->set_sizes_and_strides({1}, {1});
  }

  auto dim = at::maybe_wrap_dim(dim_, self.dim(), /*wrap_scalar=*/true);

  auto value_acc = source;
  auto slice = at::index_select(self, dim, indices);
  value_acc += slice;

  auto index_int = habana_helpers::cast_tensor_to_integer(indices);

  auto index_broadcast =
      make_index_same_size_as_value(index_int, value_acc, dim);
  self = scatter_inplace_src_hpu(self, dim, index_broadcast, value_acc);

  PT_KERNEL_END;
  return self;
}

/*************************************************************************
 * @brief Kernel implementation for index_put(indices, value, accumulate=False)
 *→ Tensor
 * @param self - Input tensor 1-4D bf16/fp32
 * @param indices - Tensors used to index into self
 * @param value - Tensor with values to be updated (of same type as self)
 * @param accumulate - Flag to indicate whether to accumulate into self
 * @param unsafe -
 ************************************************************************/
Tensor& index_put_impl_hpu_(
    Tensor& self,
    TensorList indices,
    const Tensor& value,
    bool accumulate,
    bool unsafe) {
  PT_KERNEL_BEGIN;

  TORCH_CHECK(unsafe == false, "Unsafe not supported in index_put");
  TORCH_CHECK(indices[0].dim() <= 1, "index tensor cannot be more than 1D")
  // Convert index tensor from 0D to 1D if required
  if (indices[0].dim() == 0) {
    indices[0].unsafeGetTensorImpl()->set_sizes_and_strides({1}, {1});
  }

  auto value_acc = value;
  if (accumulate) {
    auto slice = at::index_select(self, 0, indices[0]);
    value_acc += slice;
  }

  auto index_int = habana_helpers::cast_tensor_to_integer(indices[0]);

  // Insertion of updates is always along dim=0 for this operator
  int64_t dim = 0;
  auto index_broadcast =
      make_index_same_size_as_value(index_int, value_acc, dim);
  self = scatter_inplace_src_hpu(self, dim, index_broadcast, value_acc);

  PT_KERNEL_END;
  return self;
}

void IndexSelectOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    Stack& inputs,
    bool is_output_persistent) {
  TORCH_CHECK(
      inputs.size() == 3,
      "Incorrect size of input expected for IndexSelect operator");
  TORCH_CHECK(
      inputs[0].isTensor(),
      "Input type expected to be Tensor for IndexSelect operator");
  TORCH_CHECK(
      inputs[1].isInt(),
      "Input type expected to be Int for IndexSelect operator");
  TORCH_CHECK(
      inputs[2].isTensor(),
      "Input type expected to be Tensor for IndexSelect operator");

  auto index = inputs[2].toTensor();
  TORCH_CHECK(index.dim() <= 1, "index tensor cannot be more than 1D")
  bool sparse_grad = false;
  inputs.emplace_back(IValue(sparse_grad));
  GatherOperator::AllocateAndAddSynapseNode(
      graph, inputs, is_output_persistent);
}

void IndexSelectOperator::SetPTOutputs(torch::jit::Stack& inputs) {
  auto index = inputs[2].toTensor();
  TORCH_CHECK(index.dim() <= 1, "index tensor cannot be more than 1D")
  bool sparse_grad = false;
  inputs.emplace_back(IValue(sparse_grad));
  GatherOperator::SetPTOutputs(inputs);
}
/*************************************************************************
 * @brief Kernel implementation for torch.index_select(input, dim, index) →
 *Tensor
 * @param self - Input tensor 1-4D bf16/fp32
 * @param dim - The dimension in which we index
 * @param index - 1D tensor containing the indices to index
 ************************************************************************/
Tensor index_select_hpu(const Tensor& self, int64_t dim, const Tensor& index) {
  PT_KERNEL_BEGIN;

  auto index_int = habana_helpers::cast_tensor_to_integer(index);

  size_t device_id = self.device().index();
  auto& device = synapse_helpers::HPURegistrar::get_device(device_id);
  at::ScalarType scalar_type = self.scalar_type();
  std::string node_type =
      "index_select_fwd_" + habana_helpers::name_suffix_from_type(scalar_type);

  // create the operator
  IndexSelectOperator Op(device_id, scalar_type);
  std::vector<c10::IValue> stack = {IValue(self), IValue(dim), IValue(index)};
  size_t key = Op.GetRecipeKey(node_type, stack);

  // Assign Inputs to the Operator
  std::vector<at::Tensor> pt_inputs{self, index_int};

  if (device.get_recipe_handle_cache().isCached(key)) {
    PT_KERNEL_DEBUG("Cache hit key:", key);
    Op.SetPTInputs(pt_inputs);
    Op.SetPTOutputs(stack);
    Op.Execute(key);
  } else {
    PT_KERNEL_DEBUG("Key:", key);
    // create graph
    auto graph = habana_helpers::create_graph(device_id, node_type);
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

void Gather2dOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    bool is_output_persistent) {
  TORCH_CHECK(
      inputs.size() == 3,
      "Incorrect size of inputs expected for gather2d operator");
  TORCH_CHECK(inputs[0].isTensor(), "Input arg1 type expected to be tensor");
  TORCH_CHECK(inputs[1].isTensor(), "Input arg2 type expected to be tensor");
  TORCH_CHECK(inputs[2].isInt(), "Input arg4 type expected to be integer");

  auto input = inputs[0].toTensor();
  auto indices = inputs[1].toTensor();
  auto validCount = inputs[2].toInt();

  TORCH_CHECK(indices.dim() <= 1, "index tensor cannot be more than 1D")

  TORCH_CHECK(
      indices.numel() >= validCount,
      "validCount cannot be greater than number of indices provided")
  TORCH_CHECK(input.dim() == 2, "Input tensor should be 2D")

  auto shape = DimVector(input.sizes());
  shape.erase(shape.begin() + 0);
  shape.insert(shape.begin() + 0, std::min(indices.numel(), validCount));
  auto output = habana_helpers::createPTTensor(
      input,
      shape,
      input.options(),
      input.suggest_memory_format(),
      is_output_persistent);
  AllocateSynapseOutput(graph, output, is_output_persistent);
  AddNodeToSynapseGraph(graph, nullptr, 0);
}

/*************************************************************************
 * @brief Kernel implementation for gather2d custom OP
 * @param self - Input tensor 2D fp32
 * @param indices - 1D tensor containing the indices to index
 * @param validCount - number of valid indices in indices tensor
 ************************************************************************/
Tensor gather2d_hpu(
    const Tensor& input,
    const Tensor& indices,
    int64_t validCount) {
  PT_KERNEL_BEGIN;

  // Convert index tensor from 0D to 1D if required
  if (indices.dim() == 0) {
    indices.unsafeGetTensorImpl()->set_sizes_and_strides({1}, {1});
  }

  auto indices_int = habana_helpers::cast_tensor_to_integer(indices);

  // This conversion from scalar to tensor not done within
  // AllocateAndAddSynapseNode because graph mode does not have
  // support for DMA handling.
  auto validCount_int = at::empty({1}, indices_int.options());
  validCount_int.fill_(static_cast<int32_t>(validCount));

  at::ScalarType scalar_type = input.scalar_type();
  std::string node_type = "gather_with_valid_count_2d_" +
      habana_helpers::name_suffix_from_type(scalar_type);

  size_t device_id = input.device().index();

  Gather2dOperator Op(device_id, node_type);
  // Create Graph
  auto graph = habana_helpers::create_graph(device_id, node_type);

  // Assign Inputs to the Operator
  std::vector<at::Tensor> pt_inputs{input, indices_int, validCount_int};
  Op.AllocateSynapseInputs(graph, pt_inputs, true);

  // Build Params for the graph
  std::vector<c10::IValue> stack = {
      IValue(input), IValue(indices_int), IValue(validCount)};
  Op.AllocateAndAddSynapseNode(graph, stack, true);

  // compile and execute the graph
  Op.Compile(graph);

  std::vector<at::Tensor> out = Op.GetOutputs();
  TORCH_CHECK(out.size() == 1, "Incorrect size of outputs");

  PT_KERNEL_END;
  return out.at(0);
}

Tensor SliceOperator::AllocateOutputTensor(
    const Tensor& self,
    int64_t& dim,
    int64_t& start,
    int64_t& end,
    int64_t& step,
    bool is_output_persistent) {
  // convert dim to positive value if required
  dim = at::maybe_wrap_dim(dim, self.dim(), /*wrap_scalar=*/true);
  auto sizes = self.sizes().vec();
  if (start < 0) {
    start += sizes[dim];
  }
  if (end < 0) {
    end += sizes[dim];
  }
  if (start < 0) {
    start = 0;
  } else if (start >= sizes[dim]) {
    start = sizes[dim];
  }
  if (end < start) {
    end = start;
  } else if (end >= sizes[dim]) {
    end = sizes[dim];
  }

  // compute output shape
  auto len = 0;
  for (auto i = start; i < end; i += step) {
    len++;
  }
  auto shape = DimVector(self.sizes());
  shape.erase(shape.begin() + dim);
  shape.insert(shape.begin() + dim, len);

  // allocate output tensor
  auto output = habana_helpers::createPTTensor(
      self,
      shape,
      self.options(),
      self.suggest_memory_format(),
      is_output_persistent);

  return output;
}

void SliceOperator::SetPTOutputs(const torch::jit::Stack& inputs) {
  auto self = inputs[0].toTensor();
  auto dim = inputs[1].toInt();
  auto start = inputs[2].toInt();
  auto end = inputs[3].toInt();
  auto step = inputs[4].toInt();
  auto output = AllocateOutputTensor(self, dim, start, end, step, true);
  HabanaOperator::SetPTOutputs({output});
}

void SliceOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    bool is_output_persistent) {
  TORCH_CHECK(
      inputs.size() == 5,
      "Incorrect size of inputs expected for slice operator");
  TORCH_CHECK(inputs[0].isTensor(), "Input arg1 type expected to be tensor");
  TORCH_CHECK(inputs[1].isInt(), "Input arg2 type expected to be integer");
  TORCH_CHECK(inputs[2].isInt(), "Input arg3 type expected to be integer");
  TORCH_CHECK(inputs[3].isInt(), "Input arg4 type expected to be integer");
  TORCH_CHECK(inputs[4].isInt(), "Input arg5 type expected to be integer");

  auto self = inputs[0].toTensor();
  auto dim = inputs[1].toInt();
  auto start = inputs[2].toInt();
  auto end = inputs[3].toInt();
  auto step = inputs[4].toInt();

  if (dim == self.dim() - 1) {
    // check required due to GC limitation. Strided slice is possible on FCD
    // only if there is another dimension with size 1 in the tensor.
    TORCH_CHECK(step <= 1, "strided slice not supported on FCD");
  }

  auto output =
      AllocateOutputTensor(self, dim, start, end, step, is_output_persistent);
  std::vector<const at::Tensor*> pt_outputs{&output};

  synSliceParams params;
  // set defaults
  std::fill_n(params.axes, MAX_DIMENSIONS_NUM, 0);
  std::fill_n(params.starts, MAX_DIMENSIONS_NUM, 0);
  std::fill_n(params.ends, MAX_DIMENSIONS_NUM, 0);
  std::fill_n(params.steps, MAX_DIMENSIONS_NUM, 1);
  // slice triggered only on 1 dim, therefore use only index 0
  params.axes[0] = self.dim() - dim - 1;
  params.starts[0] = start;
  params.ends[0] = end;
  params.steps[0] = step;

  AllocateSynapseOutput(graph, output, is_output_persistent);
  AddNodeToSynapseGraph(graph, &params, sizeof(params));
}

/*************************************************************************
 * @brief Kernel implementation for torch slice operator
 * @param self - Input tensor
 * @param dim - Axis to slice
 * @param start - index of first element in given axis
 * @param end - index of last element in given axis
 * @param steps - number of elements to stride in given axis
 ************************************************************************/
Tensor slice_hpu(
    const Tensor& self,
    int64_t dim,
    int64_t start,
    int64_t end,
    int64_t step) {
  PT_KERNEL_BEGIN;

  // for handling trivial cases, fall-back to simple tensor meta-data
  // manipulation done in CPU implementation. This was added because
  // distributed MNIST stops working if run synapse version of slice
  // which creates new storage for output storage
  if ((self.dim() <= 1) && (step == 1)) {
    PT_KERNEL_END;
    return at::native::slice(self, dim, start, end, step);
  }

  at::ScalarType scalar_type = self.scalar_type();

  std::string node_type = "slice";
  size_t device_id = self.device().index();
  auto& device = synapse_helpers::HPURegistrar::get_device(device_id);

  SliceOperator Op(device_id, scalar_type);
  std::vector<at::Tensor> pt_inputs{self};
  std::vector<c10::IValue> stack = {
      IValue(self), IValue(dim), IValue(start), IValue(end), IValue(step)};

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

  std::vector<at::Tensor> out = Op.GetOutputs();
  TORCH_CHECK(out.size() == 1, "Incorrect size of outputs");

  PT_KERNEL_END;
  return out.at(0);
}

void SelectOperator::SetPTOutputs(const torch::jit::Stack& inputs) {
  auto self = inputs[0].toTensor();
  auto dim = inputs[1].toInt();
  auto index = inputs[2].toInt();
  auto start = index;
  auto end = index + 1;
  int64_t step = 1;
  SliceOperator slice_op(self.device().index(), self.scalar_type());
  auto slice_output =
      slice_op.AllocateOutputTensor(self, dim, start, end, step, false);

  // case for select op where tensor dimension is reduced
  // only rank 4 tensor can have channels last format
  at::MemoryFormat memory_format = at::MemoryFormat::Contiguous;

  // allocate output tensor
  auto shape = slice_output.sizes().vec();
  shape.erase(shape.begin() + dim);
  auto output = habana_helpers::createPTTensor(
      self, shape, self.options(), memory_format, true);

  HabanaOperator::SetPTOutputs({output});
}

void SelectOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    bool is_output_persistent) {
  HABANA_ASSERT(inputs.size() == 3);
  HABANA_ASSERT(inputs[0].isTensor());
  HABANA_ASSERT(inputs[1].isInt());
  HABANA_ASSERT(inputs[2].isInt());

  auto self = inputs[0].toTensor();
  auto dim = inputs[1].toInt();
  auto index = inputs[2].toInt();

  auto start = index;
  auto end = index + 1;
  int64_t step = 1;

  SliceOperator slice_op(self.device().index(), self.scalar_type());
  auto& syn_in_slice =
      slice_op.SetSynapseInput(std::move(p_context_->syn_inputs_[0]));
  std::vector<c10::IValue> stack1 = {
      IValue(self), IValue(dim), IValue(start), IValue(end), IValue(step)};
  slice_op.AllocateAndAddSynapseNode(graph, stack1, false);
  p_context_->syn_inputs_[0] = std::move(syn_in_slice);

  // Add Reshape node to graph
  ReshapeOperator reshape_op(self.device().index(), self.scalar_type());
  UNUSED auto& syn_in_reshape =
      reshape_op.SetSynapseInput(std::move(slice_op.GetSynOutputs()[0]));
  auto slice_out_tensor = slice_op.GetOutputs()[0];
  auto shape = slice_out_tensor.sizes().vec();
  shape.erase(shape.begin() + dim);
  torch::jit::Stack stack2 = {c10::IValue(slice_out_tensor),
                              c10::IValue(shape)};
  reshape_op.AllocateAndAddSynapseNode(graph, stack2, is_output_persistent);

  p_context_->syn_outputs_.emplace_back(
      std::move(reshape_op.GetSynOutputs()[0]));
  p_context_->pt_outputs_.emplace_back(std::move(reshape_op.GetOutputs()[0]));
}

/*************************************************************************
 * @brief Kernel implementation for torch select operator
 * @param self - Input tensor
 * @param dim - Axis to slice
 * @param index - index the element in given axis
 ************************************************************************/

Tensor select_hpu(const Tensor& self, int64_t dim, int64_t index) {
  PT_KERNEL_BEGIN;

  at::ScalarType scalar_type = self.scalar_type();
  HABANA_ASSERT(
      ((scalar_type == c10::ScalarType::Int) ||
       (scalar_type == c10::ScalarType::Float) ||
       (scalar_type == c10::ScalarType::BFloat16)));

  std::string node_type = "slice";
  size_t device_id = self.device().index();
  auto& device = synapse_helpers::HPURegistrar::get_device(device_id);

  SelectOperator Op(device_id, scalar_type);
  std::vector<at::Tensor> pt_inputs{self};
  std::vector<c10::IValue> stack = {IValue(self), IValue(dim), IValue(index)};

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

  std::vector<at::Tensor> out = Op.GetOutputs();
  HABANA_ASSERT(out.size() == 1);

  PT_KERNEL_END;
  return out.at(0);
}

Tensor ArangeOperator::AllocateOutput(torch::jit::Stack& inputs) {
  auto result = inputs[0].toTensor();
  auto start = inputs[1].toDouble();
  auto end = inputs[2].toDouble();
  auto step = inputs[3].toDouble();

  TORCH_CHECK(step != 0, "step value can not be 0.");
  TORCH_CHECK(!((start > end) && (step > 0)), "step must be negative.");
  TORCH_CHECK(!((start < end) && (step < 0)), "step must be positive.");

  float max, min, abs_del;
  int depth;
  max = start > end ? start : end;
  min = start > end ? end : start;
  abs_del = std::abs(step);
  depth = std::ceil((max - min) / abs_del);
  depth = depth == 0 ? 1 : depth;
  auto shape = DimVector({depth});
  auto tht_result = result.unsafeGetTensorImpl();
  THHTensor_resizeNd(tht_result, shape.size(), shape.data(), nullptr);
  return result;
}

void ArangeOperator::SetPTOutputs(torch::jit::Stack& inputs) {
  auto result = AllocateOutput(inputs);
  HabanaOperator::SetPTOutput(result);
}

void ArangeOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    bool is_output_persistent) {
  TORCH_CHECK(
      inputs.size() == 4,
      "Incorrect size of inputs expected for Arange operator");
  TORCH_CHECK(
      inputs[0].isTensor(),
      "Input arg1 expected to be tensor for Arange operator");
  TORCH_CHECK(
      inputs[1].isDouble(),
      "Input arg2 expected to be Double for Arange operator");
  TORCH_CHECK(
      inputs[2].isDouble(),
      "Input arg3 expected to be Double for Arange operator");
  TORCH_CHECK(
      inputs[3].isDouble(),
      "Input arg4 expected to be Double for Arange operator");

  auto start = inputs[1].toDouble();
  auto end = inputs[2].toDouble();
  auto step = inputs[3].toDouble();

  ns_RangeKernel::Params param;
  param.start.f = static_cast<float>(start);
  param.limit.f = static_cast<float>(end);
  param.delta.f = static_cast<float>(step);

  auto result = AllocateOutput(inputs);
  AllocateSynapseOutput(graph, result, is_output_persistent);
  AddNodeToSynapseGraph(graph, &param, sizeof(param));
}

Tensor& arange_hpu(Tensor& output, Scalar start, Scalar end, Scalar step) {
  PT_KERNEL_BEGIN;

  at::ScalarType scalar_type = output.scalar_type();
  std::string node_type =
      "range_" + habana_helpers::name_suffix_from_type(scalar_type);

  size_t device_id = output.device().index();
  auto& device = synapse_helpers::HPURegistrar::get_device(device_id);

  ArangeOperator Op(device_id, scalar_type);

  // Build Params for the graph
  std::vector<c10::IValue> stack = {IValue(output),
                                    IValue(start.to<double>()),
                                    IValue(end.to<double>()),
                                    IValue(step.to<double>())};
  size_t key = Op.GetRecipeKey(node_type, stack);

  if (device.get_recipe_handle_cache().isCached(key)) {
    PT_KERNEL_DEBUG("Cache hit key:", key);
    Op.SetPTOutputs(stack);
    Op.Execute(key);
  } else {
    PT_KERNEL_DEBUG("Key:", key);
    // Create Graph
    auto graph = habana_helpers::create_graph(device_id, node_type);
    Op.AllocateAndAddSynapseNode(graph, stack, true);
    // compile and execute the graph
    Op.Compile(graph);
  }

  std::vector<at::Tensor> out = Op.GetOutputs();
  TORCH_CHECK(out.size() == 1, "Incorrect size of outputs");

  PT_KERNEL_END;
  return out.at(0);
}

static auto& KernelRegistry =
    ::habana::KernelRegistry()
        .add(
            "aten::index_select",
            [](const int device_id, c10::ScalarType node_type) {
              return std::make_shared<IndexSelectOperator>(
                  device_id, node_type);
            })
        .add(
            "aten::gather",
            [](const int device_id, c10::ScalarType node_type) {
              return std::make_shared<GatherOperator>(device_id, node_type);
            })
        .add(
            "aten::select",
            [](const int device_id, c10::ScalarType node_type) {
              return std::make_shared<SelectOperator>(device_id, node_type);
            })
        .add(
            "aten::arange.start_out",
            [](const int device_id, c10::ScalarType node_type) {
              return std::make_shared<ArangeOperator>(device_id, node_type);
            });

static auto registry =
    torch::RegisterOperators()
        .op(torch::RegisterOperators::options()
                .schema(
                    "aten::index_select(Tensor self, int dim, Tensor index) -> Tensor")
                .impl_unboxedOnlyKernel<
                    decltype(index_select_hpu),
                    &index_select_hpu>(DispatchKey::HABANATensorId)
                .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
        .op(torch::RegisterOperators::options()
                .schema(
                    "aten::_index_put_impl_(Tensor(a!) self, Tensor?[] indices, Tensor values, bool accumulate=False, bool unsafe=False) -> Tensor(a!)")
                .impl_unboxedOnlyKernel<
                    decltype(index_put_impl_hpu_),
                    &index_put_impl_hpu_>(DispatchKey::HABANATensorId)
                .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
        .op(torch::RegisterOperators::options()
                .schema(
                    "aten::index_add_(Tensor(a!) self, int dim, Tensor index, Tensor source) -> Tensor(a!)")
                .impl_unboxedOnlyKernel<
                    decltype(index_add_hpu_),
                    &index_add_hpu_>(DispatchKey::HABANATensorId)
                .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
        .op(torch::RegisterOperators::options()
                .schema(
                    "aten::scatter_.src(Tensor(a!) self, int dim, Tensor index, Tensor src) -> Tensor(a!)")
                .impl_unboxedOnlyKernel<
                    decltype(scatter_inplace_src_hpu),
                    &scatter_inplace_src_hpu>(DispatchKey::HABANATensorId)
                .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
        .op(torch::RegisterOperators::options()
                .schema(
                    "aten::gather(Tensor self, int dim, Tensor index, *, bool sparse_grad=False) -> Tensor")
                .impl_unboxedOnlyKernel<
                    decltype(gather_src_hpu),
                    &gather_src_hpu>(DispatchKey::HABANATensorId)
                .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
        .op(torch::RegisterOperators::options()
                .schema(
                    "aten::slice.Tensor(Tensor(a) self, int dim=0, int start=0, int end=9223372036854775807, int step=1) -> Tensor(a)")
                .impl_unboxedOnlyKernel<decltype(slice_hpu), &slice_hpu>(
                    DispatchKey::HABANATensorId)
                .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
        .op(torch::RegisterOperators::options()
                .schema(
                    "aten::select.int(Tensor(a) self, int dim, int index) -> Tensor(a)")
                .impl_unboxedOnlyKernel<decltype(select_hpu), &select_hpu>(
                    DispatchKey::HABANATensorId)
                .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
        .op(torch::RegisterOperators::options()
                .schema(
                    "aten::arange.start_out(Scalar start, Scalar end, Scalar step=1, *, Tensor(a!) out) -> Tensor(a!)")
                .impl_unboxedOnlyKernel<decltype(arange_hpu), &arange_hpu>(
                    DispatchKey::HABANATensorId)
                .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA));
