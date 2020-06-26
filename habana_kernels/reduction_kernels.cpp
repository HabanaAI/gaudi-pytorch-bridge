/******************************************************************************
 * Copyright (C) 2020 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */
// #include <ATen/native/TensorIterator.h> // TODO: fix this include
#include <bitset>

#include <perf_lib_layer_params.h>
#include <torch/script.h>

#include "habana_device/HPUCheck.h"
#include "habana_device/hpu_cached_devices.h"
#include "habana_helpers/graph.h"
#include "habana_helpers/tensor_utils.h"
#include "habana_kernels/kernel_utils.h"
#include "habana_kernels/reduction_kernels.h"
#include "habana_kernels/resize.h"
#include "habana_kernels/simple_generic_kernel.h"
#include "habana_kernels/kernel_utils.h"

using namespace torch;
// TODO: DimMask = TensorIterator::DimMask
using DimMask = std::bitset<64>;

// Copy paste from PT
namespace {
inline int64_t maybe_wrap_dim(
    int64_t dim,
    int64_t dim_post_expr,
    bool wrap_scalar = true) {
  return c10::maybe_wrap_dim(dim, dim_post_expr, wrap_scalar);
}

DimMask make_dim_mask(IntArrayRef dims, int64_t ndim) {
  auto mask = DimMask();
  if (dims.empty()) {
    mask.flip();
  } else {
    for (int64_t dim : dims) {
      mask.set(maybe_wrap_dim(dim, ndim));
    }
  }
  return mask;
}

void allocate_reduction_result(
    Tensor& result,
    const Tensor& self,
    DimMask mask,
    bool keepdim,
    ScalarType dtype) {
  auto shape = DimVector(self.sizes());
  for (int dim = shape.size() - 1; dim >= 0; dim--) {
    if (mask[dim]) {
      if (keepdim) {
        shape[dim] = 1;
      } else {
        shape.erase(shape.begin() + dim);
      }
    }
  }

  // Following code is required to convert Pytorch 0d tensor
  // to a 1d tensor. This is required because synapse_helpers
  // tensor_builder does not support 0d tensors
  if (shape.size() == 0) {
    shape.push_back(1);
  }

  if (result.defined()) {
    auto tht_result = result.unsafeGetTensorImpl();
    THHTensor_resizeNd(tht_result, shape.size(), shape.data(), nullptr);
    // result.resize_(shape);
  } else {
    result = at::empty(shape, self.options().dtype(dtype));
  }
}

ScalarType get_dtype(
    Tensor& result,
    const Tensor& self,
    optional<ScalarType> dtype,
    bool promote_integers = false) {
  if (dtype.has_value()) {
    return dtype.value();

  } else if (result.defined()) {
    return result.scalar_type();
  }
  ScalarType src_type = self.scalar_type();
  if (promote_integers && at::isIntegralType(src_type, /*includeBool=*/true)) {
    return kLong;
  }
  return src_type;
}

/*Tensor review_reduce_result(
    const Tensor& result,
    int ndim,
    DimMask mask,
    bool keepdim) {
  if (keepdim) {
    return result;
  }
  auto shape = DimVector(result.sizes());
  auto stride = DimVector(result.strides());
  for (int dim = 0; dim < ndim; dim++) {
    if (mask[dim]) {
      shape.insert(shape.begin() + dim, 1);
      stride.insert(stride.begin() + dim, 0);
    }
  }
  return result.as_strided(shape, stride);
}*/
} // namespace

void ReduceOperator::SetPTOutputs(torch::jit::Stack& inputs) {
  Tensor output = inputs[0].toTensor();
  Tensor self = inputs[1].toTensor();
  auto dim = inputs[2].toIntList();
  bool keepdim = inputs[3].toBool();
  auto dtype = inputs[4].toOptional<ScalarType>();

  int64_t data[dim.size()];
  std::copy(dim.begin(), dim.end(), data);
  IntArrayRef dim_arr(data, dim.size());
  auto ndim = self.dim();
  auto mask = make_dim_mask(dim_arr, ndim);

  allocate_reduction_result(
      output, self, mask, keepdim, get_dtype(output, self, dtype, false));
  TORCH_CHECK(
      output.scalar_type() == self.scalar_type(),
      "Habana reduction ops don't support casts yet");
  HabanaOperator::SetPTOutputs({output});
}

void ReduceOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    bool is_output_persistent) {
  TORCH_CHECK(
      inputs.size() == 5,
      "Incorrect size of inputs expected for reduction operator");
  TORCH_CHECK(
      inputs[0].isTensor(),
      "Input arg1 expected to be tensor for reduction operator");
  TORCH_CHECK(
      inputs[1].isTensor(),
      "Input arg2 expected to be tensor for reduction operator");
  TORCH_CHECK(
      inputs[2].isIntList(),
      "Input arg3 expected to be IntList for reduction operator");
  TORCH_CHECK(
      inputs[3].isBool(),
      "Input arg4 expected to be Bool for reduction operator");

  Tensor output = inputs[0].toTensor();
  Tensor self = inputs[1].toTensor();
  auto dim = inputs[2].toIntList();
  bool keepdim = inputs[3].toBool();
  auto dtype = inputs[4].toOptional<ScalarType>();

  int64_t data[dim.size()];
  std::copy(dim.begin(), dim.end(), data);
  IntArrayRef dim_arr(data, dim.size());
  auto ndim = self.dim();
  auto mask = make_dim_mask(dim_arr, ndim);

  allocate_reduction_result(
      output, self, mask, keepdim, get_dtype(output, self, dtype, false));
  TORCH_CHECK(
      output.scalar_type() == self.scalar_type(),
      "Habana reduction ops don't support casts yet");

  std::vector<synapse_helpers::tensor> syn_helper_intermediate;
  std::vector<synTensor> syn_intermediate;
  std::vector<int64_t> dims = self.sizes().vec();
  AllocateSynapseOutput(graph, output, is_output_persistent);

  // add syn_input tensor
  synapse_helpers::tensor& synInput = p_context_->syn_inputs_[0];
  syn_intermediate.emplace_back(synInput.get());
  // create syn_intermediate tensors of required shape
  unsigned loopend = keepdim ? dim.size() - 1 : dim.size();
  for (unsigned i = 0; i < loopend; i++) {
    dims[dim[i]] = 1;
    c10::IntArrayRef shape(dims.data(), self.dim());
    syn_helper_intermediate.emplace_back(habana_helpers::create_tensor(
        shape,
        graph.get_graph_handle(),
        false,
        self.device().index(),
        self.scalar_type()));
    syn_intermediate.emplace_back(syn_helper_intermediate[i].get());
  }
  // add syn_output tensor
  synapse_helpers::tensor& synOutput = p_context_->syn_outputs_[0];
  syn_intermediate.emplace_back(synOutput.get());

  std::string node_type = this->guid_;
  for (unsigned i = 0; i < dim.size(); i++) {
    ns_Reduction::Params params{};
    params.reductionDimension = ndim - dim[i] - 1;

    std::vector<synTensor> syn_in{syn_intermediate[i]};
    std::vector<synTensor> syn_out{syn_intermediate[i + 1]};
    graph.add_node(
        std::move(syn_in),
        std::move(syn_out),
        &params,
        sizeof(params),
        std::move(node_type));
  }

  if (!keepdim) {
    std::string node_type = "reshape";
    std::vector<synTensor> syn_in{syn_intermediate[dim.size()]};
    std::vector<synTensor> syn_out{syn_intermediate[dim.size() + 1]};
    graph.add_node(
        std::move(syn_in),
        std::move(syn_out),
        nullptr,
        0,
        std::move(node_type));
  }
}

void SumDimOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    bool is_output_persistent) {
  TORCH_CHECK(
      inputs.size() == 4,
      "Incorrect size of inputs expected for SumDim operator");
  TORCH_CHECK(
      inputs[0].isTensor(),
      "Input arg1 expected to be tensor for SumDim operator");
  TORCH_CHECK(
      inputs[1].isIntList(),
      "Input arg3 expected to be IntList for SumDim operator");
  TORCH_CHECK(
      inputs[2].isBool(), "Input arg4 expected to be Bool for SumDim operator");

  Tensor self = inputs[0].toTensor();
  auto dim = inputs[1].toIntList();
  bool keepdim = inputs[2].toBool();

  auto ndim = self.dim();
  TORCH_CHECK(
      keepdim || static_cast<int64_t>(dim.size()) != ndim,
      "Reduction to 0d tensor not supported yet");

  Tensor output;
  inputs.insert(inputs.begin(), IValue(output));

  ReduceOperator::AllocateAndAddSynapseNode(
      graph, inputs, is_output_persistent);
}

void SumDimOperator::SetPTOutputs(torch::jit::Stack& inputs) {
  Tensor output;
  inputs.insert(inputs.begin(), IValue(output));
  ReduceOperator::SetPTOutputs(inputs);
}

Tensor sum_dim_IntList_hpu(
    const Tensor& self,
    IntArrayRef dim,
    bool keepdim,
    c10::optional<ScalarType> dtype) {
  PT_KERNEL_BEGIN;

  at::ScalarType scalar_type = self.scalar_type();
  std::string node_type =
      "reduce_sum_fwd_" + habana_helpers::name_suffix_from_type(scalar_type);

  // Create the operator
  size_t device_id = self.device().index();
  auto& device = synapse_helpers::HPURegistrar::get_device(device_id);
  std::vector<const at::Tensor*> pt_inputs{&self};
  std::vector<c10::IValue> stack = {
      IValue(self), IValue(dim), IValue(keepdim), IValue(dtype)};
  // Create the operator
  SumDimOperator Op(device_id, node_type);
  size_t key = Op.GetRecipeKey(node_type, stack);

  if (device.get_recipe_handle_cache().isCached(key)) {
    PT_KERNEL_DEBUG("Cache hit key:", key);
    Op.SetPTInputs(pt_inputs);
    Op.SetPTOutputs(stack);
    Op.Execute(key);
  } else {
    PT_KERNEL_DEBUG("key:", key);
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

void SumDimOutOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    bool is_output_persistent) {
  TORCH_CHECK(
      inputs.size() == 5,
      "Incorrect size of inputs expected for SumDimOut operator");
  TORCH_CHECK(
      inputs[0].isTensor(),
      "Input arg1 expected to be tensor for SumDimOut operator");
  TORCH_CHECK(
      inputs[1].isTensor(),
      "Input arg2 expected to be tensor for SumDimOut operator");
  TORCH_CHECK(
      inputs[2].isIntList(),
      "Input arg3 expected to be IntList for SumDimOut operator");
  TORCH_CHECK(
      inputs[3].isBool(),
      "Input arg4 expected to be Bool for SumDimOut operator");

  Tensor self = inputs[1].toTensor();
  auto dim = inputs[2].toIntList();
  bool keepdim = inputs[3].toBool();

  auto ndim = self.dim();
  TORCH_CHECK(
      keepdim || static_cast<int64_t>(dim.size()) != ndim,
      "Reduction to 0d tensor not supported yet");

  ReduceOperator::AllocateAndAddSynapseNode(
      graph, inputs, is_output_persistent);
}

void SumDimOutOperator::SetPTOutputs(torch::jit::Stack& inputs) {
  ReduceOperator::SetPTOutputs(inputs);
}

Tensor& sum_IntList_out_hpu(
    Tensor& output,
    const Tensor& self,
    IntArrayRef dim,
    bool keepdim,
    c10::optional<ScalarType> dtype) {
  PT_KERNEL_BEGIN;

  at::ScalarType scalar_type = self.scalar_type();
  std::string node_type =
      "reduce_sum_fwd_" + habana_helpers::name_suffix_from_type(scalar_type);

  size_t device_id = self.device().index();
  auto& device = synapse_helpers::HPURegistrar::get_device(device_id);
  std::vector<const at::Tensor*> pt_inputs{&self};
  // Build Params for the graph
  std::vector<c10::IValue> stack = {IValue(output),
                                    IValue(self),
                                    IValue(dim),
                                    IValue(keepdim),
                                    IValue(dtype)};
  // Create the operator
  SumDimOutOperator Op(device_id, node_type);
  size_t key = Op.GetRecipeKey(node_type, stack);

  if (device.get_recipe_handle_cache().isCached(key)) {
    PT_KERNEL_DEBUG("Cache hit key:", key);
    Op.SetPTInputs(pt_inputs);
    Op.SetPTOutputs(stack);
    Op.Execute(key);
  } else {
    PT_KERNEL_DEBUG("key:", key);
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

void MeanDimOperator::SetPTOutputs(torch::jit::Stack& inputs) {
  Tensor output;
  inputs.insert(inputs.begin(), IValue(output));
  ReduceOperator::SetPTOutputs(inputs);
}

void MeanDimOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    bool is_output_persistent) {
  TORCH_CHECK(
      inputs.size() == 4,
      "Incorrect size of inputs expected for MeanDim operator");
  TORCH_CHECK(
      inputs[0].isTensor(),
      "Input arg1 expected to be tensor for MeanDim operator");
  TORCH_CHECK(
      inputs[1].isIntList(),
      "Input arg3 expected to be IntList for MeanDim operator");
  TORCH_CHECK(
      inputs[2].isBool(),
      "Input arg4 expected to be Bool for MeanDim operator");

  Tensor self = inputs[0].toTensor();
  auto dim = inputs[1].toIntList();
  bool keepdim = inputs[2].toBool();

  auto ndim = self.dim();
  TORCH_CHECK(
      keepdim || static_cast<int64_t>(dim.size()) != ndim,
      "Reduction to 0d tensor not supported yet");

  Tensor output;
  inputs.insert(inputs.begin(), IValue(output));

  ReduceOperator::AllocateAndAddSynapseNode(
      graph, inputs, is_output_persistent);
}

Tensor mean_dim_hpu(
    const Tensor& self,
    IntArrayRef dim,
    bool keepdim,
    c10::optional<ScalarType> dtype) {
  PT_KERNEL_BEGIN;

  at::ScalarType scalar_type = self.scalar_type();
  std::string node_type =
      "reduce_mean_fwd_" + habana_helpers::name_suffix_from_type(scalar_type);

  size_t device_id = self.device().index();
  auto& device = synapse_helpers::HPURegistrar::get_device(device_id);
  std::vector<const at::Tensor*> pt_inputs{&self};
  // Build Params for the graph
  std::vector<c10::IValue> stack = {
      IValue(self), IValue(dim), IValue(keepdim), IValue(dtype)};
  // Create the operator
  MeanDimOperator Op(device_id, node_type);
  size_t key = Op.GetRecipeKey(node_type, stack);

  if (device.get_recipe_handle_cache().isCached(key)) {
    PT_KERNEL_DEBUG("Cache hit key:", key);
    Tensor output;
    Op.SetPTInputs(pt_inputs);
    Op.SetPTOutputs(stack);
    Op.Execute(key);
  } else {
    PT_KERNEL_DEBUG("key:", key);
    // Create Graph
    auto graph = habana_helpers::create_graph(device_id, node_type);

    // Assign Inputs to the Operator
    Op.AllocateSynapseInputs(graph, pt_inputs, true);

    // Add nodes to the graph
    Op.AllocateAndAddSynapseNode(graph, stack, true);

    // compile and execute the graph
    Op.Compile(graph);
  }

  std::vector<at::Tensor> out = Op.GetOutputs();
  TORCH_CHECK(out.size() == 1, "Incorrect size of outputs");

  PT_KERNEL_END;
  return out.at(0);
}

void MeanDimOutOperator::SetPTOutputs(torch::jit::Stack& inputs) {
  ReduceOperator::SetPTOutputs(inputs);
}

void MeanDimOutOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    bool is_output_persistent) {
  TORCH_CHECK(
      inputs.size() == 5,
      "Incorrect size of inputs expected for MeanDimOut operator");
  TORCH_CHECK(
      inputs[0].isTensor(),
      "Input arg1 expected to be tensor for MeanDimOut operator");
  TORCH_CHECK(
      inputs[1].isTensor(),
      "Input arg2 expected to be tensor for MeanDimOut operator");
  TORCH_CHECK(
      inputs[2].isIntList(),
      "Input arg3 expected to be IntList for MeanDimOut operator");
  TORCH_CHECK(
      inputs[3].isBool(),
      "Input arg4 expected to be Bool for MeanDimOut operator");

  Tensor self = inputs[1].toTensor();
  auto dim = inputs[2].toIntList();
  bool keepdim = inputs[3].toBool();

  auto ndim = self.dim();
  TORCH_CHECK(
      keepdim || static_cast<int64_t>(dim.size()) != ndim,
      "Reduction to 0d tensor not supported yet");

  ReduceOperator::AllocateAndAddSynapseNode(
      graph, inputs, is_output_persistent);
}
Tensor& mean_dim_out_hpu(
    Tensor& output,
    const Tensor& self,
    IntArrayRef dim,
    bool keepdim,
    c10::optional<ScalarType> dtype) {
  PT_KERNEL_BEGIN;

  at::ScalarType scalar_type = self.scalar_type();
  std::string node_type =
      "reduce_mean_fwd_" + habana_helpers::name_suffix_from_type(scalar_type);

  size_t device_id = self.device().index();
  auto& device = synapse_helpers::HPURegistrar::get_device(device_id);
  std::vector<const at::Tensor*> pt_inputs{&self};
  // Build Params for the graph
  std::vector<c10::IValue> stack = {IValue(output),
                                    IValue(self),
                                    IValue(dim),
                                    IValue(keepdim),
                                    IValue(dtype)};
  // Create the operator
  MeanDimOutOperator Op(device_id, node_type);
  size_t key = Op.GetRecipeKey(node_type, stack);

  if (device.get_recipe_handle_cache().isCached(key)) {
    PT_KERNEL_DEBUG("Cache hit key:", key);
    Tensor output;
    Op.SetPTInputs(pt_inputs);
    Op.SetPTOutputs(stack);
    Op.Execute(key);
  } else {
    PT_KERNEL_DEBUG("key:", key);
    // Create Graph
    auto graph = habana_helpers::create_graph(device_id, node_type);

    // Assign Inputs to the Operator
    Op.AllocateSynapseInputs(graph, pt_inputs, true);

    // Add nodes to the graph
    Op.AllocateAndAddSynapseNode(graph, stack, true);

    // compile and execute the graph
    Op.Compile(graph);
  }

  std::vector<at::Tensor> out = Op.GetOutputs();
  TORCH_CHECK(out.size() == 1, "Incorrect size of outputs");
  PT_KERNEL_END;
  return out.at(0);
}

void SumOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    bool is_output_persistent) {
  TORCH_CHECK(
      inputs.size() == 2, "Incorrect size of inputs expected for Sum operator");
  TORCH_CHECK(
      inputs[0].isTensor(),
      "Input arg1 expected to be tensor for Sum operator");

  Tensor self = inputs[0].toTensor();

  Tensor output;
  auto ndim = self.dim();
  int64_t data[4];
  for (int i = 0; i < ndim; i++) {
    data[i] = i;
  }
  IntArrayRef dim(data, ndim);
  bool keepdim = false;

  inputs.insert(inputs.begin(), IValue(output));
  inputs.insert(inputs.begin() + 2, IValue(dim));
  inputs.insert(inputs.begin() + 3, IValue(keepdim));

  ReduceOperator::AllocateAndAddSynapseNode(
      graph, inputs, is_output_persistent);
}

void SumOperator::SetPTOutputs(torch::jit::Stack& inputs) {
  Tensor self = inputs[0].toTensor();
  Tensor output;
  auto ndim = self.dim();
  int64_t data[4];
  for (int i = 0; i < ndim; i++) {
    data[i] = i;
  }
  IntArrayRef dim(data, ndim);
  bool keepdim = false;

  inputs.insert(inputs.begin(), IValue(output));
  inputs.insert(inputs.begin() + 2, IValue(dim));
  inputs.insert(inputs.begin() + 3, IValue(keepdim));
  ReduceOperator::SetPTOutputs(inputs);
}

Tensor sum_hpu(const Tensor& self, c10::optional<ScalarType> dtype) {
  PT_KERNEL_BEGIN;

  at::ScalarType scalar_type = self.scalar_type();
  std::string node_type =
      "reduce_sum_fwd_" + habana_helpers::name_suffix_from_type(scalar_type);

  std::vector<const at::Tensor*> pt_inputs{&self};
  std::vector<c10::IValue> stack = {IValue(self), IValue(dtype)};
  size_t device_id = self.device().index();
  auto& device = synapse_helpers::HPURegistrar::get_device(device_id);
  // Create the operator
  SumOperator Op(device_id, node_type);
  size_t key = Op.GetRecipeKey(node_type, stack);

  if (device.get_recipe_handle_cache().isCached(key)) {
    PT_KERNEL_DEBUG("Cache hit key:", key);
    Op.SetPTInputs(pt_inputs);
    Op.SetPTOutputs(stack);
    Op.Execute(key);
  } else {
    PT_KERNEL_DEBUG("key:", key);

    // Create Graph
    auto graph = habana_helpers::create_graph(device_id, node_type);

    // Assign Inputs to the Operator
    Op.AllocateSynapseInputs(graph, pt_inputs, true);

    // Add nodes to the graph
    Op.AllocateAndAddSynapseNode(graph, stack, true);

    // compile and execute the graph
    Op.Compile(graph);
  }

  std::vector<at::Tensor> out = Op.GetOutputs();
  TORCH_CHECK(out.size() == 1, "Incorrect size of outputs");

  out.at(0).unsafeGetTensorImpl()->set_sizes_and_strides({}, {});
  PT_KERNEL_END;
  return out.at(0);
}

void MeanOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    bool is_output_persistent) {
  TORCH_CHECK(
      inputs.size() == 2,
      "Incorrect size of inputs expected for Mean operator");
  TORCH_CHECK(
      inputs[0].isTensor(),
      "Input arg1 expected to be tensor for Mean operator");

  Tensor self = inputs[0].toTensor();

  Tensor output;
  auto ndim = self.dim();
  int64_t data[4];
  for (int i = 0; i < ndim; i++) {
    data[i] = i;
  }
  IntArrayRef dim(data, ndim);
  bool keepdim = false;

  inputs.insert(inputs.begin(), IValue(output));
  inputs.insert(inputs.begin() + 2, IValue(dim));
  inputs.insert(inputs.begin() + 3, IValue(keepdim));

  ReduceOperator::AllocateAndAddSynapseNode(
      graph, inputs, is_output_persistent);
}

void MeanOperator::SetPTOutputs(torch::jit::Stack& inputs) {
  Tensor self = inputs[0].toTensor();
  Tensor output;
  auto ndim = self.dim();
  int64_t data[4];
  for (int i = 0; i < ndim; i++) {
    data[i] = i;
  }
  IntArrayRef dim(data, ndim);
  bool keepdim = false;

  inputs.insert(inputs.begin(), IValue(output));
  inputs.insert(inputs.begin() + 2, IValue(dim));
  inputs.insert(inputs.begin() + 3, IValue(keepdim));
  ReduceOperator::SetPTOutputs(inputs);
}

Tensor mean_hpu(const Tensor& self, c10::optional<ScalarType> dtype) {
  PT_KERNEL_BEGIN;

  at::ScalarType scalar_type = self.scalar_type();
  std::string node_type =
      "reduce_mean_fwd_" + habana_helpers::name_suffix_from_type(scalar_type);

  size_t device_id = self.device().index();
  auto& device = synapse_helpers::HPURegistrar::get_device(device_id);
  std::vector<const at::Tensor*> pt_inputs{&self};
  // Build Params for the graph
  std::vector<c10::IValue> stack = {IValue(self), IValue(dtype)};
  // Create the operator
  MeanOperator Op(device_id, node_type);
  size_t key = Op.GetRecipeKey(node_type, stack);

  if (device.get_recipe_handle_cache().isCached(key)) {
    PT_KERNEL_DEBUG("Cache hit key:", key);
    Tensor output;
    Op.SetPTInputs(pt_inputs);
    Op.SetPTOutputs(stack);
    Op.Execute(key);
  } else {
    PT_KERNEL_DEBUG("key:", key);
    // Create Graph
    auto graph = habana_helpers::create_graph(device_id, node_type);

    // Assign Inputs to the Operator
    Op.AllocateSynapseInputs(graph, pt_inputs, true);

    // Add nodes to the graph
    Op.AllocateAndAddSynapseNode(graph, stack, true);

    // compile and execute the graph
    Op.Compile(graph);
  }

  std::vector<at::Tensor> out = Op.GetOutputs();
  TORCH_CHECK(out.size() == 1, "Incorrect size of outputs");

  out.at(0).unsafeGetTensorImpl()->set_sizes_and_strides({}, {});
  PT_KERNEL_END;
  return out.at(0);
}

void AnyDimOutOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    bool is_output_persistent) {
  TORCH_CHECK(
      inputs.size() == 4,
      "Incorrect size of inputs expected for AnyDimOut operator");
  TORCH_CHECK(
      inputs[0].isTensor(),
      "Input arg1 expected to be tensor for AnyDimOut operator");
  TORCH_CHECK(
      inputs[1].isTensor(),
      "Input arg2 expected to be tensor for AnyDimOut operator");
  TORCH_CHECK(
      inputs[2].isInt(),
      "Input arg3 expected to be Int for AnyDimOut operator");
  TORCH_CHECK(
      inputs[3].isBool(),
      "Input arg4 expected to be Bool for AnyDimOut operator");

  auto output = inputs[0].toTensor();
  auto self = inputs[1].toTensor();
  auto dim = inputs[2].toInt();
  auto keepdim = inputs[3].toBool();

  // Converting dim to dim array
  // SumDimOperator accepts dim array only
  // dim array has been used to resize output tensor along the input dim
  int64_t data[1];
  data[0] = dim;
  IntArrayRef dim_arr(data, 1);

  TORCH_CHECK(
      self.scalar_type() == c10::ScalarType::Bool,
      "Input arg2 expected to be of type Bool Tensor for AnyDimOut operator");



  // Cast Input tensor to Float tensor
  auto out_float=at::empty(self.sizes(), self.options().dtype(c10::ScalarType::Float));
  std::string node_type = "cast_i8_to_f32";

  // Create the operator
  CastOperator intToFloatOp(this->p_context_->device_id_,node_type);
  auto& float_syn = intToFloatOp.SetSynapseInput(std::move(p_context_->syn_inputs_[0]));

  // Build Params for the graph
  std::vector<c10::IValue> stack{IValue(self), IValue(out_float)};
  intToFloatOp.AllocateAndAddSynapseNode(graph, stack, false);

  synapse_helpers::tensor& float_syn_tensor = intToFloatOp.GetSynOutputs()[0];
  std::vector<at::Tensor> out = intToFloatOp.GetOutputs();
  Tensor output_float = out.at(0);
  p_context_->syn_inputs_[0] = std::move(float_syn);

  stack.clear();
  out.clear();



  //Reduction operation
  at::ScalarType scalar_type =c10::ScalarType::Float;
  node_type =
      "reduce_sum_fwd_" + habana_helpers::name_suffix_from_type(scalar_type);

  // Create the operator
  SumDimOperator sumOp(this->p_context_->device_id_,node_type);
  sumOp.SetSynapseInput(std::move(float_syn_tensor));

  // Build Params for the graph
  c10::optional<ScalarType> dtype = output_float.scalar_type();
  stack.emplace_back(IValue(output_float));
  stack.emplace_back(IValue(dim_arr));
  stack.emplace_back(IValue(keepdim));
  stack.emplace_back(IValue(dtype));

  sumOp.AllocateAndAddSynapseNode(graph, stack, false);

  synapse_helpers::tensor& reduce_syn_tensor = sumOp.GetSynOutputs()[0];
  out = sumOp.GetOutputs();
  Tensor output_reduce = out.at(0);
  stack.clear();
  out.clear();


  // Resize output tensor based on reduction dimension
  auto ndim = self.dim();
  auto mask = make_dim_mask(dim_arr, ndim);
  allocate_reduction_result(output, self, mask, keepdim, c10::ScalarType::Char);



  // Cast Reduced Float tensor to Int tensor
  node_type = "cast_f32_to_i8";

  // Create the operator
  CastOperator floatToIntOp(this->p_context_->device_id_,node_type);
  floatToIntOp.SetSynapseInput(std::move(reduce_syn_tensor));

  // Build Params for the graph
  stack.emplace_back(IValue(output_reduce));
  stack.emplace_back(IValue(output));

  floatToIntOp.AllocateAndAddSynapseNode(graph, stack, is_output_persistent);
  synapse_helpers::tensor& int_syn_tensor = floatToIntOp.GetSynOutputs()[0];

  p_context_->syn_outputs_.emplace_back(std::move(int_syn_tensor));
  p_context_->pt_outputs_.emplace_back(std::move(output));
}

/*************************************************************************
 * @brief Kernel implementation for reduction kernel torch.any(output, self,
 *dim, keepdim)
 * @param [out] output - output tensor, bool
 * @param [in] self - input tensor, bool
 * @param [in] dim - along which dimension to reduce, int64_t
 * @param [in] keepdim - output tensor has dim retained or not, bool,
 *default = false
 ************************************************************************/
Tensor& any_dim_out_hpu(
    Tensor& output,
    const Tensor& self,
    int64_t dim,
    bool keepdim) {
  PT_KERNEL_BEGIN;

  self.to(c10::ScalarType::Char);

  std::string node_type = "reduce_any"; // No TPC kernel is present for any

  // Create the operator
  size_t device_id = self.device().index();
  AnyDimOutOperator Op(device_id, node_type);

  // Create Graph
  auto graph = habana_helpers::create_graph(device_id, node_type);

  // Assign Inputs to the Operator
  std::vector<const at::Tensor*> pt_inputs{&self};
  Op.AllocateSynapseInputs(graph, pt_inputs, true);

  // Build Params for the graph
  std::vector<c10::IValue> stack = {IValue(output),
                                    IValue(self),
                                    IValue(dim),
                                    IValue(keepdim)};
  Op.AllocateAndAddSynapseNode(graph, stack, true);

  // compile and execute the graph
  Op.Compile(graph);

  std::vector<at::Tensor> out = Op.GetOutputs();
  TORCH_CHECK(out.size() == 1, "Incorrect size of outputs");

  out.at(0).to(c10::ScalarType::Bool);
  PT_KERNEL_END;
  return out.at(0);
}

void AnyDimOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    bool is_output_persistent) {
  TORCH_CHECK(
      inputs.size() == 3,
      "Incorrect size of inputs expected for AnyDim operator");
  TORCH_CHECK(
      inputs[0].isTensor(),
      "Input arg1 expected to be tensor for AnyDim operator");
  TORCH_CHECK(
      inputs[1].isInt(),
      "Input arg2 expected to be Int for AnyDim operator");
  TORCH_CHECK(
      inputs[2].isBool(),
      "Input arg3 expected to be Bool for AnyDim operator");

  auto self = inputs[0].toTensor();
  Tensor output =at::empty({0}, self.options().dtype(c10::ScalarType::Char));
  inputs.insert(inputs.begin(), IValue(output));

  AnyDimOutOperator::AllocateAndAddSynapseNode(graph, inputs, is_output_persistent);
}

/*************************************************************************
 * @brief Kernel implementation for reduction kernel output =
 *torch.any(self, dim, keepdim)
 * @param [out] output - output tensor, bool
 * @param [in] self - input tensor, bool
 * @param [in] dim - along which dimension to reduce, int64_t
 * @param [in] keepdim - output tensor has dim retained or not, bool,
 *default = false
 ************************************************************************/
Tensor any_dim_hpu(const Tensor& self, int64_t dim, bool keepdim) {
  PT_KERNEL_BEGIN;

  self.to(c10::ScalarType::Char);

  std::string node_type = "reduce_any"; // No TPC kernel is present for any

  // Create the operator
  size_t device_id = self.device().index();
  AnyDimOperator Op(device_id, node_type);

  // Create Graph
  auto graph = habana_helpers::create_graph(device_id, node_type);

  // Assign Inputs to the Operator
  std::vector<const at::Tensor*> pt_inputs{&self};
  Op.AllocateSynapseInputs(graph, pt_inputs, true);

  // Build Params for the graph
  std::vector<c10::IValue> stack = {IValue(self),
                                    IValue(dim),
                                    IValue(keepdim)};
  Op.AllocateAndAddSynapseNode(graph, stack, true);

  // compile and execute the graph
  Op.Compile(graph);

  std::vector<at::Tensor> out = Op.GetOutputs();
  TORCH_CHECK(out.size() == 1, "Incorrect size of outputs");
  PT_KERNEL_END;
  return out.at(0).to(c10::ScalarType::Bool);
}

void AnyOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    bool is_output_persistent) {
  TORCH_CHECK(
      inputs.size() == 1,
      "Incorrect size of inputs expected for Any operator");
  TORCH_CHECK(
      inputs[0].isTensor(),
      "Input arg1 expected to be tensor for Any operator");

  auto self = inputs[0].toTensor();
  Tensor output;

  TORCH_CHECK(
      self.scalar_type() == c10::ScalarType::Bool,
      "Input arg2 expected to be of type Bool Tensor for AnyDimOut operator");



  // Cast Input tensor to Float tensor
  auto out_float=at::empty(self.sizes(), self.options().dtype(c10::ScalarType::Float));
  std::string node_type = "cast_i8_to_f32";

  // Create the operator
  CastOperator intToFloatOp(this->p_context_->device_id_,node_type);
  auto& float_syn = intToFloatOp.SetSynapseInput(std::move(p_context_->syn_inputs_[0]));

  // Build Params for the graph
  std::vector<c10::IValue> stack{IValue(self), IValue(out_float)};

  intToFloatOp.AllocateAndAddSynapseNode(graph, stack, false);

  synapse_helpers::tensor& float_syn_tensor = intToFloatOp.GetSynOutputs()[0];
  std::vector<at::Tensor> out = intToFloatOp.GetOutputs();
  Tensor output_float = out.at(0);
  p_context_->syn_inputs_[0] = std::move(float_syn);

  stack.clear();
  out.clear();



  //Reduction operation
  at::ScalarType scalar_type =c10::ScalarType::Float;
  node_type =
      "reduce_sum_fwd_" + habana_helpers::name_suffix_from_type(scalar_type);

  // Create the operator
  SumOperator sumOp(this->p_context_->device_id_,node_type);
  sumOp.SetSynapseInput(std::move(float_syn_tensor));

  // Build Params for the graph
  c10::optional<ScalarType> dtype = output_float.scalar_type();
  stack.emplace_back(IValue(output_float));
  stack.emplace_back(IValue(dtype));

  sumOp.AllocateAndAddSynapseNode(graph, stack, false);

  synapse_helpers::tensor& reduce_syn_tensor = sumOp.GetSynOutputs()[0];
  out = sumOp.GetOutputs();
  Tensor output_reduce = out.at(0);
  stack.clear();
  out.clear();



  // Resize output tensor based on reduction dimension
  auto ndim = self.dim();
  int64_t data[4];
  for (int i = 0; i < ndim; i++) {
    data[i] = i;
  }
  IntArrayRef dim_arr(data, ndim);
  auto mask = make_dim_mask(dim_arr, ndim);
  allocate_reduction_result(output, self, mask, false, c10::ScalarType::Char);



  // Cast Reduced Float tensor to Int tensor
  node_type = "cast_f32_to_i8";

  // Create the operator
  CastOperator floatToIntOp(this->p_context_->device_id_,node_type);
  floatToIntOp.SetSynapseInput(std::move(reduce_syn_tensor));

  // Build Params for the graph
  stack.emplace_back(IValue(output_reduce));
  stack.emplace_back(IValue(output));

  floatToIntOp.AllocateAndAddSynapseNode(graph, stack, is_output_persistent);
  synapse_helpers::tensor& int_syn_tensor = floatToIntOp.GetSynOutputs()[0];
  p_context_->syn_outputs_.emplace_back(std::move(int_syn_tensor));
  p_context_->pt_outputs_.emplace_back(std::move(output));
}
/*************************************************************************
 * @brief Kernel implementation for reduction kernel output =
 *torch.any(self)
 * @param [out] output - output tensor, bool
 * @param [in] self - input tensor, bool
 ************************************************************************/
Tensor any_hpu(const Tensor& self) {
  PT_KERNEL_BEGIN;

  self.to(c10::ScalarType::Char);

  std::string node_type = "reduce_any"; // No TPC kernel is present for any

  // Create the operator
  size_t device_id = self.device().index();
  AnyOperator Op(device_id, node_type);

  // Create Graph
  auto graph = habana_helpers::create_graph(device_id, node_type);

  // Assign Inputs to the Operator
  std::vector<const at::Tensor*> pt_inputs{&self};
  Op.AllocateSynapseInputs(graph, pt_inputs, true);

  // Build Params for the graph
  std::vector<c10::IValue> stack = {IValue(self)};
  Op.AllocateAndAddSynapseNode(graph, stack, true);

  // compile and execute the graph
  Op.Compile(graph);

  std::vector<at::Tensor> out = Op.GetOutputs();
  TORCH_CHECK(out.size() == 1, "Incorrect size of outputs");
  out.at(0).unsafeGetTensorImpl()->set_sizes_and_strides({}, {});
  PT_KERNEL_END;
  return out.at(0).to(c10::ScalarType::Bool);

}

static auto registry =
    torch::RegisterOperators()
        .op(torch::RegisterOperators::options()
                .schema(
                    "aten::sum.dim_IntList(Tensor self, int[1] dim, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor")
                .impl_unboxedOnlyKernel<
                    decltype(sum_dim_IntList_hpu),
                    &sum_dim_IntList_hpu>(DispatchKey::HABANATensorId)
                .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
        .op(torch::RegisterOperators::options()
                .schema(
                    "aten::sum.IntList_out(Tensor self, int[1] dim, bool keepdim=False, *, ScalarType? dtype=None, Tensor(a!) out) -> Tensor(a!)")
                .impl_unboxedOnlyKernel<
                    decltype(sum_IntList_out_hpu),
                    &sum_IntList_out_hpu>(DispatchKey::HABANATensorId)
                .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
        .op(torch::RegisterOperators::options()
                .schema(
                    "aten::mean.dim(Tensor self, int[1] dim, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor")
                .impl_unboxedOnlyKernel<decltype(mean_dim_hpu), &mean_dim_hpu>(
                    DispatchKey::HABANATensorId)
                .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
        .op(torch::RegisterOperators::options()
                .schema(
                    "aten::mean.out(Tensor self, int[1] dim, bool keepdim=False, *, ScalarType? dtype=None, Tensor(a!) out) -> Tensor(a!)")
                .impl_unboxedOnlyKernel<
                    decltype(mean_dim_out_hpu),
                    &mean_dim_out_hpu>(DispatchKey::HABANATensorId)
                .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
        .op(torch::RegisterOperators::options()
                .schema(
                    "aten::sum(Tensor self, *, ScalarType? dtype=None) -> Tensor")
                .impl_unboxedOnlyKernel<decltype(sum_hpu), &sum_hpu>(
                    DispatchKey::HABANATensorId)
                .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
        .op(torch::RegisterOperators::options()
                .schema(
                    "aten::mean(Tensor self, *, ScalarType? dtype=None) -> Tensor")
                .impl_unboxedOnlyKernel<decltype(mean_hpu), &mean_hpu>(
                    DispatchKey::HABANATensorId)
                .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
        .op(torch::RegisterOperators::options()
                .schema(
                    "aten::any.dim(Tensor self, int dim, bool keepdim=False) -> Tensor")
                .impl_unboxedOnlyKernel<decltype(any_dim_hpu), &any_dim_hpu>(
                    DispatchKey::HABANATensorId)
                .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
        .op(torch::RegisterOperators::options()
                .schema("aten::any(Tensor self) -> Tensor")
                .impl_unboxedOnlyKernel<decltype(any_hpu), &any_hpu>(
                    DispatchKey::HABANATensorId)
                .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
        .op(torch::RegisterOperators::options()
                .schema(
                    "aten::any.out(Tensor self, int dim, bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!)")
                .impl_unboxedOnlyKernel<
                    decltype(any_dim_out_hpu),
                    &any_dim_out_hpu>(DispatchKey::HABANATensorId)
                .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA));
