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
#include <torch/csrc/jit/ir/irparser.h>
#include <torch/script.h>

#include "habana_bridge/kernel/hpu_habana_launch_op_pt.h"
#include "habana_device/HPUCheck.h"
#include "habana_device/hpu_cached_devices.h"
#include "habana_helpers/graph.h"
#include "habana_helpers/tensor_utils.h"
#include "habana_kernels/basic_kernels.h"
#include "habana_kernels/kernel_utils.h"
#include "habana_kernels/reduction_kernels.h"
#include "habana_kernels/resize.h"
#include "habana_kernels/simple_generic_kernel.h"
#include "habana_kernels/tensor_shape_kernels.h"
#include "habana_lazy/hlexec.h"
#include "habana_lazy/passes/transform_graph.h"

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
    ScalarType dtype,
    bool is_result_persistent) {
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
    if (result.numel() || is_result_persistent)
      THHTensor_resizeNd(tht_result, shape.size(), shape.data(), nullptr);
    else {
      THHTensor_resizeNd_nonpersistent(
          tht_result, shape.size(), shape.data(), nullptr);
    }
  } else {
    result = at::empty(
        shape, self.options().dtype(dtype), self.suggest_memory_format());
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
      output, self, mask, keepdim, get_dtype(output, self, dtype, false), true);
  /*TORCH_CHECK(
      output.scalar_type() == self.scalar_type(),
      "Habana reduction ops don't support casts yet");*/
  std::vector<at::Tensor> v{output};
  HabanaOperator::SetPTOutputs(v);
}

void ReduceOperator::sort_dims(
    c10::List<int64_t>& in_dim,
    int64_t dim,
    int64_t dims_to_reduce) {
  for (int64_t i = 0; i < dims_to_reduce; i++) {
    in_dim[i] = c10::maybe_wrap_dim(in_dim[i], dim, true);
  }
  std::sort(in_dim.begin(), in_dim.end());
  auto last = std::unique(in_dim.begin(), in_dim.end());
  in_dim.erase(last, in_dim.end());
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
  auto in_dim = inputs[2].toIntList();
  bool keepdim = inputs[3].toBool();
  auto dtype = inputs[4].toOptional<ScalarType>();
  auto num_dims_to_reduce = in_dim.size();

  // wrap dims to positive values, sort dim list and remove any duplicates
  sort_dims(in_dim, self.dim(), num_dims_to_reduce);

  // check whether all dims in list are the higher "continuous" dimensions
  // if yes, "flatten" higher dims to a single unrolled-size dim.
  // Note-1 that this is an optimization to avoid any precision loss we may
  // get due to separate back 2 back reductions along single dimensions.
  // Note-2 cases such as [0,1,3] where there is in additional dim to reduce
  // in addition to continuous dims is not supported with flattening and falls
  // back to regular flow
  std::vector<int64_t> next_val{0, 1, 2, 3, 4};
  bool flatten_higher_dims = false;
  for (auto i = 0u; i < num_dims_to_reduce && num_dims_to_reduce > 1; i++) {
    if (in_dim[i] == next_val[i]) {
      flatten_higher_dims = true;
    } else {
      flatten_higher_dims = false;
      break;
    }
  }

  if (flatten_higher_dims) {
    // reshaped_sizes is used to hold appropriate dim sizes
    // reshaped_sizes is passed to Reshape operator stack as
    // an IntArrayRef variable.
    unsigned reshaped_in_dim_size = 1;
    std::vector<int64_t> reshaped_self_sizes;
    auto original_self_sizes = self.sizes().vec();

    auto flatten_size = std::accumulate(
        original_self_sizes.begin(),
        original_self_sizes.begin() + num_dims_to_reduce,
        1,
        std::multiplies<int>());
    if (keepdim) {
      // we need to keep a size of '1' for upper dims, flattened value at last
      // pos of "dim array", and original sizes for lower dimensions
      // example: sizes [8,3,2,2] with dim=[0,1,2] and keepdim=true becomes
      // [1,1,48,2]
      for (unsigned i = 0; i < num_dims_to_reduce - 1; i++) {
        reshaped_self_sizes.emplace_back(1);
      }
    }
    // else all upper dims sizes are flattened into a single dim at 0
    // example: sizes [8,3,2,2] with dim=[0,1,2] and keepdim=true becomes
    // [48,2]

    reshaped_self_sizes.emplace_back(flatten_size);
    for (unsigned i = num_dims_to_reduce; i < self.dim(); i++) {
      reshaped_self_sizes.emplace_back(original_self_sizes[i]);
    }
    // reshape to "reshaped-sizes" before reduction
    c10::IntArrayRef shape(
        reshaped_self_sizes.data(), reshaped_self_sizes.size());
    // Reshape operator to flatten higher dims to single dim.
    // The reshape node in the else part doesn't actually reshape, but is a pass
    // through. We assume that the reshape in the else part (when upper dims are
    // not merged), will be optimized out by GC
    ReshapeOperator ReshapeOp(this->p_context_->device_id_, self.scalar_type());
    auto& reshape_syn =
        ReshapeOp.SetSynapseInput(std::move(p_context_->syn_inputs_[0]));
    // Build Params for the graph
    std::vector<c10::IValue> stack;
    stack.emplace_back(IValue(self));
    stack.emplace_back(IValue(shape));
    ReshapeOp.AllocateAndAddSynapseNode(graph, stack, false);
    p_context_->syn_inputs_[0] = std::move(reshape_syn);

    auto self_reshaped = ReshapeOp.GetOutputs()[0];
    int64_t reshaped_in_dim_data[reshaped_in_dim_size];
    if (!keepdim) {
      reshaped_in_dim_data[0] = 0;
    } else {
      std::copy(
          in_dim.begin() + num_dims_to_reduce - 1,
          in_dim.end(),
          reshaped_in_dim_data);
    }

    IntArrayRef reshaped_in_dim(reshaped_in_dim_data, reshaped_in_dim_size);
    auto mask = make_dim_mask(reshaped_in_dim, self_reshaped.dim());
    allocate_reduction_result(
        output,
        self_reshaped,
        mask,
        keepdim,
        get_dtype(output, self_reshaped, dtype, false),
        is_output_persistent);
    /*TORCH_CHECK(
        output.scalar_type() == self_reshaped.scalar_type(),
        "Habana reduction ops don't support casts yet");*/
    AllocateSynapseOutput(graph, output, is_output_persistent);

    std::tie(std::ignore, p_context_->syn_outputs_[0]) = CreateReductionGraph(
        graph,
        self_reshaped,
        std::move(ReshapeOp.GetSynOutputs()[0]),
        std::move(p_context_->syn_outputs_[0]),
        reshaped_in_dim,
        keepdim,
        output.scalar_type());
  } else {
    int64_t in_dim_copy[in_dim.size()];
    std::copy(in_dim.begin(), in_dim.end(), in_dim_copy);
    IntArrayRef in_dim_arr(in_dim_copy, in_dim.size());
    auto mask = make_dim_mask(in_dim_arr, self.dim());
    allocate_reduction_result(
        output,
        self,
        mask,
        keepdim,
        get_dtype(output, self, dtype, false),
        is_output_persistent);
    /*TORCH_CHECK(
        output.scalar_type() == self.scalar_type(),
        "Habana reduction ops don't support casts yet");*/
    AllocateSynapseOutput(graph, output, is_output_persistent);

    std::tie(p_context_->syn_inputs_[0], p_context_->syn_outputs_[0]) =
        CreateReductionGraph(
            graph,
            self,
            std::move(p_context_->syn_inputs_[0]),
            std::move(p_context_->syn_outputs_[0]),
            in_dim_arr,
            keepdim,
            output.scalar_type());
  }
}

std::tuple<synapse_helpers::tensor_or_ref, synapse_helpers::tensor_or_ref>
ReduceOperator::CreateReductionGraph(
    synapse_helpers::graph& graph,
    Tensor& pyt_tensor,
    synapse_helpers::tensor_or_ref syn_tensor_in,
    synapse_helpers::tensor_or_ref syn_tensor_out,
    IntArrayRef in_dim,
    bool keepdim,
    ScalarType dtype) {
  // In the code below syn_helper_intermediate[0] holds the reshaped/original
  // input, syn_helper_intermediate[<last_index>] holds the final output and
  // all others in between holds intermediate output/net-stage-input in the
  // dim by dim reductions.
  std::vector<synapse_helpers::tensor> syn_helper_intermediate;
  std::vector<synTensor> syn_intermediate;
  std::vector<int64_t> pyt_shape = pyt_tensor.sizes().vec();
  // add syn_input tensor
  synapse_helpers::tensor& synInput = syn_tensor_in;
  syn_intermediate.emplace_back(synInput.get());
  // create syn_intermediate tensors of required shape
  unsigned loopend = keepdim ? in_dim.size() - 1 : in_dim.size();
  for (unsigned i = 0; i < loopend; i++) {
    pyt_shape[in_dim[i]] = 1;
    c10::IntArrayRef shape(pyt_shape.data(), pyt_shape.size());
    syn_helper_intermediate.emplace_back(habana_helpers::create_tensor(
        shape,
        graph.get_graph_handle(),
        false,
        pyt_tensor.device().index(),
        dtype));
    syn_intermediate.emplace_back(syn_helper_intermediate[i].get());
  }
  // add syn_output tensor
  synapse_helpers::tensor& synOutput = syn_tensor_out;
  syn_intermediate.emplace_back(synOutput.get());

  // add reduction nodes corresponding to intermediate stages
  std::string node_type = this->guid_;
  for (unsigned i = 0; i < in_dim.size(); i++) {
    ns_Reduction::Params params{};
    params.reductionDimension = pyt_tensor.dim() - in_dim[i] - 1;

    std::vector<synTensor> syn_in{syn_intermediate[i]};
    std::vector<synTensor> syn_out{syn_intermediate[i + 1]};
    graph.add_node(
        std::move(syn_in),
        std::move(syn_out),
        &params,
        sizeof(params),
        std::move(node_type));
  }
  // if dim need not be kept add a final reshape to remove the "1" sized upper
  // dims
  if (!keepdim) {
    std::string node_type = "reshape";
    std::vector<synTensor> syn_in{syn_intermediate[in_dim.size()]};
    std::vector<synTensor> syn_out{syn_intermediate[in_dim.size() + 1]};
    graph.add_node(
        std::move(syn_in),
        std::move(syn_out),
        nullptr,
        0,
        std::move(node_type));
  }

  return std::make_tuple(std::move(syn_tensor_in), std::move(syn_tensor_out));
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

  auto self = inputs[0].toTensor();
  Tensor output = habana_helpers::createPTTensor(
      self,
      {0},
      self.options(),
      self.suggest_memory_format(),
      is_output_persistent);
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
  std::vector<at::Tensor> pt_inputs{self};
  std::vector<c10::IValue> stack = {
      IValue(self), IValue(dim), IValue(keepdim), IValue(dtype)};
  // Create the operator
  SumDimOperator Op(device_id, scalar_type);
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
  std::vector<at::Tensor> pt_inputs{self};
  // Build Params for the graph
  std::vector<c10::IValue> stack = {
      IValue(output),
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
  return output;
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

  Tensor output = habana_helpers::createPTTensor(
      self,
      {0},
      self.options(),
      self.suggest_memory_format(),
      is_output_persistent);
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
  std::vector<at::Tensor> pt_inputs{self};
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
  std::vector<at::Tensor> pt_inputs{self};
  // Build Params for the graph
  std::vector<c10::IValue> stack = {
      IValue(output),
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
  return output;
}

void ProdDimOperator::SetPTOutputs(torch::jit::Stack& inputs) {
  Tensor output;
  auto dim = inputs[1].toInt();
  // Replacing Int value with single element IntList
  IntArrayRef dimArr(&dim, 1);
  inputs[1] = IValue(dimArr);
  inputs.insert(inputs.begin(), IValue(output));
  ReduceOperator::SetPTOutputs(inputs);
}

void ProdDimOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    bool is_output_persistent) {
  TORCH_CHECK(
      inputs.size() == 4,
      "Incorrect size of inputs expected for ProdDim operator");
  TORCH_CHECK(
      inputs[0].isTensor(),
      "Input arg1 expected to be tensor for ProdDim operator");
  TORCH_CHECK(
      inputs[1].isInt(), "Input arg3 expected to be Int for ProdDim operator");
  TORCH_CHECK(
      inputs[2].isBool(),
      "Input arg4 expected to be Bool for ProdDim operator");

  Tensor self = inputs[0].toTensor();
  auto dim = inputs[1].toInt();
  // Replacing Int value with single element IntList
  IntArrayRef dimArr(&dim, 1);
  inputs[1] = IValue(dimArr);

  bool keepdim = inputs[2].toBool();
  auto ndim = self.dim();
  TORCH_CHECK(
      keepdim || 1 != ndim, "Use torch.prod for Reduction to 0d tensor");

  Tensor output = habana_helpers::createPTTensor(
      self,
      {0},
      self.options(),
      self.suggest_memory_format(),
      is_output_persistent);
  inputs.insert(inputs.begin(), IValue(output));

  ReduceOperator::AllocateAndAddSynapseNode(
      graph, inputs, is_output_persistent);
}

Tensor prod_dim_hpu(
    const Tensor& self,
    int64_t dim,
    bool keepdim,
    c10::optional<ScalarType> dtype) {
  PT_KERNEL_BEGIN;

  at::ScalarType scalar_type = self.scalar_type();
  std::string node_type =
      "reduce_prod_fwd_" + habana_helpers::name_suffix_from_type(scalar_type);

  size_t device_id = self.device().index();
  auto& device = synapse_helpers::HPURegistrar::get_device(device_id);
  std::vector<at::Tensor> pt_inputs{self};
  // Build Params for the graph
  std::vector<c10::IValue> stack = {
      IValue(self), IValue(dim), IValue(keepdim), IValue(dtype)};
  // Create the operator
  ProdDimOperator Op(device_id, scalar_type);
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
  Tensor output = habana_helpers::createPTTensor(
      self,
      {0},
      self.options(),
      self.suggest_memory_format(),
      is_output_persistent);

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

Tensor sum_hpu(const Tensor& self_in, c10::optional<ScalarType> dtype) {
  PT_KERNEL_BEGIN;

  // Cast Boolean (I8) inputs to Float since TPC kernel supports only f32 and
  // bf16. Only use-case for sum() on Bool input seen is to count the number of
  // "True" or "False" entries where this solution should be fine.
  auto self = self_in;
  if (self_in.scalar_type() == c10::ScalarType::Bool) {
    self = habana_helpers::hpu_cast_tensor(
        self_in, at::scalarTypeToTypeMeta(c10::ScalarType::Float));
  }

  at::ScalarType scalar_type = self.scalar_type();
  std::string node_type =
      "reduce_sum_fwd_" + habana_helpers::name_suffix_from_type(scalar_type);

  std::vector<at::Tensor> pt_inputs{self};
  std::vector<c10::IValue> stack = {IValue(self), IValue(dtype)};
  size_t device_id = self.device().index();
  auto& device = synapse_helpers::HPURegistrar::get_device(device_id);
  // Create the operator
  SumOperator Op(device_id, scalar_type);
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
  Tensor output = habana_helpers::createPTTensor(
      self,
      {0},
      self.options(),
      self.suggest_memory_format(),
      is_output_persistent);
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
  if (self.dim() == 0) {
    PT_KERNEL_END;
    return self;
  }
  size_t device_id = self.device().index();
  auto& device = synapse_helpers::HPURegistrar::get_device(device_id);
  std::vector<at::Tensor> pt_inputs{self};
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

void ProdOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    bool is_output_persistent) {
  TORCH_CHECK(
      inputs.size() == 2,
      "Incorrect size of inputs expected for Prod operator");
  TORCH_CHECK(
      inputs[0].isTensor(),
      "Input arg1 expected to be tensor for Prod operator");

  Tensor self = inputs[0].toTensor();
  Tensor output = habana_helpers::createPTTensor(
      self,
      {0},
      self.options(),
      self.suggest_memory_format(),
      is_output_persistent);
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

void ProdOperator::SetPTOutputs(torch::jit::Stack& inputs) {
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

Tensor prod_hpu(const Tensor& self, c10::optional<ScalarType> dtype) {
  PT_KERNEL_BEGIN;

  at::ScalarType scalar_type = self.scalar_type();
  std::string node_type =
      "reduce_prod_fwd_" + habana_helpers::name_suffix_from_type(scalar_type);
  if (self.dim() == 0) {
    PT_KERNEL_END;
    return self;
  }
  size_t device_id = self.device().index();
  auto& device = synapse_helpers::HPURegistrar::get_device(device_id);
  std::vector<at::Tensor> pt_inputs{self};
  // Build Params for the graph
  std::vector<c10::IValue> stack = {IValue(self), IValue(dtype)};
  // Create the operator
  ProdOperator Op(device_id, scalar_type);
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
  std::string node_type = "cast_i8_to_f32";

  // Create the operator
  CastOperator intToFloatOp(this->p_context_->device_id_, node_type);
  auto& float_syn =
      intToFloatOp.SetSynapseInput(std::move(p_context_->syn_inputs_[0]));

  // Build Params for the graph
  std::vector<c10::IValue> stack{IValue(self), IValue(c10::ScalarType::Float)};
  intToFloatOp.AllocateAndAddSynapseNode(graph, stack, false);

  synapse_helpers::tensor& float_syn_tensor = intToFloatOp.GetSynOutputs()[0];
  auto output_float = intToFloatOp.GetOutputs()[0];
  p_context_->syn_inputs_[0] = std::move(float_syn);
  stack.clear();

  // Reduction operation - Create the operator
  SumDimOperator sumOp(this->p_context_->device_id_, c10::ScalarType::Float);
  sumOp.SetSynapseInput(std::move(float_syn_tensor));

  // Build Params for the graph
  c10::optional<ScalarType> dtype = output_float.scalar_type();
  stack.emplace_back(IValue(output_float));
  stack.emplace_back(IValue(dim_arr));
  stack.emplace_back(IValue(keepdim));
  stack.emplace_back(IValue(dtype));
  sumOp.AllocateAndAddSynapseNode(graph, stack, false);
  synapse_helpers::tensor& reduce_syn_tensor = sumOp.GetSynOutputs()[0];
  auto output_reduce = sumOp.GetOutputs()[0];
  stack.clear();

  // Resize output tensor based on reduction dimension
  auto ndim = self.dim();
  auto mask = make_dim_mask(dim_arr, ndim);
  allocate_reduction_result(
      output, self, mask, keepdim, c10::ScalarType::Char, is_output_persistent);

  // Cast Reduced Float tensor to Int tensor
  node_type = "cast_f32_to_i8";

  // Create the operator
  CastOutOperator floatToIntOp(this->p_context_->device_id_, node_type);
  floatToIntOp.SetSynapseInput(std::move(reduce_syn_tensor));
  AllocateSynapseOutput(graph, output, is_output_persistent);
  floatToIntOp.SetSynapseInput(std::move(p_context_->syn_outputs_.back()));
  // Build Params for the graph
  stack.emplace_back(IValue(output_reduce));
  stack.emplace_back(IValue(output));

  floatToIntOp.AllocateAndAddSynapseNode(graph, stack, is_output_persistent);
  synapse_helpers::tensor& int_syn_tensor = floatToIntOp.GetSynOutputs()[0];
  p_context_->syn_outputs_.pop_back();
  p_context_->pt_outputs_.pop_back();
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
  std::vector<at::Tensor> pt_inputs{self};
  Op.AllocateSynapseInputs(graph, pt_inputs, true);

  // Build Params for the graph
  std::vector<c10::IValue> stack = {
      IValue(output), IValue(self), IValue(dim), IValue(keepdim)};
  Op.AllocateAndAddSynapseNode(graph, stack, true);

  // compile and execute the graph
  Op.Compile(graph);

  std::vector<at::Tensor> out = Op.GetOutputs();
  TORCH_CHECK(out.size() == 1, "Incorrect size of outputs");

  out.at(0).to(c10::ScalarType::Bool);
  PT_KERNEL_END;
  return output;
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
      inputs[1].isInt(), "Input arg2 expected to be Int for AnyDim operator");
  TORCH_CHECK(
      inputs[2].isBool(), "Input arg3 expected to be Bool for AnyDim operator");

  auto self = inputs[0].toTensor();
  auto output = habana_helpers::createPTTensor(
      self,
      {0},
      self.options(),
      self.suggest_memory_format(),
      c10::ScalarType::Char,
      is_output_persistent);
  inputs.insert(inputs.begin(), IValue(output));

  AnyDimOutOperator::AllocateAndAddSynapseNode(
      graph, inputs, is_output_persistent);
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
  std::vector<at::Tensor> pt_inputs{self};
  Op.AllocateSynapseInputs(graph, pt_inputs, true);

  // Build Params for the graph
  std::vector<c10::IValue> stack = {IValue(self), IValue(dim), IValue(keepdim)};
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
      inputs.size() == 1, "Incorrect size of inputs expected for Any operator");
  TORCH_CHECK(
      inputs[0].isTensor(),
      "Input arg1 expected to be tensor for Any operator");

  auto self = inputs[0].toTensor();

  TORCH_CHECK(
      self.scalar_type() == c10::ScalarType::Bool,
      "Input arg2 expected to be of type Bool Tensor for AnyDimOut operator");

  // Cast Input tensor to Float tensor
  std::string node_type = "cast_i8_to_f32";

  // Create the operator
  CastOperator intToFloatOp(this->p_context_->device_id_, node_type);
  auto& float_syn =
      intToFloatOp.SetSynapseInput(std::move(p_context_->syn_inputs_[0]));

  // Build Params for the graph
  std::vector<c10::IValue> stack{IValue(self), IValue(c10::ScalarType::Float)};
  intToFloatOp.AllocateAndAddSynapseNode(graph, stack, false);
  synapse_helpers::tensor& float_syn_tensor = intToFloatOp.GetSynOutputs()[0];
  auto output_float = intToFloatOp.GetOutputs()[0];
  p_context_->syn_inputs_[0] = std::move(float_syn);
  stack.clear();

  // Reduction operation - Create the operator
  SumOperator sumOp(this->p_context_->device_id_, c10::ScalarType::Float);
  sumOp.SetSynapseInput(std::move(float_syn_tensor));

  // Build Params for the graph
  c10::optional<ScalarType> dtype = output_float.scalar_type();
  stack.emplace_back(IValue(output_float));
  stack.emplace_back(IValue(dtype));
  sumOp.AllocateAndAddSynapseNode(graph, stack, false);
  synapse_helpers::tensor& reduce_syn_tensor = sumOp.GetSynOutputs()[0];
  auto output_reduce = sumOp.GetOutputs()[0];
  stack.clear();

  // Cast Reduced Float tensor to Int tensor
  node_type = "cast_f32_to_i8";

  // Create the operator
  CastOperator floatToIntOp(this->p_context_->device_id_, node_type);
  floatToIntOp.SetSynapseInput(std::move(reduce_syn_tensor));

  // Build Params for the graph
  stack.emplace_back(IValue(output_reduce));
  stack.emplace_back(IValue(c10::ScalarType::Char));
  floatToIntOp.AllocateAndAddSynapseNode(graph, stack, is_output_persistent);
  synapse_helpers::tensor& int_syn_tensor = floatToIntOp.GetSynOutputs()[0];
  p_context_->syn_outputs_.emplace_back(std::move(int_syn_tensor));
  p_context_->pt_outputs_.emplace_back(floatToIntOp.GetOutputs()[0]);
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
  std::vector<at::Tensor> pt_inputs{self};
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

/**
 * @brief This function adds synapse nodes corresponding to
 *aten::_grad_sum_to_size operator
 * @param self - (FP32/BF16) Input tensor
 * @param shape - (IntArray) Shape of output tensor
 **/
void GradSumToSizeOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    bool is_output_persistent) {
  TORCH_CHECK(
      inputs.size() == 2,
      "Incorrect size of inputs expected for _grad_sum_to_size operator");
  TORCH_CHECK(inputs[0].isTensor(), "Input arg1 expected to be tensor");
  TORCH_CHECK(inputs[1].isIntList(), "Input arg2 expected to be tensor");

  auto self = inputs[0].toTensor();
  auto shape = inputs[1].toIntList();
  auto device_id = self.device().index();
  auto scalar_type = self.scalar_type();

  std::vector<int64_t> reduce_dims;
  const at::IntArrayRef sizes = self.sizes();
  const int64_t leading_dims = sizes.size() - shape.size();
  for (int64_t i = 0; i < leading_dims; ++i) {
    reduce_dims.push_back(i);
  }
  for (int64_t i = leading_dims; i < static_cast<int64_t>(sizes.size()); ++i) {
    if (shape[i - leading_dims] == 1 && sizes[i] != 1) {
      reduce_dims.push_back(i);
    }
  }

  SumDimOperator sum_op(device_id, scalar_type);
  if (!reduce_dims.empty()) {
    auto& syn_arg0 =
        sum_op.SetSynapseInput(std::move(p_context_->syn_inputs_[0]));
    torch::jit::Stack stack = {
        IValue(self), IValue(reduce_dims), IValue(true), IValue(scalar_type)};
    sum_op.AllocateAndAddSynapseNode(
        graph, stack, leading_dims ? false : is_output_persistent);
    p_context_->syn_inputs_[0] = std::move(syn_arg0);
  }

  if (leading_dims) {
    ReshapeOperator reshape_op(self.device().index(), self.scalar_type());
    UNUSED auto& syn_arg0 =
        reshape_op.SetSynapseInput(std::move(sum_op.GetSynOutputs()[0]));
    torch::jit::Stack stack = {IValue(sum_op.GetOutputs()[0]), IValue(shape)};
    reshape_op.AllocateAndAddSynapseNode(graph, stack, is_output_persistent);
    p_context_->syn_outputs_.emplace_back(
        std::move(reshape_op.GetSynOutputs()[0]));
    p_context_->pt_outputs_.emplace_back(std::move(reshape_op.GetOutputs()[0]));
  } else {
    if (!reduce_dims.empty()) {
      p_context_->syn_outputs_.emplace_back(
          std::move(sum_op.GetSynOutputs()[0]));
      p_context_->pt_outputs_.emplace_back(std::move(sum_op.GetOutputs()[0]));
    } else {
      // The target shape is identical to the shape of the input tensor.
      // Adding an identity node which results in creation of output as
      // as tensor aliased to input (within GC)
      IdentityOperator identityOp(device_id, scalar_type);
      auto& syn_arg0 =
          identityOp.SetSynapseInput(std::move(p_context_->syn_inputs_[0]));

      torch::jit::Stack stack = {IValue(self)};
      identityOp.AllocateAndAddSynapseNode(graph, stack, is_output_persistent);

      p_context_->syn_inputs_[0] = std::move(syn_arg0);
      p_context_->syn_outputs_.emplace_back(
          std::move(identityOp.GetSynOutputs()[0]));
      p_context_->pt_outputs_.emplace_back(
          std::move(identityOp.GetOutputs()[0]));
    }
  }
}

/*************************************************************************
 * @brief Kernel implementation for aten.all(self)
 * @param self - tensor_0
 ************************************************************************/
Tensor all_hpu(const Tensor& self) {
  PT_KERNEL_BEGIN;
  // create OP graph and populate the stack with inputs
  auto graph = std::make_shared<torch::jit::Graph>();
  const auto graph_string = R"IR(
  graph(%a):
    %b : Tensor = aten::all(%a)
    return (%b))IR";
  torch::jit::parseIR(graph_string, graph.get());
  torch::jit::Stack stack = {IValue(self)};

  habana_lazy::exec::OptPassCfg::GetInstance()->enable_subgraph_rewrite = true;
  habana_lazy::transform_graph(graph);
  habana_lazy::exec::OptPassCfg::GetInstance()->enable_subgraph_rewrite = false;
  // reset instance count so that graph_id always remains same
  // this ensures that we get a cache hit if inputs have not changed
  HabanaLaunchOpPT::instance_count_ = 0;

  // Execute OP graph
  HabanaLaunchOpPT launch{graph, false};
  launch.run(stack);

  // Pop output from stack
  PT_KERNEL_END;
  return stack.back().toTensor();
}

/*************************************************************************
 * @brief Kernel implementation for aten.all(self, dim, keepdim)
 * @param self - tensor_0
 ************************************************************************/
Tensor all_dim_hpu(const Tensor& self, int64_t dim, bool keepdim) {
  PT_KERNEL_BEGIN;
  // create OP graph and populate the stack with inputs
  auto graph = std::make_shared<torch::jit::Graph>();
  const auto graph_string = R"IR(
  graph(%a, %dim : int, %keepdim : bool):
    %b : Tensor = aten::all(%a, %dim, %keepdim)
    return (%b))IR";
  torch::jit::parseIR(graph_string, graph.get());
  torch::jit::Stack stack = {IValue(self), IValue(dim), IValue(keepdim)};

  habana_lazy::exec::OptPassCfg::GetInstance()->enable_subgraph_rewrite = true;
  habana_lazy::transform_graph(graph);
  habana_lazy::exec::OptPassCfg::GetInstance()->enable_subgraph_rewrite = false;
  // reset instance count so that graph_id always remains same
  // this ensures that we get a cache hit if inputs have not changed
  HabanaLaunchOpPT::instance_count_ = 0;
  // Execute OP graph
  HabanaLaunchOpPT launch{graph, false};
  launch.run(stack);

  // Pop output from stack
  PT_KERNEL_END;
  return stack.back().toTensor();
}

void ArgMaxOperator::SetPTOutputs(torch::jit::Stack& inputs) {
  Tensor self = inputs[0].toTensor();
  auto dim = inputs[1].toOptional<int64_t>();
  std::vector<int64_t> dimArr;
  if (dim.has_value()) {
    // Replacing Int value with single element IntList
    dimArr.push_back(dim.value());
  } else {
    auto ndim = self.dim();
    for (int i = 0; i < ndim; i++) {
      dimArr.push_back(i);
    }
  }
  inputs[1] = IValue(dimArr);
  Tensor output = habana_helpers::createPTTensor(
      self,
      {0},
      self.options(),
      self.suggest_memory_format(),
      c10::ScalarType::Int,
      true);
  inputs.insert(inputs.begin(), IValue(output));
  inputs.emplace_back(IValue(output.scalar_type()));
  ReduceOperator::SetPTOutputs(inputs);
}

void ArgMaxOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    bool is_output_persistent) {
  PT_KERNEL_BEGIN;
  TORCH_CHECK(
      inputs.size() == 3,
      "Incorrect size of inputs expected for ArgMax operator");
  TORCH_CHECK(
      inputs[0].isTensor(),
      "Input arg1 expected to be tensor for ArgMax operator");
  TORCH_CHECK(
      inputs[2].isBool(), "Input arg4 expected to be Bool for ArgMax operator");

  Tensor self = inputs[0].toTensor();
  auto dim = inputs[1].toOptional<int64_t>();
  std::vector<int64_t> dimArr;

  if (dim.has_value()) {
    // Replacing Int value with single element IntList
    dimArr.push_back(dim.value());
  } else {
    auto ndim = self.dim();
    for (int i = 0; i < ndim; i++) {
      dimArr.push_back(i);
    }
  }
  inputs[1] = IValue(dimArr);
  Tensor output = habana_helpers::createPTTensor(
      self,
      {0},
      self.options(),
      self.suggest_memory_format(),
      c10::ScalarType::Int,
      is_output_persistent);
  inputs.insert(inputs.begin(), IValue(output));
  inputs.emplace_back(IValue(output.scalar_type()));
  ReduceOperator::AllocateAndAddSynapseNode(
      graph, inputs, is_output_persistent);
  PT_KERNEL_END;
}

Tensor argmax_hpu(
    const Tensor& self,
    c10::optional<int64_t> dim,
    bool keepdim) {
  PT_KERNEL_BEGIN;
  at::ScalarType scalar_type = self.scalar_type();
  std::string node_type =
      "argmax_fwd_" + habana_helpers::name_suffix_from_type(scalar_type);
  size_t device_id = self.device().index();
  auto& device = synapse_helpers::HPURegistrar::get_device(device_id);
  std::vector<at::Tensor> pt_inputs{self};
  // Build Params for the graph
  std::vector<c10::IValue> stack = {IValue(self), IValue(dim), IValue(keepdim)};
  // Create the operator
  ArgMaxOperator Op(device_id, scalar_type);
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

static auto& KernelRegistry =
    ::habana::KernelRegistry()
        .add(
            "aten::_grad_sum_to_size",
            [](const int device_id, c10::ScalarType node_type) {
              return std::make_shared<GradSumToSizeOperator>(
                  device_id, node_type);
            })
        .add(
            "aten::sum",
            [](const int device_id, c10::ScalarType node_type) {
              return std::make_shared<SumOperator>(device_id, node_type);
            })
        .add(
            "aten::sum.dim_IntList",
            [](const int device_id, c10::ScalarType node_type) {
              return std::make_shared<SumDimOperator>(device_id, node_type);
            })
        .add(
            "hpu::sum_dim_IntList",
            [](const int device_id, c10::ScalarType node_type) {
              return std::make_shared<SumDimOperator>(device_id, node_type);
            })
        .add(
            "aten::prod",
            [](const int device_id, c10::ScalarType node_type) {
              return std::make_shared<ProdOperator>(device_id, node_type);
            })
        .add(
            "hpu::prod_dim_Int",
            [](const int device_id, c10::ScalarType node_type) {
              return std::make_shared<ProdDimOperator>(device_id, node_type);
            })
        .add(
            "aten::prod.dim_Int",
            [](const int device_id, c10::ScalarType node_type) {
              return std::make_shared<ProdDimOperator>(device_id, node_type);
            })
        .add(
            "aten::argmax",
            [](const int device_id, c10::ScalarType node_type) {
              return std::make_shared<ArgMaxOperator>(device_id, node_type);
            });
