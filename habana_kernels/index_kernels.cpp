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
#include "habana_kernels/simple_generic_kernel.h"
#include "kernel_utils.h"

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
  std::vector<const at::Tensor*> pt_inputs{&index_expanded};
  std::vector<const at::Tensor*> pt_outputs{&index_broadcast};
  synapse_simple_generic_kernel(
      pt_outputs, pt_inputs, "broadcast", nullptr, 0, SynapsePassType::NO_PASS);

  return index_broadcast;
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

  TORCH_CHECK(sparse_grad == false, "spare_grad is not supported")

  auto dim = at::maybe_wrap_dim(dim_, self.dim(), /*wrap_scalar=*/true);

  auto index_int = habana_helpers::cast_tensor_to_integer(index);

  auto shape = DimVector(self.sizes());
  shape.erase(shape.begin() + dim);
  shape.insert(shape.begin() + dim, index.numel());
  auto output = at::empty(shape, self.options(), self.suggest_memory_format());

  ns_GatherKernel::Params params;
  params.axis = self.dim() - dim - 1;

  std::vector<const at::Tensor*> pt_inputs{&self, &index_int};
  std::vector<const at::Tensor*> pt_outputs{&output};

  synapse_simple_generic_kernel(
      pt_outputs,
      pt_inputs,
      "gather",
      &params,
      sizeof(params),
      SynapsePassType::FORWARD_PASS);

  PT_KERNEL_END;
  return output;
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

  std::vector<const at::Tensor*> pt_inputs{&self, &index, &src};

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

/*************************************************************************
 * @brief Kernel implementation for torch.index_select(input, dim, index) →
 *Tensor
 * @param self - Input tensor 1-4D bf16/fp32
 * @param dim - The dimension in which we index
 * @param index - 1D tensor containing the indices to index
 ************************************************************************/
Tensor index_select_hpu(const Tensor& self, int64_t dim, const Tensor& index) {
  PT_KERNEL_BEGIN;

  TORCH_CHECK(index.dim() <= 1, "index tensor cannot be more than 1D")
  // Convert index tensor from 0D to 1D if required
  if (index.dim() == 0) {
    index.unsafeGetTensorImpl()->set_sizes_and_strides({1}, {1});
  }

  dim = at::maybe_wrap_dim(dim, self.dim(), /*wrap_scalar=*/true);

  auto output = self.gather(dim, index, false);

  PT_KERNEL_END;
  return output;
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
  auto output =
      at::empty(shape, input.options(), input.suggest_memory_format());

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
  std::vector<const at::Tensor*> pt_inputs{
      &input, &indices_int, &validCount_int};
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
    int64_t& step) {
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
  auto output = at::empty(shape, self.options(), self.suggest_memory_format());

  return output;
}

void SliceOperator::SetPTOutputs(const torch::jit::Stack& inputs) {
  auto self = inputs[0].toTensor();
  auto dim = inputs[1].toInt();
  auto start = inputs[2].toInt();
  auto end = inputs[3].toInt();
  auto step = inputs[4].toInt();
  auto output = AllocateOutputTensor(self, dim, start, end, step);
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

  auto output = AllocateOutputTensor(self, dim, start, end, step);
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

  at::ScalarType scalar_type = self.scalar_type();
  std::string node_type = "slice";
  size_t device_id = self.device().index();
  auto& device = synapse_helpers::HPURegistrar::get_device(device_id);

  SliceOperator Op(device_id, scalar_type);
  std::vector<const at::Tensor*> pt_inputs{&self};
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
                .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA));