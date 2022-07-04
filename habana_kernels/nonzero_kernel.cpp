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

#include "habana_bridge/kernel/hpu_shape_inference.h"
#include "habana_device/HPUCheck.h"
#include "habana_device/hpu_cached_devices.h"
#include "habana_helpers/logging.h"
#include "habana_helpers/tensor_utils.h"
#include "habana_kernels/compare_kernels.h"
#include "habana_kernels/index_kernels.h"
#include "habana_kernels/kernel_utils.h"
#include "habana_kernels/nonzero_kernel.h"
#include "habana_kernels/simple_generic_kernel.h"

using namespace torch;
using namespace habana;

void NonZeroOperator::SetPTOutputs(torch::jit::Stack& inputs) {
  TORCH_CHECK(
      inputs.size() == 1,
      "Incorrect size of inputs expected for NonZero operator");
  TORCH_CHECK(
      inputs[0].isTensor(),
      "Input arg0 expected to be tensor for NonZero operator");

  auto self = inputs[0].toTensor();
  auto input_shape = self.sizes();
  int dimensions = input_shape.size();
  int elements = self.numel();
  auto output_shape = DimVector{elements, dimensions};
  auto shape_tensor_shape = DimVector{5};
  // Create PT output stage 2
  auto cordinates_of_true = habana_helpers::createPTTensor(
      self,
      output_shape,
      self.options(),
      self.suggest_memory_format(),
      c10::ScalarType::Int,
      true);
  auto shape_tensor = habana_helpers::createPTTensor(
      self,
      shape_tensor_shape,
      self.options(),
      self.suggest_memory_format(),
      c10::ScalarType::Int,
      true);

  std::vector<at::Tensor> outputs{cordinates_of_true, shape_tensor};
  HabanaOperator::SetPTOutputs(outputs);
}

float NonZeroOperator::round_dims(
    const at::Tensor& input_tensor,
    int group_size) {
  auto group_size_f = static_cast<float>(group_size);
  auto last_dim_rounded =
      std::ceil(input_tensor.sizes()[input_tensor.dim() - 1] / group_size_f) *
      group_size_f;
  return last_dim_rounded;
}

std::vector<int64_t> NonZeroOperator::compute_output_st_shape(
    const at::Tensor& input_tensor) {
  constexpr int group_size = 64;
  auto last_dim_rounded = NonZeroOperator::round_dims(input_tensor, group_size);
  auto out_st_shape = input_tensor.sizes().vec();
  auto group_size_aligned_dim =
      (long int)last_dim_rounded / (long int)group_size;
  out_st_shape.pop_back();
  out_st_shape.emplace_back(group_size_aligned_dim);
  out_st_shape.emplace_back(group_size);
  return out_st_shape;
}

std::vector<int64_t> NonZeroOperator::compute_output_shape(
    const at::Tensor& self) {
  auto input_shape = self.sizes();
  int dimensions = input_shape.size();
  auto elements = self.numel();
  if ((synapse_helpers::HPURegistrar::get_device().type() !=
       synDeviceType::synDeviceGreco) and
      (self.dim() <= 4) and (self.dim() > 0)) {
    elements = 1;
    auto last_dim_rounded = round_dims(self, 64);
    for (unsigned i = 0; i < self.sizes().size() - 1; i++) {
      elements *= self.sizes()[i];
    }
    elements = elements * last_dim_rounded;
  }
  std::vector<int64_t> output_shape{elements, dimensions};
  return output_shape;
}

OutputShapeInfRetType NonZeroOperator::ComputeOutputShape(
    torch::jit::Stack& inputs) {
  auto self = inputs[0].toTensor();
  if (self.dim() > 4) {
    auto output_shape = compute_output_shape(self);
    std::vector<int64_t> shape_tensor_shape = {5};
    OutputShapeInfRetType out;
    auto metaData = TensorMetaData(
        output_shape,
        HabanaOperator::CalculateStrides(
            output_shape, self.suggest_memory_format()),
        self.scalar_type(),
        self.suggest_memory_format());

    out.AddOutputTensor(metaData);
    auto shape_metaData = TensorMetaData(
        shape_tensor_shape,
        HabanaOperator::CalculateStrides(
            shape_tensor_shape, self.suggest_memory_format()),
        self.scalar_type(),
        self.suggest_memory_format());
    out.AddShapeTensor(shape_metaData);
    return out;

  } else {
    SetGuid(
        "non_zero_v2_fwd_" +
        habana_helpers::name_suffix_from_type(self.scalar_type()));
    OutputShapeInfRetType out;
    // (i) This output_describing_shape_tensor is created to be used by
    // "reshape" node within CGUID. This should be created within CGUID in
    // future. (ii) This shape tensor should not be created in as part of
    // accumulation (lazy_kernels) else relationship between input tensor and
    // shape tensor st = f(input) is not preserved in all cases (e.g. min, max
    // shape inference with Calculated or Local Historic policies). (iii)
    // Creating shape tensor in back-end kernel is ok for cases where shape
    // tensor is strictly a function of another input tensor(s) and not a scalar
    // value coming from framework.
    // (iv) Please consult with vgoel@habana.ai before removing or modifying
    // this shape_tensor.
    auto st_shape = compute_output_st_shape(self);

    auto shape_metaData1 = TensorMetaData(
        st_shape,
        HabanaOperator::CalculateStrides(
            st_shape, self.suggest_memory_format()),
        self.scalar_type(),
        self.suggest_memory_format());
    out.AddShapeTensor(shape_metaData1);
    auto output_shape = compute_output_shape(self);
    std::vector<int64_t> shape_tensor_shape = {5};

    auto metaData = TensorMetaData(
        output_shape,
        HabanaOperator::CalculateStrides(
            output_shape, self.suggest_memory_format()),
        self.scalar_type(),
        self.suggest_memory_format());

    out.AddOutputTensor(metaData);
    auto metaData2 = TensorMetaData(
        shape_tensor_shape,
        HabanaOperator::CalculateStrides(
            shape_tensor_shape, self.suggest_memory_format()),
        self.scalar_type(),
        self.suggest_memory_format());
    out.AddOutputTensor(metaData2);
    return out;
  }
}

void NonZeroOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    const OutputMetaDataVector& output_metadata) {
  TORCH_CHECK(
      inputs.size() == 2,
      "Incorrect size of inputs expected for NonZero operator");
  TORCH_CHECK(
      inputs[0].isTensor(),
      "Input arg0 expected to be tensor for NonZero operator");
  TORCH_CHECK(
      output_metadata.size() == 2,
      "output_metadata expected to be vector of size 2");

  auto self = inputs[0].toTensor();
  if ((synapse_helpers::HPURegistrar::get_device().type() ==
       synDeviceType::synDeviceGreco) or
      (self.dim() > 4)) {
    auto output_shape = compute_output_shape(self);
    auto shape_tensor_shape = DimVector{5};
    auto cordinates_of_true = habana_helpers::createPTTensor(
        self,
        output_shape,
        self.options(),
        self.suggest_memory_format(),
        c10::ScalarType::Int,
        output_metadata.at(0).persistent);
    auto shape_tensor = habana_helpers::createPTTensor(
        self,
        shape_tensor_shape,
        self.options(),
        self.suggest_memory_format(),
        c10::ScalarType::Int,
        output_metadata.at(1).persistent);
    // shape_tensor is of type UINT32 not supported by ScalarType, use
    // synDataType
    synDataType synType = syn_type_uint32;
    AllocateSynapseOutput(graph, cordinates_of_true, output_metadata.at(0));
    AllocateSynapseOutput(
        graph,
        shape_tensor,
        synType,
        output_metadata.at(1),
        graph.is_dynamic_graph() ? true : false);
    AddNodeToSynapseGraph(graph, nullptr, 0);
  } else {
    SetGuid(
        "non_zero_v2_fwd_" +
        habana_helpers::name_suffix_from_type(self.scalar_type()));

    // (i) This output_describing_shape_tensor is created to be used by
    // "reshape" node within CGUID. This should be created within CGUID in
    // future. (ii) This shape tensor should not be created in as part of
    // accumulation (lazy_kernels) else relationship between input tensor and
    // shape tensor st = f(input) is not preserved in all cases (e.g. min, max
    // shape inference with Calculated or Local Historic policies). (iii)
    // Creating shape tensor in back-end kernel is ok for cases where shape
    // tensor is strictly a function of another input tensor(s) and not a
    // scalar value coming from framework. (iv) Please consult with
    // vgoel@habana.ai before removing or modifying this shape_tensor.
    auto st_shape = compute_output_st_shape(self);
    Tensor reshape_shape_tensor = habana_helpers::createPTTensor(
        self, st_shape, self.options(), self.suggest_memory_format(), false);
    AllocateSynapseShapeTensor(graph, reshape_shape_tensor);

    auto output_shape = compute_output_shape(self);
    auto shape_tensor_shape = DimVector{5};
    auto cordinates_of_true = habana_helpers::createPTTensor(
        self,
        output_shape,
        self.options(),
        self.suggest_memory_format(),
        c10::ScalarType::Int,
        output_metadata.at(0).persistent);
    auto shape_tensor = habana_helpers::createPTTensor(
        self,
        shape_tensor_shape,
        self.options(),
        self.suggest_memory_format(),
        c10::ScalarType::Int,
        output_metadata.at(1).persistent);
    // shape_tensor is of type UINT32 not supported by ScalarType, use
    // synDataType
    synDataType synType = syn_type_uint32;
    AllocateSynapseOutput(graph, cordinates_of_true, output_metadata.at(0));
    AllocateSynapseOutput(
        graph, shape_tensor, synType, output_metadata.at(1), false);

    ns_NonzeroV2::Params params = {};
    params.group_size = 64;
    AddNodeToSynapseGraph(graph, &params, sizeof(params));
  }
}

/*************************************************************************
 * @brief Kernel implementation for torch.nonzero operator
 * @param self - Input tensor
 ************************************************************************/
Tensor nonzero_hpu(const Tensor& self_in) {
  PT_KERNEL_BEGIN;

  at::ScalarType scalar_type = self_in.scalar_type();
  auto input_shape = self_in.sizes();
  int dimensions = input_shape.size();

  // Output required of type Int64
  if (self_in.numel() == 0) {
    auto shape = DimVector{0, dimensions};
    auto output = habana_helpers::createPTTensor(
        self_in,
        shape,
        self_in.options(),
        self_in.suggest_memory_format(),
        c10::ScalarType::Long,
        true);
    PT_KERNEL_END;
    return output;
  }
  auto self = habana_helpers::downcast_to_int_if_needed(self_in);
  std::string node_type =
      "non_zero_fwd_" + habana_helpers::name_suffix_from_type(scalar_type);
  size_t device_id = self.device().index();
  auto& device = synapse_helpers::HPURegistrar::get_device(device_id);

  // Create the operator
  NonZeroOperator Op(device_id, self.scalar_type());
  std::vector<at::Tensor> pt_inputs{self};
  std::vector<c10::IValue> stack = {IValue(self)};

  size_t key = Op.GetRecipeKey(node_type, stack);

  if (device.get_recipe_handle_cache().isCached(key)) {
    Op.Execute(key, pt_inputs, stack);
  } else {
    // Create Graph
    OutputMetaDataVector output_metadata(2);
    output_metadata.at(0).persistent = true;
    output_metadata.at(1).persistent = true;
    Op.CreateGraphAndCompile(key, pt_inputs, stack, output_metadata, true);
  }

  std::vector<at::Tensor> out = Op.GetOutputs();
  TORCH_CHECK(out.size() == 2, "Incorrect size of outputs");
  // Slice the output to select only relevant data
  // The shape tensor = [Dimension_detail, num_relevantElement, N/U, N/U, N/U]
  // Since output tensor[elements,Dimension] already has correct dimension
  // we need to slice along row to get relevant elements to outputs
  // i.e slice the output from where_stage_2 of shape (elementsxDimensions)
  // to get correct output (relevantElementsxDimensions)
  auto end = out.at(1)[1].item<int64_t>();
  if (end == 0) {
    out.at(0).unsafeGetTensorImpl()->set_sizes_and_strides(
        IntArrayRef{0, dimensions}, IntArrayRef{1, 1});
    PT_KERNEL_END;
    return out.at(0);
  }
  auto result = out.at(0).slice(0, 0, end, 1);
  // Remove this cast once index_put_ implementation using scatter_nd is
  // available
  auto output = habana_helpers::cast_tensor_to_long(result);
  PT_KERNEL_END;
  return output;
}

static auto& NonZeroKernelRegistry =
    habana::KernelRegistry().add("hpu::nonzero", KERNEL_FN(NonZeroOperator));
