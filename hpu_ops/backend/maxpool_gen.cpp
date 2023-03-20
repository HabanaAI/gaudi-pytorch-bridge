/*******************************************************************************
 * Copyright (C) 2022-2023 Habana Labs, Ltd. an Intel Company
 * All Rights Reserved.
 *
 * Unauthorized copying of this file or any element(s) within it, via any medium
 * is strictly prohibited.
 * This file contains Habana Labs, Ltd. proprietary and confidential information
 * and is subject to the confidentiality and license agreements under which it
 * was provided.
 *
 *******************************************************************************
 */

#include "backend/helpers/create_tensor.h"
#include "backend/helpers/tensor_utils.h"
#include "generated/backend/max_pool2d_with_indices.h"
#include "generated/backend/max_pool2d_with_indices_backward.h"
#include "generated/backend/max_pool3d_with_indices.h"
#include "generated/backend/max_pool3d_with_indices_backward.h"

namespace habana {

enum MaxpoolVariant {
  MAXPOOL2D = 2,
  MAXPOOL3D = 3,
};

static bool is_greco_device() {
  return (
      synapse_helpers::HPURegistrar::get_device().type() ==
      synDeviceType::synDeviceGreco);
}

static void DummyOutput(
    synapse_helpers::graph& graph,
    PytorchKernelContextPtr& p_context_,
    bool persistent,
    bool external) {
  p_context_->syn_outputs_.emplace_back(habana_helpers::create_tensor(
      p_context_->pt_outputs_.at(1), graph, persistent, external));
}

static int OutputShapeComputation(
    int input_shape,
    int kernel,
    int stride,
    int padding,
    int dilation,
    bool ceilMode) {
  return (
      ((input_shape + 2 * padding - dilation * (kernel - 1) - 1 +
        (ceilMode ? stride - 1 : 0)) /
       stride) +
      1);
}

sizes_vec MaxPool3DIndicesOutputShape(const at::Stack& stack) {
  std::vector<long int> pad = {0, 0, 0};
  std::vector<long int> dil = {1, 1, 1};
  auto self = stack.at(0).toTensor();
  auto kernel = stack.at(1).toIntVector();
  auto stride = stack.at(2).toIntVector().size() == 0
      ? kernel
      : stack.at(2).toIntVector();
  auto padding =
      stack.at(3).toIntVector().size() == 0 ? pad : stack.at(3).toIntVector();
  auto dilation =
      stack.at(4).toIntVector().size() == 0 ? dil : stack.at(4).toIntVector();
  const bool ceil_mode = stack.at(5).toBool();

  TORCH_CHECK(
      self.dim() == 5 || self.dim() == 4,
      "Maxpool3d expects Input size must be 5 or 4, but got ",
      self.dim());
  TORCH_CHECK(
      padding.size() == 3,
      "Maxpool3d expects padding size is 3 but got ",
      padding.size());
  TORCH_CHECK(
      kernel.size() == 3,
      "Maxpool3d expects kernel size is 3 but got ",
      kernel.size());
  TORCH_CHECK(
      stride.size() == 3,
      "Maxpool3d expects stride size is 3 but got ",
      stride.size());
  TORCH_CHECK(
      dilation.size() == 3,
      "Maxpool3d expects dilation size is 3 but got ",
      dilation.size());

  std::vector<int64_t> input_shape = self.sizes().vec();
  std::vector<int64_t> output_shape = self.sizes().vec();

  int n = kernel.size();
  // updating the width, height, & depth dimension
  for (int i = 0; i < n; i++) {
    output_shape.rbegin()[i] = OutputShapeComputation(
        input_shape.rbegin()[i],
        kernel[n - i - 1],
        stride[n - i - 1],
        padding[n - i - 1],
        dilation[n - i - 1],
        ceil_mode);
  }

  // ensure that the last pooling starts inside the image
  // needed to avoid problems in ceil mode
  if (ceil_mode) {
    for (int i = 0; i < n; i++) {
      if ((output_shape.rbegin()[i] - 1) * stride[n - i - 1] >=
          input_shape.rbegin()[i] + padding[n - i - 1])
        --output_shape.rbegin()[i];
    }
  }
  return {output_shape, output_shape};
}

sizes_vec MaxPool2DOutputShape(const at::Stack& stack) {
  std::vector<long int> pad = {0, 0};
  std::vector<long int> dil = {1, 1};
  auto self = stack.at(0).toTensor();
  auto kernel = stack.at(1).toIntVector();
  auto stride = stack.at(2).toIntVector().size() == 0
      ? kernel
      : stack.at(2).toIntVector();
  auto padding =
      stack.at(3).toIntVector().size() == 0 ? pad : stack.at(3).toIntVector();
  auto dilation =
      stack.at(4).toIntVector().size() == 0 ? dil : stack.at(4).toIntVector();
  const bool ceil_mode = stack.at(5).toBool();
  TORCH_CHECK(
      self.dim() == 4 || self.dim() == 3,
      "Maxpool2d expects Input size must be 4 or 3, but got ",
      self.dim());
  TORCH_CHECK(
      kernel.size() == 2,
      "Maxpool2d expects Kernel size must 2, but got ",
      kernel.size());
  TORCH_CHECK(
      stride.size() == 2,
      "Maxpool2d expects Stride size must 2, but got ",
      stride.size());
  TORCH_CHECK(
      padding.size() == 2,
      "Maxpool2d expects Padding size must 2, but got ",
      padding.size());
  TORCH_CHECK(
      dilation.size() == 2,
      "Maxpool2d expects Dilation size must 2, but got ",
      dilation.size());
  std::vector<int64_t> input_shape = self.sizes().vec();
  std::vector<int64_t> output_shape = self.sizes().vec();

  int n = kernel.size();
  // updating the width & height dimension
  for (int i = 0; i < n; i++) {
    output_shape.rbegin()[i] = OutputShapeComputation(
        input_shape.rbegin()[i],
        kernel[n - i - 1],
        stride[n - i - 1],
        padding[n - i - 1],
        dilation[n - i - 1],
        ceil_mode);
  }

  // ensure that the last pooling starts inside the image
  // needed to avoid problems in ceil mode
  if (ceil_mode) {
    for (int i = 0; i < n; i++) {
      if ((output_shape.rbegin()[i] - 1) * stride[n - i - 1] >=
          input_shape.rbegin()[i] + padding[n - i - 1])
        --output_shape.rbegin()[i];
    }
  }

  return {output_shape, output_shape};
}

sizes_vec MaxPoolOutputShapeBwd(const at::Stack& stack) {
  auto self = stack.at(1).toTensor();
  std::vector<int64_t> input_shape = self.sizes().vec();
  return {input_shape};
}

// This method is used to apply transpose in the given input shape.
// which is expected by TPC.
static std::vector<int64_t> TransposeShape(
    std::vector<int64_t> input_shape,
    MaxpoolVariant variant) {
  // variant = 3 for maxpool3d and variant = 2 for maxpool2d
  if (variant == MaxpoolVariant::MAXPOOL3D) {
    if (input_shape.size() == 5) {
      // Converting N C D H W to N D H W C
      std::vector<int64_t> output_shape = {
          input_shape[0],
          input_shape[2],
          input_shape[3],
          input_shape[4],
          input_shape[1]};
      return {output_shape};
    } else {
      // Converting C D H W to D H W C
      std::vector<int64_t> output_shape = {
          input_shape[1], input_shape[2], input_shape[3], input_shape[0]};
      return {output_shape};
    }
  } else {
    if (input_shape.size() == 4) {
      // Converting N C H W to N H W C
      std::vector<int64_t> output_shape = {
          input_shape[0], input_shape[2], input_shape[3], input_shape[1]};
      return {output_shape};
    } else {
      // Converting C H W to H W C
      std::vector<int64_t> output_shape = {
          input_shape[1], input_shape[2], input_shape[0]};
      return {output_shape};
    }
  }
}

static std::shared_ptr<void> FillSpatialReduction3DParams(
    std::vector<int64_t>& kernel,
    std::vector<int64_t>& stride,
    std::vector<int64_t>& padding,
    std::vector<int64_t>& dilation,
    bool ceil_mode,
    size_t& size) {
  PARAMS_STUB(ns_SpatialReduction3D::Params);
  params->pad_w_begin = padding[2];
  params->pad_w_end = padding[2];
  params->pad_h_begin = padding[1];
  params->pad_h_end = padding[1];
  params->pad_d_begin = padding[0];
  params->pad_d_end = padding[0];
  params->kernel_w = kernel[2];
  params->kernel_h = kernel[1];
  params->kernel_d = kernel[0];
  params->stride_w = stride[2];
  params->stride_h = stride[1];
  params->stride_d = stride[0];
  params->dilation_w = dilation[2];
  params->dilation_h = dilation[1];
  params->dilation_d = dilation[0];
  if (ceil_mode)
    params->pooling_convention =
        EPoolingConvention::POOLING_CONVENTION_FULL_PYTORCH;
  else
    params->pooling_convention = EPoolingConvention::POOLING_CONVENTION_VALID;
  return params;
}

std::shared_ptr<void> FillSpatialReduction3DParamsFwd(
    const at::Stack& stack,
    size_t& size) {
  std::vector<long int> pad = {0, 0, 0};
  std::vector<long int> dil = {1, 1, 1};
  auto kernel = stack.at(1).toIntVector();
  auto stride = stack.at(2).toIntVector().size() == 0
      ? kernel
      : stack.at(2).toIntVector();
  auto padding =
      stack.at(3).toIntVector().size() == 0 ? pad : stack.at(3).toIntVector();
  auto dilation =
      stack.at(4).toIntVector().size() == 0 ? dil : stack.at(4).toIntVector();
  const bool ceil_mode = stack.at(5).toBool();

  return FillSpatialReduction3DParams(
      kernel, stride, padding, dilation, ceil_mode, size);
}

std::shared_ptr<void> FillSpatialReduction3DParamsBwd(
    const at::Stack& stack,
    size_t& size) {
  std::vector<long int> pad = {0, 0, 0};
  std::vector<long int> dil = {1, 1, 1};
  auto kernel = stack.at(2).toIntVector();
  auto stride = stack.at(3).toIntVector().size() == 0
      ? kernel
      : stack.at(3).toIntVector();
  auto padding =
      stack.at(4).toIntVector().size() == 0 ? pad : stack.at(4).toIntVector();
  auto dilation =
      stack.at(5).toIntVector().size() == 0 ? dil : stack.at(5).toIntVector();
  const bool ceil_mode = stack.at(6).toBool();

  return FillSpatialReduction3DParams(
      kernel, stride, padding, dilation, ceil_mode, size);
}

static std::shared_ptr<void> FillSpatialReduction2DParams(
    std::vector<int64_t>& kernel,
    std::vector<int64_t>& stride,
    std::vector<int64_t>& padding,
    std::vector<int64_t>& dilation,
    bool ceil_mode,
    size_t& size) {
  PARAMS_STUB(ns_SpatialReduction::Params);
  params->pad_w_begin = padding[1];
  params->pad_w_end = padding[1];
  params->pad_h_begin = padding[0];
  params->pad_h_end = padding[0];
  params->kernel_w = kernel[1];
  params->kernel_h = kernel[0];
  params->stride_w = stride[1];
  params->stride_h = stride[0];
  params->dilation_w = dilation[1];
  params->dilation_h = dilation[0];
  if (ceil_mode)
    params->pooling_convention =
        EPoolingConvention::POOLING_CONVENTION_FULL_PYTORCH;
  else
    params->pooling_convention = EPoolingConvention::POOLING_CONVENTION_VALID;
  return params;
}

std::shared_ptr<void> FillSpatialReduction2DParamsFwd(
    const at::Stack& stack,
    size_t& size) {
  std::vector<long int> pad = {0, 0};
  std::vector<long int> dil = {1, 1};
  auto kernel = stack.at(1).toIntVector();
  auto stride = stack.at(2).toIntVector().size() == 0
      ? kernel
      : stack.at(2).toIntVector();
  auto padding =
      stack.at(3).toIntVector().size() == 0 ? pad : stack.at(3).toIntVector();
  auto dilation =
      stack.at(4).toIntVector().size() == 0 ? dil : stack.at(4).toIntVector();
  const bool ceil_mode = stack.at(5).toBool();

  return FillSpatialReduction2DParams(
      kernel, stride, padding, dilation, ceil_mode, size);
}

std::shared_ptr<void> FillSpatialReduction2DParamsBwd(
    const at::Stack& stack,
    size_t& size) {
  std::vector<long int> pad = {0, 0};
  std::vector<long int> dil = {1, 1};
  auto kernel = stack.at(2).toIntVector();
  auto stride = stack.at(3).toIntVector().size() == 0
      ? kernel
      : stack.at(3).toIntVector();
  auto padding =
      stack.at(4).toIntVector().size() == 0 ? pad : stack.at(4).toIntVector();
  auto dilation =
      stack.at(5).toIntVector().size() == 0 ? dil : stack.at(5).toIntVector();
  const bool ceil_mode = stack.at(6).toBool();

  return FillSpatialReduction2DParams(
      kernel, stride, padding, dilation, ceil_mode, size);
}

static std::vector<synapse_helpers::tensor> ShapeTranspose(
    OpBackend* op,
    synapse_helpers::graph& graph,
    std::vector<synTensor> input,
    std::vector<int64_t> output_shape,
    at::ScalarType scalar_type,
    synTransposeParams trans_params,

    c10::optional<int> is_final_node = c10::nullopt,
    std::string name = std::string()) {
  return OpBackend::BuildNode(
      op,
      graph,
      {"transpose",
       {input.at(0)},
       {{output_shape, scalar_type, is_final_node}},
       &trans_params,
       sizeof(trans_params),
       name});
}

static synTransposeParams GenerateTransposePermutation(int dim) {
  synTransposeParams trans_params{};
  trans_params.tensorDim = dim;
  for (int i = 0; i < dim; ++i) {
    trans_params.permutation[i] = static_cast<TransposePermutationDim>(i);
  }
  return trans_params;
}

static synTransposeParams ChangeTransposePermutation(
    synTransposeParams trans_params,
    std::vector<int>& permutation_order_list,
    int dim) {
  for (int i = 0; i < dim; ++i) {
    trans_params.permutation[i] =
        static_cast<TransposePermutationDim>(permutation_order_list[i]);
  }
  return trans_params;
}

static c10::ScalarType FindIndexType(c10::ScalarType input_tensor_type) {
  if (input_tensor_type == c10::ScalarType::BFloat16)
    return c10::ScalarType::Short;
  return c10::ScalarType::Byte;
}

static std::vector<std::vector<int>> GetTransposePermutationOrder(
    MaxpoolVariant variant,
    int dim) {
  std::vector<std::vector<int>> permutation_order = {{}, {}};
  if (variant == MaxpoolVariant::MAXPOOL3D) {
    if (dim == 5) {
      // N C D H W to N D H W C permutation order = 3, 0, 1, 2, 4
      permutation_order[0] = {3, 0, 1, 2, 4};
      // N D H W C to N C D H W permutation order = 1, 2, 3, 0, 4
      permutation_order[1] = {1, 2, 3, 0, 4};
    } else {
      // C D H W to D H W C permutation order = 3, 0, 1, 2
      permutation_order[0] = {3, 0, 1, 2};
      // N D H W C to N C D H W permutation order = 1, 2, 3, 0,
      permutation_order[1] = {1, 2, 3, 0};
    }
  } else {
    if (dim == 4) {
      // N C H W to N H W C permutation order = 2, 0, 1, 3
      permutation_order[0] = {2, 0, 1, 3};
      // C H W to H W C permutation order = 2, 0, 1
      permutation_order[1] = {1, 2, 0, 3};
    } else {
      // N H W C to N C H W premutation order = 1, 2, 0, 3
      permutation_order[0] = {2, 0, 1};
      // H W C to C H W premutation order = 1, 2, 0
      permutation_order[1] = {1, 2, 0};
    }
  }
  return permutation_order;
}

static std::vector<synapse_helpers::tensor> Maxpool3dWithIndicesFwdCommonFunc(
    OpBackend* op,
    synapse_helpers::graph& graph,
    const at::Stack& stack,
    std::vector<synTensor> input,
    const c10::ScalarType& scalar_type) {
  const torch::Tensor& self = stack.at(0).toTensor();
  const auto& final_out_shape = MaxPool3DIndicesOutputShape(stack);
  size_t size = 0;
  const auto& params = FillSpatialReduction3DParamsFwd(stack, size);
  auto index_type = FindIndexType(self.scalar_type());
  std::vector<synapse_helpers::tensor> output;
  // TODO: SW-86955 move build op to code gen

  // maxpool3d guid will return tuple of tensors (indices tensor, output
  // tensor)
  auto maxpool3d = OpBackend::BuildNode(
      op,
      graph,
      {"maxpool_3d_fwd_" + habana_helpers::name_suffix_from_type(scalar_type),
       {input.at(0)},
       {{final_out_shape[0], index_type, 0},
        {final_out_shape[0], scalar_type, 1}},
       params.get(),
       size});

  // It's reversed, as the calling function expects it this way
  output.emplace_back(std::move(maxpool3d.at(1)));
  output.emplace_back(std::move(maxpool3d.at(0)));
  return output;
}

// Since the out varriant intices tensor has some issue
// (https://jira.habana-labs.com/browse/SW-74263)
void MaxPool3DWithIndicesOut::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  auto output = Maxpool3dWithIndicesFwdCommonFunc(
      this, graph, stack, {syn_in(0)}, ScalarType());

  syn_out(0) = std::move(output.at(0));
  syn_out(1) = std::move(output.at(1));

  // Maxpool3d with indices fwd Output  order should be like (Output Tensor,
  // Output Index) so swapping is needed.
  std::swap(p_context_->pt_outputs_[0], p_context_->pt_outputs_[1]);
}

void MaxPool3DWithIndicesBwd::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  const auto& out_shape = ComputeOutputShapes(stack);
  size_t size = 0;
  const auto& params = FillParams(stack, size);
  std::vector<synTensor> grad = {syn_in(0), syn_in(2)};
  this->CreateShapeTensorInput(graph, this->ScalarType(), out_shape[0], grad);

  auto grad_output = BuildOp(
      graph,
      "maxpool_3d_bwd_" + habana_helpers::name_suffix_from_type(ScalarType()),
      grad,
      {{out_shape[0], ScalarType(), 0}},
      params.get(),
      size);

  syn_out(0) = std::move(grad_output.at(0));
}

// Since the out varriant intices tensor has some issue
// (https://jira.habana-labs.com/browse/SW-74263)
void MaxPool2DWithIndicesOut::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  const auto& out_shape = ComputeOutputShapes(stack);
  const torch::Tensor& self = stack.at(0).toTensor();
  size_t size = 0;
  const auto& params = FillParams(stack, size);

  auto index_type = FindIndexType(self.scalar_type());
  std::string name = std::string();
  if (GET_ENV_FLAG_NEW(PT_HPU_INFERENCE_MODE))
    name = habana_helpers::get_tensor_range(syn_in(0), graph);

  // maxpool2d guid will return tuple of tensors except greco device
  // (indices tensor, output tensor)
  // For greco device, only `output tensor` will be returned

  const bool greco_device = is_greco_device();
  const bool is_dynamic = GET_ENV_FLAG_NEW(PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES);

  std::vector<NodeAttr::NodeOutputAttr> output_attr;
  int64_t maxpool_out_index;
  if (greco_device) {
    p_context_->syn_outputs_.pop_back();
    // dummy output in place of indices tensor
    DummyOutput(
        graph,
        p_context_,
        IsOutputPersistent(1),
        GetOutputMetaData(1).external);
    if (is_dynamic)
      output_attr.push_back({out_shape[0], ScalarType()});
    else
      output_attr.push_back({out_shape[0], ScalarType(), 0});
    maxpool_out_index = 0;
  } else {
    if (is_dynamic) {
      output_attr.push_back({out_shape[1], index_type});
      output_attr.push_back({out_shape[0], ScalarType()});
    } else {
      output_attr.push_back({out_shape[1], index_type, 1});
      output_attr.push_back({out_shape[0], ScalarType(), 0});
    }
    maxpool_out_index = 1;
  }

  auto maxpool2d = BuildOp(
      graph,
      "maxpool_2d_fwd_" + habana_helpers::name_suffix_from_type(ScalarType()),
      {syn_in(0)},
      output_attr,
      params.get(),
      size,
      name);

  // Identity Kernel was added in dynamic case alone
  // Without this, we will get tensor missing error.
  if (is_dynamic) {
    auto output = BuildOp(
        graph,
        "identity",
        {maxpool2d[maxpool_out_index].get()},
        {{out_shape[maxpool_out_index], ScalarType(), 0}});
    syn_out(0) = std::move(output.at(0));
  } else {
    syn_out(0) = std::move(maxpool2d[maxpool_out_index]);
  }

  if (!greco_device) {
    if (is_dynamic) {
      auto output_indx = BuildOp(
          graph,
          "identity",
          {maxpool2d[0].get()},
          {{out_shape[1], index_type, 1}});
      syn_out(1) = std::move(output_indx.at(0));
    } else {
      syn_out(1) = std::move(maxpool2d[0]);
    }
  }
}

// Since the out varriant intices tensor has some issue
// (https://jira.habana-labs.com/browse/SW-74263)
void MaxPool2DWithIndicesBwd::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  const auto& out_shape = ComputeOutputShapes(stack);
  size_t size = 0;
  const auto& params = FillParams(stack, size);

  std::vector<synTensor> grad = {syn_in(0), syn_in(2)};
  this->CreateShapeTensorInput(graph, this->ScalarType(), out_shape[0], grad);

  auto maxpool2d_gradout = BuildOp(
      graph,
      "maxpool_2d_bwd_" + habana_helpers::name_suffix_from_type(ScalarType()),
      grad,
      {{out_shape[0], ScalarType(), 0}},
      params.get(),
      size);

  syn_out(0) = std::move(maxpool2d_gradout.at(0));
}
} // namespace habana
