/*******************************************************************************
 * Copyright (C) 2023-2024 Habana Labs, Ltd. an Intel Company
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
#include "backend/helpers/lowering_util.h"
#include "backend/synapse_helpers/layout_utils.h"
#include "generated/backend/convolution_backward_overrideable.h"
#include "hpu_ops/backend/reduction_template.h"
#include "hpu_ops/common/convolution_gen.h"

using namespace synapse_helpers::layouts;

namespace habana {

static synapse_helpers::tensor ComputeBiasGrad(
    habana::OpBackend* op,
    synapse_helpers::graph& graph,
    bool is_conv_3d,
    at::Tensor& grad_output,
    std::vector<synTensor>&& syn_grad_output,
    synapse_helpers::tensor& ten_output) {
  int channel_dim = is_conv_3d ? INPUT_3D_C_IDX : INPUT_C_IDX;

  int ndims = grad_output.ndimension();
  auto [maskWithoutChannelDim, bitPosChannelDimInTpcOrder] =
      getMaskWithBitPosOutInTpcOrderAndBitPosInTpcOrder(channel_dim, ndims);

  // Check whether all dims in list are the higher "continuous" dimensions
  // if yes, "flatten" higher dims to a single unrolled-size dim.
  // Note-1 that this is an optimization to avoid any precision loss we may
  // get due to separate back 2 back reductions along single dimensions.
  // Note-2 cases such as [0,1,3] where there is in additional dim to reduce
  // in addition to continuous dims is not supported with flattening and falls
  // back to regular flow
  bool flatten_higher_dims =
      ((ndims >= 3) && (bitPosChannelDimInTpcOrder <= 0)) ||
      ((ndims == 2) && (bitPosChannelDimInTpcOrder < 0));

  if (flatten_higher_dims) {
    // TODO: remove or replace if we want to support flatten higher dims
    return std::move(ten_output);
  } else {
    ns_Reduction::ParamsV2 params;
    params.keepDim = false;
    params.reductionDimensionMask = maskWithoutChannelDim;

    auto multi_dim_reduce_sum = OpBackend::BuildNode(
        op,
        graph,
        {get_guid_with_precision("reduce_sum_multi_dim", op->ScalarType()),
         std::move(syn_grad_output),
         {{{grad_output.sizes()[channel_dim]}, grad_output.scalar_type(), 2}},
         &params,
         sizeof(params)});

    return std::move(multi_dim_reduce_sum[0]);
  }
}

static std::shared_ptr<void> SynapseConvParamsBuilder(
    const c10::IntArrayRef& weight, // HWCK
    const c10::IntArrayRef& stride, // HW
    const c10::IntArrayRef& padding, // HW
    const c10::IntArrayRef& dilation, // HW
    int64_t groups,
    size_t& size) {
  PARAMS_STUB(synConvolutionParams);

  params->dH = stride[0];
  params->dW = stride[1];
  params->kH = weight[WEIGHT_KERNEL_R_IDX];
  params->kW = weight[WEIGHT_KERNEL_S_IDX];
  params->dilH = dilation[0];
  params->dilW = dilation[1];
  params->setPadT(padding[0]);
  params->setPadB(padding[0]);
  params->setPadL(padding[1]);
  params->setPadR(padding[1]);
  params->nGroups = groups;

  return params;
}

static std::shared_ptr<void> SynapseConv3dParamsBuilder(
    const c10::IntArrayRef& weight, // DHWCK
    const c10::IntArrayRef& stride, // DHW
    const c10::IntArrayRef& padding, // DHW
    const c10::IntArrayRef& dilation, // DHW
    int64_t groups,
    size_t& size) {
  constexpr uint32_t d_axis = 0;
  constexpr uint32_t h_axis = 1;
  constexpr uint32_t w_axis = 2;

  const int64_t filter_D = weight[WEIGHT_KERNEL_3D_Q_IDX];
  const int64_t filter_H = weight[WEIGHT_KERNEL_3D_R_IDX];
  const int64_t filter_W = weight[WEIGHT_KERNEL_3D_S_IDX];
  const int64_t stride_D = stride[d_axis];
  const int64_t stride_H = stride[h_axis];
  const int64_t stride_W = stride[w_axis];
  const int64_t dilation_D = dilation[d_axis];
  const int64_t dilation_H = dilation[h_axis];
  const int64_t dilation_W = dilation[w_axis];

  PARAMS_STUB(synConvolution3DParams);

  params->kernel[CONV_KERNEL_WIDTH] = filter_W;
  params->kernel[CONV_KERNEL_HEIGHT] = filter_H;
  params->kernel[CONV_KERNEL_DEPTH] = filter_D;
  params->stride[CONV_STRIDE_WIDTH] = stride_W;
  params->stride[CONV_STRIDE_HEIGHT] = stride_H;
  params->stride[CONV_STRIDE_DEPTH] = stride_D;
  params->dilation[CONV_DIL_WIDTH] = dilation_W;
  params->dilation[CONV_DIL_HEIGHT] = dilation_H;
  params->dilation[CONV_DIL_DEPTH] = dilation_D;
  params->padding[CONV_PAD_LEFT] = padding[w_axis];
  params->padding[CONV_PAD_RIGHT] = padding[w_axis];
  params->padding[CONV_PAD_TOP] = padding[h_axis];
  params->padding[CONV_PAD_BOTTOM] = padding[h_axis];
  params->padding[CONV_PAD_FRONT] = padding[d_axis];
  params->padding[CONV_PAD_BACK] = padding[d_axis];
  params->nGroups = groups;

  return params;
}

static std::shared_ptr<void> FillConvolutionBackwardOverrideableParams(
    size_t input_rank,
    std::vector<int64_t> weight_shape,
    std::vector<int64_t> stride,
    std::vector<int64_t> padding,
    std::vector<int64_t> dilation,
    int64_t groups,
    size_t& size) {
  if (input_rank == 3) {
    weight_shape.push_back(1);
    stride.push_back(1);
    padding.push_back(0);
    dilation.push_back(1);
  }

  if (input_rank == 5) {
    return SynapseConv3dParamsBuilder(
        weight_shape, stride, padding, dilation, groups, size);
  } else {
    return SynapseConvParamsBuilder(
        weight_shape, stride, padding, dilation, groups, size);
  }
}

static OutputMetaData CreateMetaData(
    const at::Tensor& input,
    const c10::List<bool>& output_mask_in,
    const int index) {
  OutputMetaData meta;

  meta.shape = input.sizes().vec();
  meta.dtype = input.scalar_type();
  meta.mem_format = input.suggest_memory_format();
  meta.undefined = output_mask_in.size() && !output_mask_in.get(index);

  return meta;
}

OutputMetaDataVector ConvolutionOverrideableMetaBwd(const at::Stack& stack) {
  auto grad_output = stack_tensor(stack, 0);
  auto input = stack_tensor(stack, 1);
  auto weight = stack_tensor(stack, 2);
  auto output_mask_in = c10::List<bool>();

  auto elem = stack.at(9);
  if (!elem.isBoolList())
    elem = stack.at(10);
  if (elem.isBoolList())
    output_mask_in = elem.toBoolList();

  OutputMetaData input_meta = CreateMetaData(input, output_mask_in, 0);
  OutputMetaData weight_meta = CreateMetaData(weight, output_mask_in, 1);
  OutputMetaData grad_output_meta =
      CreateMetaData(grad_output, output_mask_in, 2);
  grad_output_meta.shape = std::vector<int64_t>{grad_output_meta.shape[1]};
  grad_output_meta.mem_format = at::MemoryFormat::Contiguous;

  return {input_meta, weight_meta, grad_output_meta};
}

static int64_t ComputeOutputSize(
    const int64_t input_dim,
    const int64_t padding,
    const int64_t dilation,
    const int64_t kernel_size,
    const int64_t stride,
    const bool transposed) {
  if (!transposed) {
    return (input_dim + 2 * padding - dilation * (kernel_size - 1) - 1) /
        stride +
        1;
  } else {
    // conv2d fwd output shape computation done as per formula provided below
    // https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose2d.html#torch.nn.ConvTranspose2d
    return (input_dim - 1) * stride - 2 * padding +
        dilation * (kernel_size - 1) + 1;
  }
}

void ConvolutionBackwardOverrideable::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  size_t params_size = 0;
  at::Tensor grad_output = stack_tensor(stack, 0); // Result of convolution fwd
  at::Tensor input = stack_tensor(stack, 1);
  at::Tensor weight = stack_tensor(stack, 2);

  // Both convolution_backward_overrideable and convolution_backward ops
  // are implemented by this backend. The difference in these two is that
  // the latter takes additional argument at idx 3, which, as pytorch docs
  // says:
  //
  // bias_sizes_opt: if specified, indicates that a bias was used in the forward
  // pass and contains the shape
  //   of the bias. While the bias shape can be computed from other inputs, it
  //   is provided to this function for ease of use. The bias shape is
  //   (weight.shape[0]) for normal convolution and (weight.shape[1] * groups)
  //   for transposed convolution.
  //
  // Since it's not needed, it's just being ignored below, by shifting the rest
  // of inputs' indices.
  const int index_shift =
      GetGuid().find("convolution_backward") != std::string::npos ? 1 : 0;
  const auto stride = stack[3 + index_shift].toIntList().vec();
  const auto padding = stack[4 + index_shift].toIntList().vec();
  const auto dilation = stack[5 + index_shift].toIntList().vec();
  const bool transposed = stack[6 + index_shift].toBool();
  const int64_t groups = stack[8 + index_shift].toInt();
  const auto output_mask_in = stack[9 + index_shift].toBoolList();

  const bool is_conv_1d = input.dim() == 3;
  const bool is_conv_3d = input.dim() == 5;

  const auto output_meta = ConvolutionOverrideableMetaBwd(stack);
  auto out0_shape = output_meta[0].shape;
  auto out1_shape = output_meta[1].shape;

  if (is_conv_1d) {
    out0_shape.push_back(1);
    out1_shape.push_back(1);
  }

  const auto& params = FillConvolutionBackwardOverrideableParams(
      input.dim(),
      weight.sizes().vec(),
      stride,
      padding,
      dilation,
      groups,
      params_size);
  // In case of dynamic graph and dry run check if the input and output sizes
  // are valid fix for SW-94417
  if (graph.is_dynamic_graph() && graph.is_dry_run()) {
    auto shape_in = input.sizes().vec();
    auto shape_wt = weight.sizes().vec();
    auto K = transposed ? shape_wt[1] * groups : shape_wt[0];
    std::vector<int64_t> out_shape{shape_in[0], K};
    for (size_t i = 0; i < shape_in.size() - 2; ++i) {
      out_shape.push_back(ComputeOutputSize(
          shape_in[i + 2],
          padding[i],
          dilation[i],
          shape_wt[i + 2],
          stride[i],
          transposed));
    }
    auto grad_output_sizes = grad_output.sizes().vec();
    bool validateRes = std::equal(
        out_shape.begin(),
        out_shape.end(),
        grad_output_sizes.begin(),
        grad_output_sizes.end());
    TORCH_CHECK(
        validateRes,
        "Mismatch in Grad Out size{",
        grad_output_sizes,
        "} and calculated size{",
        out_shape,
        "} according to input and params");
  }

  auto BuildOpFor =
      [&](synapse_helpers::graph& graph,
          std::string&& guid,
          bool is_conv_3d,
          std::vector<synTensor>&& node_inputs,
          std::vector<NodeAttr::NodeOutputAttr>&& node_output_attr,
          void* params) {
        if (guid.find("spatial_convolution") != std::string::npos ||
            guid.find("dedx") != std::string::npos) {
          if (is_conv_3d) {
            std::vector<synapse_helpers::layouts::SynapseLayoutFormat>
                input_layouts = {
                    synapse_helpers::layouts::SynapseLayoutFormat::WHDCN,
                    synapse_helpers::layouts::SynapseLayoutFormat::SRQCK,
                    synapse_helpers::layouts::SynapseLayoutFormat::WHDCN};
            if (graph.is_dynamic_graph() &&
                guid.find("dedx") != std::string::npos) {
              input_layouts.emplace_back(
                  synapse_helpers::layouts::SynapseLayoutFormat::WHDCN);
            }
            SetSynapseLayouts(
                input_layouts,
                {synapse_helpers::layouts::SynapseLayoutFormat::WHDCN});
          } else {
            std::vector<synapse_helpers::layouts::SynapseLayoutFormat>
                input_layouts = {
                    synapse_helpers::layouts::SynapseLayoutFormat::WHCN,
                    synapse_helpers::layouts::SynapseLayoutFormat::SRCK};
            if (graph.is_dynamic_graph() &&
                guid.find("dedx") != std::string::npos) {
              input_layouts.emplace_back(
                  synapse_helpers::layouts::SynapseLayoutFormat::WHCN);
            }
            SetSynapseLayouts(
                input_layouts,
                {synapse_helpers::layouts::SynapseLayoutFormat::WHCN});
          }
        } else if (guid.find("dedw") != std::string::npos) {
          if (is_conv_3d) {
            SetSynapseLayouts(
                {synapse_helpers::layouts::SynapseLayoutFormat::WHDCN,
                 synapse_helpers::layouts::SynapseLayoutFormat::WHDCN,
                 synapse_helpers::layouts::SynapseLayoutFormat::WHDCN},
                {synapse_helpers::layouts::SynapseLayoutFormat::SRQCK});
          } else {
            SetSynapseLayouts(
                {synapse_helpers::layouts::SynapseLayoutFormat::WHCN,
                 synapse_helpers::layouts::SynapseLayoutFormat::WHCN},
                {synapse_helpers::layouts::SynapseLayoutFormat::SRCK});
          }
        }

        guid = get_guid_with_precision(guid, ScalarType());

        return std::move(BuildOp(
                             graph,
                             std::move(guid),
                             std::move(node_inputs),
                             std::move(node_output_attr),
                             params,
                             params_size)
                             .at(0));
      };

  std::string guid;

  IF_CONV1D_EXPAND_TO_2D(grad_output, 0);
  IF_CONV1D_EXPAND_TO_2D(input, 1);
  IF_CONV1D_EXPAND_TO_2D(weight, 2);

#define COND_FINAL_RES_IDX(condition, false_val)                      \
  condition ? c10::optional<int>{c10::nullopt} : c10::optional<int> { \
    false_val                                                         \
  }
  if (transposed) {
    if (output_mask_in[0]) {
      guid = "spatial_convolution";
      auto convOp = BuildOpFor(
          graph,
          std::move(guid),
          is_conv_3d,
          {grad_output_expanded, weight_expanded},
          {{out0_shape, ScalarType(), COND_FINAL_RES_IDX(is_conv_1d, 0)}},
          params.get());

      IF_CONV1D_SQUEEZE_TO_ORIG_AND_SET_OUT(convOp, out0_shape, 0);
    } else {
      AddUndefinedOutputTensor();
    }

    if (output_mask_in[1]) {
      guid = is_conv_3d ? "dedw3d" : "dedw";
      auto dedwOp = BuildOpFor(
          graph,
          std::move(guid),
          is_conv_3d,
          {input_expanded, grad_output_expanded},
          {{out1_shape, ScalarType(), COND_FINAL_RES_IDX(is_conv_1d, 1)}},
          params.get());

      IF_CONV1D_SQUEEZE_TO_ORIG_AND_SET_OUT(dedwOp, out1_shape, 1);
    } else {
      AddUndefinedOutputTensor();
    }
  } else {
    if (output_mask_in[0]) {
      std::vector syn_inputs = {grad_output_expanded, weight_expanded};

      // Allocate Shape Tensor
      CreateShapeTensorInput(graph, ScalarType(), out0_shape, syn_inputs);

      guid = is_conv_3d ? "dedx3d" : "dedx";
      auto convOp = BuildOpFor(
          graph,
          std::move(guid),
          is_conv_3d,
          std::move(syn_inputs),
          {{out0_shape, ScalarType(), COND_FINAL_RES_IDX(is_conv_1d, 0)}},
          params.get());

      IF_CONV1D_SQUEEZE_TO_ORIG_AND_SET_OUT(convOp, out0_shape, 0);
    } else {
      AddUndefinedOutputTensor();
    }

    if (output_mask_in[1]) {
      guid = is_conv_3d ? "dedw3d" : "dedw";
      auto dedwOp = BuildOpFor(
          graph,
          std::move(guid),
          is_conv_3d,
          {grad_output_expanded, input_expanded},
          {{out1_shape, ScalarType(), COND_FINAL_RES_IDX(is_conv_1d, 1)}},
          params.get());

      IF_CONV1D_SQUEEZE_TO_ORIG_AND_SET_OUT(dedwOp, out1_shape, 1);
    } else {
      AddUndefinedOutputTensor();
    }
  }

  // Bias grad computation is the same for conv2d bwd and conv2d_transpose bwd
  if (output_mask_in[2]) {
    SetSynapseLayouts({}, {});
    auto biasRes = ComputeBiasGrad(
        this, graph, is_conv_3d, grad_output, {syn_in(0)}, syn_out(2));
    syn_out(2) = std::move(biasRes);
  } else {
    AddUndefinedOutputTensor();
  }
}

} // namespace habana
