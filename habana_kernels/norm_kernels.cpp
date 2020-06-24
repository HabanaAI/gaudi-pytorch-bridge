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
#include <perf_lib_layer_params.h>
#include <torch/script.h>
#include <memory>

#include "habana_device/HPUCheck.h"
#include "habana_device/hpu_cached_devices.h"
#include "habana_helpers/tensor_utils.h"
#include "habana_helpers/unused_macro.h"
#include "habana_kernels/kernel_utils.h"
#include "habana_kernels/resize.h"
#include "habana_kernels/simple_generic_kernel.h"

using namespace torch;

/**********************************************************************
*@brief Changes dimensions of the input tensor as per specified dimension.
*This is done by adding dummy x1 dimensions. Eg: NC -> NCHW is done by
* resizing to NxCx1x1. Resizing is done inplace
@param input - Tensor, 2D/3D/4D, float/bf16
@param num_out_dim - uint, Range ~[2,4]
***********************************************************************/

Tensor batch_norm_resize(
    const Tensor& input,
    uint num_out_dim,
    c10::MemoryFormat memory_format) {
  auto num_in_dim = input.dim();
  Tensor input_resize = at::alias(input);

  auto shape = DimVector(input_resize.sizes());
  auto strides = DimVector(input_resize.strides());
  switch (memory_format) {
    case c10::MemoryFormat::ChannelsLast: {
      if (num_out_dim > num_in_dim) {
        auto last = shape.back();
        shape.pop_back();
        // Create view_sizes initialized to part which has size=1 for upper dims
        auto view_sizes = std::vector<int64_t>(num_out_dim - num_in_dim, 1);
        // and append to shape
        shape.insert(shape.end(), view_sizes.begin(), view_sizes.end());
        shape.push_back(last);
        input_resize = input_resize.view(shape);
      } else {
        // Remove the additional x1 dimensions
        // TODO: The logic here won't work when size of any intermediate
        // (non-start,end) dimensions is 1
        std::vector<int64_t> new_shape;
        new_shape.push_back(shape[0]);
        for (uint cnt = 1; cnt < num_out_dim - 1; cnt++) {
          if (1 != shape.back())
            new_shape.push_back(shape[cnt]);
        }
        new_shape.push_back(shape[num_out_dim - 1]);
        input_resize = input_resize.view(new_shape);
      }
      break;
    }
    case c10::MemoryFormat::Contiguous: {
      if (num_out_dim > num_in_dim) {
        // Create view_sizes initialized to part which has size=1 for upper dims
        auto view_sizes = std::vector<int64_t>(num_out_dim - num_in_dim, 1);
        // and append to shape
        shape.insert(shape.end(), view_sizes.begin(), view_sizes.end());
        input_resize = input_resize.view(shape);
      } else {
        // Remove the additional x1 dimensions
        std::vector<int64_t> new_shape;
        for (uint cnt = 0; cnt < num_out_dim; cnt++) {
          new_shape.push_back(shape[cnt]);
        }
        input_resize = input_resize.view(new_shape);
      }
      break;
    }
    default:
      TORCH_CHECK(
          false,
          "Unsupported memory format. Supports only ChannelsLast, Contiguous");
  }
  return input_resize;
}

/**********************************************************************
*@brief Pushes the optional tensor to device if defined.Otherwise,
* create an empty device tensor
@param input - Optional 1D tensor
@param size - size of 1D tensor. used in case the tensor is not defined
@param device - Device param
@param output - 1D HPU tensor
**********************************************************************/

inline Tensor get_batch_norm_optional_tensors(
    const Tensor& input,
    uint size,
    Device device) {
  Tensor output;
  if (input.defined() == true) {
    output = input.to(DeviceType::HABANA);
  } else {
    output = at::empty(
        {size}, TensorOptions().dtype(c10::ScalarType::Float).device(device));
  }
  return output;
}

/*******************************************************************
*@brief Implements forward pass for batch norm
*INPUTS
@param input - IFM, 2D/3D/4D, bf16/FP32, NHWC
@param weight - Gamma in PyT, 1D, FP32, C (optional)
@param bias - beta in PyT, 1D, FP32, C (optional)
@param running_mean - Filtered mean 1D, FP32, C (optional)
@param running_var - Filtered variance, 1D, FP32, C (optional)
@param momentum - Update factor, float
@param eps - float (for numerical stability during
divisions)
*OUTPUTS
@param output - OFM, 2D/3D/4D, bf16/FP32, NHWC
@param running_mean_out - Updated filtered mean 1D, FP32, C
@param running_var_out - Updated filtered variance 1D, FP32, C
*UNUSED variables will be enabled later once evaluation/inference mode is
*implemented
*******************************************************************/

std::tuple<Tensor, Tensor, Tensor> batch_norm_hpu(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& bias,
    const Tensor& running_mean,
    const Tensor& running_var,
    bool training,
    double momentum,
    double eps) {
  PT_KERNEL_BEGIN;

  // Build Params for the graph
  std::vector<c10::IValue> stack = {IValue(input),
                                    IValue(weight),
                                    IValue(bias),
                                    IValue(running_mean),
                                    IValue(running_var),
                                    IValue(training),
                                    IValue(momentum),
                                    IValue(eps)};

  auto num_input_dim = input.dim();
  TORCH_CHECK(num_input_dim > 1, "Expected range of input dimensions is [2,4]");
  c10::MemoryFormat memory_format = habana_helpers::get_memory_format({&input});
  // Resize input to 4D to match TPC kernel requirement.
  auto input_resize = batch_norm_resize(input, 4, memory_format);
  Tensor input_nhwc = input_resize;
  std::vector<const at::Tensor*> pt_in = {&input_resize};
  std::vector<at::Tensor*> pt_out = {&input_nhwc};
  IntArrayRef new_dim_pos = {0, 2, 3, 1};
  std::vector<const IntArrayRef*> pt_new_pos = {&new_dim_pos};
  habana_helpers::change_tensors_to_memory_format(
      pt_out, pt_in, pt_new_pos, memory_format);

  // The following tensors are autogenerated by pytorch. Hence need to be
  // pushed to device first. The below operations can be removed in graph
  // mode when the intermediate tensors lives in the device
  // AFAIK, in training mode, all the optional tensors are received.
  // Nevertheless, if we don't receive the optional tensors from PyT, create
  // empty ones to satify TPC kernel input constraints

  Tensor wt_hpu =
      get_batch_norm_optional_tensors(weight, input.sizes()[1], input.device());

  Tensor bias_hpu =
      get_batch_norm_optional_tensors(bias, input.sizes()[1], input.device());

  Tensor running_mean_hpu = get_batch_norm_optional_tensors(
      running_mean, input.sizes()[1], input.device());

  Tensor running_var_hpu = get_batch_norm_optional_tensors(
      running_var, input.sizes()[1], input.device());
  auto output_nhwc = at::empty(input_nhwc.sizes(), input_nhwc.options());
  std::vector<const at::Tensor*> pt_outputs{&output_nhwc};

  auto current_mean =
      at::empty(running_mean_hpu.sizes(), running_mean_hpu.options());
  auto current_istd =
      at::empty(running_mean_hpu.sizes(), running_mean_hpu.options());

  if (training == true) {
    std::vector<const at::Tensor*> pt_inputs{
        &input_nhwc,
        &wt_hpu,
        &bias_hpu,
    };

    Tensor running_mean_hpu_in, running_var_hpu_in, residualAdd;

    if (running_mean.defined()) {
      running_mean_hpu_in =
          at::empty(running_mean_hpu.sizes(), running_mean_hpu.options());

      running_var_hpu_in =
          at::empty(running_var_hpu.sizes(), running_var_hpu.options());

      residualAdd = at::empty(input_nhwc.sizes(), input_nhwc.options());

      // This residual add is dummy tensor to match the API requirements
      pt_inputs.push_back(&residualAdd);

      // running mean and running var cannot be in input and output list
      // simultaneously. create a copy
      habana_helpers::copy_data_within_device(
          running_mean_hpu, running_mean_hpu_in);

      habana_helpers::copy_data_within_device(
          running_var_hpu, running_var_hpu_in);

      pt_inputs.push_back(&running_mean_hpu_in);
      pt_inputs.push_back(&running_var_hpu_in);

      pt_outputs.push_back(&running_mean_hpu);
      pt_outputs.push_back(&running_var_hpu);

      pt_outputs.push_back(&current_mean);
      pt_outputs.push_back(&current_istd);
    }
    // synapse uses expAvgfactor = 1 - momentum
    struct synCudBnExParams param = {synBnOps::BN_OPS_BN,
                                     static_cast<float>(1 - momentum),
                                     static_cast<float>(eps)};

    synapse_simple_generic_kernel(
        pt_outputs,
        pt_inputs,
        "cud_bn_fwd_ex",
        &param,
        sizeof(param),
        SynapsePassType::NO_PASS);
  } else {
    // training=false - Evaluation mode
    std::vector<const at::Tensor*> pt_inputs{
        &input_nhwc,
        &bias_hpu,
        &wt_hpu,
        &running_mean_hpu,
        &running_var_hpu,
    };

    struct ns_BatchNormKernel::Params param;
    param.threshold.f = 0.0;
    param.momentum = momentum;
    param.epsilon = eps;

    synapse_simple_generic_kernel(
        pt_outputs,
        pt_inputs,
        "batch_norm_inf",
        &param,
        sizeof(param),
        SynapsePassType::NO_PASS);
  }
  Tensor output = output_nhwc;
  pt_in = {&output_nhwc};
  pt_out = {&output};
  new_dim_pos = {0, 3, 1, 2};
  pt_new_pos = {&new_dim_pos};
  habana_helpers::change_tensors_to_memory_format(
      pt_out, pt_in, pt_new_pos, memory_format);
  // Resize output
  auto output_resized = batch_norm_resize(output, num_input_dim, memory_format);

  PT_KERNEL_END;

  return std::make_tuple(output_resized, current_mean, current_istd);
}

/*******************************************************************
*@brief Implements backward pass for batch norm
*INPUTS
@param grad_out - output gradient tensor, 2D/3D/4D, bf16/FP32, NHWC
@param input - IFM, 2D/3D/4D, bf16/FP32, NHWC
@param weight - Gamma in PyT, 1D, FP32, C (optional)
@param running_mean - Filtered mean 1D, FP32, C (optional)
@param running_var - Filtered variance, 1D, FP32, C (optional)
@param save_mean - saved mean from fwd pass, 1D, FP32, C
@param save_invstd - saved inverse variance from fwd pass, 1D, FP32, C

*OUTPUTS
@param grad_in - input gradient tensor, 2D/3D/4D, bf16/FP32, NHWC
@param grad_gamma - gradient of weight 1D, FP32, C
@param grad_beta - gradient of bias 1D, FP32, C
*UNUSED variables will be enabled later once evaluation/inference mode is
*implemented
*******************************************************************/

std::tuple<Tensor, Tensor, Tensor> batch_norm_bwd_hpu(
    Tensor& grad_out,
    Tensor& input,
    Tensor& weight,
    UNUSED Tensor& running_mean,
    UNUSED Tensor& running_var,
    Tensor& save_mean,
    Tensor& save_invstd,
    bool train,
    double eps,
    UNUSED std::array<bool, 3> output_mask) {
  PT_KERNEL_BEGIN;

  bool output_mask_in[3];
  output_mask_in[0] = output_mask[0];
  output_mask_in[1] = output_mask[1];
  output_mask_in[2] = output_mask[2];
  // Build Params for the graph
  std::vector<c10::IValue> stack = {IValue(grad_out),
                                    IValue(input),
                                    IValue(weight),
                                    IValue(running_mean),
                                    IValue(running_var),
                                    IValue(save_mean),
                                    IValue(save_invstd),
                                    IValue(train),
                                    IValue(eps),
                                    IValue(output_mask_in)};

  auto num_input_dim = input.dim();
  TORCH_CHECK(num_input_dim > 1, "Expected range of input dimensions is [2,4]");
  TORCH_CHECK(
      num_input_dim == grad_out.dim(),
      "Grad out dimension not matching that of input");
  c10::MemoryFormat memory_format = habana_helpers::get_memory_format({&input});
  // Resize input and grad in to 4D to match TPC kernel requirement.
  auto input_resize = batch_norm_resize(input, 4, memory_format);
  auto grad_out_resize = batch_norm_resize(grad_out, 4, memory_format);

  Tensor input_nhwc = input_resize;
  Tensor grad_out_nhwc = grad_out_resize;
  std::vector<const at::Tensor*> pt_in{&input_resize, &grad_out_resize};
  std::vector<at::Tensor*> pt_out{&input_nhwc, &grad_out_nhwc};
  IntArrayRef new_dim_pos = {0, 2, 3, 1};
  std::vector<const IntArrayRef*> pt_new_pos{&new_dim_pos, &new_dim_pos};
  habana_helpers::change_tensors_to_memory_format(
      pt_out, pt_in, pt_new_pos, memory_format);

  Tensor wt_hpu =
      get_batch_norm_optional_tensors(weight, input.sizes()[1], input.device());
  Tensor bias_hpu = at::zeros(wt_hpu.sizes(), wt_hpu.options());
  Tensor save_mean_hpu = get_batch_norm_optional_tensors(
      save_mean, input.sizes()[1], input.device());
  Tensor save_invstd_hpu = get_batch_norm_optional_tensors(
      save_invstd, input.sizes()[1], input.device());
  std::vector<const at::Tensor*> pt_inputs{
      &input_nhwc,
      &grad_out_nhwc,
      &wt_hpu,
      &bias_hpu,
      &save_mean_hpu,
      &save_invstd_hpu,
  };
  // Prepare output tensor vector
  auto grad_in_nhwc = at::empty(input_nhwc.sizes(), input_nhwc.options());
  auto grad_beta = at::empty(wt_hpu.sizes(), wt_hpu.options());
  auto grad_gamma = at::empty(wt_hpu.sizes(), wt_hpu.options());

  std::vector<const at::Tensor*> pt_outputs{
      &grad_in_nhwc, &grad_gamma, &grad_beta};

  struct synCudBnExParams param = {
      synBnOps::BN_OPS_BN, 0, static_cast<float>(eps)};

  synapse_simple_generic_kernel(
      pt_outputs,
      pt_inputs,
      "cud_bn_bwd_ex",
      &param,
      sizeof(param),
      SynapsePassType::NO_PASS);
  Tensor grad_in = grad_in_nhwc;
  pt_in = {&grad_in_nhwc};
  pt_out = {&grad_in};
  new_dim_pos = {0, 3, 1, 2};
  pt_new_pos = {&new_dim_pos};
  habana_helpers::change_tensors_to_memory_format(
      pt_out, pt_in, pt_new_pos, memory_format);
  // Resize output
  auto grad_in_resized =
      batch_norm_resize(grad_in, num_input_dim, memory_format);

  PT_KERNEL_END;
  return std::make_tuple(grad_in_resized, grad_gamma, grad_beta);
}

/** @brief This function implements forward pass for torch.nn.LayerNorm()
 * @param input (bf16/fp32 tensor) input tensor
 * @param weight (fp32 tensor) per element scale value tensor
 * @param bias (fp32 tensor) per element bias value tensor
 * @param m (int) num of elements in outer dims not used in LayerNorm
 * @param n (int) num of elements used for computing LayerNorm
 * @param eps (double) a value added to the denominator for numerical stability.
 * Default: 1e-5
 */
std::tuple<Tensor, Tensor, Tensor> layer_norm_hpu(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& bias,
    int64_t m,
    int64_t n,
    double eps) {
  PT_KERNEL_BEGIN;

  auto wt_reshaped = weight.view(-1);
  auto bias_reshaped = bias.view(-1);
  std::vector<int64_t> shape{m, n};
  auto input_reshaped = input.view(shape);

  std::vector<const at::Tensor*> pt_inputs{
      &input_reshaped, &bias_reshaped, &wt_reshaped};

  std::vector<int64_t> shape_mean{m, 1};
  IntArrayRef meanArray(shape_mean.data(), shape_mean.size());
  auto output = at::empty(input_reshaped.sizes(), input_reshaped.options());
  auto mean = at::empty(meanArray, wt_reshaped.options());
  auto istd = at::empty(meanArray, bias_reshaped.options());

  std::vector<const at::Tensor*> pt_outputs{&output, &mean, &istd};

  struct ns_LayerNormKernel::Params param;
  param.eps = static_cast<float>(eps);
  param.epsValid = true;

  synapse_simple_generic_kernel(
      pt_outputs,
      pt_inputs,
      "layer_norm",
      &param,
      sizeof(param),
      SynapsePassType::FORWARD_PASS);

  auto output_reshaped = output.view(input.sizes().vec());

  PT_KERNEL_END;
  return std::make_tuple(
      std::move(output_reshaped), std::move(mean), std::move(istd));
}

/*************************************************************************
 * @brief Kernel implementation for LP Norm (Frobenius norm) kernel
          output = torch.norm(self, p=2)
 * @param [in] self - input tensor, 1-4D, FP32/BF16
 * @param [in] output - output tensor, 1-4D, FP32/BF16
 * @param [in] p - optional input, default = 2
 ************************************************************************/
Tensor norm_scalar_hpu(const Tensor& self, Scalar p) {
  PT_KERNEL_BEGIN;

  TORCH_CHECK(p.toFloat() > 0.0, "norm with p > 0.0 is only supported");

  auto self_hpu = self.view(-1);
  auto output = at::empty(self_hpu.sizes(), self.options());
  auto retain = at::empty(self_hpu.sizes(), self.options());

  ns_LpNormKernel::Params params{};
  params.p = p.to<float>();
  params.dim = 0;
  params.eps = 1e-5;

  std::vector<const at::Tensor*> pt_inputs{&self_hpu};
  std::vector<const at::Tensor*> pt_outputs{&output, &retain};

  synapse_simple_generic_kernel(
      pt_outputs,
      pt_inputs,
      "lpnorm",
      &params,
      sizeof(params),
      SynapsePassType::FORWARD_PASS);

  at::reciprocal_(retain);

  // PT expects 0-D
  retain.unsafeGetTensorImpl()->set_sizes_and_strides({}, {});

  PT_KERNEL_END;
  return retain;
}

static auto registry =
    torch::RegisterOperators()
        .op(torch::RegisterOperators::options()
                .schema(
                    "aten::native_batch_norm(Tensor input, Tensor? weight, Tensor? bias, Tensor? running_mean, Tensor? running_var, bool training, float momentum, float eps) -> (Tensor, Tensor, Tensor)")
                .impl_unboxedOnlyKernel<
                    decltype(batch_norm_hpu),
                    &batch_norm_hpu>(DispatchKey::HABANATensorId)
                .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
        .op(torch::RegisterOperators::options()
                .schema(
                    "aten::native_batch_norm_backward(Tensor grad_out, Tensor input, Tensor? weight, Tensor? running_mean, Tensor? running_var, Tensor? save_mean, Tensor? save_invstd, bool train, float eps, bool[3] output_mask) -> (Tensor, Tensor, Tensor)")
                .impl_unboxedOnlyKernel<
                    decltype(batch_norm_bwd_hpu),
                    &batch_norm_bwd_hpu>(DispatchKey::HABANATensorId)
                .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
        .op(torch::RegisterOperators::options()
                .schema(
                    "aten::native_layer_norm(Tensor input, Tensor? weight, Tensor? bias, int M, int N, float eps) -> (Tensor, Tensor, Tensor)")
                .impl_unboxedOnlyKernel<
                    decltype(layer_norm_hpu),
                    &layer_norm_hpu>(DispatchKey::HABANATensorId)
                .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
        .op(torch::RegisterOperators::options()
                .schema("aten::norm.Scalar(Tensor self, Scalar p=2) -> Tensor")
                .impl_unboxedOnlyKernel<
                    decltype(norm_scalar_hpu),
                    &norm_scalar_hpu>(DispatchKey::HABANATensorId)
                .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA));
