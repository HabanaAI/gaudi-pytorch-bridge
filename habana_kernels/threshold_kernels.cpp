#include <torch/script.h>

#include "habana_device/HPUCheck.h"
#include "habana_device/HPUContext.h"
// #include "habana_device/fake_tensor_builder.h"
#include "habana_device/hpu_cached_devices.h"
#include "habana_helpers/tensor_utils.h"
#include "habana_kernels/kernel_utils.h"

using namespace torch;

// computes `output = input <= threshold ? value : other`
// other is `input` in threshold() and `grad` in threshold_backward()
void synapse_threshold_out(
    Tensor& output,
    const Tensor& input,
    const Tensor& threshold,
    const Tensor& value,
    const Tensor& other) {
  // TODO: request threshold TPC kernel, current implementation is not optimized
  // current implementation algorithm:
  // 1. mask = input <= threshold
  // 2. output = mask*val + !mask*other
  const auto device_id = input.device().index();
  auto& device = synapse_helpers::HPURegistrar::get_device(device_id);

  // graph_handle scope
  synGraphHandle graph_handle;
  TORCH_HABANA_CHECK(
      synGraphCreate(&graph_handle, synDeviceType::synDeviceGaudi),
      "synGraphCreate failed");

  { // tensors scope
    std::vector<synapse_helpers::tensor> syn_helper_inputs, syn_helper_outputs,
        syn_helper_tmp;
    std::vector<synTensor> syn_inputs, syn_outputs, syn_tmp;

    std::tie(syn_helper_inputs, syn_inputs) = habana_helpers::create_tensors(
        std::vector<const at::Tensor*>{&input, &threshold, &value, &other},
        graph_handle,
        true);
    std::tie(syn_helper_outputs, syn_outputs) = habana_helpers::create_tensors(
        std::vector<const at::Tensor*>{&output}, graph_handle, true);
    std::tie(syn_helper_tmp, syn_tmp) = habana_helpers::create_tensors(
        std::vector<const at::Tensor*>{
            &output, &output, &output, &output, &output, &output},
        graph_handle,
        std::vector<bool>(6, false),
        {{}, {}, {}, {}, c10::ScalarType::Char, c10::ScalarType::Char});
    {
      const auto kernel_suffix =
          "_fwd_" + habana_helpers::name_suffix_from_type(input.scalar_type());
      const std::string leq_node_type = "less_equal" + kernel_suffix,
                        greater_node_type = "greater" + kernel_suffix,
                        mult_node_type = "mult" + kernel_suffix,
                        add_node_type = "add" + kernel_suffix,
                        cast_i8_to_fp32 = "cast_i8_to_f32";

      { // add nodes
        { // create mask and inv_mask
          // mask_i8 = input <= threshold
          // inv_mask_i8 = input > threshold
          TORCH_HABANA_CHECK(
              synNodeCreate(
                  graph_handle,
                  &syn_inputs[0],
                  &syn_tmp[4],
                  2,
                  1,
                  nullptr,
                  0,
                  leq_node_type.c_str(),
                  "",
                  nullptr,
                  nullptr),
              "synNodeCreate failed");
          TORCH_HABANA_CHECK(
              synNodeCreate(
                  graph_handle,
                  &syn_inputs[0],
                  &syn_tmp[5],
                  2,
                  1,
                  nullptr,
                  0,
                  greater_node_type.c_str(),
                  "",
                  nullptr,
                  nullptr),
              "synNodeCreate failed");
        }
        { // convert masks to fp32
          // mask = float(mask_i8)
          // inv_mask = float(inv_mask_i8)
          TORCH_HABANA_CHECK(
              synNodeCreate(
                  graph_handle,
                  &syn_tmp[4],
                  &syn_tmp[0],
                  1,
                  1,
                  nullptr,
                  0,
                  cast_i8_to_fp32.c_str(),
                  "",
                  nullptr,
                  nullptr),
              "synNodeCreate failed");
          TORCH_HABANA_CHECK(
              synNodeCreate(
                  graph_handle,
                  &syn_tmp[5],
                  &syn_tmp[1],
                  1,
                  1,
                  nullptr,
                  0,
                  cast_i8_to_fp32.c_str(),
                  "",
                  nullptr,
                  nullptr),
              "synNodeCreate failed");
        }
        { // mask*val
          // mask*val = mask * val
          std::vector<synTensor> syn_tmp_in{syn_tmp[0], syn_inputs[2]};
          TORCH_HABANA_CHECK(
              synNodeCreate(
                  graph_handle,
                  syn_tmp_in.data(),
                  &syn_tmp[2],
                  syn_tmp_in.size(),
                  1,
                  nullptr,
                  0,
                  mult_node_type.c_str(),
                  "",
                  nullptr,
                  nullptr),
              "synNodeCreate failed");
        }
        { // inv_mask*other
          // inv_mask*other = inv_mask*other
          std::vector<synTensor> syn_tmp_in{syn_tmp[1], syn_inputs[3]};
          TORCH_HABANA_CHECK(
              synNodeCreate(
                  graph_handle,
                  syn_tmp_in.data(),
                  &syn_tmp[3],
                  syn_tmp_in.size(),
                  1,
                  nullptr,
                  0,
                  mult_node_type.c_str(),
                  "",
                  nullptr,
                  nullptr),
              "synNodeCreate failed");
        }
        { // output
          // output = mask*val + inv_mask*inv_mask
          TORCH_HABANA_CHECK(
              synNodeCreate(
                  graph_handle,
                  &syn_tmp[2],
                  syn_outputs.data(),
                  2,
                  syn_outputs.size(),
                  nullptr,
                  0,
                  add_node_type.c_str(),
                  "",
                  nullptr,
                  nullptr),
              "synNodeCreate failed");
        }
      }
    }
    habana_helpers::compile_and_run(
        "threshold",
        graph_handle,
        habana_helpers::names(syn_helper_inputs),
        habana_helpers::names(syn_helper_outputs),
        {input.data_ptr(),
         threshold.data_ptr(),
         value.data_ptr(),
         other.data_ptr()},
        {output.data_ptr()},
        device_id);
  }
  TORCH_HABANA_CHECK(synGraphDestroy(graph_handle), "synGraphDestroy failed");
}

Tensor threshold_backward_hpu(
    const Tensor& grad_output,
    const Tensor& self,
    Scalar threshold) {
  LOG_FUNC_BEGIN;
  TORCH_CHECK(self.scalar_type() == c10::ScalarType::Float);

  Scalar threshold_converted = threshold;
  if (self.scalar_type() != habana_helpers::scalar_type(threshold))
    threshold_converted = threshold.toFloat();

  auto dims = self.ndimension();
  auto options = self.options();
  auto threshold_tensor = habana_helpers::scalar_to_device_tensor(
      threshold_converted, options, dims);
  auto value_tensor =
      habana_helpers::scalar_to_device_tensor(Scalar(0.0), options, dims);
  auto output = at::empty(self.sizes(), options);
  synapse_threshold_out(
      output, self, threshold_tensor, value_tensor, grad_output);
  LOG_FUNC_END;
  return output;
}

static auto registry = torch::RegisterOperators().op(
    torch::RegisterOperators::options()
        .schema(
            "aten::threshold_backward(Tensor grad_output, Tensor self, Scalar threshold) -> Tensor")
        .impl_unboxedOnlyKernel<
            decltype(threshold_backward_hpu),
            &threshold_backward_hpu>(TensorTypeId::HABANATensorId)
        .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA));