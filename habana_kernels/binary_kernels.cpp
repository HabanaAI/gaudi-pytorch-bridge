#include <torch/script.h>

#include "habana_device/HPUCheck.h"
#include "habana_device/HPUContext.h"
#include "habana_device/hpu_cached_devices.h"
#include "habana_helpers/tensor_utils.h"
#include "habana_kernels/kernel_utils.h"

using namespace torch;

// arg1 += arg2
// or arg1 += arg2 * alpha
void synapse_add_tensor_(
    const Tensor& arg1,
    const Tensor& arg2,
    c10::optional<const Tensor*> alpha) {
  auto& device =
      synapse_helpers::HPURegistrar::get_device(arg1.device().index());
  const auto device_id = device.id();

  // graph_handle scope
  synGraphHandle graph_handle;
  TORCH_HABANA_CHECK(
      synGraphCreate(&graph_handle, synDeviceType::synDeviceGaudi),
      "synGraphCreate failed");

  { // tensors scope
    std::vector<synapse_helpers::tensor> syn_helper_inputs, syn_helper_tmps;
    std::vector<synTensor> syn_inputs, syn_tmps;

    std::vector<const at::Tensor*> pt_inputs{&arg1, &arg2};
    if (alpha.has_value()) {
      pt_inputs.push_back(alpha.value());

      std::tie(syn_helper_tmps, syn_tmps) =
          habana_helpers::create_tensors({&arg2}, graph_handle, false);
    }

    std::tie(syn_helper_inputs, syn_inputs) =
        habana_helpers::create_tensors(pt_inputs, graph_handle, true);
    auto syn_helper_output = habana_helpers::duplicate_tensor_in_memory_section(
        syn_helper_inputs[0]);

    {
      const auto kernel_suffix =
          habana_helpers::name_suffix_from_type(arg1.scalar_type());
      const std::string mult_node_type = "mult_fwd_" + kernel_suffix,
                        add_node_type = "add_fwd_" + kernel_suffix;
      { // add node
        if (alpha.has_value()) {
          TORCH_HABANA_CHECK(
              synNodeCreate(
                  graph_handle,
                  &syn_inputs[1],
                  syn_tmps.data(),
                  2,
                  syn_tmps.size(),
                  nullptr,
                  0,
                  mult_node_type.c_str(),
                  "",
                  nullptr,
                  nullptr),
              "synNodeCreate failed");
        }
        std::vector<synTensor> syn_add_inputs{
            syn_inputs[0], alpha.has_value() ? syn_tmps[0] : syn_inputs[1]};

        TORCH_HABANA_CHECK(
            synNodeCreate(
                graph_handle,
                syn_add_inputs.data(),
                &syn_helper_output.get(),
                syn_add_inputs.size(),
                1,
                nullptr,
                0,
                add_node_type.c_str(),
                "",
                nullptr,
                nullptr),
            "synNodeCreate failed");
      }

      habana_helpers::compile_and_run(
          add_node_type,
          graph_handle,
          habana_helpers::names(syn_helper_inputs),
          {syn_helper_output.tensor_name_},
          alpha.has_value()
              ? std::vector<void*>{arg1.data_ptr(),
                                   arg2.data_ptr(),
                                   alpha.value()->data_ptr()}
              : std::vector<void*>{arg1.data_ptr(), arg2.data_ptr()},
          {arg1.data_ptr()},
          device_id);
    }
  }
  TORCH_HABANA_CHECK(synGraphDestroy(graph_handle), "synGraphDestroy failed");
}

// self += alpha * other
Tensor& add_tensor_hpu_(Tensor& self, const Tensor& other, Scalar alpha) {
  LOG_FUNC_BEGIN;
  TORCH_CHECK(self.scalar_type() == c10::ScalarType::Float);
  TORCH_CHECK(other.scalar_type() == c10::ScalarType::Float);

  Scalar alpha_converted = alpha;
  if (self.scalar_type() != habana_helpers::scalar_type(alpha))
    alpha_converted = alpha.toFloat();

  auto alpha_tensor = habana_helpers::scalar_to_device_tensor(
      alpha_converted, self.options(), self.ndimension());

  synapse_add_tensor_(self, other, &alpha_tensor);

  LOG_FUNC_END;
  return self;
}

static auto registry = torch::RegisterOperators().op(
    torch::RegisterOperators::options()
        .schema(
            "aten::add_.Tensor(Tensor(a!) self, Tensor other, *, Scalar alpha=1) -> Tensor(a!)")
        .impl_unboxedOnlyKernel<decltype(add_tensor_hpu_), &add_tensor_hpu_>(
            TensorTypeId::HABANATensorId)
        .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA));