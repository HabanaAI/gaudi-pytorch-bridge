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
#include <torch/script.h>
#include <memory>

#include "habana_device/HPUCheck.h"
#include "habana_device/HPUContext.h"
#include "habana_device/hpu_cached_devices.h"
#include "habana_helpers/tensor_utils.h"
#include "habana_kernels/kernel_utils.h"
#include "habana_kernels/simple_generic_kernel.h"

using namespace torch;

void check_ew_kernel_constraints(const Tensor& arg1, const Tensor& arg2) {
  TORCH_CHECK(
      arg1.device() == arg2.device(),
      "Devices don't match. arg1 device: ",
      arg1.device(),
      " arg2 device: ",
      arg2.device());
  TORCH_CHECK(
      arg1.scalar_type() == arg2.scalar_type(),
      "Types don't match. arg1 type: ",
      arg1.scalar_type(),
      " arg2 type: ",
      arg2.scalar_type());
  TORCH_CHECK(
      (arg1.sizes() == arg2.sizes()) ||
          (arg1.ndimension() == arg2.ndimension() &&
           std::all_of(
               arg2.sizes().cbegin(),
               arg2.sizes().cend(),
               [](auto val) { return val == 1; })),
      "Sizes in elementwise kernel don't match. arg1 sizes: ",
      arg1.sizes(),
      ", arg2 sizes: ",
      arg2.sizes());
}

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


/*************************************************************************
 * @brief Synapse implementation for output = torch.add(arg1, alpha, arg2)
 * @param arg1 - first input
 * @param arg2 - second input
 * @param alpha - optional input
 * @param output - output
 * output = arg1 + alpha * arg2
 * or output = arg1 + arg2
 ************************************************************************/
void synapse_add_tensor(
    const Tensor& output,
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
    std::vector<synapse_helpers::tensor> syn_helper_inputs, syn_helper_tmps, syn_helper_outputs;
    std::vector<synTensor> syn_inputs, syn_tmps, syn_outputs;

    std::vector<const at::Tensor*> pt_inputs{&arg1, &arg2};
    std::vector<const at::Tensor*> pt_outputs{&output};

    if (alpha.has_value()) {
      pt_inputs.push_back(alpha.value());

      std::tie(syn_helper_tmps, syn_tmps) =
          habana_helpers::create_tensors({&arg2}, graph_handle, false);
    }

    std::tie(syn_helper_inputs, syn_inputs) =
        habana_helpers::create_tensors(pt_inputs, graph_handle, true);

    std::tie(syn_helper_outputs, syn_outputs) =
        habana_helpers::create_tensors(pt_outputs, graph_handle, true);

    {
      const auto kernel_suffix =
          habana_helpers::name_suffix_from_type(arg1.scalar_type());
      const std::string mult_node_type = "mult_fwd_" + kernel_suffix,
                        add_node_type = "add_fwd_" + kernel_suffix;
      { // add node
        if (alpha.has_value()) {
          std::vector<synTensor> syn_mul_inputs{syn_inputs[1], syn_inputs[2]};

          TORCH_HABANA_CHECK(
              synNodeCreate(
                  graph_handle,
                  syn_mul_inputs.data(),
                  syn_tmps.data(),
                  syn_mul_inputs.size(),
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
                syn_outputs.data(),
                syn_add_inputs.size(),
                syn_outputs.size(),
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
          habana_helpers::names(syn_helper_outputs),
          habana_helpers::extract_data_ptrs(pt_inputs),
          habana_helpers::extract_data_ptrs(pt_outputs),
          device_id);
    }
  }
  TORCH_HABANA_CHECK(synGraphDestroy(graph_handle), "synGraphDestroy failed");
}

// arg1 *= arg2
void synapse_mul_tensor_(const Tensor& arg1, const Tensor& arg2) {
  auto& device =
      synapse_helpers::HPURegistrar::get_device(arg1.device().index());
  const auto device_id = device.id();

  // graph_handle scope
  synGraphHandle graph_handle;
  TORCH_HABANA_CHECK(
      synGraphCreate(&graph_handle, synDeviceType::synDeviceGaudi),
      "synGraphCreate failed");

  { // tensors scope
    std::vector<synapse_helpers::tensor> syn_helper_inputs;
    std::vector<synTensor> syn_inputs;

    std::vector<const at::Tensor*> pt_inputs{&arg1, &arg2};
    std::tie(syn_helper_inputs, syn_inputs) =
        habana_helpers::create_tensors(pt_inputs, graph_handle, true);
    auto syn_helper_output = habana_helpers::duplicate_tensor_in_memory_section(
        syn_helper_inputs[0]);

    {
      const auto kernel_suffix =
          habana_helpers::name_suffix_from_type(arg1.scalar_type());
      const std::string node_type = "mult_fwd_" + kernel_suffix;
      {
        TORCH_HABANA_CHECK(
            synNodeCreate(
                graph_handle,
                syn_inputs.data(),
                &syn_helper_output.get(),
                syn_inputs.size(),
                1,
                nullptr,
                0,
                node_type.c_str(),
                "",
                nullptr,
                nullptr),
            "synNodeCreate failed");
      }

      habana_helpers::compile_and_run(
          node_type,
          graph_handle,
          habana_helpers::names(syn_helper_inputs),
          {syn_helper_output.tensor_name_},
          std::vector<void*>{arg1.data_ptr(), arg2.data_ptr()},
          {arg1.data_ptr()},
          device_id);
    }
  }
  TORCH_HABANA_CHECK(synGraphDestroy(graph_handle), "synGraphDestroy failed");
}

// self += alpha * other
Tensor& add_tensor_hpu_(Tensor& self, const Tensor& other, Scalar alpha) {
  LOG_FUNC_BEGIN;
  check_ew_kernel_constraints(self, other);

  Scalar alpha_converted = alpha;
  if (self.scalar_type() != habana_helpers::scalar_type(alpha))
    alpha_converted = alpha.toFloat();

  auto alpha_tensor = habana_helpers::scalar_to_device_tensor(
      alpha_converted, self.options(), self.ndimension());

  synapse_add_tensor_(self, other, &alpha_tensor);

  LOG_FUNC_END;
  return self;
}

/*************************************************************************
 * @brief Kernel implementation for out = torch.add(self, alpha, other)
 * @param self - first input
 * @param other - second input
 * @param alpha - optional input
 * out = self + alpha * other
 ************************************************************************/
Tensor add_tensor_hpu(Tensor& self, const Tensor& other, Scalar alpha) {
  LOG_FUNC_BEGIN;
  check_ew_kernel_constraints(self, other);

  Scalar alpha_converted = alpha;
  if (self.scalar_type() != habana_helpers::scalar_type(alpha))
    alpha_converted = alpha.toFloat();

  auto alpha_tensor = habana_helpers::scalar_to_device_tensor(
      alpha_converted, self.options(), self.ndimension());

   auto output = at::empty(self.sizes(), self.options());
   synapse_add_tensor(output, self, other, &alpha_tensor);

  LOG_FUNC_END;
  return output;
}
// Elementwise multiplication
// self *= other
Tensor& mul_tensor_hpu_(Tensor& self, const Tensor& other) {
  LOG_FUNC_BEGIN;

  // TODO: [SW-9849] I am confused why in MNIST example we are multipling self
  // tensor with other, which is 0dim tensor (scalar), with different type
  // (double) on different device (cpu). IMO pytorch should call mul_(Tensor,
  // Scalar) instead of this function. Check this after upgrading to PT1.4
  // [SW-8961]
  auto modified_other = std::make_unique<Tensor>();
  try {
    // Throws if synapse doesn't support given type
    habana_helpers::pytorch_to_synapse_type(other.scalar_type());
  } catch (c10::Error& e) {
    if (e.msg_without_backtrace().find("Unsupported pytorch type") == 0) {
      TORCH_WARN(
          e.msg_without_backtrace(),
          ". It will be casted to ",
          self.scalar_type());
      *modified_other = other.to(self.scalar_type());
    } else {
      throw;
    }
  }

  if (self.device() != other.device())
    *modified_other =
        (modified_other->defined() ? *modified_other : other).to(self.device());

  if (other.ndimension() == 0) {
    auto expanded_sizes = std::vector<int64_t>(self.ndimension(), 1);
    std::tie(*modified_other) = at::expand_size(
        modified_other->defined() ? *modified_other : other,
        expanded_sizes,
        "matmul_with_bias_hpu");
  }

  check_ew_kernel_constraints(
      self, modified_other->defined() ? *modified_other : other);
  synapse_mul_tensor_(
      self, modified_other->defined() ? *modified_other : other);
  LOG_FUNC_END;
  return self;
}

/*************************************************************************
 * @brief Kernel implementation for torch.eq(self,other, out)
 * @param self - first input
 * @param other - second input
 * @param out -  output tensor of bool dtype
 ************************************************************************/
void eq_tensor_out_hpu(
    const Tensor& output,
    const Tensor& self,
    const Tensor& other) {
  LOG_FUNC_BEGIN;

  // change dtype bool to int8 to match TPC kernel signature
  // NOTE: This works because both bool and int8 uses 1 byte per element
  // Else we need to overload .to operator with an explicit TPC kernel for typecasting
  output.to(c10::ScalarType::Char);

  std::vector<const at::Tensor*> pt_inputs{&self, &other};
  std::vector<const at::Tensor*> pt_outputs{&output};

  synapse_simple_generic_kernel(
      pt_outputs, pt_inputs, "equal", nullptr, 0, SynapsePassType::FORWARD_PASS);

  // convert back to bool
  output.to(c10::ScalarType::Bool);

  LOG_FUNC_END;
}

/*************************************************************************
 * @brief Kernel implementation for out = torch.eq(self,other)
 * @param self - first input
 * @param other - second input
 ************************************************************************/
Tensor eq_tensor_hpu(Tensor& self, Tensor& other) {
  LOG_FUNC_BEGIN;

  auto tensor_options = self.options();
  auto output =
      at::empty(self.sizes(), tensor_options.dtype(c10::ScalarType::Char));

  eq_tensor_out_hpu(output, self, other);

  LOG_FUNC_END;
  return output;
}

/*************************************************************************
 * @brief Kernel implementation for out = torch.div(self,other)
 * @param self - first input
 * @param other - second input
 ************************************************************************/
Tensor div_tensor_hpu(const Tensor& self, const Tensor& other) {
  LOG_FUNC_BEGIN;
  auto divisor_sizes = other.sizes();
  auto divisor_tot_elems = other.numel();
  auto out = at::empty(self.sizes(), self.options());
  std::vector<const at::Tensor*> pt_inputs;
  pt_inputs.push_back(&self);

  Tensor other_tensor;
  if (1 == divisor_tot_elems) {//case where a single element tensor comes as divisor
    Scalar other_converted = other.item<float>();
    other_tensor = habana_helpers::scalar_to_device_tensor(
        other_converted, self.options(), self.ndimension());
    pt_inputs.push_back(&other_tensor);
  }
  else if (divisor_sizes != self.sizes()) {
    //TO DO: Return error or try to broadcast along appropriate dim
    check_ew_kernel_constraints((const at::Tensor&) self, other);
    return out;
  }
  else {
    pt_inputs.push_back(&other);
  }
  std::vector<const at::Tensor*> pt_outputs;
  pt_outputs.push_back(&out);
  synapse_simple_generic_kernel(pt_outputs, pt_inputs, "div", nullptr, 0, SynapsePassType::FORWARD_PASS);
  LOG_FUNC_END;
  return out;
}

/****************************************************************************
 * @brief Kernel implementation for result = torch.div(input, denom, out=out)
 * @param result - output
 * @param self - first input
 * @param other - second input
 ***************************************************************************/
Tensor& div_tensor_hpu_out(Tensor& result, const Tensor& self, const Tensor& other) {
  LOG_FUNC_BEGIN;
  auto divisor_sizes = other.sizes();
  auto divisor_tot_elems = other.numel();
  auto out = at::empty(self.sizes(), self.options());
  std::vector<const at::Tensor*> pt_inputs;
  pt_inputs.push_back(&self);

  Tensor other_tensor;
  if (1 == divisor_tot_elems) {//case where a single element tensor comes as divisor
    Scalar other_converted = other.item<float>();
    other_tensor = habana_helpers::scalar_to_device_tensor(
        other_converted, self.options(), self.ndimension());
    pt_inputs.push_back(&other_tensor);
  }
  else if (divisor_sizes != self.sizes()) {
    //TO DO: Return error or try to broadcast along appropriate dim
    check_ew_kernel_constraints(self, other);
    return result;
  }
  else {
    pt_inputs.push_back(&other);
  }
  std::vector<const at::Tensor*> pt_outputs;
  pt_outputs.push_back(&result);
  synapse_simple_generic_kernel(pt_outputs, pt_inputs, "div", nullptr, 0, SynapsePassType::FORWARD_PASS);

  LOG_FUNC_END;
  return result;
}
/*************************************************************************
 * @brief Kernel implementation for inplace torch.div_(self,other)
 * @param self - first input
 * @param other - second input
 ************************************************************************/
Tensor& div_tensor_hpu_(Tensor& self, const Tensor& other) {
  LOG_FUNC_BEGIN;
  auto divisor_sizes = other.sizes();
  auto divisor_tot_elems = other.numel();
  auto out = at::empty(self.sizes(), self.options());
  std::vector<const at::Tensor*> pt_inputs;
  pt_inputs.push_back(&self);

  Tensor other_tensor;
  if (1 == divisor_tot_elems) {//case where a single element tensor comes as divisor
    Scalar other_converted = other.item<float>();
    other_tensor = habana_helpers::scalar_to_device_tensor(
        other_converted, self.options(), self.ndimension());
    pt_inputs.push_back(&other_tensor);
  }
  else if (divisor_sizes != self.sizes()) {
    //TO DO: Return error or try to broadcast along appropriate dim
    check_ew_kernel_constraints(self, other);
    return self;
  }
  else {
    pt_inputs.push_back(&other);
  }
  synapse_simple_generic_inplace_kernel(pt_inputs, "div", nullptr, 0, SynapsePassType::FORWARD_PASS);

  LOG_FUNC_END;
  return self;
}

Tensor div_scalar_hpu(const Tensor& self, Scalar other) {//TODO: No way to test this yet from python
  LOG_FUNC_BEGIN;
  TORCH_CHECK(
      self.scalar_type() != habana_helpers::scalar_type(other),
      "Types don't match. arg1 type: ",
      self.scalar_type(),
      " arg2 type: ",
      habana_helpers::scalar_type(other));
  Scalar divisor_converted = other;
  auto divisor_tensor = habana_helpers::scalar_to_device_tensor(
      divisor_converted, self.options(), self.ndimension());

  std::vector<const at::Tensor*> pt_inputs;
  pt_inputs.push_back(&self);
  pt_inputs.push_back(&divisor_tensor);
  auto out = at::empty(self.sizes(), self.options());
  std::vector<const at::Tensor*> pt_outputs;
  pt_outputs.push_back(&out);

  synapse_simple_generic_kernel(pt_outputs, pt_inputs, "div", nullptr, 0, SynapsePassType::FORWARD_PASS);

  LOG_FUNC_END;
  return out;
}

Tensor& div_scalar_hpu_(Tensor& self, Scalar other) {//TODO: No way to test this yet from python
  LOG_FUNC_BEGIN;
  std::vector<const at::Tensor*> pt_inputs;
  TORCH_CHECK(
      self.scalar_type() != habana_helpers::scalar_type(other),
      "Types don't match. arg1 type: ",
      self.scalar_type(),
      " arg2 type: ",
      habana_helpers::scalar_type(other));
  Scalar divisor_converted = other;
  auto divisor_tensor = habana_helpers::scalar_to_device_tensor(
      divisor_converted, self.options(), self.ndimension());

  pt_inputs.push_back(&self);
  pt_inputs.push_back(&divisor_tensor);

  synapse_simple_generic_inplace_kernel(pt_inputs, "div", nullptr, 0, SynapsePassType::FORWARD_PASS);

  LOG_FUNC_END;
  return self;
}

static auto registry =
    torch::RegisterOperators()
        .op(torch::RegisterOperators::options()
                .schema(
                    "aten::add_.Tensor(Tensor(a!) self, Tensor other, *, Scalar alpha=1) -> Tensor(a!)")
                .impl_unboxedOnlyKernel<
                    decltype(add_tensor_hpu_),
                    &add_tensor_hpu_>(TensorTypeId::HABANATensorId)
                .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
        .op(torch::RegisterOperators::options()
                .schema(
                    "aten::add.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor")
                .impl_unboxedOnlyKernel<
                    decltype(add_tensor_hpu),
                    &add_tensor_hpu>(TensorTypeId::HABANATensorId)
                .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
        .op(torch::RegisterOperators::options()
                .schema(
                    "aten::mul_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)")
                .impl_unboxedOnlyKernel<
                    decltype(mul_tensor_hpu_),
                    &mul_tensor_hpu_>(TensorTypeId::HABANATensorId)
                .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
        .op(torch::RegisterOperators::options()
                .schema("aten::eq.Tensor(Tensor self, Tensor other) -> Tensor")
                .impl_unboxedOnlyKernel<
                    decltype(eq_tensor_hpu),
                    &eq_tensor_hpu>(TensorTypeId::HABANATensorId)
                .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
        .op(torch::RegisterOperators::options()
                .schema(
                    "aten::eq.Tensor_out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)")
                .impl_unboxedOnlyKernel<
                    decltype(eq_tensor_out_hpu),
                    &eq_tensor_out_hpu>(TensorTypeId::HABANATensorId)
                .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
        .op(torch::RegisterOperators::options()
                .schema(
                    "aten::div.Tensor(Tensor self, Tensor other) -> Tensor")
                .impl_unboxedOnlyKernel<
                    decltype(div_tensor_hpu),
                    &div_tensor_hpu>(TensorTypeId::HABANATensorId)
                .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
        .op(torch::RegisterOperators::options()
                .schema(
                    "aten::div.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)")
                .impl_unboxedOnlyKernel<
                    decltype(div_tensor_hpu_out),
                    &div_tensor_hpu_out>(TensorTypeId::HABANATensorId)
                .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
        .op(torch::RegisterOperators::options()
                .schema(
                    "aten::div_.Tensor(Tensor(a!) self, Tensor other) -> (Tensor(a!))")
                .impl_unboxedOnlyKernel<
                    decltype(div_tensor_hpu_),
                    &div_tensor_hpu_>(TensorTypeId::HABANATensorId)
                .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
        .op(torch::RegisterOperators::options()
                .schema(
                    "aten::div.Scalar(Tensor self, Scalar other) -> Tensor")
                .impl_unboxedOnlyKernel<
                    decltype(div_scalar_hpu),
                    &div_scalar_hpu>(TensorTypeId::HABANATensorId)
                .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
        .op(torch::RegisterOperators::options()
                .schema(
                    "aten::div_.Scalar(Tensor(a!) self, Scalar other) -> (Tensor(a!))")
                .impl_unboxedOnlyKernel<
                    decltype(div_scalar_hpu_),
                    &div_scalar_hpu_>(TensorTypeId::HABANATensorId)
                .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA));
