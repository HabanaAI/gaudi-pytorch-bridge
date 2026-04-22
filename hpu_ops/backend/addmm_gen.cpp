/**
 * Copyright (c) 2021-2025 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "generated/backend/_addmm_activation.h"
#include "generated/backend/addbmm.h"
#include "generated/backend/addmm.h"
#include "hpu_ops/backend/reduction_template.h"
#include "hpu_ops/shared_meta_common.h"

namespace habana {

sizes_vec AddMMOutshape(const at::Stack& stack) {
  auto self = stack_tensor(stack, 0);
  auto mat1 = stack_tensor(stack, 1);
  auto mat2 = stack_tensor(stack, 2);
  HABANA_ASSERT(
      self.dim() == 2 || self.dim() == 1 || self.dim() == 0,
      "addmm: Expected self to be 0-D, 1-D or 2-D, but got ",
      self.dim(),
      "-D");
  HABANA_ASSERT(
      mat1.dim() == 2,
      "addmm: Expected mat1 to be 2-D, but got ",
      mat1.dim(),
      "-D");
  HABANA_ASSERT(
      mat2.dim() == 2,
      "addmm: Expected mat2 to be 2-D, but got ",
      mat2.dim(),
      "-D");
  HABANA_ASSERT(
      mat1.sizes()[1] == mat2.sizes()[0],
      "Matrices sizes are not compatible to multiply them");
  // (n, m)@(m, p) -> (n, p)
  std::vector<int64_t> matMulShape = {mat1.sizes()[0], mat2.sizes()[1]};
  std::vector<int64_t> outshape = at::infer_size(self.sizes(), matMulShape);
  return {outshape};
}

OutputMetaDataVector AddMMMeta(const at::Stack& stack) {
  OutputMetaDataVector metaVec(1);
  auto& meta = metaVec.front();
  // Take output tensor dtype
  std::optional<at::Tensor> output_tensor = std::nullopt;
  std::optional<c10::ScalarType> output_type = std::nullopt;
  if (stack.at(stack.size() - 1).isTensor()) {
    output_tensor = stack.at(stack.size() - 1).toTensor();
    output_type = stack.at(stack.size() - 1).toTensor().scalar_type();
  }
  meta.dtype = habana_helpers::DTypeHelper::get_compute_dtype(
      stack,
      output_tensor,
      habana_helpers::DTypeHelper::DtypePromoteVariant::kPromoteToCommon,
      false,
      output_type);
  meta.shape = AddMMOutshape(stack)[0];
  return metaVec;
}

SharedMetaDataVector AddMMSharedMeta(
    const at::Stack& stack,
    habana_helpers::HabanaExecutionMode /*unused*/) {
  return MatrixMulWithAddSharedMeta(stack, "addmm", false);
}

SharedMetaDataVector AddBMMSharedMeta(
    const at::Stack& stack,
    habana_helpers::HabanaExecutionMode /*unused*/) {
  return MatrixMulWithAddSharedMeta(stack, "addbmm", false);
}

SharedMetaDataVector AddMMActivationSharedMeta(
    const at::Stack& stack,
    habana_helpers::HabanaExecutionMode /*unused*/) {
  return MatrixMulWithAddSharedMeta(stack, "addmm", true);
}

OutputMetaDataVector AddBMMMeta(const at::Stack& stack) {
  auto self = stack_tensor(stack, 0);
  auto batch1 = stack_tensor(stack, 1);
  auto batch2 = stack_tensor(stack, 2);
  HABANA_ASSERT(
      self.dim() == 2 || self.dim() == 1 || self.dim() == 0,
      "addbmm: Expected self to be 0-D, 1-D or 2-D, but got ",
      self.dim(),
      "-D");
  HABANA_ASSERT(
      batch1.dim() == 3,
      "addbmm: Expected batch1 to be 3-D, but got ",
      batch1.dim(),
      "-D");
  HABANA_ASSERT(
      batch2.dim() == 3,
      "addbmm: Expected batch2 to be 3-D, but got ",
      batch2.dim(),
      "-D");

  OutputMetaDataVector metaVec(1);
  auto& meta = metaVec.front();
  meta.shape = {
      batch1.sizes()[1], batch2.sizes()[2]}; // (b, n, m)@(b, m, p) -> (n, p)
  meta.dtype = self.scalar_type();
  return metaVec;
}

using namespace std::literals;

void AddMM::AddNode(synapse_helpers::graph& graph, const at::Stack& stack) {
  const auto meta = OutputMeta(stack);

  const float beta_val = stack.at(3).toScalar().toFloat();
  const float alpha_val = stack.at(4).toScalar().toFloat();

  const bool shouldUseParams = beta_val == 0.0 || beta_val == 1.0 ||
      alpha_val == 1.0 || alpha_val == 0.0;

  // Kernel precision type is based on the input1 of the addmm op,
  // because we want to support configuration: inputs(fp8), output(bf16/fp32).
  // Formula: out = beta * input0 + alpha * (input1 @ input2)
  // GEMM returns higher precision dtype, so input0 has to be (bf16/fp32).
  update_guid_dtype(guid_, stack_tensor(stack, 1).scalar_type());

  if (shouldUseParams) {
    ns_AddmmKernel::Params params{};
    params.alpha = alpha_val;
    params.beta = beta_val;

    auto addmv = BuildOp(
        graph,
        GetGuid(),
        {syn_in(0), syn_in(1), syn_in(2)},
        {{meta[0].shape, meta[0].dtype, 0}},
        &params,
        sizeof(params));
    syn_out(0) = std::move(addmv[0]);
  } else {
    auto alpha_tensor = ConstantHelper(graph, alpha_val, ScalarType(), 1);
    auto beta_tensor = ConstantHelper(graph, beta_val, ScalarType(), 1);
    auto addmm = BuildOp(
        graph,
        GetGuid(),
        {syn_in(0),
         syn_in(1),
         syn_in(2),
         beta_tensor.get(),
         alpha_tensor.get()},
        {{meta[0].shape, meta[0].dtype, 0}});

    syn_out(0) = std::move(addmm[0]);
  }
}

void AddMMActivation::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  const auto meta = AddMMMeta(stack)[0];

  const float beta_val = stack.at(3).toScalar().toFloat();
  const float alpha_val = stack.at(4).toScalar().toFloat();

  const bool shouldUseParams = beta_val == 0.0 || beta_val == 1.0 ||
      alpha_val == 1.0 || alpha_val == 0.0;

  const bool append_activation = alpha_val != 0 || beta_val != 0;

  std::vector<synapse_helpers::tensor> result;
  if (shouldUseParams) {
    ns_AddmmKernel::Params params{};
    params.alpha = alpha_val;
    params.beta = beta_val;

    result = BuildOp(
        graph,
        guid_,
        {syn_in(0), syn_in(1), syn_in(2)},
        {{meta.shape,
          meta.dtype,
          append_activation ? std::nullopt : std::optional(0)}},
        &params,
        sizeof(params));
  } else {
    auto alpha_tensor = ConstantHelper(graph, alpha_val, ScalarType(), 1);
    auto beta_tensor = ConstantHelper(graph, beta_val, ScalarType(), 1);
    result = BuildOp(
        graph,
        guid_,
        {syn_in(0),
         syn_in(1),
         syn_in(2),
         beta_tensor.get(),
         alpha_tensor.get()},
        {{meta.shape,
          meta.dtype,
          append_activation ? std::nullopt : std::optional(0)}});
  }

  if (append_activation) {
    bool use_gelu = stack.at(5).toBool();
    std::vector<NodeAttr::NodeOutputAttr> act_output_attr{
        {meta.shape, meta.dtype, 0}};
    if (use_gelu) {
      act_output_attr.push_back({meta.shape, meta.dtype});
    }
    auto act = BuildOp(
        graph,
        get_guid_with_precision(
            use_gelu ? "gelu_fwd"sv : "relu_fwd"sv, meta.dtype),
        {result[0].get()},
        act_output_attr);
    syn_out(0) = std::move(act[0]);
  } else {
    syn_out(0) = std::move(result[0]);
  }
}

} // namespace habana
