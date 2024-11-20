/**
* Copyright (c) 2021-2024 Intel Corporation
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
#include <ATen/core/Reduction.h>
#include <perf_lib_layer_params.h>

#include "backend/habana_device/hpu_cached_devices.h"
#include "backend/helpers/create_tensor.h"
#include "backend/helpers/tensor_utils.h"
#include "backend/synapse_helpers/recipe.h"
#include "habana_helpers/logging.h"
#include "habana_kernels/binary_inplace_kernels.h"
#include "habana_kernels/binary_kernels.h"
#include "habana_kernels/optimizer_kernels.h"
#include "habana_kernels/unary_kernels.h"
#include "habana_lazy/aten_lazy_bridge.h"
#include "hpu_ops/backend/reduction_template.h"

using namespace torch;
using namespace habana;

namespace sh = synapse_helpers;

// Input tensors
// 1    Gradient        FP32/FP16/BF16  2D
// 2    Weights         FP32            2D
// 3    Moments         FP32            2D
// 4    Indices         I32             1D
// 5    Learning rate   FP32            1D
// 6    Valid count     I32             1D
// 7    momentum        FP32
// 8    nesterov        Bool
// Output tensors
// 1    Weights         FP32            2D
// 2    Moments         FP32            2D
// TODO: TPC kernel seems to give wrong results.
void OptimizerSparseSgdOperator::AllocateAndAddSynapseNode(
    sh::graph& graph,
    torch::jit::Stack& inputs,
    const OutputMetaDataVector& output_metadata) {
  TORCH_CHECK(
      inputs.size() == 8,
      "Incorrect size of inputs for optimizer_sparse_sgd operator");
  TORCH_CHECK(inputs[0].isTensor(), "Input arg1 type expected to be tensor");
  TORCH_CHECK(inputs[1].isTensor(), "Input arg2 type expected to be tensor");
  TORCH_CHECK(inputs[2].isTensor(), "Input arg3 type expected to be tensor");
  TORCH_CHECK(inputs[3].isTensor(), "Input arg4 type expected to be tensor");
  TORCH_CHECK(inputs[4].isTensor(), "Input arg5 type expected to be tensor");
  TORCH_CHECK(inputs[5].isTensor(), "Input arg6 type expected to be tensor");
  TORCH_CHECK(inputs[6].isDouble(), "Input arg7 type expected to be float");
  TORCH_CHECK(inputs[7].isBool(), "Input arg8 type expected to be Bool");
  TORCH_CHECK(
      output_metadata.size() == 2,
      "OptimizerSparseSgdOperator: #output_metadata should be 2");

  auto weights_in = inputs[1].toTensor();
  auto moments_in = inputs[2].toTensor();
  auto mom = static_cast<float>(inputs[6].toDouble());
  auto nesterov = inputs[7].toBool();

  ns_OptimizerSparseSGD::Params params;
  params.mom = mom;
  params.nesterov = nesterov;

  // execute in-place for weights & moments
  p_context_->syn_outputs_.emplace_back(
      habana_helpers::duplicate_tensor_in_memory_section(
          p_context_->syn_inputs_[1], graph, output_metadata.at(0).external));
  p_context_->pt_outputs_.emplace_back(weights_in);

  // moments
  p_context_->syn_outputs_.emplace_back(
      habana_helpers::duplicate_tensor_in_memory_section(
          p_context_->syn_inputs_[2], graph, output_metadata.at(1).external));
  p_context_->pt_outputs_.emplace_back(moments_in);

  AddNodeToSynapseGraph(graph, &params, sizeof(params));
}

void OptimizerSparseAdagradOperator::AllocateAndAddSynapseNode(
    sh::graph& graph,
    torch::jit::Stack& inputs,
    const OutputMetaDataVector& output_metadata) {
  TORCH_CHECK(
      inputs.size() == 6,
      "Incorrect size of inputs for optimizer_adagrad_sgd operator");
  TORCH_CHECK(inputs[0].isTensor(), "Input arg1 type expected to be tensor");
  TORCH_CHECK(inputs[1].isTensor(), "Input arg2 type expected to be tensor");
  TORCH_CHECK(inputs[2].isTensor(), "Input arg3 type expected to be tensor");
  TORCH_CHECK(inputs[3].isTensor(), "Input arg4 type expected to be tensor");
  TORCH_CHECK(inputs[4].isTensor(), "Input arg5 type expected to be tensor");
  TORCH_CHECK(inputs[5].isTensor(), "Input arg6 type expected to be tensor");
  TORCH_CHECK(
      output_metadata.size() == 2,
      "OptimizerSparseAdagradOperator: #output_metadata should be 2");

  ns_OptimizerSparseAdagrad::Params params;
  // PT does not use decay param for sparse params
  // Ref:
  // https://pytorch.org/docs/stable/_modules/torch/optim/adagrad.html#Adagrad
  // Even for dense, it applies decay param to the current grad whereas TPC
  // applies to the accumulated grad
  params.decay = 1.0;
  params.eps = 1e-10f;

  // execute in-place for weights & moments
  p_context_->syn_outputs_.emplace_back(
      habana_helpers::duplicate_tensor_in_memory_section(
          p_context_->syn_inputs_[1], graph, output_metadata.at(0).external));

  auto weights_in = inputs[1].toTensor();
  p_context_->pt_outputs_.emplace_back(weights_in);

  p_context_->syn_outputs_.emplace_back(
      habana_helpers::duplicate_tensor_in_memory_section(
          p_context_->syn_inputs_[2], graph, output_metadata.at(1).external));

  auto moments_in = inputs[2].toTensor();
  p_context_->pt_outputs_.emplace_back(moments_in);

  AddNodeToSynapseGraph(graph, &params, sizeof(params));
}

void OptimizerAdamwOperator::AllocateAndAddSynapseNode(
    sh::graph& graph,
    torch::jit::Stack& inputs,
    const OutputMetaDataVector& output_metadata) {
  static_cast<void>(output_metadata);
  TORCH_CHECK(
      inputs.size() == 10,
      "Incorrect size of inputs for adamw optimizer graph creation call");

  auto gradients = inputs[0].toTensorList();
  auto weights = inputs[1].toTensorList();
  auto exp_avg = inputs[2].toTensorList();
  auto exp_avg_sq = inputs[3].toTensorList();
  auto neg_step_size = inputs[4].toTensor();
  auto beta1 = inputs[5].toScalar();
  auto beta2 = inputs[6].toScalar();
  auto epsilon = inputs[7].toScalar();
  auto modified_wd_t = inputs[8].toTensor();
  auto is_wd_modified = inputs[9].toScalar();

  /*  This are the operations we need to perform per parameter
      exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
      exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
      denom = exp_avg_sq.sqrt().add_(group["eps"])
      ratio = torch.div(exp_avg, denom)
      scaled_ratio = torch.mul(ratio, step_size)
      p.data.sub_(scaled_ratio)
      if group["weight_decay"] > 0.0:
        p.data.add_(p.data, alpha=-group["lr"] * group["weight_decay"])
  */
  auto device_id = gradients.get(0).device().index();
  auto scalar_type = gradients.get(0).scalar_type();
  auto num_params = static_cast<unsigned int>(weights.size());
  torch::jit::Stack stack;

  for (unsigned int i = 0; i < num_params; i++) {
    // Synapse Graph for single parameter update to be created here
    // All synapse input tensor references are there in a single std::vector
    // gradients ; weights ; exp_avg ; exp_avg_sq ; lr ; neg_step_size

    // if group["weight_decay"] > 0.0:
    //  p.data.add_(p.data, alpha=-group["lr"] *
    //  group["weight_decay"])
    // Since kernel receives modified_wd = 1-group["weight_decay"]*group["lr"]
    // therefore  p.data.mul_(modified_wd)

    auto mul_wt_wd =
        make_operator<habana::MulInplaceOperator>(device_id, scalar_type);

    if (is_wd_modified.toBool()) {
      // Weight tensor index == weight tensor list index + curr location i in
      // tensor list
      mul_wt_wd->SetSynapseInput(p_context_->syn_inputs_[1 * num_params + i]);
      // Weight decay tensor index == after 4 tensor lists + 1 tensors
      mul_wt_wd->SetSynapseInput(p_context_->syn_inputs_[4 * num_params + 1]);

      stack.emplace_back(IValue(weights.get(i)));
      stack.emplace_back(IValue(modified_wd_t));
      mul_wt_wd->AllocateAndAddSynapseNode(
          graph, stack, OutputMetaDataVector(1));
      stack.clear();
    }

    // exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
    auto mul_exp_avg =
        make_operator<habana::MulInplaceOperator>(device_id, scalar_type);
    mul_exp_avg->SetSynapseInput(p_context_->syn_inputs_[2 * num_params + i]);
    stack.emplace_back(IValue(exp_avg.get(i)));
    stack.emplace_back(IValue(beta1));
    mul_exp_avg->AllocateAndAddSynapseNode(
        graph, stack, OutputMetaDataVector(1));
    stack.clear();

    auto add_exp_avg =
        make_operator<habana::AddInplaceOperator>(device_id, scalar_type);
    sh::tensor& syn_in_11 =
        add_exp_avg->SetSynapseInput(mul_exp_avg->GetSynOutputs()[0]);
    add_exp_avg->SetSynapseInput(p_context_->syn_inputs_[i]);
    stack.emplace_back(IValue(mul_exp_avg->GetOutputs()[0]));
    stack.emplace_back(IValue(gradients.get(i)));
    stack.emplace_back(IValue(Scalar(1.0 - beta1.toDouble())));
    add_exp_avg->AllocateAndAddSynapseNode(
        graph, stack, OutputMetaDataVector(1));
    stack.clear();

    // exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
    auto mul_exp_avg_sq =
        make_operator<habana::MulInplaceOperator>(device_id, scalar_type);
    mul_exp_avg_sq->SetSynapseInput(
        p_context_->syn_inputs_[3 * num_params + i]);
    stack.emplace_back(IValue(exp_avg_sq.get(i)));
    stack.emplace_back(IValue(beta2));
    mul_exp_avg_sq->AllocateAndAddSynapseNode(
        graph, stack, OutputMetaDataVector(1));
    stack.clear();

    auto addcmul_exp_avg_sq =
        make_operator<habana::AddcmulInplaceOperator>(device_id, scalar_type);
    sh::tensor& syn_in_14 =
        addcmul_exp_avg_sq->SetSynapseInput(mul_exp_avg_sq->GetSynOutputs()[0]);
    addcmul_exp_avg_sq->SetSynapseInput(p_context_->syn_inputs_[i]);
    // Internally we are going to use "pow" instead of "mul",
    // therefore 3rd synapse tensor will be unused. We can give
    // a dummy tensor
    auto syn_in_3 = habana_helpers::create_tensor(
        gradients.get(i), graph, true, false, c10::nullopt);
    addcmul_exp_avg_sq->SetSynapseInput(syn_in_3);
    stack.emplace_back(IValue(mul_exp_avg_sq->GetOutputs()[0]));
    stack.emplace_back(IValue(gradients.get(i)));
    stack.emplace_back(IValue(gradients.get(i)));
    stack.emplace_back(IValue(Scalar(1.0 - beta2.toDouble())));
    addcmul_exp_avg_sq->AllocateAndAddSynapseNode(
        graph, stack, OutputMetaDataVector(1));
    stack.clear();

    // denom = exp_avg_sq.sqrt().add_(group["eps"])
    // we will actually do "add" instead of "add_". Inplace not strictly
    // required here
    auto sqrt_exp_avg_sq = make_operator<SqrtOperator>(device_id, scalar_type);
    sh::tensor& syn_in_15 = sqrt_exp_avg_sq->SetSynapseInput(
        addcmul_exp_avg_sq->GetSynOutputs()[0]);
    stack.emplace_back(IValue(addcmul_exp_avg_sq->GetOutputs()[0]));
    sqrt_exp_avg_sq->AllocateAndAddSynapseNode(
        graph, stack, OutputMetaDataVector(1));
    stack.clear();

    auto add_exp_avg_sq =
        make_operator<habana::AddOperator>(device_id, scalar_type);
    add_exp_avg_sq->SetSynapseInput(sqrt_exp_avg_sq->GetSynOutputs()[0]);
    stack.emplace_back(IValue(sqrt_exp_avg_sq->GetOutputs()[0]));
    stack.emplace_back(IValue(epsilon));
    stack.emplace_back(IValue(1.0));
    add_exp_avg_sq->AllocateAndAddSynapseNode(
        graph, stack, OutputMetaDataVector(1));
    stack.clear();

    // Replaced addcdiv with following OPs, so that -step_size
    // can be used as a tensor
    // ratio = torch.div(exp_avg, denom)
    // scaled_ratio = torch.mul(ratio, -step_size)
    // p.data.add_(scaled_ratio)
    auto div_wt = make_operator<habana::DivOperator>(device_id, scalar_type);
    sh::tensor& syn_in_17 =
        div_wt->SetSynapseInput(add_exp_avg->GetSynOutputs()[0]);
    div_wt->SetSynapseInput(add_exp_avg_sq->GetSynOutputs()[0]);
    stack.emplace_back(IValue(add_exp_avg->GetOutputs()[0]));
    stack.emplace_back(IValue(add_exp_avg_sq->GetOutputs()[0]));
    div_wt->AllocateAndAddSynapseNode(graph, stack, OutputMetaDataVector(1));
    stack.clear();

    auto mul_wt = make_operator<habana::MulOperator>(device_id, scalar_type);
    mul_wt->SetSynapseInput(div_wt->GetSynOutputs()[0]);
    mul_wt->SetSynapseInput(p_context_->syn_inputs_[4 * num_params]);
    stack.emplace_back(IValue(div_wt->GetOutputs()[0]));
    stack.emplace_back(IValue(neg_step_size));
    mul_wt->AllocateAndAddSynapseNode(graph, stack, OutputMetaDataVector(1));
    stack.clear();

    auto add_wt =
        make_operator<habana::AddInplaceOperator>(device_id, scalar_type);

    if (!is_wd_modified.toBool()) {
      // in this case weight directly comes as input to the fused kernel
      add_wt->SetSynapseInput(p_context_->syn_inputs_[1 * num_params + i]);
      add_wt->SetSynapseInput(mul_wt->GetSynOutputs()[0]);

      stack.emplace_back(IValue(weights.get(i)));
      stack.emplace_back(IValue(mul_wt->GetOutputs()[0]));
      stack.emplace_back(IValue(1.0));
      add_wt->AllocateAndAddSynapseNode(graph, stack, OutputMetaDataVector(1));
      stack.clear();
    } else {
      // use the updated weight tensor after  weight decay operation
      add_wt->SetSynapseInput(mul_wt_wd->GetSynOutputs()[0]);
      stack.emplace_back(IValue(mul_wt_wd->GetOutputs()[0]));

      add_wt->SetSynapseInput(mul_wt->GetSynOutputs()[0]);

      stack.emplace_back(IValue(mul_wt->GetOutputs()[0]));
      stack.emplace_back(IValue(1.0));
      add_wt->AllocateAndAddSynapseNode(graph, stack, OutputMetaDataVector(1));
      stack.clear();
    }

    // Note that these outputs are being filled just to keep GC
    // runtime happy No need to return these since updates on
    // weights, exp_avg, exp_avg_sq are all inplace
    if (is_wd_modified.toBool()) {
      p_context_->syn_outputs_.emplace_back(
          std::move(mul_wt_wd->GetSynOutputs()[0]));
      p_context_->pt_outputs_.emplace_back(mul_wt_wd->GetOutputs()[0]);
    }

    p_context_->syn_outputs_.emplace_back(syn_in_11);
    p_context_->pt_outputs_.emplace_back(mul_exp_avg->GetOutputs()[0]);
    p_context_->syn_outputs_.emplace_back(syn_in_17);
    p_context_->pt_outputs_.emplace_back(add_exp_avg->GetOutputs()[0]);
    p_context_->syn_outputs_.emplace_back(syn_in_14);
    p_context_->pt_outputs_.emplace_back(mul_exp_avg_sq->GetOutputs()[0]);
    p_context_->syn_outputs_.emplace_back(syn_in_15);
    p_context_->pt_outputs_.emplace_back(addcmul_exp_avg_sq->GetOutputs()[0]);
    p_context_->syn_outputs_.emplace_back(
        std::move(add_wt->GetSynOutputs()[0]));
    p_context_->pt_outputs_.emplace_back(add_wt->GetOutputs()[0]);
  }
}

void OptimizerAdagradOperator::AllocateAndAddSynapseNode(
    sh::graph& graph,
    torch::jit::Stack& inputs,
    const OutputMetaDataVector& output_metadata) {
  static_cast<void>(output_metadata);
  TORCH_CHECK(
      inputs.size() == 8,
      "Incorrect size of inputs for optimizer_adagrad operator");
  TORCH_CHECK(inputs[0].isTensor(), "Input arg1 type expected to be tensor");
  TORCH_CHECK(inputs[1].isTensor(), "Input arg2 type expected to be tensor");
  TORCH_CHECK(inputs[2].isTensor(), "Input arg3 type expected to be tensor");
  TORCH_CHECK(inputs[3].isTensor(), "Input arg4 type expected to be tensor");
  TORCH_CHECK(inputs[4].isTensor(), "Input arg5 type expected to be tensor");
  TORCH_CHECK(inputs[5].isDouble(), "Input arg6 type expected to be float");
  TORCH_CHECK(inputs[6].isDouble(), "Input arg7 type expected to be float");
  TORCH_CHECK(inputs[7].isDouble(), "Input arg8 type expected to be float");

  auto gradients = inputs[0].toTensor();
  auto weights = inputs[1].toTensor();
  auto variances = inputs[2].toTensor();
  auto epoch_num = inputs[3].toTensor();
  auto lr = inputs[4].toTensor();

  // std::cout << "weight size "
  //           << weights.sizes() << std::endl;

  ns_OptimizerAdagrad::Params params;
  params.wd = inputs[5].toDouble();
  params.lrd = inputs[6].toDouble();
  params.eps = inputs[7].toDouble();

  // execute in-place for weights & variance
  p_context_->syn_outputs_.emplace_back(
      habana_helpers::duplicate_tensor_in_memory_section(
          p_context_->syn_inputs_[1], graph, output_metadata.at(0).external));

  auto weights_in = inputs[1].toTensor();
  p_context_->pt_outputs_.emplace_back(weights_in);

  p_context_->syn_outputs_.emplace_back(
      habana_helpers::duplicate_tensor_in_memory_section(
          p_context_->syn_inputs_[2], graph, output_metadata.at(1).external));

  auto variance_in = inputs[2].toTensor();
  p_context_->pt_outputs_.emplace_back(variance_in);

  AddNodeToSynapseGraph(graph, &params, sizeof(params));
}

void OptimizerFusedAdagradOperator::AllocateAndAddSynapseNode(
    sh::graph& graph,
    torch::jit::Stack& inputs,
    const OutputMetaDataVector& output_metadata) {
  TORCH_CHECK(
      inputs.size() == 8,
      "Incorrect size of inputs for optimizer fused adagrad operator");
  TORCH_CHECK(
      inputs[0].isTensorList(), "Input arg1 type expected to be tensorlist");
  TORCH_CHECK(
      inputs[1].isTensorList(), "Input arg2 type expected to be tensorlist");
  TORCH_CHECK(
      inputs[2].isTensorList(), "Input arg3 type expected to be tensorlist");
  TORCH_CHECK(inputs[3].isTensor(), "Input arg4 type expected to be tensor");
  TORCH_CHECK(inputs[4].isTensor(), "Input arg5 type expected to be tensor");
  TORCH_CHECK(inputs[5].isDouble(), "Input arg6 type expected to be float");
  TORCH_CHECK(inputs[6].isDouble(), "Input arg7 type expected to be float");
  TORCH_CHECK(inputs[7].isDouble(), "Input arg8 type expected to be float");

  auto gradients = inputs[0].toTensorList();
  auto weights = inputs[1].toTensorList();
  auto variances = inputs[2].toTensorList();
  auto epoch_num = inputs[3].toTensor();
  auto lr = inputs[4].toTensor();

  auto num_params = static_cast<unsigned int>(gradients.size());

  torch::jit::Stack stack;
  size_t device_id = gradients.get(0).device().index();
  auto scalar_type = gradients.get(0).scalar_type();

  for (unsigned int i = 0; i < num_params; i++) {
    auto op = make_operator<OptimizerAdagradOperator>(device_id, scalar_type);
    op->SetSynapseInput(p_context_->syn_inputs_[i]);
    op->SetSynapseInput(p_context_->syn_inputs_[num_params + i]);
    op->SetSynapseInput(p_context_->syn_inputs_[2 * num_params + i]);
    op->SetSynapseInput(p_context_->syn_inputs_[3 * num_params]);
    op->SetSynapseInput(p_context_->syn_inputs_[3 * num_params + 1]);

    stack.emplace_back(IValue(gradients.get(i)));
    stack.emplace_back(IValue(weights.get(i)));
    stack.emplace_back(IValue(variances.get(i)));
    stack.emplace_back(inputs[3]);
    stack.emplace_back(inputs[4]);
    stack.emplace_back(inputs[5]);
    stack.emplace_back(inputs[6]);
    stack.emplace_back(inputs[7]);

    op->AllocateAndAddSynapseNode(
        graph, stack, SelectVectorIndices(output_metadata, {i * 2, i * 2 + 1}));

    stack.clear();

    p_context_->syn_outputs_.emplace_back(std::move(op->GetSynOutputs()[0]));
    p_context_->pt_outputs_.emplace_back(op->GetOutputs()[0]);

    p_context_->syn_outputs_.emplace_back(std::move(op->GetSynOutputs()[1]));
    p_context_->pt_outputs_.emplace_back(op->GetOutputs()[1]);

  } // for (auto i = 0;i < num_params;i++)
}

// SGD Optimizer
void OptimizerSGDOperator::AllocateAndAddSynapseNode(
    sh::graph& graph,
    torch::jit::Stack& inputs,
    const OutputMetaDataVector& output_metadata) {
  PT_OTHER_OPS_BEGIN;
  static_cast<void>(output_metadata);
  TORCH_CHECK(
      inputs.size() == 7,
      "Incorrect size of inputs for optimizer SGD operator");
  TORCH_CHECK(inputs[0].isTensor(), "Input arg1 type expected to be tensor");
  TORCH_CHECK(inputs[1].isTensor(), "Input arg2 type expected to be tensor");
  TORCH_CHECK(inputs[2].isTensor(), "Input arg3 type expected to be tensor");
  TORCH_CHECK(inputs[3].isDouble(), "Input arg4 type expected to be float");
  TORCH_CHECK(inputs[4].isDouble(), "Input arg5 type expected to be float");
  TORCH_CHECK(inputs[5].isDouble(), "Input arg6 type expected to be float");
  TORCH_CHECK(inputs[6].isBool(), "Input arg7 type expected to be bool");

  auto gradients = inputs[0].toTensor();
  auto weights = inputs[1].toTensor();
  auto lr = inputs[2].toTensor();

  ns_OptimizerSGD::Params params;
  params.wd = inputs[3].toDouble();
  params.mom = inputs[4].toDouble();
  params.damp = inputs[5].toDouble();
  params.nesterov = inputs[6].toBool();

  // execute in-place for weights
  p_context_->syn_outputs_.emplace_back(
      habana_helpers::duplicate_tensor_in_memory_section(
          p_context_->syn_inputs_[1], graph, output_metadata.at(0).external));

  auto weights_in = inputs[1].toTensor();
  p_context_->pt_outputs_.emplace_back(weights_in);

  AddNodeToSynapseGraph(graph, &params, sizeof(params));
  PT_OTHER_OPS_END;
}

void OptimizerFusedSGDOperator::AllocateAndAddSynapseNode(
    sh::graph& graph,
    torch::jit::Stack& inputs,
    const OutputMetaDataVector& output_metadata) {
  PT_OTHER_OPS_BEGIN;

  TORCH_CHECK(
      inputs.size() == 7,
      "Incorrect size of inputs for optimizer fused SGD operator");
  TORCH_CHECK(
      inputs[0].isTensorList(), "Input arg1 type expected to be tensorlist");
  TORCH_CHECK(
      inputs[1].isTensorList(), "Input arg2 type expected to be tensorlist");
  TORCH_CHECK(inputs[2].isTensor(), "Input arg3 type expected to be tensor");
  TORCH_CHECK(inputs[3].isDouble(), "Input arg4 type expected to be float");
  TORCH_CHECK(inputs[4].isDouble(), "Input arg5 type expected to be float");
  TORCH_CHECK(inputs[5].isDouble(), "Input arg6 type expected to be float");
  TORCH_CHECK(inputs[6].isBool(), "Input arg7 type expected to be bool");

  auto gradients = inputs[0].toTensorList();
  auto weights = inputs[1].toTensorList();
  auto lr = inputs[2].toTensor();

  auto num_params = static_cast<unsigned int>(gradients.size());

  torch::jit::Stack stack;
  size_t device_id = gradients.get(0).device().index();

  for (unsigned int i = 0; i < num_params; i++) {
    auto op =
        make_operator<OptimizerSGDOperator>(device_id, at::ScalarType::Float);
    op->SetSynapseInput(p_context_->syn_inputs_[i]);
    op->SetSynapseInput(p_context_->syn_inputs_[num_params + i]);
    op->SetSynapseInput(p_context_->syn_inputs_[2 * num_params]);

    stack.emplace_back(IValue(gradients.get(i)));
    stack.emplace_back(IValue(weights.get(i)));
    stack.emplace_back(inputs[2]);
    stack.emplace_back(inputs[3]);
    stack.emplace_back(inputs[4]);
    stack.emplace_back(inputs[5]);
    stack.emplace_back(inputs[6]);

    op->AllocateAndAddSynapseNode(
        graph, stack, SelectVectorIndices(output_metadata, {i}));

    stack.clear();

    p_context_->syn_outputs_.emplace_back(std::move(op->GetSynOutputs()[0]));
    p_context_->pt_outputs_.emplace_back(op->GetOutputs()[0]);

  } // for (auto i = 0;i < num_params;i++)
  PT_OTHER_OPS_END;
}

void OptimizerFusedEMAOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    const OutputMetaDataVector& output_metadata) {
  static_cast<void>(output_metadata);
  TORCH_CHECK(
      inputs.size() == 3,
      "Incorrect size of inputs for fused ema optimizer graph creation call");

  TORCH_CHECK(
      inputs[0].isTensorList(), "Input arg1 type expected to be tensorlist");
  TORCH_CHECK(
      inputs[1].isTensorList(), "Input arg2 type expected to be tensorlist");
  TORCH_CHECK(inputs[2].isTensor(), "Input arg3 type expected to be Tensor");

  auto model_inputs = inputs[0].toTensorList();
  auto updated_ema = inputs[1].toTensorList();
  auto decay = inputs[2].toTensor();

  auto device_id = updated_ema.get(0).device().index();
  auto scalar_type = updated_ema.get(0).scalar_type();
  auto num_params = static_cast<unsigned int>(updated_ema.size());

  torch::jit::Stack stack;

  OutputMetaDataVector outputMetaData(1);
  for (auto& md : outputMetaData) {
    md.persistent = true;
  }

  // EMA Kernel Computations
  // for k, v in self.ema.state_dict().items():
  // v *= d
  // v += (1. - d) * msd[k]

  // v = updated_ema
  // d = decay
  // msd = model.module.state_dict() - module_inputs

  for (unsigned int i = 0; i < num_params; i++) {
    auto mul_in_exp =
        make_operator<habana::MulOperator>(device_id, scalar_type);
    mul_in_exp->SetSynapseInput(p_context_->syn_inputs_[num_params + i]);
    mul_in_exp->SetSynapseInput(p_context_->syn_inputs_[2 * num_params]);
    stack.emplace_back(IValue(updated_ema.get(i)));
    stack.emplace_back(IValue(decay));
    mul_in_exp->AllocateAndAddSynapseNode(
        graph, stack, OutputMetaDataVector(1));
    stack.clear();

    auto sub_exp = make_operator<habana::SubOperator>(device_id, scalar_type);
    sub_exp->SetSynapseInput(p_context_->syn_inputs_[2 * num_params]);
    stack.emplace_back(IValue(1.0));
    stack.emplace_back(IValue(decay));
    stack.emplace_back(IValue(1.0));
    sub_exp->AllocateAndAddSynapseNode(graph, stack, OutputMetaDataVector(1));
    stack.clear();

    auto mul_exp = make_operator<habana::MulOperator>(device_id, scalar_type);
    mul_exp->SetSynapseInput(p_context_->syn_inputs_[i]);
    mul_exp->SetSynapseInput(sub_exp->GetSynOutputs()[0]);
    stack.emplace_back(IValue(model_inputs.get(i)));
    stack.emplace_back(IValue(sub_exp->GetOutputs()[0]));
    mul_exp->AllocateAndAddSynapseNode(graph, stack, OutputMetaDataVector(1));
    stack.clear();

    auto update_exp =
        make_operator<habana::AddOperator>(device_id, scalar_type);

    synapse_helpers::tensor& updt_syn_T =
        update_exp->SetSynapseInput(mul_exp->GetSynOutputs()[0]);
    update_exp->SetSynapseInput(mul_in_exp->GetSynOutputs()[0]);

    stack.emplace_back(IValue(mul_exp->GetOutputs()[0]));
    stack.emplace_back(IValue(mul_in_exp->GetOutputs()[0]));
    // dummy input
    stack.emplace_back(IValue(Scalar(1.0)));
    update_exp->AllocateAndAddSynapseNode(graph, stack, outputMetaData);
    stack.clear();

    p_context_->syn_outputs_.emplace_back(
        std::move(update_exp->GetSynOutputs()[0]));
    p_context_->pt_outputs_.emplace_back(update_exp->GetOutputs()[0]);

    auto synInput = habana_helpers::duplicate_tensor_in_memory_section(
        updt_syn_T, graph, output_metadata.at(0).external);
  }
}

void OptimizerSGDMomentumOperator::AllocateAndAddSynapseNode(
    sh::graph& graph,
    torch::jit::Stack& inputs,
    const OutputMetaDataVector& output_metadata) {
  PT_OTHER_OPS_BEGIN;
  static_cast<void>(output_metadata);
  TORCH_CHECK(
      inputs.size() == 9,
      "Incorrect size of inputs for optimizer SGD operator");
  TORCH_CHECK(inputs[0].isTensor(), "Input arg1 type expected to be tensor");
  TORCH_CHECK(inputs[1].isTensor(), "Input arg2 type expected to be tensor");
  TORCH_CHECK(inputs[2].isTensor(), "Input arg3 type expected to be tensor");
  TORCH_CHECK(inputs[3].isTensor(), "Input arg4 type expected to be tensor");
  TORCH_CHECK(inputs[4].isTensor(), "Input arg5 type expected to be tensor");
  TORCH_CHECK(inputs[5].isTensor(), "Input arg6 type expected to be tensor");
  TORCH_CHECK(inputs[6].isDouble(), "Input arg7 type expected to be float");
  TORCH_CHECK(inputs[7].isDouble(), "Input arg8 type expected to be float");
  TORCH_CHECK(inputs[8].isBool(), "Input arg9 type expected to be bool");

  auto gradients = inputs[0].toTensor();
  if (habana_lazy::GetHbInternalTensorImpl(gradients)) {
    PT_BRIDGE_DEBUG(
        "OptimizerSGDMomentumOperator lowering gradient HbInternal address: ",
        habana_lazy::GetHbInternalTensorImpl(gradients),
        " permute: ",
        VecToString(habana_lazy::GetHbInternalTensorImpl(gradients)
                        ->GetMemoryPermutation()));
  } else {
    PT_BRIDGE_DEBUG(
        "OptimizerSGDMomentumOperator lowering - gradients HbInternal address is null!")
  }
  auto weights = inputs[1].toTensor();
  if (habana_lazy::GetHbInternalTensorImpl(weights)) {
    PT_BRIDGE_DEBUG(
        "OptimizerSGDMomentumOperator lowering weight HbInternal address: ",
        habana_lazy::GetHbInternalTensorImpl(weights),
        " permute: ",
        VecToString(habana_lazy::GetHbInternalTensorImpl(weights)
                        ->GetMemoryPermutation()));
  } else {
    PT_BRIDGE_DEBUG(
        "OptimizerSGDMomentumOperator lowering - weights HbInternal address is null!")
  }
  auto momentum = inputs[2].toTensor();
  if (habana_lazy::GetHbInternalTensorImpl(momentum)) {
    PT_BRIDGE_DEBUG(
        "OptimizerSGDMomentumOperator lowering momentum HbInternal address: ",
        habana_lazy::GetHbInternalTensorImpl(momentum),
        " permute: ",
        VecToString(habana_lazy::GetHbInternalTensorImpl(momentum)
                        ->GetMemoryPermutation()));
  } else {
    PT_BRIDGE_DEBUG(
        "OptimizerSGDMomentumOperator lowering - momentum HbInternal address is null!")
  }
  auto epoch_num = inputs[3].toTensor();
  auto lr = inputs[4].toTensor();
  auto mom = inputs[5].toTensor();

  ns_OptimizerSGD::Params params;
  params.wd = inputs[6].toDouble();
  // we use mom tensor instead. setting to some non zero as a hack. Need fix
  // from tpc glue
  params.mom = (float)0.1;
  params.damp = inputs[7].toDouble();
  params.nesterov = inputs[8].toBool();

  // execute in-place for weights & momentum
  p_context_->syn_outputs_.emplace_back(
      habana_helpers::duplicate_tensor_in_memory_section(
          p_context_->syn_inputs_[1], graph, output_metadata.at(0).external));

  auto weights_in = inputs[1].toTensor();
  p_context_->pt_outputs_.emplace_back(weights_in);

  p_context_->syn_outputs_.emplace_back(
      habana_helpers::duplicate_tensor_in_memory_section(
          p_context_->syn_inputs_[2], graph, output_metadata.at(1).external));

  auto momentum_in = inputs[2].toTensor();
  p_context_->pt_outputs_.emplace_back(momentum_in);

  AddNodeToSynapseGraph(graph, &params, sizeof(params));
  PT_OTHER_OPS_END;
}

void OptimizerFusedSGDMomentumOperator::AllocateAndAddSynapseNode(
    sh::graph& graph,
    torch::jit::Stack& inputs,
    const OutputMetaDataVector& output_metadata) {
  PT_OTHER_OPS_BEGIN;
  TORCH_CHECK(
      inputs.size() == 9,
      "Incorrect size of inputs for optimizer fused SGD operator");
  TORCH_CHECK(
      inputs[0].isTensorList(), "Input arg1 type expected to be tensorlist");
  TORCH_CHECK(
      inputs[1].isTensorList(), "Input arg2 type expected to be tensorlist");
  TORCH_CHECK(
      inputs[2].isTensorList(), "Input arg3 type expected to be tensorlist");
  TORCH_CHECK(inputs[3].isTensor(), "Input arg4 type expected to be tensor");
  TORCH_CHECK(inputs[4].isTensor(), "Input arg5 type expected to be tensor");
  TORCH_CHECK(inputs[5].isTensor(), "Input arg6 type expected to be tensor");
  TORCH_CHECK(inputs[6].isDouble(), "Input arg7 type expected to be float");
  TORCH_CHECK(inputs[7].isDouble(), "Input arg8 type expected to be float");
  TORCH_CHECK(inputs[8].isBool(), "Input arg9 type expected to be bool");

  auto gradients = inputs[0].toTensorList();
  auto weights = inputs[1].toTensorList();
  auto momentum = inputs[2].toTensorList();
  auto epoch_num = inputs[3].toTensor();
  auto lr = inputs[4].toTensor();
  auto mom = inputs[5].toTensor();

  auto num_params = static_cast<unsigned int>(gradients.size());

  torch::jit::Stack stack;
  size_t device_id = gradients.get(0).device().index();

  for (unsigned int i = 0; i < num_params; i++) {
    auto op = make_operator<OptimizerSGDMomentumOperator>(
        device_id, at::ScalarType::Float);
    op->SetSynapseInput(p_context_->syn_inputs_[i]);
    op->SetSynapseInput(p_context_->syn_inputs_[num_params + i]);
    op->SetSynapseInput(p_context_->syn_inputs_[2 * num_params + i]);
    op->SetSynapseInput(p_context_->syn_inputs_[3 * num_params]);
    op->SetSynapseInput(p_context_->syn_inputs_[3 * num_params + 1]);
    op->SetSynapseInput(
        p_context_->syn_inputs_[3 * num_params + 2]); // mom tensor

    stack.emplace_back(IValue(gradients.get(i)));
    stack.emplace_back(IValue(weights.get(i)));
    stack.emplace_back(IValue(momentum.get(i)));
    stack.emplace_back(inputs[3]);
    stack.emplace_back(inputs[4]);
    stack.emplace_back(inputs[5]);
    stack.emplace_back(inputs[6]);
    stack.emplace_back(inputs[7]);
    stack.emplace_back(inputs[8]);

    op->AllocateAndAddSynapseNode(
        graph, stack, SelectVectorIndices(output_metadata, {i * 2, i * 2 + 1}));

    stack.clear();

    p_context_->syn_outputs_.emplace_back(std::move(op->GetSynOutputs()[0]));

    p_context_->pt_outputs_.emplace_back(op->GetOutputs()[0]);

    p_context_->syn_outputs_.emplace_back(std::move(op->GetSynOutputs()[1]));
    p_context_->pt_outputs_.emplace_back(op->GetOutputs()[1]);
  } // for (auto i = 0;i < num_params;i++)

  PT_OTHER_OPS_END;
}

namespace habana {
class OptimizerFusedLarsOperatorLazy : public OpBackend {
 public:
  OptimizerFusedLarsOperatorLazy(int device_id, c10::ScalarType scalar_type)
      : OpBackend(
            device_id,
            NO_TPC + "optimizer_fused_lars_",
            scalar_type,
            {0}, // outplace id
            {},
            {},
            false) {
    this->CreateSynContext(device_id);
    SetOutputMetaFn(OptimizerFusedLarsMeta);
  }
  static OutputMetaDataVector OptimizerFusedLarsMeta(const at::Stack&);

  void AddNode(sh::graph& graph, const at::Stack& stack) override;
};

OutputMetaDataVector OptimizerFusedLarsOperatorLazy::OptimizerFusedLarsMeta(
    const at::Stack& stack) {
  auto grads = stack.at(0).toTensorList();
  auto tlSize = grads.size();

  OutputMetaDataVector meta_vec;
  meta_vec.reserve(tlSize);

  for (const at::Tensor& grad : grads) {
    OutputMetaData meta;
    meta.shape = grad.sizes().vec();
    meta.dtype = grad.scalar_type();
    meta_vec.emplace_back(meta);
  }
  return meta_vec;
}

void OptimizerFusedLarsOperatorLazy::AddNode(
    sh::graph& graph,
    const at::Stack& stack) {
  auto params = stack.at(1).toTensorList();
  auto grads = stack.at(0).toTensorList();
  auto skipMasks = stack.at(3).toIntList();
  auto eeta = stack.at(4).toDouble();
  auto weightDecay = stack.at(5).toDouble();
  auto eps = stack.at(6).toDouble();

  auto dtype = grads.get(0).scalar_type();
  auto tlSize = grads.size();
  // syn_in[] is arranged as [[grads],[params], lr]
  // where grads and params are vectors of size tlSize
  // and lr is a single tensor corr. to the float lr value.
  auto syn_lr = syn_in(2 * tlSize);

  for (size_t i = 0; i < tlSize; ++i) {
    auto grad = grads.get(i);
    auto param = params.get(i);
    auto outshape = grad.sizes();
    auto zero_constant = ConstantHelper(graph, 0.0f, dtype, outshape);
    auto one_constant = ConstantHelper(graph, 1.0f, dtype, outshape);
    auto eetaTensor = ConstantHelper(graph, eeta, dtype, outshape);
    auto weightDecayTensor =
        ConstantHelper(graph, weightDecay, dtype, outshape);
    auto epsTensor = ConstantHelper(graph, eps, dtype, outshape);

    auto syn_grad = syn_in(i);

    if (!skipMasks[i]) {
      auto mul0 = BuildOp(
          graph,
          get_guid_with_precision("mult", dtype),
          {syn_grad, syn_lr},
          {{outshape, dtype, i}});
      syn_out(i) = std::move(mul0[0]);
      continue;
    }
    auto syn_param = syn_in(i + tlSize);
    auto n_dims = grad.dim();

    auto mul1 = BuildOp(
        graph,
        get_guid_with_precision("mult", dtype),
        {syn_param, syn_param},
        {{outshape, dtype}});

    std::vector<synTensor> reduction_inputs1 = {mul1[0].get()};
    std::vector<sh::tensor> reshape1;

    if (n_dims > 1) {
      auto reshape_outshape = grad.numel();
      reshape1.emplace_back(
          ReshapeHelper(graph, reduction_inputs1[0], reshape_outshape, dtype));
      reduction_inputs1 = {reshape1[0].get()};
    }

    ns_Reduction::Params reduce_params{};
    reduce_params.reductionDimension = 0;
    auto sum1 = BuildOp(
        graph,
        get_guid_with_precision("reduce_sum_fwd", dtype),
        std::move(reduction_inputs1),
        {{1, dtype}},
        &reduce_params,
        sizeof(reduce_params));

    auto sqrt1 = BuildOp(
        graph,
        get_guid_with_precision("sqrt_fwd", dtype),
        {sum1[0].get()},
        {{1, dtype}});

    // Norm calculation for 1-st argument viz. param: mul2, sum2, sqrt2
    auto mul2 = BuildOp(
        graph,
        get_guid_with_precision("mult", dtype),
        {syn_grad, syn_grad},
        {{outshape, dtype}});

    std::vector<synTensor> reduction_inputs2 = {mul2[0].get()};
    std::vector<sh::tensor> reshape2;

    if (n_dims > 1) {
      auto reshape_outshape = grad.numel();
      reshape2.emplace_back(
          ReshapeHelper(graph, reduction_inputs2[0], reshape_outshape, dtype));
      reduction_inputs2 = {reshape2[0].get()};
    }

    auto sum2 = BuildOp(
        graph,
        get_guid_with_precision("reduce_sum_fwd", dtype),
        std::move(reduction_inputs2),
        {{1, dtype}},
        &reduce_params,
        sizeof(reduce_params));

    auto sqrt2 = BuildOp(
        graph,
        get_guid_with_precision("sqrt_fwd", dtype),
        {sum2[0].get()},
        {{1, dtype}});

    // torch.greater(param_norm, 0)
    auto ge1 = BuildOp(
        graph,
        get_guid_with_precision("greater_fwd", dtype),
        {sqrt1[0].get(), zero_constant.get()},
        {{outshape, dtype}});

    // torch.greater(grad_norm, 0)
    auto ge2 = BuildOp(
        graph,
        get_guid_with_precision("greater_fwd", dtype),
        {sqrt2[0].get(), zero_constant.get()},
        {{outshape, dtype}});

    // eeta*paramNorm
    auto mul3 = BuildOp(
        graph,
        get_guid_with_precision("mult", dtype),
        {sqrt1[0].get(), eetaTensor.get()},
        {{outshape, dtype}});

    // paranNorm*weightDecay
    auto mul4 = BuildOp(
        graph,
        get_guid_with_precision("mult", dtype),
        {sqrt1[0].get(), weightDecayTensor.get()},
        {{outshape, dtype}});

    // weightDecay*paranNorm + eps
    auto add1 = BuildOp(
        graph,
        get_guid_with_precision("add_fwd", dtype),
        {mul4[0].get(), epsTensor.get()},
        {{outshape, dtype}});

    // gradNorm + weightDecay*paranNorm + eps
    auto add2 = BuildOp(
        graph,
        get_guid_with_precision("add_fwd", dtype),
        {add1[0].get(), sqrt2[0].get()},
        {{outshape, dtype}});

    //(eeta*param_norm) / (gradNorm + weightDecay*paranNorm + eps)
    auto div1 = BuildOp(
        graph,
        get_guid_with_precision("div_fwd", dtype),
        {mul3[0].get(), add2[0].get()},
        {{outshape, dtype}});

    auto where1 = BuildOp(
        graph,
        get_guid_with_precision("where_fwd", dtype),
        {ge2[0].get(), div1[0].get(), one_constant.get()},
        {{outshape, dtype}});

    // trust_ratio
    auto where2 = BuildOp(
        graph,
        get_guid_with_precision("where_fwd", dtype),
        {ge1[0].get(), where1[0].get(), one_constant.get()},
        {{outshape, dtype}});

    // scaled_lr = lr*trust_ratio
    auto mul5 = BuildOp(
        graph,
        get_guid_with_precision("mult", dtype),
        {where2[0].get(), syn_lr},
        {{outshape, dtype}});

    // param*weightDecayTensor
    auto mul6 = BuildOp(
        graph,
        get_guid_with_precision("mult", dtype),
        {syn_param, weightDecayTensor.get()},
        {{outshape, dtype}});

    // grad + param*weightDecayTensor
    auto add3 = BuildOp(
        graph,
        get_guid_with_precision("add_fwd", dtype),
        {syn_grad, mul6[0].get()},
        {{outshape, dtype}});

    // param*weightDecayTensor
    auto mul7 = BuildOp(
        graph,
        get_guid_with_precision("mult", dtype),
        {add3[0].get(), mul5[0].get()},
        {{outshape, dtype, i}});

    syn_out(i) = std::move(mul7[0]);
  } // for (size_t i=0; i< tlSize; ++i)
}

} // namespace habana

static auto& OptimizerKernelsKernelRegistry =
    habana::KernelRegistry()
        .add(
            "hpu::habanaOptimizerSparseSgd",
            KERNEL_FN(OptimizerSparseSgdOperator))
        .add(
            "hpu::habanaOptimizerSparseAdagrad",
            KERNEL_FN(OptimizerSparseAdagradOperator))
        .add("hpu::habanaOptimizerAdamW", KERNEL_FN(OptimizerAdamwOperator))
        .add(
            "hpu::habanaOptimizerFusedAdagrad",
            KERNEL_FN(OptimizerFusedAdagradOperator))
        .add("hpu::habanaOptimizerSgd", KERNEL_FN(OptimizerFusedSGDOperator))
        .add(
            "hpu::habanaOptimizerSgdMomentum",
            KERNEL_FN(OptimizerFusedSGDMomentumOperator))
        .add(
            "hpu::habanaOptimizerLars",
            KERNEL_FN(OptimizerFusedLarsOperatorLazy))
        .add(
            "hpu::habanaOptimizerFusedEMA",
            KERNEL_FN(OptimizerFusedEMAOperator));
