/*******************************************************************************
 * Copyright (C) 2020-2023 Habana Labs, Ltd. an Intel Company
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
#if 1 // TODO: TPC kernel seems to give wrong results.
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

#else
#endif

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
    synapse_helpers::graph& graph,
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
    synapse_helpers::graph& graph,
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
  auto scalar_type = gradients.get(0).scalar_type();

  for (unsigned int i = 0; i < num_params; i++) {
    auto op = make_operator<OptimizerSGDOperator>(device_id, scalar_type);
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

void OptimizerSGDMomentumOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
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
    synapse_helpers::graph& graph,
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
  auto scalar_type = gradients.get(0).scalar_type();

  for (unsigned int i = 0; i < num_params; i++) {
    auto op =
        make_operator<OptimizerSGDMomentumOperator>(device_id, scalar_type);
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

TORCH_LIBRARY_FRAGMENT(hpu, m) {
  m.def(
      "habanaOptimizerFusedSGDMomentum(Tensor[] gradients, Tensor(a!)[] weights_in, Tensor(b!)[] momentum_in, Tensor epoch_num, Tensor(c!) learning_rate, Tensor mom, float wd, float damp, bool nesterov) -> ()");
}

static auto& OptimizerKernelsKernelRegistry =
    habana::KernelRegistry()
        .add(
            "hpu::habanaOptimizerSparseSgd",
            KERNEL_FN(OptimizerSparseSgdOperator))
        .add(
            "hpu::habanaOptimizerSparseAdagrad",
            KERNEL_FN(OptimizerSparseAdagradOperator))
        .add(
            "hpu::habanaOptimizerFusedAdagrad",
            KERNEL_FN(OptimizerFusedAdagradOperator))
        .add(
            "hpu::habanaOptimizerFusedSGD",
            KERNEL_FN(OptimizerFusedSGDOperator))
        .add(
            "hpu::habanaOptimizerFusedSGDMomentum",
            KERNEL_FN(OptimizerFusedSGDMomentumOperator));
