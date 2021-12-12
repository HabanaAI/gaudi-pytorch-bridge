/******************************************************************************
 * Copyright (C) 2022 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */
#include "sbs_runner.h"
#include <exception>
#include "aten_lazy_bridge.h"

namespace habana_lazy {
static at::IValue gatherInputForCPUOp(const at::Tensor& input) {
  auto hl_input = GetHbLazyTensor(input);
  c10::optional<at::Tensor> pTensor = hl_input.GetCPUTensorData();
  if (pTensor != c10::nullopt) {
    PT_LAZY_DEBUG("pTensor != c10::nullopt");
    // Decision point to use HPU data
    if ((GET_ENV_FLAG_NEW(PT_SBS) == SBSModes::SBS_MODE_USE_HPU_INPUT) &&
        (hl_input.GetSBSLiveTensorIndication())) {
      PT_LAZY_DEBUG("HPU input, decision point. Type: ", input.scalar_type());
      return std::move(input.to(c10::kCPU));
    } else {
      PT_LAZY_DEBUG(
          "CPU input, or not decision point. Type: ",
          pTensor.value().scalar_type());
      return std::move(pTensor.value());
    }
  } else {
    PT_LAZY_DEBUG("pTensor == c10::nullopt");
    return std::move(input.to(c10::kCPU));
  }
}

static torch::jit::Value* addInputToGraph(
    const at::Tensor& input,
    std::shared_ptr<torch::jit::Graph>& graph) {
  auto hl_input = GetHbLazyTensor(input);
  auto inp = hl_input.GetIrValue();

  auto t = graph->addInput(inp.ToString());
  HABANA_ASSERT(!inp.m_data_ptr.expired());
  std::shared_ptr<Data> d = inp.m_data_ptr.lock();

  t->setType(c10::TensorType::create(
      d->logical_element_type, d->device, d->sizes.size(), false));
  t->setDebugName(inp.ToString());

  return t;
}

static std::shared_ptr<torch::jit::Operator> createCPUOperator(
    ir::NodePtr node,
    const std::vector<at::IValue>& inputs) {
  std::vector<torch::jit::Value*> node_inputs(
      inputs.size()); // includes input and metadata
  auto graph = std::make_shared<torch::jit::Graph>();
  auto nodeMetadata = node->GetMetaData();

  for (size_t i = 0; i < inputs.size(); ++i) {
    auto input = inputs[i];
    if (nodeMetadata.count(i)) {
      PT_LAZY_DEBUG("input index: ", i, " is metadata");
      if (input.isTensor()) // undefined tensor as decided in
                            // lazy_kernels.h::create_node()
      {
        node_inputs[i] =
            graph->addInput(); // addInputToGraph(input.toTensor(), graph);
      } else {
        node_inputs[i] = graph->insertConstant(input);
      }
    } else if (input.isTensor()) {
      PT_LAZY_DEBUG("input index: ", i, " is tensor");
      node_inputs[i] = addInputToGraph(input.toTensor(), graph);
    } else if (input.isTensorList()) {
      TORCH_CHECK(false, "got tensor list: i=", i, " this is unhandled now");
    } // scalar or something else
    // logic taken from LazyOp::create_node()
    else {
      PT_LAZY_DEBUG(
          "input index: ",
          i,
          " is something else. Treating as scalar. type: ",
          input.toScalar().type());
      node_inputs[i] = graph->insertConstant(input.toScalar());
    }
  }

  at::ArrayRef<torch::jit::Value*> args(node_inputs);
  auto jit_node = graph->create(node->op(), args, node->GetNumOutputs());
  if (!jit_node) {
    PT_LAZY_WARN(
        "SBS: CPU Op was not found for: " +
        std::string(node->op().toQualString()));
    return nullptr;
  }
  if (!node->GetName().empty()) {
    jit_node->s_(c10::attr::name, node->GetName());
  }

  auto op = std::make_shared<torch::jit::Operator>(jit_node->getOperator());
  if (op) {
    PT_LAZY_DEBUG(
        "SBS: CPU Op was found for: " + std::string(node->op().toQualString()),
        " Schema: ",
        torch::jit::canonicalSchemaString(op->schema()));
  } else {
    PT_LAZY_WARN(
        "SBS: CPU Op was not found for: " +
        std::string(node->op().toQualString()));
  }

  return op;
}

void SBSRunner::populateInputForCPUOp(
    const std::vector<at::IValue>& inputs,
    const ir::MetaData& metadata,
    std::vector<at::IValue>& stack) {
  stack.resize(inputs.size());
  for (size_t i = 0; i < inputs.size(); ++i) {
    auto input = inputs[i];
    if (metadata.count(i)) {
      if (input.isTensor()) {
        PT_LAZY_DEBUG("input index: ", i, " is metadata, tensor");
        stack[i] = at::IValue();
      } else {
        PT_LAZY_DEBUG("input index: ", i, " is metadata, not tensor");
        stack[i] = std::move(input);
      }
    } else if (input.isTensor()) {
      PT_LAZY_DEBUG("input index: ", i, " is tensor");
      stack[i] = gatherInputForCPUOp(input.toTensor());
    } else if (input.isTensorList()) {
      TORCH_CHECK(false, "got tensor list: i=", i, " this is unhandled");
    } else if (input.isScalar()) {
      PT_LAZY_DEBUG("input index: ", i, " is scalar");
      stack[i] = std::move(input.toScalar());
    } else {
      PT_LAZY_DEBUG(
          "input index: ", i, " is something else, this is unhandled");
    }

    if (stack[i].isTensor()) {
      TORCH_CHECK(
          stack[i].toTensor().device().type() == c10::DeviceType::CPU,
          "Input tensor to CPU Op is not in CPU. Stack index: ",
          i);
    }
  }
}

void SBSRunner::run(
    at::TensorList results,
    const std::vector<at::IValue>& inputs,
    const std::vector<at::IValue>& prealloc_stack) {
  PT_LAZY_TRACE;
  if (GET_ENV_FLAG_NEW(PT_SBS) == SBSModes::SBS_MODE_DISABLED) {
    return;
  }
  // Get the CPU Op
  // getting ir node, we'll need it to make the jit node
  auto hl_result = GetHbLazyTensor(results[0]);
  auto node = hl_result.CurrentIrValue().mp_node;

  if (!node) {
    return;
  }
  PT_LAZY_DEBUG("Trying SBS for op ", node->GetName());
  auto jit_op = createCPUOperator(node, inputs);
  if (!jit_op) {
    return;
  }

  std::vector<at::IValue> stack = prealloc_stack;
  if (stack.empty()) {
    populateInputForCPUOp(inputs, node->GetMetaData(), stack);
  }

  TORCH_CHECK(
      stack.size() == inputs.size(), "HPU and CPU inputs size should equal");

  try {
    PT_LAZY_DEBUG(
        "SBS: Running CPU Op Schema: ",
        torch::jit::canonicalSchemaString(jit_op->schema()));
    // Run CPU kernel
    jit_op->getOperation()(stack);
  } catch (std::exception& e) {
    std::string error_str = e.what();
    PT_LAZY_WARN(
        "SBS: Failed to run CPU Op type: ",
        node->op().toQualString(),
        ". Node name: ",
        node->GetName(),
        " - Details :\n",
        error_str);
    return;
  }

  PT_LAZY_DEBUG(
      "SBS: CPU Op Schema: ",
      torch::jit::canonicalSchemaString(jit_op->schema()),
      " finished, output stack size: ",
      stack.size());

  TORCH_CHECK(
      stack.size() == results.size(), "HPU and CPU output size should equal");

  // Connect CPU result to HPU result
  for (size_t i = 0; i < stack.size(); ++i) {
    auto& output = stack[i];
    auto& result = results[i];
    if (output.isTensor()) {
      TORCH_CHECK(
          output.toTensor().device().type() == c10::DeviceType::CPU,
          "CPU output tensor is not in cpu. stack index =",
          i);
      auto cpu_res = output.toTensor();
      auto hl_result = GetHbLazyTensor(result);
      PT_LAZY_DEBUG(
          "Setting CPU tensor. ID: ",
          hl_result.getTensorUniqueId(),
          " (op name: ",
          node->GetName(),
          " type: ",
          node->op().toQualString(),
          ")");
      hl_result.SetCPUTensorData(cpu_res);
    } else if (output.isTensorList()) {
      for (const at::Tensor& tensor : output.toTensorList()) {
        // TODO: Are we sure this is the right thing?
        auto cpu_res = tensor;
        auto hl_result = GetHbLazyTensor(result);
        PT_LAZY_DEBUG(
            "Setting CPU tensor. ID: ",
            hl_result.getTensorUniqueId(),
            " (op name: ",
            node->GetName(),
            " type: ",
            node->op().toQualString(),
            ")");
        hl_result.SetCPUTensorData(cpu_res);
      }
    }
  }
}

} // namespace habana_lazy