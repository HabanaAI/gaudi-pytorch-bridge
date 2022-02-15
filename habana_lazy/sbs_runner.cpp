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
#include <memory>
#include <sstream>
#include "aten_lazy_bridge.h"
#include "passes/permute_graph.h"
#include "sbs_debug.h"

namespace habana_lazy {

std::map<std::string, std::shared_ptr<SBSInterface>>
    SBSInterface::m_special_sbs_ops = {
        {"aten::convolution_overrideable",
         std::static_pointer_cast<SBSInterface>(
             std::make_shared<SBSDisabledOp>())},
        {"hpu::nonzero",
         std::static_pointer_cast<SBSInterface>(
             std::make_shared<SBSDisabledOp>())},
        {"aten::ones_like",
         std::static_pointer_cast<SBSInterface>(
             std::make_shared<SBSDisabledOp>())},
        // Failed in LazyIndexKernelTest.IndexTest
        {"hpu::index",
         std::static_pointer_cast<SBSInterface>(
             std::make_shared<SBSDisabledOp>())},
};

std::shared_ptr<SBSInterface> SBSInterface::getSBSHandler(std::string op_type) {
  auto iter = m_special_sbs_ops.find(op_type);
  if (iter != m_special_sbs_ops.end()) {
    return iter->second;
  }
  return std::make_shared<SBSRunner>();
}

bool SBSInterface::LogError(
    const std::string& op_name,
    const std::string& message_short,
    const std::string& message_detailed) {
  return SBSDebug::getInstance().LogError(
      op_name, message_short, message_detailed);
}

void SBSDisabledOp::run(
    at::TensorList results,
    UNUSED const std::vector<at::IValue>& inputs,
    UNUSED const std::vector<at::IValue>& prealloc_stack) {
  auto hl_result = GetHbLazyTensor(results[0]);
  LogError(hl_result.CurrentIrValue().ToString(), "SBS is disabled for op");
}

at::IValue SBSRunner::gatherInputForCPUOp(
    const at::Tensor& input,
    size_t index) {
  auto hl_input = GetHbLazyTensor(input);
  c10::optional<at::Tensor> pTensor = hl_input.GetCPUTensorData();
  if (pTensor != c10::nullopt) {
    PT_LAZY_DEBUG("SBS: pTensor != c10::nullopt");
    // Decision point to use HPU data
    if ((GET_ENV_FLAG_NEW(PT_SBS) == SBSModes::SBS_MODE_USE_HPU_INPUT) &&
        (hl_input.GetSBSLiveTensorIndication())) {
      PT_LAZY_DEBUG(
          "SBS: HPU input, decision point. Type: ", input.scalar_type());
      return std::move(prepareCPUTensor(input.to(c10::kCPU), index));
    } else {
      PT_LAZY_DEBUG(
          "SBS: CPU input, or not decision point. Type: ",
          pTensor.value().scalar_type());
      return std::move(pTensor.value());
    }
  } else {
    PT_LAZY_DEBUG("SBS: pTensor == c10::nullopt");
    return std::move(prepareCPUTensor(input.to(c10::kCPU), index));
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

void SBSRunner::populateInputForCPUOp(
    const std::vector<at::IValue>& inputs,
    const ir::MetaData& metadata,
    std::vector<at::IValue>& stack) {
  if (GET_ENV_FLAG_NEW(PT_SBS) == SBSModes::SBS_MODE_DISABLED) {
    return;
  }
  PT_LAZY_DEBUG("SBS: populateInputForCPUOp: inputs size=", inputs.size());
  stack.resize(inputs.size());
  for (size_t i = 0; i < inputs.size(); ++i) {
    auto input = inputs[i];
    if (metadata.count(i)) {
      if (input.isTensor()) {
        PT_LAZY_DEBUG("SBS: input index: ", i, " is metadata, tensor");
        stack[i] = at::IValue();
      } else {
        PT_LAZY_DEBUG("SBS: input index: ", i, " is metadata, not tensor");
        stack[i] = std::move(input);
      }
    } else if (input.isTensor()) {
      PT_LAZY_DEBUG("SBS: input index: ", i, " is tensor");
      stack[i] = gatherInputForCPUOp(input.toTensor(), i);
    } else if (input.isTensorList()) {
      TORCH_CHECK(false, "SBS: Got tensor list: i=", i, " this is unhandled");
    } else if (input.isScalar()) {
      PT_LAZY_DEBUG("SBS: input index: ", i, " is scalar");
      stack[i] = std::move(input.toScalar());
    } else {
      PT_LAZY_DEBUG(
          "SBS: input index: ", i, " is something else, this is unhandled");
    }

    if (stack[i].isTensor()) {
      TORCH_CHECK(
          stack[i].toTensor().device().type() == c10::DeviceType::CPU,
          "SBS: Input tensor to CPU Op is not in CPU. Stack index: ",
          i);
    }
  }

  PT_LAZY_DEBUG("SBS: populateInputForCPUOp: input stack size=", stack.size());
}

void handleTensorForCPUInput(
    const at::Tensor& input,
    std::vector<at::IValue>& inputs_modified) {
  if (!input.defined()) {
    // setting this undefined tensor, it's meant to be that way
    inputs_modified.push_back(input);
    return;
  }
  if (input.device().type() != c10::DeviceType::HPU) {
    // a special case when tensor is still on CPU - see set_inputs()
    inputs_modified.push_back(std::move(input.to(c10::kHPU)));
    return;
  }
  auto hl_input = GetHbLazyTensor(input);
  c10::optional<at::Tensor> pTensor = hl_input.GetCPUTensorData();
  if ((pTensor != c10::nullopt) && hl_input.GetSBSLiveTensorIndication()) {
    inputs_modified.push_back(std::move(pTensor.value().to(c10::kHPU)));
  } else {
    if (hl_input.GetSBSLiveTensorIndication()) {
      PT_LAZY_WARN(
          "SBS: Tensor is live (decision point), but has no CPU (SBS is not supported). Name: ",
          hl_input.CurrentIrValue().ToString())
    }
    // There's no CPU input or this is not a decision point
    // >> we'll take the HPU data
    inputs_modified.push_back(std::move(input));
  }
}

void SBSRunner::setCPUInputs(const std::vector<at::IValue>& inputs) {
  // PT_SBS=2 means inject CPU inputs to HPU
  if (GET_ENV_FLAG_NEW(PT_SBS) != SBSModes::SBS_MODE_USE_CPU_INPUT) {
    return;
  }
  std::vector<at::IValue> inputs_modified;
  for (auto& input : inputs) {
    if (input.isTensor()) {
      handleTensorForCPUInput(input.toTensor(), inputs_modified);
    } else if (input.isTensorList()) {
      for (const at::Tensor& tensor : input.toTensorList()) {
        handleTensorForCPUInput(tensor, inputs_modified);
      }
    } else {
      inputs_modified.push_back(std::move(input));
    }
  }
}

void SBSRunner::run(
    at::TensorList results,
    const std::vector<at::IValue>& inputs,
    const std::vector<at::IValue>& prealloc_stack) {
  PT_LAZY_TRACE;
  // Get the CPU Op
  // getting ir node, we'll need it to make the jit node
  auto hl_result = GetHbLazyTensor(results[0]);
  auto node = hl_result.CurrentIrValue().mp_node;
  auto ir_name = hl_result.CurrentIrValue().ToString();

  if (!node) {
    LogError(ir_name, "IR Node doesn't exist");
    return;
  }
  PT_LAZY_DEBUG("SBS: Trying SBS for op ", node->GetName());
  auto jit_op = createCPUOperator(ir_name, node, inputs);
  if (!jit_op) {
    return;
  }

  std::vector<at::IValue> stack = prealloc_stack;
  if (stack.empty()) {
    populateInputForCPUOp(inputs, node->GetMetaData(), stack);
  }

  TORCH_CHECK(
      stack.size() == inputs.size(),
      "SBS: HPU and CPU inputs size should equal");

  try {
    PT_LAZY_DEBUG(
        "SBS: Running CPU Op Schema: ",
        torch::jit::canonicalSchemaString(jit_op->schema()));
    // Run CPU kernel
    jit_op->getOperation()(stack);
  } catch (std::exception& e) {
    std::string error_str = e.what();
    std::stringstream ss;
    ss << "Failed to run CPU Op. Details :\n" << error_str;
    LogError(
        ir_name,
        "Failed to run CPU Op. Check lazy log for details and call stack",
        ss.str());
    return;
  }

  PT_LAZY_DEBUG(
      "SBS: CPU Op Schema: ",
      torch::jit::canonicalSchemaString(jit_op->schema()),
      " finished, output stack size: ",
      stack.size());

  TORCH_CHECK(
      stack.size() == results.size(),
      "SBS: HPU and CPU output size should equal");

  // Connect CPU result to HPU result
  for (size_t i = 0; i < stack.size(); ++i) {
    auto& output = stack[i];
    auto& result = results[i];
    if (output.isTensor()) {
      TORCH_CHECK(
          output.toTensor().device().type() == c10::DeviceType::CPU,
          "SBS: CPU output tensor is not in cpu. stack index =",
          i);
      auto cpu_res = output.toTensor();
      auto hl_result = GetHbLazyTensor(result);
      PT_LAZY_DEBUG(
          "SBS: Setting CPU tensor. ID: ",
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
            "SBS: Setting CPU tensor. ID: ",
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

std::shared_ptr<torch::jit::Operator> SBSRunner::createCPUOperator(
    std::string ir_name,
    ir::NodePtr node,
    const std::vector<at::IValue>& inputs) {
  std::vector<torch::jit::Value*> node_inputs(
      inputs.size()); // includes input and metadata
  auto graph = std::make_shared<torch::jit::Graph>();
  auto nodeMetadata = node->GetMetaData();

  for (size_t i = 0; i < inputs.size(); ++i) {
    auto input = inputs[i];
    if (nodeMetadata.count(i)) {
      PT_LAZY_DEBUG("SBS: input index: ", i, " is metadata");
      if (input.isTensor()) // undefined tensor as decided in
                            // lazy_kernels.h::create_node()
      {
        node_inputs[i] = graph->addInput();
      } else {
        node_inputs[i] = graph->insertConstant(input);
      }
    } else if (input.isTensor()) {
      PT_LAZY_DEBUG("SBS: input index: ", i, " is tensor");
      node_inputs[i] = addInputToGraph(input.toTensor(), graph);
    } else if (input.isTensorList()) {
      TORCH_CHECK(
          false, "SBS: Got tensor list: i=", i, " this is unhandled now");
    } // scalar or something else
    // logic taken from LazyOp::create_node()
    else {
      PT_LAZY_DEBUG(
          "SBS: input index: ",
          i,
          " is something else. Treating as scalar. type: ",
          input.toScalar().type());
      node_inputs[i] = graph->insertConstant(input.toScalar());
    }
  }

  auto cpu_op = buildCPUOpSymbol(node->op());

  at::ArrayRef<torch::jit::Value*> args(node_inputs);
  auto jit_node = graph->create(cpu_op, args, node->GetNumOutputs());
  if (!jit_node) {
    LogError(ir_name, "CPU Op was not found (jit node is null)");
    return nullptr;
  }
  if (!node->GetName().empty()) {
    jit_node->s_(c10::attr::debug_name, node->GetName());
  }

  auto op = std::make_shared<torch::jit::Operator>(jit_node->getOperator());
  if (op) {
    PT_LAZY_DEBUG(
        "SBS: CPU Op was found for: " + std::string(node->op().toQualString()),
        " Schema: ",
        torch::jit::canonicalSchemaString(op->schema()));
  } else {
    LogError(ir_name, "CPU Op was not found (jit op is null)");
  }

  return op;
}

c10::Symbol SBSRunner::buildCPUOpSymbol(const c10::Symbol& hpu_op) {
  return hpu_op;
}

at::Tensor SBSRunner::prepareCPUTensor(
    const at::Tensor& tensor,
    UNUSED size_t index) {
  return tensor;
}

} // namespace habana_lazy