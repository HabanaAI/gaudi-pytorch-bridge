/******************************************************************************
 * Copyright (C) 2022 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */

#include "habana_lazy/lazy_graph_hash_builder.h"
#include "habana_lazy/lazy_executor.h"

namespace habana_lazy {

GraphHashBuilder* GraphHashBuilder::instance = nullptr;

// We are working off ir::Node, we would move to a faster implementation later
// We want the hash code to work based on basics like op and its metadata alone
// without the input connections
uint64_t OpArrayEntry::getNodeHash() {
  // the index for the op
  uint64_t hash = index;
  // Hash for op and metadata
  hash = at::hash_combine(hash, node->get_hash_without_connections());
  return hash;
}

uint64_t OpArrayEntry::getNodeOpHash() {
  // the index for the op
  uint64_t hash = index;
  // hash Op id
  hash = at::hash_combine(hash, static_cast<uint32_t>(m_op));
  // Hash metadata
  for (auto& m : meta_data) {
    hash = at::hash_combine(m.first, hash);
    if (m.second.isList()) {
      for (auto& v : m.second.toListRef()) {
        hash = ival_hash(v, hash);
      }
    } else {
      hash = ival_hash(m.second, hash);
    }
  }
  return hash;
}

void OpArrayEntry::populateMetaData(
    const std::vector<c10::IValue>& input_tensors) {
  size_t index = 0;
  for (auto& input : input_tensors) {
    if (isMetadataCandidate(input)) {
      meta_data.insert({index, input});
    }
    index++;
  }
}

void GraphHashBuilder::updateRunningHash() {
  PT_LAZY_TRACE;
  // This must be done after the node and inputs are added to this class
  // auto& node_entry = getLatestEntry();
  // HABANA_ASSERT(node_entry.is_node_addition_done());
  // combine the node_hash with fwd_running_hash...
  fwd_running_hash = at::hash_combine(fwd_running_hash, node_hash);

  // Combine the input hash with fwd_running_hash...
  fwd_running_hash = at::hash_combine(fwd_running_hash, input_hash);
}

void GraphHashBuilder::addNode(ir::Node* node) {
  PT_LAZY_TRACE;
  OpArrayEntry entry;
  entry.addNode(node);
  entry.updateIndex(nodes_array.size());
  node_hash = entry.getNodeHash();
  nodes_array.emplace_back(entry);
}

void GraphHashBuilder::addNode(const c10::Symbol& node_symbol) {
  PT_LAZY_TRACE;
  OpArrayEntry entry;
  entry.addNode(node_symbol);
  entry.updateIndex(nodes_array.size());
  nodes_array.emplace_back(entry);
}

void GraphHashBuilder::prepareInputs(
    const std::vector<uint64_t>& input_map,
    std::vector<ir::Value>& inputs) {
  PT_LAZY_TRACE;
  assert(input_map.size());
  inputs.reserve(input_map.size());
  for (auto idx : input_map) {
    std::shared_ptr<Data> d = graph_input_tensors.at(idx).lock();
    inputs.emplace_back(d->ir_value);
  }
}

void GraphHashBuilder::prepareInputsStackMap(
    const std::vector<ir::Value>& inputs) {
  for (auto in : inputs) {
    std::shared_ptr<Data> d = in.m_data_ptr.lock();
    auto uid = d->unique_id;
    auto itr = std::find(
        graph_input_stack_uids.begin(), graph_input_stack_uids.end(), uid);
    if (itr != graph_input_stack_uids.end()) {
      auto indx = itr - graph_input_stack_uids.begin();
      graph_input_stack_uid_map.emplace_back(indx);
    } else {
      assert(0);
    }
  }
}

uint64_t GraphHashBuilder::getFwdRunningHash() {
  return fwd_running_hash;
}

int64_t GraphHashBuilder::getTensorRunningId(const at::Tensor& tensor) {
  int64_t tid = 0;
  auto hbimpl = dynamic_cast<HbLazyTensorImpl*>(tensor.unsafeGetTensorImpl());
  if (hbimpl) {
    HbLazyTensor hl_t = hbimpl->tensor();
    if (hl_t.getDataPtr()->running_cntr == -1) {
      hl_t.getDataPtr()->running_cntr = getRunningCntr();
    }
    input_hash = at::hash_combine(input_hash, hl_t.getDataPtr()->running_cntr);
  } else {
    HABANA_ASSERT((hbimpl == nullptr), "GetHbLazyTensor for a non lazy tensor");
  }
  return tid;
}

void GraphHashBuilder::invalidateDeviceTids(c10::Device& device) {
  if (!GET_ENV_FLAG_NEW(PT_HPU_ENABLE_GRAPH_RUNNING_HASH))
    return;

  HbContext* devctx = habana_lazy::HbContextArena::Get()->GetHbContext(device);
  for (auto& uid_wptr : devctx->tensors_data) {
    std::shared_ptr<Data> data = uid_wptr.second.lock();
    if (data != nullptr) {
      data->running_cntr = -1;
    }
  }
  for (auto& uid_wptr : devctx->tensors_data_opt) {
    std::shared_ptr<Data> data = uid_wptr.second.lock();
    if (data != nullptr) {
      data->running_cntr = -1;
    }
  }
}

void GraphHashBuilder::graph(
    const c10::Symbol& op_name,
    const std::vector<c10::IValue>& inputs,
    const std::vector<at::Tensor>& outputs) {
  PT_LAZY_TRACE;
  addNode(op_name);
  auto node_entry = getLatestEntry();
  node_entry.populateMetaData(inputs);
  node_hash = node_entry.getNodeOpHash();
  addInputTensors(inputs);
  addOutPutTensors(outputs);
}

void GraphHashBuilder::addInputTensors(
    const std::vector<c10::IValue>& input_tensors) {
  PT_LAZY_TRACE;
  input_hash = 0;
  auto idx = 0;
  for (auto& t : input_tensors) {
    input_hash = at::hash_combine(input_hash, idx);
    if (t.isTensor() && t.toTensor().defined()) {
      // prepare hash using tID
      auto tid = getTensorRunningId(t.toTensor());
      input_hash = at::hash_combine(input_hash, tid);
      // needed to preare InputStack for cache Hit case
      auto hbo1 = GetOrCreateHbLazyTensor(t.toTensor(), t.toTensor().device());
      auto ir_value = hbo1.CurrentIrValue();
      if (ir_value) {
        if (!ir_value.m_data_ptr.expired()) {
          auto shared_ptr = ir_value.m_data_ptr.lock();
          graph_input_tensors.emplace_back(shared_ptr);
          graph_input_stack_uids.emplace_back(shared_ptr->unique_id);
        }
      }
    } else if (t.isTensorList()) {
      auto tvec = t.toTensorVector();
      for (auto& tensor : tvec) {
        // prepare hash using tID
        auto tid = getTensorRunningId(tensor);
        input_hash = at::hash_combine(input_hash, tid);
        // preare InputStack for cache Hit case
        auto hb_t = GetOrCreateHbLazyTensor(tensor, tensor.device());
        auto ir_value = hb_t.CurrentIrValue();
        if (ir_value) {
          if (!ir_value.m_data_ptr.expired()) {
            auto shared_ptr = ir_value.m_data_ptr.lock();
            graph_input_tensors.emplace_back(shared_ptr);
            graph_input_stack_uids.emplace_back(shared_ptr->unique_id);
          }
        }
      }
    }
    idx++;
  }
}

void GraphHashBuilder::addOutPutTensors(
    const std::vector<at::Tensor>& output_tensors) {
  PT_LAZY_TRACE;
  size_t idx = 0;
  for (auto& tensor : output_tensors) {
    input_hash = at::hash_combine(input_hash, idx);
    getTensorRunningId(tensor);
  }
}

void GraphHashBuilder::validateAccumJitOps(
    std::shared_ptr<torch::jit::Graph> mp_g) {
  if (!GET_ENV_FLAG_NEW(PT_HPU_ENABLE_GRAPH_RUNNING_HASH))
    return;
  /*for (const auto& fwd_op: nodes_array) {
    auto op = fwd_op.getOp().toQualString();
    std::cout << " fwd graph buolder Op: " << op << std::endl;
  }*/
  torch::jit::graph_node_list graph_nodes = mp_g->nodes();
  for (auto* node : graph_nodes) {
    if (node->kind().is_prim()) {
      continue;
    }
    if ((node->kind() == torch::jit::prim::Constant) ||
        (node->kind() == torch::jit::prim::ListConstruct) ||
        (node->kind() == torch::jit::prim::ListUnpack)) {
      continue;
    }
    bool match_found = false;
    for (const auto& fwd_op : nodes_array) {
      auto op = fwd_op.getOp().toQualString();
      if (strcmp(node->kind().toQualString(), op) == 0) {
        match_found = true;
        break;
      }
    }
    if (!match_found) {
      // assert here
      std::cout << "Op not found in FWD accum: " << node->kind().toQualString()
                << std::endl;
    }
  }
  // mp_g->print(std::cout, false);
}

} // namespace habana_lazy