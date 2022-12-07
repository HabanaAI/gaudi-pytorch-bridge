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

bool OpArrayEntry::isMetadataCandidate(const at::IValue& input) const {
  return input.isBool() || input.isDevice() || input.isIntList() ||
      input.isScalar() || input.isDoubleList() || input.isBoolList() ||
      input.isString() || input.isNone() ||
      (input.isList() &&
       !input.toList().elementType()->cast<at::TensorType>() &&
       !input.toList().elementType()->cast<at::OptionalType>()->ofTensor());
}

size_t OpArrayEntry::ival_hash(const torch::jit::IValue& v, size_t h) {
  if (v.isInt()) {
    return at::hash_combine(h, at::get_hash(habana::mod_exp(v.toInt())));
  } else if (v.isString()) {
    return at::hash_combine(h, at::get_hash(v.toStringView()));
  } else if (v.isBool()) {
    return at::hash_combine(h, at::get_hash(habana::mod_exp(v.toBool())));
  } else if (v.isScalar()) {
    return at::hash_combine(
        h, c10::WeakIValue(v).hash()); // hash() moved to WeakIvalue
  } else {
    if (!v.isNone() && !v.isDevice()) {
      PT_LAZY_WARN(
          "Metadata of type ",
          v.type()->str(),
          " is not hashed. Might get false Lazy IR Cache hits, ",
          "if the value of the constant metadata changes");
    }
  }
  return h;
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

void GraphHashBuilder::prepareInputs(
    const std::vector<uint64_t>& input_map,
    std::vector<ir::Value>& inputs) {
  PT_LAZY_TRACE;
  assert(input_map.size());
  inputs.reserve(input_map.size());
  for (auto idx : input_map) {
    std::shared_ptr<Data> d = graph_input_tensors[idx].lock();
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
    TORCH_CHECK(
        itr != graph_input_stack_uids.end(),
        "missing tensor in stack map. id: ",
        uid);
    auto indx = itr - graph_input_stack_uids.begin();
    graph_input_stack_uid_map.emplace_back(indx);
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
      // we add dtype and memoryformat, mostly this is going to be pure graph
      // inputs
      torch::jit::ArgumentSpec as(1, 0);
      as.addTensor(tensor, true);
      input_hash = at::hash_combine(input_hash, as.hashCode());
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

void GraphHashBuilder::addNode(const c10::Symbol& node_symbol) {
  PT_LAZY_TRACE;
  OpArrayEntry entry;
  entry.addNode(node_symbol);
  entry.updateIndex(nodes_array.size());
  nodes_array.emplace_back(entry);
}

void GraphHashBuilder::graph(
    const c10::Symbol& op_name,
    const std::vector<c10::IValue>& inputs) {
  PT_LAZY_TRACE;
  addNode(op_name);
  auto node_entry = getLatestEntry();
  node_entry.populateMetaData(inputs);
  node_hash = node_entry.getNodeOpHash();
  addInputTensors(inputs);
}

void GraphHashBuilder::updateGraphInputTMap(const at::Tensor tensor) {
  auto impl = dynamic_cast<HbLazyTensorImpl*>(tensor.unsafeGetTensorImpl());
  if (impl == nullptr) {
    return;
  }
  auto hbo1 = impl->tensor();
  auto shared_ptr = hbo1.getDataPtr();
  // this can be further optimized with unorder map -> vector complexityO(1)
  if (std::find(
          graph_input_stack_uids.begin(),
          graph_input_stack_uids.end(),
          shared_ptr->unique_id) == graph_input_stack_uids.end()) {
    graph_input_tensors.emplace_back(shared_ptr);
    graph_input_stack_uids.emplace_back(shared_ptr->unique_id);
  }

  HbLazyTensor hl_t;

  {
    auto id = hbo1.getTensorUniqueId();
    auto context = habana_lazy_executor.getDeviceExecutionContext(0);
    // shallow copy tensor map
    {
      auto t_shallow_copy_opt = hbo1.getDataPtr()->tensor_shallow_copy;
      if (t_shallow_copy_opt.has_value()) {
        impl = GetHbLazyTensorImpl(t_shallow_copy_opt.value());
        hl_t = impl->tensor();
        auto shared_ptr = hl_t.getDataPtr();
        // this can be further optimized with unorder map -> vector
        // complexityO(1)
        if (std::find(
                graph_input_stack_uids.begin(),
                graph_input_stack_uids.end(),
                shared_ptr->unique_id) == graph_input_stack_uids.end()) {
          graph_input_tensors.emplace_back(shared_ptr);
          graph_input_stack_uids.emplace_back(shared_ptr->unique_id);
        }
      }
    }

    {
      LOCK_VIEW_TABLE_MUTEX(context->viewContext);
      // view or recent base

      auto params_ptr = context->viewContext.GetViewTableEntry(id);
      if (params_ptr != nullptr) {
        auto recent_base =
            HbLazyTensorViews::get_recent_base_tensor(params_ptr->base);
        hl_t = GetHbLazyTensor(recent_base);
        auto shared_ptr = hl_t.getDataPtr();
        // this can be further optimized with unorder map -> vector
        // complexityO(1)
        if (std::find(
                graph_input_stack_uids.begin(),
                graph_input_stack_uids.end(),
                shared_ptr->unique_id) == graph_input_stack_uids.end()) {
          graph_input_tensors.emplace_back(shared_ptr);
          graph_input_stack_uids.emplace_back(shared_ptr->unique_id);
        }
      } else {
        // Recent version map
        auto recent_base = HbLazyTensorViews::get_recent_base_tensor(tensor);
        hl_t = GetHbLazyTensor(recent_base);
        auto shared_ptr = hl_t.getDataPtr();
        // this can be further optimized with unorder map -> vector
        // complexityO(1)
        if (std::find(
                graph_input_stack_uids.begin(),
                graph_input_stack_uids.end(),
                shared_ptr->unique_id) == graph_input_stack_uids.end()) {
          graph_input_tensors.emplace_back(shared_ptr);
          graph_input_stack_uids.emplace_back(shared_ptr->unique_id);
        }
      }
    }
  }
}

size_t GraphHashBuilder::addInputTensor(at::Tensor t, size_t hash) {
  // prepare hash using tID
  auto tid = getTensorRunningId(t);
  hash = at::hash_combine(hash, tid);
  auto hbo1 = GetHbLazyTensor(t);
  hash = HbLazyTensorViews::updateViewHash(hbo1.getTensorUniqueId(), hash);

  // needed to preare InputStack for cache Hit case
  updateGraphInputTMap(t);

  // add scalar type, dim to hash
  hash = at::hash_combine(hash, static_cast<size_t>(t.scalar_type()));
  hash = at::hash_combine(hash, t.dim());

  return hash;
}

void GraphHashBuilder::addInputTensors(
    const std::vector<c10::IValue>& input_tensors) {
  PT_LAZY_TRACE;
  input_hash = 0;
  auto idx = 0;
  for (auto& t : input_tensors) {
    input_hash = at::hash_combine(input_hash, idx);

    if (t.isTensor()) {
      if (t.toTensor().defined()) {
        input_hash = addInputTensor(t.toTensor(), input_hash);
      }
    } else if (t.isTensorList()) {
      auto tvec = t.toTensorVector();
      for (auto& tensor : tvec) {
        input_hash = addInputTensor(tensor, input_hash);
      }
    } else if (t.isList()) {
      for (auto& v : t.toListRef()) {
        if (v.isTensor()) {
          if (v.toTensor().defined()) {
            input_hash = addInputTensor(v.toTensor(), input_hash);
          }
        }
      }
    }
    idx++;
  }
}

void GraphHashBuilder::validateAccumJitOps(
    std::shared_ptr<torch::jit::Graph> mp_g) {
  if (!GET_ENV_FLAG_NEW(PT_HPU_ENABLE_GRAPH_RUNNING_HASH))
    return;
  {
    std::set<std::string> fwdOpL;
    for (const auto& fwd_op : nodes_array) {
      auto op = fwd_op.getOp().toQualString();
      fwdOpL.emplace(op);
    }
  }
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
  mp_g->print(std::cout, false);
}

} // namespace habana_lazy
