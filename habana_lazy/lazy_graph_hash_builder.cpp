/*******************************************************************************
 * Copyright (C) 2022-2023 Habana Labs, Ltd. an Intel Company
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

    PT_LAZY_DEBUG("Updated meta hash ", hash);
  }
  return hash;
}

void OpArrayEntry::populateMetaData(
    const std::vector<c10::IValue>& input_tensors) {
  size_t index = 0;
  for (auto& input : input_tensors) {
    // TODO: Fetch tensor from generator before running hash.
    // We cannot hash the generator but we need the Tensor it produces because
    // it's a graph input. Until that is fixed Generator-typed args will reach
    // here and will be ignored.
    /*HABANA_ASSERT(
        !input.isGenerator(),
        "Must fetch Tensor from Generator before calculating running hash");*/
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

void GraphHashBuilder::prepareInputs(
    const std::vector<uint64_t>& input_map,
    std::vector<ir::Value>& inputs) {
  PT_LAZY_TRACE;
  assert(input_map.size());
  inputs.reserve(input_map.size());
  for (auto idx : input_map) {
    PT_LAZY_DEBUG(
        "preparing input tensor at index ",
        idx,
        " (",
        graph_input_tensors.size(),
        ")");
    HABANA_ASSERT(idx < graph_input_tensors.size());
    std::shared_ptr<Data> d = graph_input_tensors[idx].lock();
    inputs.emplace_back(d->ir_value);
  }
}

void GraphHashBuilder::prepareInputsStackMap(
    const std::vector<ir::Value>& inputs) {
  PT_LAZY_TRACE;
  PT_LAZY_DEBUG("Preparing input stack map for hash ", fwd_running_hash);
  for (auto in : inputs) {
    std::shared_ptr<Data> d = in.m_data_ptr.lock();
    auto uid = d->unique_id;
    PT_LAZY_DEBUG(
        "input processing tensor uid=",
        uid,
        ", running=",
        d->running_cntr,
        ", name=",
        in.ToString());
    auto itr = std::find(
        graph_input_stack_uids.begin(), graph_input_stack_uids.end(), uid);
    TORCH_CHECK(
        itr != graph_input_stack_uids.end(),
        "missing tensor in stack map. id: ",
        uid);
    auto indx = itr - graph_input_stack_uids.begin();
    graph_input_stack_uid_map.emplace_back(indx);
    PT_LAZY_DEBUG(
        "input entry ",
        graph_input_stack_uid_map.size() - 1,
        " is index ",
        indx);
    if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) != 2) {
      HABANA_ASSERT(
          d->running_cntr != -1,
          "Producing graph metadata with input that has no running ID");
    }
  }
}

uint64_t GraphHashBuilder::getFwdRunningHash() {
  return fwd_running_hash;
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
  for (auto& uid : devctx->tensors_data_opt_order) {
    std::shared_ptr<Data> data = devctx->getDataPtr(uid);
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

void GraphHashBuilder::updateRunningHash(
    const c10::Symbol& op_name,
    const std::vector<c10::IValue>& inputs) {
  PT_LAZY_TRACE;
  size_t prev_hash{fwd_running_hash};

  addNode(op_name);
  auto node_entry = getLatestEntry();
  node_entry.populateMetaData(inputs);
  fwd_running_hash =
      at::hash_combine(fwd_running_hash, node_entry.getNodeOpHash());
  addInputTensors(inputs);
  PT_LAZY_DEBUG(
      "Updating running hash of op",
      op_name,
      " number of arguments ",
      inputs.size(),
      " ",
      prev_hash,
      "->",
      fwd_running_hash);
}

void GraphHashBuilder::rememberIfInput(const at::Tensor& tensor) {
  const auto& hl_t = GetHbLazyTensor(tensor);
  std::shared_ptr<Data> input_data{hl_t.getDataPtr()};
  // this can be further optimized with unorder map -> vector complexityO(1)
  if (input_data->running_cntr == -1) {
    input_data->running_cntr = getRunningCntr();
    PT_LAZY_DEBUG(
        "Tensor ",
        hl_t.getTensorUniqueId(),
        " got running id ",
        hl_t.getTensorRunningId());
    // TODO also here we may push back tensor to graph_input_tensors without
    // looking at uid
    torch::jit::ArgumentSpec as(1, 0);
    as.addTensor(tensor, true);
    fwd_running_hash = at::hash_combine(fwd_running_hash, as.hashCode());
  }
  HABANA_ASSERT(input_data != nullptr);
  if (std::find(
          graph_input_stack_uids.begin(),
          graph_input_stack_uids.end(),
          input_data->unique_id) == graph_input_stack_uids.end()) {
    PT_LAZY_DEBUG(
        "Remembering tensor uid=",
        hl_t.getTensorUniqueId(),
        " rid=",
        hl_t.getTensorRunningId(),
        " ",
        hl_t.CurrentIrValue().ToString(),
        " at index ",
        graph_input_tensors.size());
    graph_input_tensors.emplace_back(input_data);
    graph_input_stack_uids.emplace_back(input_data->unique_id);
  }
}

void GraphHashBuilder::addInputTensors(const at::Tensor& tensor) {
  PT_LAZY_TRACE;
  auto impl = dynamic_cast<HbLazyTensorImpl*>(tensor.unsafeGetTensorImpl());
  if (impl == nullptr) {
    return;
  }
  auto hbo1 = impl->tensor();
  rememberIfInput(tensor);

  HbLazyTensor hl_t;
  {
    auto id = hbo1.getTensorUniqueId();

    // shallow copy tensor map
    {
      auto& t_shallow_copy_opt = hbo1.getDataPtr()->tensor_shallow_copy;
      if (t_shallow_copy_opt.has_value()) {
        auto& t = t_shallow_copy_opt.value().back();
        impl = GetHbLazyTensorImpl(t);
        hl_t = impl->tensor();
        PT_LAZY_DEBUG("addInputTensors also shallow copy uid ", id);
        rememberIfInput(t);
      }
    }

    {
      // view or recent base
      auto& params_opt = hl_t.getDataPtr()->stride_params;
      if (params_opt.has_value()) {
        auto& params = params_opt.value();
        auto recent_base =
            HbLazyTensorViews::get_recent_base_tensor(params.base);
        hl_t = GetHbLazyTensor(recent_base);
        fwd_running_hash =
            HbLazyTensorViews::updateViewHash(hbo1, fwd_running_hash);
        PT_LAZY_DEBUG(
            "addInputTensors also view base tensor uid ",
            hl_t.getTensorUniqueId());
        rememberIfInput(recent_base);
      } else {
        // Recent version map
        auto recent_base = HbLazyTensorViews::get_recent_base_tensor(tensor);
        hl_t = GetHbLazyTensor(recent_base);
        PT_LAZY_DEBUG(
            "addInputTensors also recent version map uid ",
            hl_t.getTensorUniqueId());
        rememberIfInput(recent_base);
      }
    }
  }
}

void GraphHashBuilder::hashCombineTensor(
    const at::Tensor& tensor,
    size_t& hash) {
  // prepare hash using tID
  auto hbimpl = dynamic_cast<HbLazyTensorImpl*>(tensor.unsafeGetTensorImpl());
  HABANA_ASSERT((hbimpl != nullptr), "GetHbLazyTensor for a non lazy tensor");
  HbLazyTensor hl_t = hbimpl->tensor();
  // we add dtype and memoryformat, mostly this is going to be pure graph
  // inputs
  hash = at::hash_combine(hash, hl_t.getDataPtr()->running_cntr);

  // add scalar type, dim to hash
  hash = at::hash_combine(hash, static_cast<size_t>(tensor.scalar_type()));
  hash = at::hash_combine(hash, tensor.dim());
  PT_LAZY_DEBUG(
      "Updated input hash ",
      hash,
      " ",
      hl_t.CurrentIrValue().ToString(),
      " uid=",
      hl_t.getTensorUniqueId());
}

void GraphHashBuilder::addInputTensors(
    const std::vector<c10::IValue>& input_tensors) {
  PT_LAZY_TRACE;
  auto idx = 0;
  for (auto& t : input_tensors) {
    fwd_running_hash =
        at::hash_combine(fwd_running_hash, idx++); // this is pointless

    if (t.isTensor()) {
      if (t.toTensor().defined()) {
        addInputTensors(t.toTensor());
      }
    } else if (t.isTensorList()) {
      auto tvec = t.toTensorVector();
      for (auto& tensor : tvec) {
        addInputTensors(tensor);
      }
    } else if (t.isList()) {
      for (auto& v : t.toListRef()) {
        if (v.isTensor()) {
          if (v.toTensor().defined()) {
            addInputTensors(v.toTensor());
          }
        }
      }
    }
  }
}

int64_t GraphHashBuilder::combineSyncData(
    const std::vector<HbLazyTensor>& tensors,
    const std::vector<int>& indices) {
  PT_LAZY_TRACE;
  PT_LAZY_DEBUG("Fwd_running_hash before view combine : ", fwd_running_hash);

  fwd_running_hash = at::hash_combine(fwd_running_hash, indices.size());

  // special handling for a single view node execution
  // a.view().to(cpu)
  if (indices.size() == 1) {
    fwd_running_hash = HbLazyTensorViews::updateViewHash(
        tensors[indices[0]], fwd_running_hash);
    for (auto& s : tensors[indices[0]].GetSizes()) {
      fwd_running_hash = at::hash_combine(fwd_running_hash, s);
    }
  }

  for (auto idx : indices)
    fwd_running_hash = at::hash_combine(fwd_running_hash, idx);
  PT_LAZY_DEBUG("\nFwd_running_hash : ", fwd_running_hash);
  return fwd_running_hash;
}

void GraphHashBuilder::reset() {
  nodes_array.clear();

  graph_input_tensors.clear();
  graph_input_stack_uids.clear();
  graph_input_stack_uid_map.clear();

  fwd_running_hash = 0;
  fwd_inputs_running_hash = 0;

  global_cntr = 0;
  PT_LAZY_DEBUG("Cleared Graph Hash Builder");
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
