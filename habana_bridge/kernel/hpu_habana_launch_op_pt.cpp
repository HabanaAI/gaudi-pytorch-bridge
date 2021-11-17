/******************************************************************************
 * Copyright (C) 2020 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */

#include "habana_bridge/kernel/hpu_habana_launch_op_pt.h"

#include <algorithm>
#include <iomanip>
#include <sstream>
#include <typeinfo>
#include <unordered_map>

#include <ATen/record_function.h>
#include <torch/csrc/jit/ir/constants.h>
#include <torch/csrc/jit/runtime/interpreter.h>

#include <absl/container/flat_hash_map.h>
#include <absl/container/flat_hash_set.h>
#include <absl/container/inlined_vector.h>
#include <absl/hash/hash.h>
#include <absl/memory/memory.h>
#include <absl/types/optional.h>

#include "habana_device/HPUAllocator.h"
#include "habana_device/HPUCheck.h"
#include "habana_device/tensor_builder.h"

#include "habana_helpers/graph.h"
#include "habana_helpers/logging.h"
#include "habana_helpers/tensor_utils.h"
#include "habana_helpers/unused_macro.h"
#include "habana_kernels/kernel_utils.h"
#include "habana_kernels/lazy_kernels_declarations.h"
#include "habana_kernels/random_gen_kernels.h"
#include "habana_kernels/unary_kernels.h"
#include "habana_lazy/aten_lazy_bridge.h"
#include "habana_lazy/hlexec.h"

#include "habana_lazy/aten_lazy_bridge.h"
#include "habana_lazy/hpu_lazy_tensors.h"
#include "synapse_helpers/env_flags.h"

using namespace torch::jit;

namespace habana {

// static initializations
const std::unordered_set<std::string> HabanaMetaOpList::meta_ops = {
    // Add aten string here for ops to support
    // e.g  :: "aten::view"
    "aten::size",
    "prim::dtype"};

std::unordered_set<std::string> HabanaLaunchOpPT::watchlist_ = {};
//--------------------------------------

void adjustSizesforPT(at::Tensor* tensor, bool is_output) {
  auto sizes = tensor->sizes().vec();
  auto strides = tensor->strides().vec();
  int64_t dim_out_pos[] = {0, 3, 1, 2};
  int64_t dim_in_pos[] = {0, 2, 3, 1};
  int64_t dim_out_pos_3d[] = {0, 4, 1, 2, 3};
  int64_t dim_in_pos_3d[] = {0, 2, 3, 4, 1};
  auto is_5d_layout = sizes.size() == 5 ? true : false;
  at::IntArrayRef out_pos;
  at::IntArrayRef in_pos;
  if (is_5d_layout) {
    out_pos = dim_out_pos_3d;
    in_pos = dim_in_pos_3d;
  } else {
    out_pos = dim_out_pos;
    in_pos = dim_in_pos;
  }

  at::IntArrayRef new_pos_arr = is_output ? out_pos : in_pos;
  auto new_pos = new_pos_arr.vec();
  std::vector<long int> swapped_sizes = {
      sizes[new_pos[0]],
      sizes[new_pos[1]],
      sizes[new_pos[2]],
      sizes[new_pos[3]]};
  if (is_5d_layout) {
    swapped_sizes.push_back(sizes[new_pos[4]]);
  }

  tensor->unsafeGetTensorImpl()->set_sizes_contiguous(swapped_sizes);
  // make the output layouts correct for PT

  if (is_output) {
    auto format = is_5d_layout ? c10::MemoryFormat::ChannelsLast3d
                               : c10::MemoryFormat::ChannelsLast;
    tensor->unsafeGetTensorImpl()->empty_tensor_restride(format);
  } else {
    tensor->unsafeGetTensorImpl()->empty_tensor_restride(
        c10::MemoryFormat::Contiguous);
  }
}

bool dropCachedRecipe_LRU(size_t& recipe_count) {
  bool dropped{false};
  dropped = RecipeCacheLRU::get_cache().drop_lru(recipe_count);
  return dropped;
}

std::string makeOpName(const char* name) {
  std::string op_name =
      (name ? std::string(name) : std::string("HabanaLaunchOp"));
  std::replace(op_name.begin(), op_name.end(), ':', '_');
  return op_name;
}

std::string makeIdStr(const char* name, size_t graph_index) {
  std::ostringstream oss;
  oss << makeOpName(name) << '_' << graph_index;
  return oss.str();
}

HabanaLaunchOpPT::HabanaLaunchOpPT(const torch::jit::Node* node, bool dbg)
    : HabanaLaunchOpPT(
          node->g(attr::Subgraph),
          dbg,
          node->kind().toQualString()) {}

HabanaLaunchOpPT::HabanaLaunchOpPT(
    std::shared_ptr<torch::jit::Graph> graph,
    bool dbg,
    size_t graph_index,
    const char* name)
    : HabanaLaunchOpPT(
          std::move(graph),
          dbg,
          makeOpName(name),
          makeIdStr(name, graph_index)) {}

HabanaLaunchOpPT::HabanaLaunchOpPT(
    std::shared_ptr<torch::jit::Graph> graph,
    bool dbg,
    const std::string& name)
    : op_name(name), jit_ir_graph{std::move(graph)}, debug(dbg), id_str(name) {}

HabanaLaunchOpPT::HabanaLaunchOpPT(
    std::shared_ptr<torch::jit::Graph> graph,
    bool dbg,
    const std::string& name,
    const std::string& id)
    : op_name(name), jit_ir_graph{std::move(graph)}, debug(dbg), id_str(id) {
  refine_ds_enabled_ = GET_ENV_FLAG(PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES);

  PT_BRIDGE_DEBUG("Creating : ", id_str);
  if (!HPUDeviceAllocator::drop_cached_recipe_cb) {
    HPUDeviceAllocator::drop_cached_recipe_cb = dropCachedRecipe_LRU;
  }

  valptr_to_persistent_map = {};

  tensor_dump_numel_ = -2;

  char* snumel = getenv("HABANA_PGM_DUMP_TENSOR_NUMEL");
  if (snumel != nullptr) {
    tensor_dump_numel_ = atoi(snumel);
    char* wfile_name = getenv("HABANA_PGM_WATCHLIST_FILE");
    if (watchlist_.empty() && wfile_name) {
      std::ifstream wfile(wfile_name);
      TORCH_CHECK(
          wfile.is_open(), "Unable to open watchlist file ", wfile_name);

      std::string opname;
      while (wfile) {
        getline(wfile, opname);
        watchlist_.insert(opname);
      }
      wfile.close();
    }
  }

  enable_tensor_dump_ = (tensor_dump_numel_ >= -1) ? true : false;
  enable_caching_ = true;

  // The enable_caching_ can be overridden with PT_HPU_PGM_ENABLE_CACHE
  const auto val = GET_ENV_FLAG(PT_HPU_PGM_ENABLE_CACHE);
  if (val == 0) {
    enable_caching_ = false;
  }
  use_persistent_tensors = GET_ENV_FLAG_NEW(HABANA_USE_PERSISTENT_TENSOR);

  if (enable_tensor_dump_) {
    struct stat st = {};
    std::string dir_name{"./tensor_dumps"};
    mode_t dir_mode{0755};

    if (stat(dir_name.c_str(), &st) == -1) {
      auto ret = mkdir(dir_name.c_str(), dir_mode);
      TORCH_CHECK(0 == ret, std::string("failed to create " + dir_name));
    }

    dir_name += std::string("/") + id_str;

    if (stat(dir_name.c_str(), &st) == -1) {
      auto ret = mkdir(dir_name.c_str(), dir_mode);
      TORCH_CHECK(0 == ret, std::string("failed to create " + dir_name));
    }

    tdmp_dir_name_ = dir_name;

    {
      std::ostringstream oss;
      oss << tdmp_dir_name_ << "/"
          << (enable_caching_ ? "tensors_chon" : "tensors_choff")
          << "_pre.tdmp";
      tdmp_file_name_pre_ = oss.str();

      std::ofstream tensor_file;
      tensor_file.open(tdmp_file_name_pre_.c_str());
      tensor_file << "---- id_str : " << id_str << '\n'
                  << "---- tensor dump of the following graph" << '\n';
      tensor_file << jit_ir_graph->toString() << "----" << '\n' << '\n';
      tensor_file.close();
    }

    {
      std::ostringstream oss;
      oss << tdmp_dir_name_ << "/"
          << (enable_caching_ ? "tensors_chon" : "tensors_choff") << ".tdmp";
      tdmp_file_name_ = oss.str();

      std::ofstream tensor_file;
      tensor_file.open(tdmp_file_name_.c_str());
      tensor_file << "---- id_str : " << id_str << '\n'
                  << "---- tensor dump of the following graph" << '\n';
      tensor_file << jit_ir_graph->toString() << "----" << '\n' << '\n';
      tensor_file.close();
    }

    if (tensor_dump_numel_ > 0) {
      htensor_wbuff_size = sizeof(float) * tensor_dump_numel_;
    }
  }
}

HabanaLaunchOpPT::HabanaLaunchOpPT(std::shared_ptr<torch::jit::Graph> graph)
    : jit_ir_graph{std::move(graph)} {
  refine_ds_enabled_ = GET_ENV_FLAG(PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES);

  PT_BRIDGE_DEBUG("Creating HabanaLaunchOp for Optimized Lazy Eager Path");

  if (!HPUDeviceAllocator::drop_cached_recipe_cb) {
    HPUDeviceAllocator::drop_cached_recipe_cb = dropCachedRecipe_LRU;
  }

  valptr_to_persistent_map = {};
  tensor_dump_numel_ = -2;
  enable_tensor_dump_ = false;
  enable_caching_ = true;
  use_persistent_tensors = false;
}

HabanaLaunchOpPT::~HabanaLaunchOpPT() {
  PT_BRIDGE_DEBUG("Destroying : ", id_str);
}

LayoutFormat getLayoutFromDims(const std::vector<int64_t>& dims) {
  std::unordered_map<const LayoutFormat, const std::vector<int64_t>>
      toDevicePermuteOrder = {
          {LayoutFormat::NHWC, {0, 2, 3, 1}},
          {LayoutFormat::NCHW, {0, 1, 2, 3}},
          {LayoutFormat::HWCK, {2, 3, 1, 0}}};
  //[ToDo] use find method instead, need to define vectorhasher
  for (const auto& l : toDevicePermuteOrder) {
    if (l.second == dims)
      return l.first;
  }
  return LayoutFormat::ANY;
}

LayoutFormat getPTTensorLayout(at::Tensor& tensor) {
  auto mem_format = tensor.suggest_memory_format();
  if (mem_format == at::MemoryFormat::ChannelsLast ||
      mem_format == at::MemoryFormat::ChannelsLast3d) {
    return LayoutFormat::NHWC;
  } else {
    return LayoutFormat::NCHW;
  }
}

bool HabanaLaunchOpPT::IsOutputToRestride(torch::jit::Value* value) {
  auto uses = value->uses();
  for (auto u : uses) {
    auto restride_node = u.user;
    if ((strcmp(restride_node->kind().toQualString(), "hpu::restride_cl") ==
         0) ||
        (strcmp(restride_node->kind().toQualString(), "hpu::restride") == 0)) {
      return true;
    }
  }
  return false;
}

torch::jit::Value* HabanaLaunchOpPT::GetRestridedOutvalue(
    torch::jit::Value* val) {
  for (auto u : val->uses()) {
    auto restride_node = u.user;
    if ((strcmp(restride_node->kind().toQualString(), "hpu::restride_cl") ==
         0) ||
        (strcmp(restride_node->kind().toQualString(), "hpu::restride") == 0)) {
      return restride_node->output(0);
    }
  }
  return nullptr;
}

bool HabanaLaunchOpPT::IsOutputToPermute(torch::jit::Value* value) {
  auto uses = value->uses();
  for (auto u : uses) {
    auto permute_node = u.user;
    if (strcmp(permute_node->kind().toQualString(), "aten::permute") == 0) {
      return true;
    }
  }
  return false;
}

torch::jit::Value* HabanaLaunchOpPT::GetPermuteOutvalue(
    torch::jit::Value* val) {
  for (auto u : val->uses()) {
    auto restride_node = u.user;
    if (strcmp(restride_node->kind().toQualString(), "aten::permute") == 0) {
      return restride_node->output(0);
    }
  }
  return nullptr;
}

bool HabanaLaunchOpPT::isPermuteInGraphOutputs(torch::jit::Value* value) {
  // return if graph output is restrided node output
  if (IsOutputToPermute(value)) {
    auto value_permuted = GetPermuteOutvalue(value);
    TORCH_CHECK(nullptr != value_permuted, "Permuted value output is null");
    auto graph_outs = jit_ir_graph->outputs();
    for (auto value_out : graph_outs) {
      if (value_permuted->unique() == value_out->unique()) {
        return true;
      }
    }
  }
  return false;
}

torch::jit::Node* HabanaLaunchOpPT::GetUnpackNodeFromTensorList(
    torch::jit::Value* val) {
  for (auto u : val->uses()) {
    auto node = u.user;
    if (strcmp(node->kind().toQualString(), "prim::ListUnpack") == 0) {
      return node;
    }
  }
  return nullptr;
}

bool HabanaLaunchOpPT::isInGraphOutputs(torch::jit::Value* value) {
  auto graph_outs = jit_ir_graph->outputs();
  for (auto value_out : graph_outs) {
    if (value->unique() == value_out->unique()) {
      return true;
    }
  }
  // return if graph output is restrided node output
  if (IsOutputToRestride(value)) {
    auto value_restrided = GetRestridedOutvalue(value);
    TORCH_CHECK(nullptr != value_restrided, "Restrided value output is null");
    auto graph_outs = jit_ir_graph->outputs();
    for (auto value_out : graph_outs) {
      if (value_restrided->unique() == value_out->unique()) {
        return true;
      }
    }
  }
  return false;
}

bool HabanaLaunchOpPT::isInGraphOutputs(torch::jit::Node* node, size_t index) {
  auto node_outs = node->outputs();
  TORCH_CHECK(index <= node_outs.size());

  return isInGraphOutputs(node_outs[index]);
}

bool HabanaLaunchOpPT::nodeOutputPersistencePerValue(
    torch::jit::Node* node,
    torch::jit::Value* value_out) {
  bool is_persistent = false;
  if (use_persistent_tensors || isInGraphOutputs(value_out) ||
      isPermuteInGraphOutputs(value_out)) {
    // Highest priority is given to the env variable, and if
    // part of the graph output
    auto in_graph_output = isInGraphOutputs(value_out);
    if (in_graph_output) {
      PT_BRIDGE_DEBUG(
          "Persistent tensor for ",
          node->kind().toQualString(),
          " for value %",
          value_out->debugName(),
          " appears in graph output");
    }
    is_persistent = true;
  } else if (
      valptr_to_persistent_map.find(value_out) !=
      valptr_to_persistent_map.end()) {
    if (valptr_to_persistent_map[value_out]) {
      PT_BRIDGE_DEBUG(
          "Persistent tensor for ",
          node->kind().toQualString(),
          " for value %",
          value_out->debugName(),
          " created for an in-place op");
    }
    is_persistent = valptr_to_persistent_map[value_out];
  } else {
    // If no specific flag is set, the mark as false
    is_persistent = false;
  }

  return is_persistent;
}

std::vector<bool> HabanaLaunchOpPT::nodeOutputPersistence(
    torch::jit::Node* node) {
  auto node_outs = node->outputs();
  std::vector<bool> is_persistent_vec{};
  // If node output is tensor list
  // tensorList and Unpack pair is supported
  if (node->output(0)->type() == torch::ListType::ofTensors() &&
      node->outputs().size() == 1) {
    auto unpack_node = GetUnpackNodeFromTensorList(node->output(0));
    if (unpack_node != nullptr) {
      for (auto value_out : unpack_node->outputs()) {
        auto is_persistent =
            nodeOutputPersistencePerValue(unpack_node, value_out);
        is_persistent_vec.emplace_back(is_persistent);
      }
    } else {
      PT_BRIDGE_DEBUG("TensorList is not input to ListUnpack Node");
      HABANA_ASSERT(0);
    }
  } else {
    for (auto value_out : node_outs) {
      auto is_persistent = nodeOutputPersistencePerValue(node, value_out);
      is_persistent_vec.emplace_back(is_persistent);
    }
  }
  return is_persistent_vec;
}

void HabanaLaunchOpPT::HandleMappedTensor(
    CValPtr value_in,
    const HabanaOperatorPtr& habana_op,
    SharedSynTensorOrRefListPtr& tensorList) {
  auto syn_tensor_input = pt_to_synapse_tensors.find(value_to_ivalue[value_in]);

  for (synapse_helpers::tensor& tensor : *(syn_tensor_input->second)) {
    synapse_helpers::tensor& syn_tensor = habana_op->SetSynapseInput(tensor);
    tensorList->emplace_back(tensor_or_ref(syn_tensor));
  }

  pt_to_synapse_tensors.erase(value_to_ivalue[value_in]);
  pt_to_synapse_tensors.emplace(value_to_ivalue[value_in], tensorList);
}

synapse_helpers::tensor& HabanaLaunchOpPT::AllocateSynapseTensor(
    const HabanaOperatorPtr& habana_op,
    at::Tensor& pt_tensor) {
  auto impl = habana_lazy::GetHbInternalTensorImpl(pt_tensor);

  if (impl && impl->isShapeTensor()) {
    auto& syn_tensor = habana_op->AllocateSynapseInput(
        *syn_graph_ptr, pt_tensor, true, impl->getTensorType());
    return syn_tensor;
  } else {
    habana_helpers::TensorShape min_shape, max_shape;

    void* pt_tensor_buffer_start = pt_tensor.storage().data_ptr().get();
    bool is_duplicate_syn_tensor{
        (pt_tensor_buffer_start != nullptr &&
         buff_to_syn_tensor_map.count(pt_tensor_buffer_start))};
    if (is_duplicate_syn_tensor) {
      auto syn_tensor_it = buff_to_syn_tensor_map.find(pt_tensor_buffer_start);
      synapse_helpers::tensor& st = syn_tensor_it->second;
      habana_op->set_is_duplicate_input_flag(true);
      habana_op->add_syn_input_tensor_orig(st);
    }
    auto& syn_tensor =
        habana_op->AllocateSynapseInput(*syn_graph_ptr, pt_tensor, true);

    if (is_duplicate_syn_tensor) {
      habana_op->set_is_duplicate_input_flag(false);
      habana_op->clear_syn_input_tensor_orig();
    }

    if (pt_tensor_buffer_start != nullptr) {
      buff_to_syn_tensor_map.emplace(
          pt_tensor_buffer_start, tensor_or_ref(syn_tensor));
    }
    return syn_tensor;
  }
}
void HabanaLaunchOpPT::HandleUnmappedTensor(
    CValPtr value_in,
    const HabanaOperatorPtr& habana_op,
    SharedSynTensorOrRefListPtr& tensorList) {
  std::vector<at::Tensor> pyTensorList;
  const auto& ivalue = value_to_ivalue[value_in];
  if (ivalue->isTensor()) {
    pyTensorList.emplace_back(ivalue->toTensor());
  } else {
    const auto& pytList = ivalue->toListRef();
    for (const auto& pyTensor : pytList) {
      if (!pyTensor.isNone())
        pyTensorList.emplace_back(pyTensor.toTensor());
    }
  }

  std::vector<PtTensorInfo> tiv;
  for (auto& pt_tensor : pyTensorList) {
    if (!pt_tensor.defined()) {
      continue;
    }

    auto& syn_tensor = AllocateSynapseTensor(habana_op, pt_tensor);

    tensorList->emplace_back(tensor_or_ref(syn_tensor));

    std::string irn = "%" + value_in->debugName();
    PtTensorInfo ti(
        pt_tensor,
        syn_tensor.name(),
        irn,
        watch_tensor_flag_,
        syn_tensor.id(),
        syn_tensor.tensor_type());
    tiv.push_back(ti);

    if (enable_caching_) {
      void* buffp = ti.get_buffer_start();
      if (ti.is_ZST() == false) {
        buff_to_input_ivpsh_map.emplace(buffp, ivalue);
      }
    }
  }

  if (!tensorList->empty()) {
    auto it = pt_to_synapse_tensors.emplace(ivalue, tensorList);
    if (it.second == false) {
      pt_to_synapse_tensors[ivalue] = tensorList;
    }

    if (enable_caching_) {
      input_tiv_map.emplace(value_to_ivalue[value_in], tiv);
      if ((strcmp(
               value_in->node()->kind().toQualString(), "hpu::restride_cl") ==
           0) ||
          (strcmp(value_in->node()->kind().toQualString(), "hpu::restride") ==
           0)) {
        auto restride_node = value_in->node();
        auto restride_value_in = restride_node->input(0);
        if (isInGraphInputs(restride_value_in) != -1) {
          input_tiv_map.emplace(value_to_ivalue[restride_value_in], tiv);
        }
      }
    } else {
      input_tivs.emplace_back(tiv);
    }
  }
}

void HabanaLaunchOpPT::HandleMappedandUnmappedTensor(
    CValPtr value_in,
    const HabanaOperatorPtr& habana_op,
    SharedSynTensorOrRefListPtr& tensorList) {
  auto is_already_mapped =
      pt_to_synapse_tensors.find(value_to_ivalue[value_in]) !=
      std::end(pt_to_synapse_tensors);
  if (is_already_mapped) {
    HandleMappedTensor(value_in, habana_op, tensorList);
  } else {
    HandleUnmappedTensor(value_in, habana_op, tensorList);
  }
}

void HabanaLaunchOpPT::GetSynapseInputs(
    const HabanaOperatorPtr& habana_op,
    torch::jit::Node* node) {
  auto node_ins = node->inputs();
  int input_idx = 0;
  for (const auto value_in : node_ins) {
    if (value_to_ivalue[value_in] &&
        (value_to_ivalue[value_in]->isTensor() ||
         value_to_ivalue[value_in]->isTensorList())) {
      // Find if an input tensor is already mapped
      // NB: It seems Habana doesn't support shared input to
      // different nodes in graph
      // note: else path is only of listcontruct is fused with another op like
      // cat. This case occurs in lazy eval but not in torch trace mode
      if (value_to_ivalue[value_in]->isTensor() ||
          (value_in->node()->kind() != torch::jit::prim::ListConstruct)) {
        SharedSynTensorOrRefListPtr tensor_ref_list_ptr_sh =
            std::make_shared<SynTensorOrRefList>();
        HandleMappedandUnmappedTensor(
            value_in, habana_op, tensor_ref_list_ptr_sh);
      } else {
        // tensorlist
        auto prev_node = value_in->node();
        if (prev_node->kind() == torch::jit::prim::ListConstruct) {
          for (auto& value_in : prev_node->inputs()) {
            if (value_to_ivalue[value_in]->isTensor()) {
              SharedSynTensorOrRefListPtr tensor_ref_list_ptr_sh =
                  std::make_shared<SynTensorOrRefList>();
              HandleMappedandUnmappedTensor(
                  value_in, habana_op, tensor_ref_list_ptr_sh);
            }
          }
        }
      } // else
      input_idx++;
    } // if (value_to_ivalue[value_in] && ..
    else if (
        (!strcmp("aten::_fused_dropout", node->kind().toQualString()) &&
         1 == input_idx) ||
        (!strcmp("hpu::randperm_out", node->kind().toQualString()) &&
         0 == input_idx)) {
      auto stack = getStackForNode(node);
      // Create the seed tensor
      // TODO : check for the generator when the generator could be passed
      at::Tensor seed_tensor;
      // as an IValues
      if (!strcmp("hpu::randperm_out", node->kind().toQualString()))
        seed_tensor = RandpermOperator::GenerateAndCopySeedToHPU(stack, true);
      else
        seed_tensor = DropoutOperator::GenerateAndCopySeedToHPU(stack, true);

      auto& syn_tensor =
          habana_op->AllocateSynapseInput(*syn_graph_ptr, seed_tensor, true);

      auto dma_cb = habana_op->getDMAInputTensorCB();
      std::ostringstream oss;
      oss << "%dma_input" << '_' << dma_input_idx;
      dma_input_idx++;
      std::string irn{oss.str()};
      PtTensorInfo ti(
          seed_tensor,
          syn_tensor.name(),
          irn,
          watch_tensor_flag_,
          syn_tensor.id(),
          DATA_TENSOR,
          dma_cb);
      auto dma_tensor_idx = aten_dma_inputs.size();
      ti.set_dma_tensor_idx(dma_tensor_idx);
      dma_input_tensorinfos.emplace_back(ti);
      // Saving as persistent intermediate tensor
      aten_dma_inputs.push_back(seed_tensor);
      input_idx++;
    }
  } // for (const auto value_in : node_ins)
}

void HabanaLaunchOpPT::ProcessPersistentNodeOutput(
    const IValPtrShared& ivpsh,
    const ValPtr& vp,
    const synapse_helpers::tensor& out_syntensor) {
  // If the node output is persistent, there are the following possibilities
  // 1. The output is not in graph output, hence it is an intermediate
  //    which is persistent.
  //    A: Add it to duplicate_input_tivs as this would be a duplicate of a
  //       persistent input tensor if a parent exists
  //    B: Add it to intermediate tensor list if no parent available.
  //       Right now thats the only data structure that supports adding it
  //       to patching
  //
  // 2. The output is in graph output. In this scenario, it could be -
  //    A: It is an output created by the PT kernel that goes to the
  //       graph output.
  //       Add it to output_tensorinfos as it is to be counted as an
  //       output tensor of the recipe.
  //       With enable_caching_, this is maintained in
  //       output_tensorinfo_map
  //    B: It is a duplicate of an input. An example:
  //           graph(%id:0 : Float(*),
  //                 ..
  //             %1 : FLoat(*) = aten::add_(%id:0, ...)
  //                 ..
  //             return (%1, ...)
  //       Here, the aten::add_ creates a duplicate for output from the
  //       input, hence %1 is a duplicate of input %id:0. The duplicate
  //       output also goes to graph output.
  //       Add it to duplicate_input_to_outtinfo_map with
  //       enable_caching_
  //    C: It is a duplicate of a persistent intermediate. An example:
  //           graph(%id:9 : Tensor,
  //                 %id:6 : Tensor,
  //                 %id:3 : Tensor):
  //             %3 : int = prim::Constant[value=1]()
  //             %4 : Tensor = aten::sigmoid(%id:3)
  //             %5 : Tensor = aten::sub(%4, %id:6, %3)
  //             %6 : Tensor = hpu::control_edge_(%5)
  //             %7 : Tensor = aten::mul_(%6, %id:9)
  //             return (%7)
  //       Here, the aten::mul_ creates a duplicate for output from the
  //       persistent intermediate %6. The duplicate output goes to graph
  //       output.
  //       Add it to duplicate_intermediate_to_outtinfo_map with
  //       enable_caching_
  //    D: It is a duplicate of an existing output

  auto ti = PtTensorInfo(
      ivpsh,
      out_syntensor.name(),
      vp,
      watch_tensor_flag_,
      out_syntensor.id(),
      out_syntensor.tensor_type());
  void* buffp = ti.get_buffer_start();

  if (false == isInGraphOutputs(vp)) {
    if (ti.is_ZST() == false && buff_to_input_ivpsh_map.count(buffp)) {
      // Case 1.A: intermediate persistent tensor which an alias of an input
      PT_BRIDGE_DEBUG("Adding to duplicate_input_tivs ", ti);
      duplicate_input_tivs.emplace_back(ti);
    } else {
      if (ti.is_ZST() == false && buff_to_output_ivpsh_map.count(buffp)) {
        duplicate_outtinfos.emplace_back(ti);
      } else {
        // Case 1.B: intermediate persistent tensor
        if (ti.is_view_tensor()) {
          PT_BRIDGE_DEBUG(
              "Starting persistent intermediate is view tensor ",
              "with non zero offset ",
              ti.get_offset());
        }
        AddAtenIntermediate(ivpsh, ti);
      }
    }
  } else {
    if (!enable_caching_) {
      // Case 2.A: graph output tensor
      output_tensorinfos.emplace_back(ti);
    } else {
      // Is this a duplicate tensor going to graph output?
      // See if this the buffer pointer matches any input, then -
      // Check whether it is an alias of any input
      if (ti.is_ZST() == false && buff_to_input_ivpsh_map.count(buffp)) {
        // Case 2.B: Graph output that is duplicate of input
        PT_BRIDGE_DEBUG("Adding to duplicate_input_to_outtinfo_map ", ti);
        duplicate_input_to_outtinfo_map.emplace(ivpsh, ti);
      } else if (
          ti.is_ZST() == false && buff_to_intermediate_ivpsh_map.count(buffp)) {
        // Case 2.C: Graph output that is duplicate of a persistent
        // intermediate
        PT_BRIDGE_DEBUG(
            "Adding to duplicate_intermediate_to_outtinfo_map ", ti);
        duplicate_intermediate_to_outtinfo_map.emplace(ivpsh, ti);
      } else if (
          ti.is_ZST() == false && buff_to_output_ivpsh_map.count(buffp)) {
        // Case 2.D: Graph output that is duplicate of a previous output
        PT_BRIDGE_DEBUG("Adding to duplicate_output_to_outtinfo_map ", ti);
        duplicate_output_to_outtinfo_map.emplace(ivpsh, ti);
      } else {
        // Case 2.A: graph output tensor, enable_tensor_release_
        PT_BRIDGE_DEBUG("Adding to output_tensorinfo_map ", ti);
        output_tensorinfo_map.emplace(ivpsh, ti);
        if (ti.is_ZST() == false) {
          buff_to_output_ivpsh_map.emplace(buffp, ivpsh);
        }
      }
    }
  }
}

void HabanaLaunchOpPT::ProcessSynapseOutputs(
    const HabanaOperatorPtr& habana_op,
    torch::jit::Node* node) {
  auto output_nodes = node->outputs();
  auto habana_kernel_meta_data = habana_op->GetKernelMetaData();

  if (node->output(0)->type() == torch::ListType::ofTensors() &&
      node->outputs().size() == 1) {
    auto unpack_node = GetUnpackNodeFromTensorList(node->output(0));
    if (unpack_node != nullptr) {
      output_nodes = unpack_node->outputs();
    } else {
      PT_BRIDGE_DEBUG("TensorList is not input to ListUnpack Node");
      HABANA_ASSERT(0);
    }
  }

  const auto& output_tensors_pt = habana_op->GetOutputs();
  const auto& excluded_out_indices =
      habana_op->GetSynOutputIndicesExcludedInNode();

  if (node->kind().toQualString() != std::string("aten::gelu")) {
    TORCH_CHECK(
        output_nodes.size() ==
            output_tensors_pt.size() - excluded_out_indices.size(),
        "HabanaFusionOp Lowering of node : ",
        node->kind().toQualString(),
        " Number of output nodes ",
        output_nodes.size(),
        " doesnt match the generated ",
        output_tensors_pt.size() - excluded_out_indices.size());
  }

  size_t output_nodes_idx = 0, output_tensor_idx = 0;

  for (synapse_helpers::tensor& out_tensor_syn : habana_op->GetSynOutputs()) {
    if (excluded_out_indices.find(output_tensor_idx) ==
        excluded_out_indices.end()) {
      IValPtrShared ivpsh =
          std::make_shared<IVal>(output_tensors_pt[output_tensor_idx]);
      value_to_ivalue[output_nodes[output_nodes_idx]] = ivpsh;

      // For some kernels, like the inplace ones, the kernel output is always
      // created as persistent. Patching table needs to be updated accordingly
      // for such tensors.
      if (use_persistent_tensors || out_tensor_syn.is_persistent()) {
        ProcessPersistentNodeOutput(
            ivpsh, output_nodes[output_nodes_idx], out_tensor_syn);
      }

      SharedSynTensorOrRefListPtr tensorList =
          std::make_shared<SynTensorOrRefList>();
      tensorList->emplace_back(tensor_or_ref(out_tensor_syn));
      pt_to_synapse_tensors.emplace(
          value_to_ivalue[output_nodes[output_nodes_idx]], tensorList);

      output_nodes_idx++;
    }
    output_tensor_idx++;
  }
}

void HabanaLaunchOpPT::ProcessSynapseShapeTensors(
    const HabanaOperatorPtr& habanaOp,
    torch::jit::Node* node) {
  for (synapse_helpers::tensor& maybe_syn_shape_tensor :
       habanaOp->GetSynInputs()) {
    if (maybe_syn_shape_tensor.is_shape_tensor() ||
        maybe_syn_shape_tensor.is_input_shape_tensor()) {
      std::string irn{"%shapeInput_"};
      irn += std::to_string(shape_index);
      shape_index++;
      PtTensorInfo ti(maybe_syn_shape_tensor, irn);
      shape_tensor_tinfos.emplace_back(ti);
    }
  }
  // Add shape tensor for all Operator created inside habanaOp
  std::vector<HabanaOperatorPtr> habana_kernels = habanaOp->GetKernels();
  for (auto& habana_op : habana_kernels) {
    ProcessSynapseShapeTensors(habana_op, node);
  }
}

at::IntArrayRef getDimsForLayout(
    LayoutFormat channel_order,
    LayoutFormat current_order) {
  at::IntArrayRef dims;

  if (current_order == LayoutFormat::NCHW) {
    if (channel_order == LayoutFormat::NHWC) {
      static const int64_t dimarr[] = {0, 2, 3, 1};
      dims = dimarr;
    } else if (channel_order == LayoutFormat::HWCK) {
      static const int64_t dimarr[] = {2, 3, 1, 0};
      dims = dimarr;
    } else {
      TORCH_CHECK(
          0, " Habana Fusion op permute called for unsupported channel order");
    }
  } else if (current_order == LayoutFormat::NHWC) {
    if (channel_order == LayoutFormat::NCHW) {
      static const int64_t dimarr[] = {0, 3, 1, 2};
      dims = dimarr;
    } else if (channel_order == LayoutFormat::HWCK) {
      static const int64_t dimarr[] = {1, 2, 3, 0};
      dims = dimarr;
    } else {
      TORCH_CHECK(
          0, " Habana Fusion op permute called for unsupported channel order");
    }
  } else if (current_order == LayoutFormat::HWCK) {
    if (channel_order == LayoutFormat::NCHW) {
      static const int64_t dimarr[] = {3, 2, 0, 1};
      dims = dimarr;
    } else if (channel_order == LayoutFormat::NHWC) {
      static const int64_t dimarr[] = {3, 0, 1, 2};
      dims = dimarr;
    } else {
      TORCH_CHECK(
          0, " Habana Fusion op permute called for unsupported channel order");
    }
  } else {
    TORCH_CHECK(
        0, " Habana Fusion op permute called for unsupported channel order");
  }

  return dims;
}

int64_t HabanaLaunchOpPT::isInGraphInputs(torch::jit::Value* value) {
  auto graph_ins = jit_ir_graph->inputs();
  auto it = std::find_if(
      graph_ins.cbegin(),
      graph_ins.cend(),
      [&](const torch::jit::Value* value_in) {
        return (value->unique() == value_in->unique());
      });

  if (it != graph_ins.cend()) {
    return (it - graph_ins.begin());
  }

  return -1;
}

void HabanaLaunchOpPT::create_duplicate_syn_tensor(
    at::Tensor* tensor,
    torch::jit::Value* value_in,
    bool persistence) {
  auto syn_tensorlist_input =
      pt_to_synapse_tensors.find(value_to_ivalue[value_in]);
  TORCH_CHECK(
      syn_tensorlist_input->second->size() == 1,
      "not implemented the handling of syn_tensorlist_input size ",
      syn_tensorlist_input->second->size());
  synapse_helpers::tensor& syn_tensor_input =
      syn_tensorlist_input->second->back();

  auto dtype = tensor->scalar_type();
  // if both are persistent, use same memeory section
  if (syn_tensor_input.is_persistent() && persistence) {
    // create a tensor variant on the same memory section as the input
    auto variant = synapse_helpers::tensor_builder(
                       tensor->sizes(),
                       tensor->strides(),
                       habana_helpers::pytorch_to_synapse_type(dtype))
                       .mark_persistence(true)
                       .with_memory_section(syn_tensor_input.memorysection())
                       .build(
                           synapse_helpers::HPURegistrar::get_device(
                               tensor->device().index()),
                           syn_tensor_input.graph());

    meta_syn_tensors.push_back(
        absl::get<synapse_helpers::tensor>(std::move(variant)));

    PtTensorInfo ti(
        value_to_ivalue[value_in],
        meta_syn_tensors.back().name(),
        value_in,
        watch_tensor_flag_,
        meta_syn_tensors.back().id(),
        meta_syn_tensors.back().tensor_type());
    if (!isInGraphOutputs(value_in)) {
      duplicate_input_tivs.emplace_back(ti);
    } else {
      if (!enable_caching_) {
        output_tensorinfos.emplace_back(ti);
      } else {
        duplicate_outtinfos.emplace_back(ti);
      }
    }
  } else {
    pt_to_synapse_tensors.erase(value_to_ivalue[value_in]);
    auto variant =
        habana_helpers::create_tensor(*tensor, *syn_graph_ptr, persistence);
    meta_syn_tensors.push_back((std::move(variant)));
  }

  auto& syn_tensor = meta_syn_tensors.back();
  pt_to_synapse_tensors.erase(value_to_ivalue[value_in]);
  SharedSynTensorOrRefListPtr tensorList =
      std::make_shared<SynTensorOrRefList>();
  tensorList->emplace_back(tensor_or_ref(syn_tensor));
  pt_to_synapse_tensors.emplace(value_to_ivalue[value_in], tensorList);
}
void adjustInputWeight(at::Tensor* tensor, bool is_input) {
  if (tensor->dim() != 4)
    return;

  auto sizes = tensor->sizes().vec();
  auto strides = tensor->strides().vec();
  int64_t dims_in[] = {2, 3, 1, 0};
  int64_t dims_out[] = {3, 2, 0, 1};
  at::IntArrayRef in = dims_in;
  at::IntArrayRef out = dims_out;
  // TODO : Remove these hardcoded dims, maybe take it from config file?
  at::IntArrayRef new_pos_arr = is_input ? in : out;
  auto new_pos = new_pos_arr.vec();
  std::vector<long int> swapped_sizes = {
      sizes[new_pos[0]],
      sizes[new_pos[1]],
      sizes[new_pos[2]],
      sizes[new_pos[3]]};
  std::vector<long int> swapped_strides = {
      strides[new_pos[0]],
      strides[new_pos[1]],
      strides[new_pos[2]],
      strides[new_pos[3]]};
  tensor->unsafeGetTensorImpl()->set_sizes_and_strides(
      swapped_sizes, swapped_strides);
}

IValPtrShared castConstantTensor(IValPtrShared ival) {
  auto tensor = ival->toTensor();
  auto dtype = tensor.scalar_type();
  bool cast = dtype == c10::ScalarType::Long || dtype == c10::ScalarType::Double
      ? true
      : false;
  c10::ScalarType dst_type = dtype;
  if (cast) {
    dst_type = dtype == c10::ScalarType::Long ? c10::ScalarType::Int
                                              : c10::ScalarType::Float;
    tensor = tensor.to(dst_type);
  }
  auto new_tensor = tensor.to(c10::kHPU);
  IValPtrShared ivptrsh = std::make_shared<IVal>(IValue(new_tensor));
  return ivptrsh;
}

void HabanaLaunchOpPT::handleRestrideNode(
    torch::jit::Node* node,
    bool is_restride_cl) {
  auto value_in = node->input(0);
  auto value_out = node->output(0);
  HABANA_ASSERT(value_to_ivalue.find(value_in) != std::end(value_to_ivalue));
  HABANA_ASSERT(value_to_ivalue[value_in]->isTensor());
  auto tensor = value_to_ivalue[value_in]->toTensor();
  auto is_5d_layout = tensor.dim() == 5 ? true : false;

  if ((tensor.dim() == 4) || (tensor.dim() == 5)) {
    auto sizes = tensor.sizes().vec();
    auto new_pos = toIValue(node->input(1))->toIntVector();
    std::vector<int64_t> swapped_sizes;
    for (auto& pos : new_pos) {
      swapped_sizes.emplace_back(sizes[pos]);
    }
    auto strides = tensor.strides().vec();
    std::vector<long int> swapped_strides;
    for (auto& pos : new_pos) {
      swapped_strides.emplace_back(strides[pos]);
    }
    tensor.unsafeGetTensorImpl()->set_sizes_and_strides(
        swapped_sizes, swapped_strides);
    tensor.unsafeGetTensorImpl()->empty_tensor_restride(
        c10::MemoryFormat::Contiguous);

    if (isInGraphOutputs(value_out)) {
      auto format = is_5d_layout ? c10::MemoryFormat::ChannelsLast3d
                                 : c10::MemoryFormat::ChannelsLast;
      if (!is_restride_cl) {
        if (tensor.dim() == 4 || tensor.dim() == 5) {
          auto hb_grad_weight = habana_lazy::GetHbInternalTensorImpl(tensor);
          hb_grad_weight->SetTensorLayout(habana_lazy::LayoutFormat::kHWCK);
        }
        tensor.unsafeGetTensorImpl()->empty_tensor_restride(
            c10::MemoryFormat::Contiguous);
      } else {
        tensor.unsafeGetTensorImpl()->empty_tensor_restride(format);
      }
    } else {
      tensor.unsafeGetTensorImpl()->empty_tensor_restride(
          c10::MemoryFormat::Contiguous);
    }
  }
  auto ivpsh = value_to_ivalue[value_in];
  auto ivpsh_restrided = std::make_shared<IVal>(tensor);
  PT_BRIDGE_DEBUG(
      "processing restride node, input %",
      value_in->debugName(),
      " output",
      value_out->debugName());

  if (isInGraphOutputs(value_out)) {
    TORCH_CHECK(
        pt_to_synapse_tensors.count(ivpsh),
        " Could not find the syn tensor corresponding to %",
        value_in->debugName());

    auto& syn_tensor_vec = pt_to_synapse_tensors[ivpsh];
    synapse_helpers::tensor& syn_tensor = syn_tensor_vec->at(0);
    auto ti = PtTensorInfo(
        ivpsh_restrided,
        syn_tensor.name(),
        value_in,
        watch_tensor_flag_,
        syn_tensor.id());

    value_to_ivalue.erase(value_in);

    if (enable_caching_) {
      void* buffp = ti.get_buffer_start();
      if (output_tensorinfo_map.count(ivpsh)) {
        // Case 2.A: graph output tensor
        PT_BRIDGE_DEBUG(
            "removing ivalue for restride input %",
            value_in->debugName(),
            " from output_tensorinfo_map");
        output_tensorinfo_map.erase(ivpsh);

        PT_BRIDGE_DEBUG(
            "adding ivalue for restride output %",
            value_out->debugName(),
            " to output_tensorinfo_map");
        output_tensorinfo_map.emplace(ivpsh_restrided, ti);
      } else if (
          ti.is_ZST() == false && buff_to_intermediate_ivpsh_map.count(buffp)) {
        // Case 2.C: Graph output that is duplicate of a persistent
        // intermediate
        PT_BRIDGE_DEBUG(
            "updating buff_to_intermediate_ivpsh_map entry for ",
            buffp,
            " with ivalue for restride output %",
            value_out->debugName());

        buff_to_intermediate_ivpsh_map.erase(buffp);
        buff_to_intermediate_ivpsh_map.emplace(buffp, ivpsh_restrided);
        TORCH_CHECK(
            duplicate_intermediate_to_outtinfo_map.count(ivpsh),
            " entry for restride input %",
            value_in->debugName(),
            " not found in duplicate_intermediate_to_outtinfo_map");

        PT_BRIDGE_DEBUG(
            "updating duplicate_intermediate_to_outtinfo_map entry for ",
            buffp,
            " with ivalue for restride output %",
            value_out->debugName());
        duplicate_intermediate_to_outtinfo_map.erase(ivpsh);
        duplicate_intermediate_to_outtinfo_map.emplace(ivpsh_restrided, ti);
      } else if (ti.is_ZST() == false && buff_to_input_ivpsh_map.count(buffp)) {
        // Case 2.B: Graph output that is duplicate of input
        PT_BRIDGE_DEBUG(
            "updating buff_to_input_ivpsh_map entry for ",
            buffp,
            " with ivalue for restride output %",
            value_out->debugName());

        buff_to_input_ivpsh_map.erase(buffp);
        buff_to_input_ivpsh_map.emplace(buffp, ivpsh_restrided);

        TORCH_CHECK(
            duplicate_input_to_outtinfo_map.count(ivpsh),
            " entry for restride input %",
            value_in->debugName(),
            " not found in duplicate_input_to_outtinfo_map");

        PT_BRIDGE_DEBUG(
            "updating duplicate_input_to_outtinfo_map entry for ",
            buffp,
            " with ivalue for restride output %",
            value_out->debugName());
        duplicate_input_to_outtinfo_map.erase(ivpsh);
        duplicate_input_to_outtinfo_map.emplace(ivpsh_restrided, ti);
      } else if (
          ti.is_ZST() == false && buff_to_output_ivpsh_map.count(buffp)) {
        // Case 2.D: Graph output that is duplicate of a previous output
        PT_BRIDGE_DEBUG(
            "updating buff_to_output_ivpsh_map entry for ",
            buffp,
            " with ivalue for restride output %",
            value_out->debugName());

        buff_to_output_ivpsh_map.erase(buffp);
        buff_to_output_ivpsh_map.emplace(buffp, ivpsh_restrided);

        TORCH_CHECK(
            duplicate_output_to_outtinfo_map.count(ivpsh),
            " entry for restride input %",
            value_in->debugName(),
            " not found in duplicate_output_to_outtinfo_map");

        PT_BRIDGE_DEBUG(
            "updating duplicate_output_to_outtinfo_map entry for ",
            buffp,
            " with ivalue for restride output %",
            value_out->debugName());
        duplicate_output_to_outtinfo_map.erase(ivpsh);
        duplicate_output_to_outtinfo_map.emplace(ivpsh_restrided, ti);
      } else {
        TORCH_CHECK(
            false,
            " unhandled scenario for restride input %",
            value_in->debugName(),
            (ti.is_ZST() ? " is ZST" : " is non ZST"),
            ", not found in any duplicate detection or output map");
      }
    }

    value_to_ivalue[value_in] = ivpsh_restrided;
    value_to_ivalue[value_out] = ivpsh_restrided;
  } else {
    PT_BRIDGE_DEBUG(
        "restride node output %", value_out->debugName(), " is non persistent");
    value_to_ivalue[value_out] = ivpsh_restrided;
  }
}

void HabanaLaunchOpPT::handlePrimNodes(torch::jit::Node* node) {
  if (node->kind() == torch::jit::prim::Constant) {
    auto node_vals = node->outputs();
    for (const auto value : node_vals) {
      IValPtrShared ivptrsh = std::make_shared<IVal>(toIValue(value).value());
      if (value->type()->kind() == c10::TypeKind::TensorType) {
        auto ivptrsh_updated = castConstantTensor(ivptrsh);
        value_to_ivalue[value] = ivptrsh_updated;
        std::string irn{"%intermediate_"};
        irn += std::to_string(intermediate_index);
        intermediate_index++;

        auto tensor = ivptrsh_updated->toTensor();
        meta_syn_tensors.push_back(habana_helpers::create_tensor(
            tensor, *syn_graph_ptr, true, tensor.scalar_type()));
        SharedSynTensorOrRefListPtr tensorList =
            std::make_shared<SynTensorOrRefList>();
        tensorList->emplace_back(tensor_or_ref(meta_syn_tensors.back()));
        pt_to_synapse_tensors.emplace(value_to_ivalue[value], tensorList);
        intermediate_tinfos.emplace_back(PtTensorInfo(
            tensor,
            meta_syn_tensors.back().name(),
            irn,
            watch_tensor_flag_,
            meta_syn_tensors.back().id(),
            meta_syn_tensors.back().tensor_type()));
        aten_intermediates.push_back(tensor);
      } else {
        value_to_ivalue[value] = ivptrsh;
      }
    }
  } else if (node->kind() == torch::jit::prim::ListConstruct) {
    const auto& node_ins = node->inputs();
    IValPtrShared ivptrsh_list;

    // ListConstruct can have optional and non-optional tensors as item types
    if (node->output()->type()->containedTypes()[0]->kind() ==
        OptionalType::Kind) {
      c10::List<c10::optional<at::Tensor>> opttensorList;
      for (const auto& value_in : node_ins) {
        auto ivptrsh = value_to_ivalue[value_in];
        if (ivptrsh->isTensor()) {
          opttensorList.emplace_back(ivptrsh->toTensor());
        } else {
          opttensorList.emplace_back(c10::nullopt);
        }
      }

      // convert opttensorList to Ivalue and update the stack
      ivptrsh_list = std::make_shared<IVal>(opttensorList);
    } else {
      c10::List<at::Tensor> tensorList;
      for (const auto& value_in : node_ins) {
        auto ivptrsh = value_to_ivalue[value_in];
        if (ivptrsh->isTensor()) {
          tensorList.emplace_back(ivptrsh->toTensor());
        }
      }

      // convert tensorList to Ivalue and update the stack
      ivptrsh_list = std::make_shared<IVal>(tensorList);
    }
    auto node_vals = node->outputs();
    HABANA_ASSERT(node_vals.size() == 1);
    value_to_ivalue[node_vals[0]] = ivptrsh_list;
  } else if (node->kind() == torch::jit::prim::ListUnpack) {
    // currently lowering code supports only TensorList+Unpack combination
    // [ToDo] Standalone ListUnpack support is not added here
  } else {
    HABANA_ASSERT(
        (node->kind() == torch::jit::prim::Constant) ||
        (node->kind() == torch::jit::prim::ListConstruct) ||
        (node->kind() == torch::jit::prim::ListUnpack));
  }
}

torch::jit::Stack HabanaLaunchOpPT::getStackForNode(torch::jit::Node* node) {
  torch::jit::Stack stack_in;
  auto node_inputs = node->inputs();
  for (auto input : node_inputs) {
    if (value_to_ivalue.count(input)) {
      stack_in.insert(stack_in.end(), *value_to_ivalue[input]);
    } else {
      stack_in.insert(stack_in.end(), IValue());
    }
  }
  return stack_in;
}

c10::ScalarType HabanaLaunchOpPT::getNodeScalarType(torch::jit::Node* node) {
  // return the data type of first input tensor
  for (auto input : node->inputs()) {
    if (value_to_ivalue.count(input) && value_to_ivalue[input]->isTensor()) {
      return value_to_ivalue[input]->toTensor().scalar_type();
    }
  }
  // Default return float for now if no tensor found
  return c10::ScalarType::Float;
}

void HabanaLaunchOpPT::handleMetaOps(torch::jit::Node* node) {
  // Call the meta op via CPU impl
  // Some ops dont support c10 op.callBoxed so we need to call via JIT
  torch::jit::Stack stack;
  auto node_ins = node->inputs();
  IValPtrShared input_ptr{nullptr};
  // LayoutFormat out_layout{}, out_origin_layout{};
  // void* in_data, *out_data;

  for (const auto value_in : node_ins) {
    stack.insert(stack.end(), *value_to_ivalue[value_in]);
    if (value_to_ivalue[value_in]->isTensor()) {
      auto tensor = value_to_ivalue[value_in]->toTensor();
      HABANA_ASSERT(
          pt_to_synapse_tensors.find(value_to_ivalue[value_in]) !=
          std::end(pt_to_synapse_tensors))

      // Below code is commented for now, since we dont handle
      // any meta ops that would create a new pytorch/syanpse tensor
      // If we need view to be handled as a meta op, need to enable
      // the below code
      /*
      if (pt_to_synapse_tensors.find(value_to_ivalue[value_in]) ==
          std::end(pt_to_synapse_tensors)) {
        in_data = tensor.data_ptr();
        input_ptr = value_to_ivalue[value_in];
        auto dtype = tensor.scalar_type();
        meta_syn_tensors.push_back(habana_helpers::create_tensor(
            tensor, *syn_graph_ptr, true, dtype));
        SharedSynTensorOrRefListPtr tensorList =
            std::make_shared<SynTensorOrRefList>();
        tensorList->emplace_back(tensor_or_ref(meta_syn_tensors.back()));
        pt_to_synapse_tensors.emplace(value_to_ivalue[value_in], tensorList);

        if (enable_caching_) {
          input_tiv_map.emplace(
              value_to_ivalue[value_in],
              PtTensorInfo(
                  value_to_ivalue[value_in],
                  meta_syn_tensors.back().name(),
                  value_in,
                  watch_tensor_flag_));
          buff_to_input_ivpsh_map.emplace(in_data, value_to_ivalue[value_in]);
        } else {
          input_tivs.emplace_back(PtTensorInfo(
              value_to_ivalue[value_in],
              meta_syn_tensors.back().name(),
              value_in,
              watch_tensor_flag_));
        }
      }*/
    }
  }
  torch::jit::Operator jit_op = node->getOperator();
  // auto offset =
  jit_op.getOperation()(stack);

  // TORCH_CHECK(offset == 0);

  auto node_outs = node->outputs();
  auto outputs = last(stack, node_outs.size());
  int i = 0;
  for (const auto val_out : node_outs) {
    IValPtrShared ival = std::make_shared<IVal>(outputs[i]);
    value_to_ivalue[val_out] = ival;
    HABANA_ASSERT(ival->isTensor() == false);
    // Below code is commented for now, since we dont handle
    // any meta ops that would create a new pytorch/syanpse tensor
    // If we need view to be handled as a meta op, need to enable
    // the below code
    /*if (ival->isTensor()) {
      auto tensor = ival->toTensor();
      create_duplicate_syn_tensor(&tensor, val_out, true);
      out_data = tensor.data_ptr();
    }*/
    i++;
  }
  /* TORCH_CHECK(
      in_data == out_data, "HabanaFusion : Data pointer changed in Meta
     op");*/
}

void HabanaLaunchOpPT::OrderInputs() {
  if (enable_caching_) {
    // Order the input_tivs according to the order of suggraph inputs
    size_t i = pt_stack_sh.size() - num_inputs;
    for (; i < pt_stack_sh.size(); i++) {
      IValPtrShared ivpsh = pt_stack_sh.at(i);
      if (ivpsh->isTensor() || ivpsh->isTensorList()) {
        auto it = input_tiv_map.find(ivpsh);
        if (it != input_tiv_map.end()) {
          input_tivs.push_back(it->second);
        } else {
          TORCH_CHECK(false, "synapse tensor not found for input index", i);
        }
      }
    }
    TORCH_CHECK(
        input_tivs.size() == num_tensor_inputs,
        "number of input tensors ",
        num_tensor_inputs,
        " mismatch with #input_tivs ",
        input_tivs.size());
  }
}

void HabanaLaunchOpPT::FlattenAndLinkInputTIVs(RecipeValueSpec& rv) {
  // dtensorinfos maintain the flattened tinfo list
  rv.dtensorinfos =
      std::make_shared<std::vector<PtTensorInfo>>(std::vector<PtTensorInfo>());

  std::unordered_map<void*, size_t> buff_to_inputtividx_map;
  for (auto& tiv : input_tivs) {
    if (absl::holds_alternative<PtTensorInfo>(tiv)) {
      const auto ti = absl::get<PtTensorInfo>(tiv);
      rv.dtensorinfos->push_back(ti);
      if (enable_caching_) {
        void* buffp = ti.get_buffer_start();
        buff_to_inputtividx_map.emplace(buffp, rv.dtensorinfos->size() - 1);
      }
    } else if (absl::holds_alternative<std::vector<PtTensorInfo>>(tiv)) {
      for (const auto& ti : absl::get<std::vector<PtTensorInfo>>(tiv)) {
        rv.dtensorinfos->push_back(ti);
        if (enable_caching_) {
          void* buffp = ti.get_buffer_start();
          buff_to_inputtividx_map.emplace(buffp, rv.dtensorinfos->size() - 1);
        }
      }
    } else {
      TORCH_CHECK(false, "Error condition for input tiv");
    }
  }
  // At this point inputs tinfos are populated
  rv.num_inputs = rv.dtensorinfos->size();

  // Link the input tivs with the duplicate
  size_t nduplicates{0};
  for (auto& tiv : duplicate_input_tivs) {
    if (absl::holds_alternative<PtTensorInfo>(tiv)) {
      auto ti = absl::get<PtTensorInfo>(tiv);
      if (enable_caching_) {
        void* buffp = ti.get_buffer_start();
        auto it_parent = buff_to_inputtividx_map.find(buffp);

        std::ostringstream err;
        err << ti;

        TORCH_CHECK(
            buff_to_inputtividx_map.end() != it_parent,
            "parent tinfo is missing for input duplicate ",
            err.str());

        ti.set_duplicate_flag(true);
        size_t parent_idx = it_parent->second;
        TORCH_CHECK(
            parent_idx < num_inputs,
            "out of bound parent index : ",
            parent_idx,
            " for ",
            ti.get_syn_name());
        ti.set_parent_index(parent_idx);
        PT_BRIDGE_DEBUG(
            "FlattenAndLinkInputTIVs: Input duplicate: parent idx ",
            parent_idx,
            " parent buffer ptr ",
            rv.dtensorinfos->at(parent_idx).get_buffer(),
            " duplicate_tiv buffer ptr ",
            ti.get_buffer());
      }
      rv.dtensorinfos->push_back(ti);
      nduplicates++;
    } else {
      TORCH_CHECK(false, "duplicate tiv must be a tensor");
    }
  }
  TORCH_CHECK(
      nduplicates == duplicate_input_tivs.size(),
      "#duplicate_input_tivs ",
      duplicate_input_tivs.size(),
      " is not matching with num_induplicates ",
      nduplicates);

  rv.num_induplicates = nduplicates;

  // At this point inputs and duplicate tinfos are populated
  TORCH_CHECK(
      (rv.num_inputs + rv.num_induplicates == rv.dtensorinfos->size()),
      "num_inputs ",
      rv.num_inputs,
      "num_induplicates ",
      rv.num_induplicates,
      " are not adding up to #dtensorinfos ",
      rv.dtensorinfos->size());
}

void HabanaLaunchOpPT::OrderOutputTinfos(RecipeValueSpec& rv) {
  bool has_empty_name = false;

  std::unordered_map<void*, size_t> buff_to_outputtinfoidx_map;
  // push the actual output tinfos
  size_t output_idx{0};
  for (auto output : jit_ir_graph->outputs()) {
    auto oit = value_to_ivalue.find(output);
    TORCH_CHECK(
        oit != value_to_ivalue.end(),
        "value_to_ivalue does not have an entry for %",
        output->debugName());

    IValPtrShared ivpsh = oit->second;
    TORCH_CHECK(nullptr != ivpsh, "IValPtrShared for subgraph output is null");

    // Checking where we can find the outputs
    {
      if (output_tensorinfo_map.count(ivpsh)) {
        auto it = output_tensorinfo_map.find(ivpsh);
        it->second.set_output_index(output_idx);
        output_tensorinfos.push_back(it->second);
        if (it->second.get_syn_name().empty()) {
          has_empty_name = true;
        }
        output_tensorinfo_map.erase(ivpsh);
      } else if (duplicate_input_to_outtinfo_map.count(ivpsh)) {
        auto it_dup = duplicate_input_to_outtinfo_map.find(ivpsh);
        it_dup->second.set_output_index(output_idx);
      } else if (duplicate_intermediate_to_outtinfo_map.count(ivpsh)) {
        auto it_dup = duplicate_intermediate_to_outtinfo_map.find(ivpsh);
        it_dup->second.set_output_index(output_idx);
      } else if (duplicate_output_to_outtinfo_map.count(ivpsh)) {
        auto it_dup = duplicate_output_to_outtinfo_map.find(ivpsh);
        it_dup->second.set_output_index(output_idx);
      } else {
        TORCH_CHECK(
            0,
            "Unaccounted output %",
            output->debugName(),
            " at index ",
            output_idx,
            ". Cached recipe execution might break");
      }
    }

    // add ivpsh to outputs
    rv.aten_outputs->push_back(ivpsh);
    output_idx++;
  }

  TORCH_CHECK(!has_empty_name, "empty tensor name");

  size_t intermediates_start = rv.num_inputs + rv.num_induplicates +
      rv.num_dma_inputs + rv.num_shape_tensors;

  TORCH_CHECK(
      output_tensorinfo_map.empty(),
      "output_tensorinfo_map still contains ",
      output_tensorinfo_map.size(),
      " tensors.");

  // Add the intermediates to rv.dtensorinfos
  std::unordered_map<void*, size_t> buff_to_interim_tividx_map;
  size_t interim_tinfo_idx{intermediates_start};
  if (!intermediate_tinfos.empty()) {
    rv.num_intermediates = intermediate_tinfos.size();
    for (auto& ti : intermediate_tinfos) {
      void* buffp = ti.get_buffer_start();
      // Duplicate analysis for the persistent intermediates
      if (buff_to_interim_tividx_map.count(buffp)) {
        ti.set_duplicate_flag(true);
        ti.set_parent_index(buff_to_interim_tividx_map[buffp]);
      } else {
        buff_to_interim_tividx_map.emplace(buffp, interim_tinfo_idx);
      }
      interim_tinfo_idx++;
    }

    rv.dtensorinfos->insert(
        rv.dtensorinfos->end(),
        intermediate_tinfos.begin(),
        intermediate_tinfos.end());
  }

  // At this point, within dtensorinfos, tinfos for inputs,
  // input duplicates and intermediates are added.
  size_t intermediates_end = intermediates_start + rv.num_intermediates;
  size_t outputs_start = intermediates_end;

  // Add the outputs to rv.dtensorinfos
  for (auto& ti : output_tensorinfos) {
    rv.dtensorinfos->push_back(ti);
    void* buffp = ti.get_buffer_start();

    buff_to_outputtinfoidx_map.emplace(buffp, rv.dtensorinfos->size() - 1);
  }
  rv.num_outputs = output_tensorinfos.size();

  // At this point, within dtensorinfos, tinfos for inputs, input_duplicates,
  // intermediates and outputs are added.
  size_t outputs_end = outputs_start + rv.num_outputs;

  // Link the out tivs with the duplicate.
  // Remember these are not outputs but tensors
  // that go back to the FusedOp from the outputs
  size_t nduplicates{0};
  for (auto& ti : duplicate_outtinfos) {
    void* buffp = ti.get_buffer_start();
    auto it_parent = buff_to_outputtinfoidx_map.find(buffp);

    std::ostringstream err;
    err << ti;

    TORCH_CHECK(
        buff_to_outputtinfoidx_map.end() != it_parent,
        "parent tinfo is missing for output duplicate ",
        err.str());

    ti.set_duplicate_flag(true);
    size_t parent_idx = it_parent->second;
    TORCH_CHECK(
        parent_idx >= outputs_start && parent_idx < outputs_end,
        "for output duplicate ",
        ti.get_syn_name(),
        "parent index should be within [",
        outputs_start,
        ',',
        outputs_end,
        ')');
    ti.set_parent_index(parent_idx);
    rv.dtensorinfos->push_back(ti);
    nduplicates++;
  }
  rv.num_outduplicates = nduplicates;

  // Create the input tensor tiv to idx map, this is required
  // to match the input to output duplicates against their parent idx.
  std::unordered_map<void*, size_t> buff_to_inputtividx_map;
  size_t in_idx = 0;
  for (auto& tiv : input_tivs) {
    if (absl::holds_alternative<PtTensorInfo>(tiv)) {
      const auto ti = absl::get<PtTensorInfo>(tiv);
      void* buffp = ti.get_buffer_start();
      buff_to_inputtividx_map.emplace(buffp, in_idx++);
    } else if (absl::holds_alternative<std::vector<PtTensorInfo>>(tiv)) {
      for (const auto& ti : absl::get<std::vector<PtTensorInfo>>(tiv)) {
        void* buffp = ti.get_buffer_start();
        buff_to_inputtividx_map.emplace(buffp, in_idx++);
      }
    } else {
      TORCH_CHECK(false, "Error condition for input tiv");
    }
  }

  // Link the inout tivs with the duplicate
  // These are tensors that are duplicated from an input and is part
  // of the graph output
  nduplicates = 0;
  for (auto& mi : duplicate_input_to_outtinfo_map) {
    auto ivpsh = mi.first;
    auto& ti = mi.second;
    void* buffp = ti.get_buffer_start();
    auto it_parent = buff_to_inputtividx_map.find(buffp);

    std::ostringstream err;
    err << ti;

    TORCH_CHECK(
        buff_to_inputtividx_map.end() != it_parent,
        "parent tinfo is missing for input_to_out duplicate ",
        err.str());

    ti.set_duplicate_flag(true);
    auto parent_idx = it_parent->second;
    TORCH_CHECK(
        parent_idx < rv.num_inputs,
        "for in_to_out duplicate ",
        ti.get_syn_name(),
        "parent index ",
        parent_idx,
        " should be within [",
        0,
        ',',
        rv.num_inputs,
        ')');
    ti.set_parent_index(parent_idx);
    rv.dtensorinfos->push_back(ti);
    nduplicates++;
  }

  rv.num_input_to_outduplicates = nduplicates;

  // Link the out tivs which are duplicate of persistent intermediates
  // These are tensors that are duplicated from a persistent interim and is
  // part of the graph output
  nduplicates = 0;
  for (auto& mi : duplicate_intermediate_to_outtinfo_map) {
    auto ivpsh = mi.first;
    auto& ti = mi.second;
    void* buffp = ti.get_buffer_start();
    auto it_parent = buff_to_interim_tividx_map.find(buffp);

    std::ostringstream err;
    err << ti;

    TORCH_CHECK(
        buff_to_interim_tividx_map.end() != it_parent,
        "parent tinfo is missing for interim_to_out duplicate ",
        err.str());

    ti.set_duplicate_flag(true);
    auto parent_idx = it_parent->second;
    TORCH_CHECK(
        (parent_idx >= intermediates_start && parent_idx < intermediates_end),
        "for interim to out duplicate ",
        ti.get_syn_name(),
        "parent index ",
        parent_idx,
        " should be within [",
        intermediates_start,
        ',',
        intermediates_end,
        ')');
    ti.set_parent_index(parent_idx);
    rv.dtensorinfos->push_back(ti);
    nduplicates++;
  }
  rv.num_intermediate_to_outduplicates = nduplicates;

  // Link the out tivs which are duplicate of actual outputs
  // These are tensors that are duplicated from outputs created by individual
  // ops using aten::empty like calls and is part of the graph output
  nduplicates = 0;
  for (auto& mi : duplicate_output_to_outtinfo_map) {
    auto ivpsh = mi.first;
    auto& ti = mi.second;
    void* buffp = ti.get_buffer_start();
    auto it_parent = buff_to_outputtinfoidx_map.find(buffp);

    std::ostringstream err;
    err << ti;

    TORCH_CHECK(
        buff_to_outputtinfoidx_map.end() != it_parent,
        "parent tinfo is missing for output_to_out duplicate ",
        err.str());

    ti.set_duplicate_flag(true);
    auto parent_idx = it_parent->second;
    TORCH_CHECK(
        parent_idx >= outputs_start && parent_idx < outputs_end,
        parent_idx < rv.num_inputs,
        "for out_to_out duplicate ",
        ti.get_syn_name(),
        "parent index ",
        parent_idx,
        " should be within [",
        outputs_start,
        ',',
        outputs_end,
        ')');
    ti.set_parent_index(parent_idx);
    rv.dtensorinfos->push_back(ti);
    nduplicates++;
  }
  rv.num_output_to_outduplicates = nduplicates;

  rv.num_tinfos = rv.dtensorinfos->size();
}

std::string HabanaLaunchOpPT::DumpNode(torch::jit::Node* node) {
  std::ostringstream o;
  node->print(o, 0, nullptr);
  auto str = o.str();
  if (node->input(0)->type() != torch::ListType::ofTensors()) {
    for (auto value_in : node->inputs()) {
      if (value_to_ivalue[value_in]->isTensor()) {
        auto tensor = value_to_ivalue[value_in]->toTensor();
        std::ostringstream o;
        o << "input Tensor ";
        o << value_in->debugName();
        o << "  size: ";
        o << tensor.sizes();
        str.append(o.str());
        str.append("\n");
      }
    }
  }
  if (node->output(0)->type() != torch::ListType::ofTensors()) {
    for (auto value_out : node->outputs()) {
      if (value_to_ivalue[value_out]->isTensor()) {
        auto tensor = value_to_ivalue[value_out]->toTensor();
        std::ostringstream o;
        o << "outout Tensor ";
        o << value_out->debugName();
        o << "  size: ";
        o << tensor.sizes();
        str.append(o.str());
        str.append("\n");
      }
    }
  }
  return str;
}

void HabanaLaunchOpPT::CompileAndExecuteHabanaFusedOpKernel(
    synapse_helpers::graph& syn_graph,
    bool is_shape_inference) {
  // figure out the right device id
  auto& device = synapse_helpers::HPURegistrar::get_device();
  synapse_helpers::detail::tensor_name_generator::reset();
  synDeviceId device_id = device.id();
  syn_graph_ptr = &syn_graph;

  // for each node in IR graph, at this point the graph is a list with nodes
  // topoloically sorted
  // TODO : check if we need to reorder nodes in any case
  torch::jit::graph_node_list graph_nodes = jit_ir_graph->nodes();
  // This is an optimization pass to mark all the nodes with sepcial layout
  // like weights which have HWCK Only activated in lazy mode for now
  if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) != 0) {
    runMetaDataAdjustmentPasses(jit_ir_graph->nodes());
  }

  for (auto* node : graph_nodes) {
    watch_tensor_flag_ = false;
    std::string opname(node->kind().toQualString());

    if (watchlist_.empty() || watchlist_.find(opname) != watchlist_.end()) {
      watch_tensor_flag_ = true;
    }

    // If its a meta op we need to call the CPU impl and capture changes
    // Only valid for single tensor ops
    // Can we avoid the string match here?
    if (HabanaMetaOpList::isHabanaMetaOp(node->kind().toQualString())) {
      handleMetaOps(node);
      continue;
    }

    // Prim nodes require special handling and are a special case
    if (node->kind().is_prim()) {
      handlePrimNodes(node);
      continue;
    }

      if ((strcmp(node->kind().toQualString(), "hpu::restride_cl") == 0) ||
          (strcmp(node->kind().toQualString(), "hpu::restride") == 0)) {
        bool is_restride_cl =
            (strcmp(node->kind().toQualString(), "hpu::restride_cl") == 0)
            ? true
            : false;
        handleRestrideNode(node, is_restride_cl);
        continue;
      }
    // Get kernel context
    const auto& op = node->schema().operator_name();
    HabanaOperatorPtr HabanaKernel =
        KernelRegistry().get(device_id, op, getNodeScalarType(node));

    TORCH_CHECK(HabanaKernel, op, " isn't registered in KernelRegistry!");

    PT_BRIDGE_DEBUG("Going to add ", *node);

    // clear the accumulated synapse node indices corresponding to permute.
    // Otherwise this results in spurious control edges
    syn_graph_ptr->clear_node_indices();

    // Set output metadata for node
    const auto& outputs = node->outputs();
    OutputMetaDataVector outputs_metadata;
    std::transform(
        outputs.begin(),
        outputs.end(),
        std::back_inserter(outputs_metadata),
        [](CValPtr value) -> OutputMetaData { return OutputMetaData(*value); });
    HabanaKernel->SetOutputMetadata(outputs_metadata);

    // set op name in synapse graph
    std::unique_ptr<synapse_helpers::graph::OpNameContext> op_name_context;
    if (node->hasAttribute(c10::attr::name)) {
      op_name_context = std::make_unique<synapse_helpers::graph::OpNameContext>(
          syn_graph, node->s(c10::attr::name));
    }

    // Create/attach the synapse inputs from aten tensors
    GetSynapseInputs(HabanaKernel, node);

    torch::jit::Stack input_stack = getStackForNode(node);

    // setup the config params for the kernels
    auto outputPersistent = nodeOutputPersistence(node);

    if (outputPersistent.size() == 1) {
      HabanaKernel->AllocateAndAddSynapseNode(
          syn_graph, input_stack, outputPersistent[0]);
    } else {
      HabanaKernel->AllocateAndAddSynapseNode(
          syn_graph, input_stack, outputPersistent);
    }

    jit_to_synapse_node_idx_map.emplace(
        node, syn_graph_ptr->get_node_indices());
    syn_graph_ptr->clear_node_indices();

    if (!is_shape_inference) {
      ProcessSynapseShapeTensors(HabanaKernel, node);
    }

    // Get the output tensors created back from the kernel and do the
    // subsequent processing.
    // We set type so that the created tensor is propagated throughout graph
    ProcessSynapseOutputs(HabanaKernel, node);

    PT_BRIDGE_DEBUG(DumpNode(node));
    // The kernel corresponding to current IR node, HabanaKernel, might create
    // one or more appended tensors. These are tensors which do not have a
    // corresponding ValPtr in the IR graph. These are either duplicate of
    // some inputs or persistent intermediates required by the kernel.
    auto patch_info = HabanaKernel->getAppendedTensorInfos();
    if (!patch_info.empty() && !is_shape_inference) {
      for (const auto& p : patch_info) {
        auto tensor_name = std::get<0>(p);
        auto tensor = std::get<1>(p);
        auto tensor_id = std::get<2>(p);
        void* buffp = tensor.data_ptr();

        // Check whether it is an alias of any input
        auto it = buff_to_input_ivpsh_map.find(buffp);
        if (it != buff_to_input_ivpsh_map.end()) {
          std::string irn{"%appended_indup_"};
          irn += std::to_string(appended_index);
          appended_index++;

          PtTensorInfo ti(
              tensor, tensor_name, irn, watch_tensor_flag_, tensor_id);
          auto& ivpsh = it->second;

          if (enable_caching_) {
            auto mit = input_tiv_map.find(ivpsh);
            TORCH_CHECK(input_tiv_map.end() != mit, "tinfo missing for input");

            TORCH_CHECK(ivpsh->isTensor(), "non tensor parent found");
          }

          duplicate_input_tivs.emplace_back(ti);
        } else {
          std::string irn{"%intermediate_"};
          irn += std::to_string(intermediate_index);
          intermediate_index++;

          IValPtrShared ivpsh = std::make_shared<IVal>(tensor);
          AddAtenIntermediate(ivpsh, tensor_name, irn, tensor_id);
          PT_BRIDGE_DEBUG(
              "Added appended tensor for ",
              node->kind().toQualString(),
              " as persistent intermediate");
        }
      }
    }

    // Adding to a vector as we share context through shared pointers and we
    // dont want to call delete untill we are done with whole graph
    habana_kernels.push_back(HabanaKernel);
  }

  // if its a shape inference pass, we dont need to do any additional processing
  // for launching the graph
  if (is_shape_inference) {
    return;
  }

  // Process control edges
  HabanaLaunchOpPT::ProcessControlEdges();

  if (syn_graph.is_empty()) {
    UpdateOutputs();
    return;
  }

  std::chrono::steady_clock::time_point t_start;
  t_start = std::chrono::steady_clock::now();
  auto&& error_variant{syn_graph.compile()};
  auto t_compile = std::chrono::steady_clock::now() - t_start;
  uint64_t t_compile_ns =
      std::chrono::duration_cast<std::chrono::nanoseconds>(t_compile).count();

  if (ABSL_PREDICT_FALSE(
          absl::holds_alternative<synapse_helpers::synapse_error>(
              error_variant))) {
    auto& error = absl::get<synapse_helpers::synapse_error>(error_variant);
    PT_BRIDGE_FATAL(
        "syn compile encountered : ",
        error.error,
        " ",
        error.status,
        " compile time ",
        t_compile_ns,
        " ns");
  }

  RecipeValueSpec::increment_compile_count();

  auto cur_recipe = get_value(std::move(error_variant));

  std::shared_ptr<RecipeValueSpec> rvalpsh =
      std::make_shared<RecipeValueSpec>(cur_recipe);

  RecipeValueSpec& rv = *rvalpsh;
  if (!syn_graph.is_empty()) {
    // first time, we need to get workspace size of the recipe, that was
    // compiled
    auto&& ws_size_result{
        synapse_helpers::graph::query_workspace_size(*cur_recipe)};
    if (ABSL_PREDICT_FALSE(
            absl::holds_alternative<synapse_helpers::synapse_error>(
                ws_size_result))) {
      auto& error = absl::get<synapse_helpers::synapse_error>(ws_size_result);
      PT_BRIDGE_FATAL(
          "workspace size query failed: ", error.error, " ", error.status);
      TORCH_CHECK(false, "workspace size query failed");
    }
    rv.workspace_size = get_value(ws_size_result);
  }

  // input_tivs need to be reordered for patching
  OrderInputs();

  // rv.num_inputs and rv.num_induplicates will be set by
  // FlattenAndLinkInputTIVs
  FlattenAndLinkInputTIVs(rv);

  if (!dma_input_tensorinfos.empty()) {
    rv.num_dma_inputs = dma_input_tensorinfos.size();
    rv.dtensorinfos->insert(
        rv.dtensorinfos->end(),
        dma_input_tensorinfos.begin(),
        dma_input_tensorinfos.end());
  }

  if (!shape_tensor_tinfos.empty()) {
    rv.num_shape_tensors = shape_tensor_tinfos.size();
    rv.dtensorinfos->insert(
        rv.dtensorinfos->end(),
        shape_tensor_tinfos.begin(),
        shape_tensor_tinfos.end());
  }

  // tinfos for outputs are populated during compile
  // need to be reordered only when the tensor handles are released
  rv.aten_outputs = std::make_shared<std::vector<IValPtrShared>>(
      std::vector<IValPtrShared>());
  if (!enable_caching_) {
    if (!intermediate_tinfos.empty()) {
      rv.num_intermediates = intermediate_tinfos.size();
      rv.dtensorinfos->insert(
          rv.dtensorinfos->end(),
          intermediate_tinfos.begin(),
          intermediate_tinfos.end());
    }

    // At this point, tinfos for inputs, input duplicates and intermediates
    // are populated
    TORCH_CHECK(
        (rv.num_inputs + rv.num_induplicates + rv.num_dma_inputs +
             rv.num_shape_tensors + rv.num_intermediates ==
         rv.dtensorinfos->size()),
        "num_inputs ",
        rv.num_inputs,
        " num_induplicates ",
        rv.num_induplicates,
        " num_dma_inputs ",
        rv.num_dma_inputs,
        " num_intermediates ",
        rv.num_intermediates,
        " num_shape_tensors ",
        rv.num_shape_tensors,
        " are not adding up to #dtensorinfos ",
        rv.dtensorinfos->size());

    rv.dtensorinfos->insert(
        rv.dtensorinfos->end(),
        output_tensorinfos.begin(),
        output_tensorinfos.end());

    rv.num_outputs = output_tensorinfos.size();
    rv.num_tinfos = rv.dtensorinfos->size();

    for (auto output : jit_ir_graph->outputs()) {
      auto oit = value_to_ivalue.find(output);
      TORCH_CHECK(
          oit != value_to_ivalue.end(),
          "value_to_ivalue does not have an entry for %",
          output->debugName());
      IValPtrShared ivpsh = oit->second;
      rv.aten_outputs->push_back(ivpsh);
    }
  } else {
    // TODO :
    //   preclude any interim tinfo from adding to output_tensorinfo_map
    OrderOutputTinfos(rv);
  }

  for (auto& ti : *rv.dtensorinfos) {
    if (!ti.is_duplicate()) {
      rv.ntensorbytes += ti.get_size();
    }
  }

  // At this point, tinfos for inputs, input duplicates, dma_inputs,
  // intermediates, outputs and output duplicates are populated
  auto total_tinfos = rv.num_inputs + rv.num_induplicates + rv.num_dma_inputs +
      rv.num_shape_tensors + rv.num_intermediates + rv.num_outputs +
      rv.num_outduplicates + rv.num_input_to_outduplicates +
      rv.num_intermediate_to_outduplicates + rv.num_output_to_outduplicates;

  TORCH_CHECK(
      total_tinfos == rv.dtensorinfos->size(),
      __LINE__,
      " ::",
      " num_inputs ",
      rv.num_inputs,
      " num_induplicates ",
      rv.num_induplicates,
      " num_dma_inputs ",
      rv.num_dma_inputs,
      " num_intermediates ",
      rv.num_intermediates,
      " num_outputs ",
      rv.num_outputs,
      " num_outduplicates ",
      rv.num_outduplicates,
      " num_input_to_outduplicates ",
      rv.num_input_to_outduplicates,
      " num_intermediate_to_outduplicates ",
      rv.num_intermediate_to_outduplicates,
      " num_output_to_outduplicates ",
      rv.num_output_to_outduplicates,
      " are not adding up to #dtensorinfos ",
      rv.dtensorinfos->size());

  if (enable_tensor_dump_) {
    if (0 == htensor_wbuff_size) {
      for (size_t i = 0; i < rv.num_tinfos; ++i) {
        htensor_wbuff_size =
            std::max(htensor_wbuff_size, rv.dtensorinfos->at(i).get_size());
      }
    }

    if (!htensor_wbuff) {
      synStatus status;
      status = synHostMalloc(
          device_id, htensor_wbuff_size, 0, (void**)&(htensor_wbuff));
      TORCH_CHECK(status == synSuccess, "host-malloc failed");
    }
    rv.htensor_wbuff = htensor_wbuff;
    rv.htensor_wbuff_size = htensor_wbuff_size;
  }

  if (enable_tensor_dump_) {
    DumpTensors_pre(rv);
  }

  rv.populate_syn_tensor_ids();

  if (refine_ds_enabled_) {
    // Initiate recipe execution time collection
    InitiateSynlaunchTimeCapture(rv);
    // Add the jit_ir_graph to current_dbipsh_
    current_dbipsh_->SetJitIRGraphPtr(jit_ir_graph);
    current_dbipsh_->UpdateCompileTime(t_compile_ns, current_bucket_id_);
  }

  PT_BRIDGE_DEBUG(
      "HabanaOp recipe cache :: launching new recipe", rv.header_str());

  std::shared_ptr<std::vector<IValPtrShared>> intermediate_tensors_ptr =
      std::make_shared<std::vector<IValPtrShared>>(
          std::vector<IValPtrShared>());

  for (auto& tensor : aten_intermediates) {
    IValPtrShared ivpsh = std::make_shared<IVal>(tensor);
    intermediate_tensors_ptr->push_back(ivpsh);
  }

  rv.launch(input_refs, intermediate_tensors_ptr);
  rv.update_hit_count();

  if (enable_tensor_dump_) {
    DumpTensors(rv);
  }

  if (enable_caching_) {
    // Add the <key,value> pair to the map
    if (false == refine_ds_enabled_) {
      std::shared_ptr<RecipeArgumentSpec> rargpsh =
          std::make_shared<RecipeArgumentSpec>(
              false, input_refs, jit_ir_graph, "");
      rv.key = rargpsh->hashCode();
      RecipeCacheLRU::get_cache().add(rargpsh, rvalpsh);
    } else {
      std::shared_ptr<RecipeArgumentSpec> rargpsh =
          std::make_shared<RecipeArgumentSpec>(
              input_refs, jit_ir_graph, cur_ds_token_);
      rv.key = rargpsh->hashCode();
      rvalpsh->dynamic_graph = syn_graph.is_dynamic_graph();
      RecipeCacheLRU::get_cache().add(rargpsh, rvalpsh);
    }
    PT_BRIDGE_DEBUG(
        "HabanaOp recipe cache :: adding new recipe to cache :: ", rv.key);
  }
  if (enable_caching_ && refine_ds_enabled_) {
    PT_DYNAMIC_SHAPE_DEBUG(
        current_dbipsh_->digest_str(),
        "Recipe Header::",
        rv.header_str(),
        "\n",
        rv.digest_str(),
        "\n",
        "--------------------");
  } else {
    PT_BRIDGE_DEBUG(rv.digest_str());
  }

  UpdateOutputs(rv);
}

void HabanaLaunchOpPT::CreateDynamicBucketInputShapes(
    habana_helpers::DynamicBucketInfo::InpTensorShapes& shape_map) {
  for (size_t i = 0; i < input_refs.size(); i++) {
    auto input = input_refs[i];
    if (input.isTensor()) {
      at::Tensor pt_tensor = input.toTensor();
      habana_helpers::TensorShape shape(
          pt_tensor.sizes(), pt_tensor.scalar_type());
      shape_map[i] = shape;
    }
  }
}

void HabanaLaunchOpPT::AdjustInputLayout() {
  for (size_t j = 0; j < pt_stack_sh.size(); j++) {
    auto value_input = jit_ir_graph->inputs().at(j);

    if (pt_stack_sh[j]->isTensor()) {
      // Taking alias as that allows us to detach it from PT and do metadata
      // changes It gives us more control over tensor changes, but caution
      // is needed. Its might be a bit dangerous, but only way to
      // communicate layour changes PT doesnt allow any stride changes we
      // want, we can review it with PT folks

      auto impl =
          habana_lazy::GetHbInternalTensorImpl(pt_stack_sh[j]->toTensor());
      bool is_shape_tensor = impl && impl->isShapeTensor();

      /*
       * If we detach from pytorch, we loose the shape tensor related info
       * For shape tensors, we dont create an alias, since these tensors are
       * created by the frontend, we dont need tp detach.
       * Assumption here is that the Frontend will not create a shape tensor
       * with zero dims
       */
      auto tensor = is_shape_tensor ? pt_stack_sh[j]->toTensor()
                                    : at::alias(pt_stack_sh[j]->toTensor());

      // WE dont support 0D tensors internally, so convert to 1D internally
      if (tensor.dim() == 0) {
        HABANA_ASSERT(is_shape_tensor == false);
        tensor.unsafeGetTensorImpl()->set_sizes_contiguous({1});
      }

      IValPtrShared ivptrsh = std::make_shared<IVal>(tensor);
      value_to_ivalue[value_input] = ivptrsh;
      pt_stack_sh[j] = ivptrsh;
    } else {
      value_to_ivalue[value_input] = pt_stack_sh[j];
    }
  }
}

torch::jit::Stack HabanaLaunchOpPT::CreateStack(
    const torch::jit::Stack& stack,
    habana_helpers::DynamicBucketInfo::InpTensorShapes& dynamic_shapes) {
  PT_BRIDGE_BEGIN;
  torch::jit::Stack new_stack;

  for (size_t i = 0; i < stack.size(); ++i) {
    if (dynamic_shapes.count(i)) {
      auto& tensor = stack[i].toTensor();
      auto impl = habana_lazy::GetHbInternalTensorImpl(tensor);
      //
      // TODO: When creating a new stack, we need to look, if this
      // can be done using storage less pytorch tensor, need to fix
      // this
      auto new_tensor = at::empty(
          dynamic_shapes.at(i).get_dims(),
          tensor.options(),
          tensor.suggest_memory_format());

      /*
       * Every new tensor is created using Habana Tensor Implementer.
       * Ensure propogation of shape tensor information for the new
       * tensor created for the stack.
       */
      auto new_impl = habana_lazy::GetHbInternalTensorImpl(new_tensor);
      HABANA_ASSERT(new_impl);
      if (impl) {
        new_impl->setTensorType(impl->getTensorType());
      }
      new_stack.push_back(torch::jit::IValue(new_tensor));
    } else {
      new_stack.push_back(stack[i]);
    }
  }
  PT_BRIDGE_END;
  return new_stack;
}

void HabanaLaunchOpPT::InitiateSynlaunchTimeCapture(RecipeValueSpec& rv) {
  // Initiate recipe execution time collection
  if (current_dbipsh_->NeedRunTimeSlot(current_bucket_id_)) {
    auto& syn_device = synapse_helpers::HPURegistrar::get_device();
    if (GET_ENV_FLAG_NEW(PT_ENABLE_SYNLAUNCH_TIME_CAPTURE)) {
      auto& time_event_handle_cache = syn_device.get_time_event_handle_cache();
      if (time_event_handle_cache.get_total_events_count() <
          synapse_helpers::event_handle_cache::
              get_num_events_high_watermark()) {
        rv.time_slot_ = std::make_shared<synapse_helpers::TimeSlot>(
            syn_device.get_cached_time_event_handle(),
            syn_device.get_cached_time_event_handle(),
            static_cast<synStreamHandle>(syn_device.get_compute_stream()));
        current_dbipsh_->RegisterTimeSlot(rv.time_slot_, current_bucket_id_);
      } else {
        PT_BRIDGE_WARN(
            "High water mark for synapse events ",
            synapse_helpers::event_handle_cache::
                get_num_events_high_watermark(),
            " reached, will not create any time event");
        rv.time_slot_ = nullptr;
      }
    }
  }
}
void HabanaLaunchOpPT::ProcessHabanaFusedOpWithDS() {
  PT_BRIDGE_BEGIN;

  std::shared_ptr<RecipeArgumentSpec> rargpsh =
      std::make_shared<RecipeArgumentSpec>(jit_ir_graph, input_refs);
  PT_DYNAMIC_SHAPE_DEBUG(
      "====================\n",
      "Processing with dynamic shape enabled\n",
      "JIT IR graph_hash_code : ",
      rargpsh->graphHashCode());

  current_dbipsh_ = DynamicBucketInfoMap::get_instance().get(rargpsh);
  if (nullptr == current_dbipsh_) {
    PT_DYNAMIC_SHAPE_DEBUG("Creating new DynamicBucketInfo");
    auto dbi = habana_helpers::DynamicBucketInfo();
    current_dbipsh_ = std::make_shared<habana_helpers::DynamicBucketInfo>(dbi);
    DynamicBucketInfoMap::get_instance().add(rargpsh, current_dbipsh_);
  }

  DynamicShapeInfo graph_input_info;
  CreateDynamicBucketInputShapes(graph_input_info.act_input_tshapes);
  PT_DYNAMIC_SHAPE_DEBUG(
      "Input shapes::\n",
      graph_input_info.act_input_tshapes,
      "--------------------");

  if (GET_ENV_FLAG_NEW(PT_HPU_ENABLE_BUCKET_REFINEMENT)) {
    current_dbipsh_->CheckForSplitBucket();
  }
  current_dbipsh_->CollectDynamicDims(graph_input_info.act_input_tshapes);
  current_bucket_id_ =
      current_dbipsh_->GetBucketId(graph_input_info.act_input_tshapes);
  cur_ds_token_ = current_dbipsh_->GetTokenForBucketId(current_bucket_id_);

  PT_DYNAMIC_SHAPE_DEBUG(
      jit_ir_graph->toString(),
      "current bucket id : ",
      current_bucket_id_);

  auto ranges = current_dbipsh_->CalculateShapes(current_bucket_id_);
  if (ranges.empty()) {
    PT_DYNAMIC_SHAPE_DEBUG(
        "exact graph with token : ",
        cur_ds_token_,
        "\n",
        "--------------------");
  } else {
    PT_DYNAMIC_SHAPE_DEBUG(
        "dynamic graph with token : ",
        cur_ds_token_,
        '\n',
        "Input range ::\n",
        ranges.DebugString(),
        "--------------------");

    graph_input_info.min_input_tshapes.insert(
        ranges.min_shapes.begin(), ranges.min_shapes.end());
    graph_input_info.max_input_tshapes.insert(
        ranges.max_shapes.begin(), ranges.max_shapes.end());
    graph_input_info.current_bucket_id = current_bucket_id_;
    graph_input_info.min_policy = current_dbipsh_->GetMinPolicy();
    graph_input_info.max_policy = current_dbipsh_->GetMaxPolicy();
  }

  // Check for cached recipe
  if (enable_caching_) {
    std::shared_ptr<RecipeArgumentSpec> spec_key =
        std::make_shared<RecipeArgumentSpec>(
            input_refs, jit_ir_graph, cur_ds_token_);

    std::shared_ptr<RecipeValueSpec> rvpsh = GetCachedRecipe(spec_key);

    if (ABSL_PREDICT_TRUE(rvpsh)) {
      // Cache hit for a dynamic bucket
      // Steps:
      // 1. Infer shapes of all persistent tensors which are not input
      // 2. Patch using the exact shape
      // 3. Launch
      // 4. Update outputs
      current_dbipsh_->IncrementHitCount(current_bucket_id_);

      RecipeValueSpec& rv = *rvpsh;
      rv.update_hit_count();

      // Initiate recipe execution time collection
      InitiateSynlaunchTimeCapture(rv);

      if (rv.dynamic_graph) {
        // For Dynamic shapes in case of cache hit, we need to run
        // shape inference for determining the output shape and
        // persistent intermediates
        PT_DYNAMIC_SHAPE_DEBUG("Running output shape inference pass");
        try_run_shape_inference(
            ShapeInfo::InferencePass::OUTPUT_SHAPE, graph_input_info);
      }

      std::shared_ptr<std::vector<IValPtrShared>> intermediate_tensors_ptr =
          std::make_shared<std::vector<IValPtrShared>>(
              std::vector<IValPtrShared>());

      std::shared_ptr<std::vector<IValPtrShared>> dma_inputs_ptr =
          std::make_shared<std::vector<IValPtrShared>>(
              std::vector<IValPtrShared>());

      rv.update_patching_table(
          input_refs,
          intermediate_tensors_ptr,
          dma_inputs_ptr,
          m_map_shape.m_actual_shapes);

      if (enable_tensor_dump_) {
        DumpTensors_pre(rv);
      }
      rv.launch(input_refs, intermediate_tensors_ptr, dma_inputs_ptr);

      if (enable_tensor_dump_) {
        DumpTensors(rv);
      }

      // Update the stack from the recipe itself
      UpdateOutputs(rv);
      ReturnCachedRecipe(rv);
      PT_DYNAMIC_SHAPE_DEBUG(
          "HabanaOp recipe cache hit :: key ",
          spec_key->hashCode(),
          "\n",
          current_dbipsh_->digest_str(),
          "Recipe Header::",
          rv.header_str(),
          "\n",
          rv.digest_str(),
          "\n",
          "--------------------");

      clear();
      PT_BRIDGE_END;
      return;
    } else {
      PT_DYNAMIC_SHAPE_DEBUG(
          "HabanaOp recipe cache miss :: key ", spec_key->hashCode());
    }
  }

  CompileAndRunDynamicGraph(graph_input_info);
  clear();
  PT_BRIDGE_END;
}

void HabanaLaunchOpPT::DumpTensors_pre(RecipeValueSpec& rv) {
  if (enable_tensor_dump_) {
    std::ofstream tensor_file;
    tensor_file.open(
        tdmp_file_name_pre_.c_str(), std::ios::out | std::ios::app);
    for (size_t i = 0; i < rv.num_tinfos; ++i) {
      if (rv.dtensorinfos->at(i).watch_enabled()) {
        rv.d2h_dbuff(i);
        rv.print_hbuff(i, tensor_file, iteration_count_, tensor_dump_numel_);
      }
    }
    tensor_file.close();
  }
}

void HabanaLaunchOpPT::DumpTensors(RecipeValueSpec& rv) {
  if (enable_tensor_dump_) {
    std::ofstream tensor_file;
    tensor_file.open(tdmp_file_name_.c_str(), std::ios::out | std::ios::app);
    for (size_t i = 0; i < rv.num_tinfos; ++i) {
      if (rv.dtensorinfos->at(i).watch_enabled()) {
        rv.d2h_dbuff(i);
        rv.print_hbuff(i, tensor_file, iteration_count_, tensor_dump_numel_);
      }
    }
    tensor_file.close();
  }
}

void HabanaLaunchOpPT::PrintRecipeInputs() {
  std::ostream& O = std::cout;

  O << "aten_inputs #" << num_inputs << "::" << '\n';
  size_t idx{0};
  for (size_t i = pt_stack_sh.size() - num_inputs; i < pt_stack_sh.size();
       i++) {
    auto vp = jit_ir_graph->inputs().at(i);
    O << idx++ << " : %" << vp->debugName() << " : ";
    PrintATenTensor(pt_stack_sh.at(i));
  }
}

void HabanaLaunchOpPT::UpdateOutputs() {
  drop(*pt_stack, num_inputs);
  for (auto output : jit_ir_graph->outputs()) {
    auto oit = value_to_ivalue.find(output);
    TORCH_CHECK(
        oit != value_to_ivalue.end(),
        "value_to_ivalue does not have an entry for %",
        output->debugName());
    IValPtrShared ivpsh = oit->second;
    pt_stack->insert(pt_stack->end(), *ivpsh);
  }
}

void HabanaLaunchOpPT::UpdateOutputs(RecipeValueSpec& rv) {
  // Update the stack from the recipe itself
  drop(*pt_stack, num_inputs);
  for (const auto& ivpsh : *(rv.aten_outputs)) {
    pt_stack->insert(pt_stack->end(), *ivpsh);
  }

  if (enable_caching_) {
    // Release the tensor handles from the recipe
    rv.aten_outputs = nullptr;
    // Retain the aten_intermediates as these tensors are otherwise
    // going to be released while the recipe is in execution.
    // Keep these tensors cached with the recipe and reuse on execution.
    // Even when two separate graphs hit the same cache entry, it is fine
    // to reuse the intermediate tensors as they only exist within the
    // scope of the recipe execution and multiple recipes execute
    // serially on the same compute stream.
  }
}

template <typename T>
void HabanaLaunchOpPT::clearMember(T& m_container) {
  T empty;
  using std::swap;
  swap(m_container, empty);
}

void HabanaLaunchOpPT::clear(bool is_shape_inference) {
  if (is_shape_inference == false) {
    pt_stack = nullptr;
    pt_stack_sh.clear();
    num_tensor_inputs = 0;
    habana::ShapeInference::Reset();
  }

  value_to_ivalue.clear();
  watchlist_.clear();
  syn_graph_ptr = nullptr;

  habana_kernels.clear();

  input_tivs.clear();
  duplicate_input_tivs.clear();
  input_tiv_map.clear();

  intermediate_tinfos.clear();
  dma_input_tensorinfos.clear();
  shape_tensor_tinfos.clear();
  output_tensorinfos.clear();
  duplicate_outtinfos.clear();
  duplicate_input_to_outtinfo_map.clear();
  duplicate_intermediate_to_outtinfo_map.clear();

  aten_intermediates.clear();
  aten_dma_inputs.clear();

  output_tensorinfo_map.clear();

  valptr_to_persistent_map.clear();

  pt_to_synapse_tensors.clear();
  meta_syn_tensors.clear();
  buff_to_input_ivpsh_map.clear();
  buff_to_intermediate_ivpsh_map.clear();
  buff_to_output_ivpsh_map.clear();
  buff_to_syn_tensor_map.clear();

  jit_to_synapse_node_idx_map.clear();
}

void RecipeValueSpec::create_outdup(
    size_t ti_idx,
    std::unordered_map<size_t, IValPtrShared>& parent_ivpsh_map,
    std::string map_name) {
  // The aten_output_num is the total number of outputs
  size_t aten_output_num = num_outputs + num_input_to_outduplicates +
      num_intermediate_to_outduplicates + num_output_to_outduplicates;

  PtTensorInfo& ti = dtensorinfos->at(ti_idx);
  auto output_idx = ti.get_output_index();
  TORCH_CHECK(
      output_idx < aten_output_num,
      "output index ",
      output_idx,
      " is greater than #outputs ",
      aten_output_num);

  size_t parent_idx = ti.get_parent_index();
  TORCH_CHECK(
      parent_ivpsh_map.count(parent_idx),
      "Actual input with idx ",
      parent_idx,
      " not found in ",
      map_name);

  auto ivpsh_parent = parent_ivpsh_map[parent_idx];
  auto parent_tensor = ivpsh_parent->toTensor();

  auto& pt_sizes{ti.get_shape()};
  auto& pt_strides{ti.get_strides()};
  long pt_offset = (long)ti.get_offset() / parent_tensor.itemsize();
  auto pt_opt_offset = c10::make_optional(pt_offset);

  at::Tensor pt_outdup =
      at::as_strided(parent_tensor, pt_sizes, pt_strides, pt_opt_offset);

  ti.patch(pt_outdup);

  IValPtrShared ivpsh = std::make_shared<IVal>(pt_outdup);
  aten_outputs->at(output_idx) = ivpsh;
}

void HabanaLaunchOpPT::run(torch::jit::Stack& stack) {
  PT_BRIDGE_BEGIN;
  num_inputs = jit_ir_graph->inputs().size();
  num_tensor_inputs = 0;
  input_refs = last(stack, num_inputs);
  iteration_count_++;
  auto& device = synapse_helpers::HPURegistrar::get_device();

  //
  // Set the habana operators to capture data
  if (refine_ds_enabled_) {
    habana::ShapeInference::Capture(&m_map_shape);
  }

  // Keep a handle to the stack for future use
  pt_stack = &stack;

  size_t j = stack.size() - num_inputs;
  for (; j < stack.size(); j++) {
    IValPtrShared ivpsh = std::make_shared<IVal>(stack[j]);
    pt_stack_sh.push_back(ivpsh);
    if (ivpsh->isTensor() || ivpsh->isTensorList()) {
      num_tensor_inputs++;
    }
  }

  // Fusion pass should ensure all nodes are on Habana, if all nodes not on
  // habana device, we should assert
  bool is_all_hpu = true;
  for (auto& input : input_refs) {
    if (input.isTensor()) {
      is_all_hpu = input.toTensor().device().type() != c10::DeviceType::HPU
          ? false
          : is_all_hpu;
    }
  }

  // We dont support running some ops on CPU while running fused op on Habana
  // All tensors should be alocated to habana before entering this phase
  TORCH_CHECK(
      is_all_hpu == true, " Habana Fusion needs all tensors to be in HPU ");

  PT_BRIDGE_DEBUG(
      "Lowering JIT IR Graph ====\n",
      jit_ir_graph->toString(),
      "JIT IR Graph ----\n");

  // Handle everything related to graph when dynamic flag is set.
  if (refine_ds_enabled_) {
    ProcessHabanaFusedOpWithDS();
    return;
  }

  // caching :: begin
  if (enable_caching_) {
    std::shared_ptr<RecipeArgumentSpec> spec_key =
        std::make_shared<RecipeArgumentSpec>(
            false, input_refs, jit_ir_graph, "");

    std::shared_ptr<RecipeValueSpec> rvpsh = GetCachedRecipe(spec_key);

    if (ABSL_PREDICT_TRUE(rvpsh)) {
      RecipeValueSpec& rv = *rvpsh;
      rv.update_hit_count();

      PT_BRIDGE_DEBUG(
          "HabanaOp recipe cache hit ::",
          rv.header_str(),
          "\n",
          rv.digest_str());

      std::shared_ptr<std::vector<IValPtrShared>> intermediate_tensors_ptr =
          std::make_shared<std::vector<IValPtrShared>>(
              std::vector<IValPtrShared>());

      std::shared_ptr<std::vector<IValPtrShared>> dma_inputs_ptr =
          std::make_shared<std::vector<IValPtrShared>>(
              std::vector<IValPtrShared>());

      rv.update_patching_table(
          input_refs,
          intermediate_tensors_ptr,
          dma_inputs_ptr,
          m_map_shape.m_actual_shapes);

      if (enable_tensor_dump_) {
        DumpTensors_pre(rv);
      }
      rv.launch(input_refs, intermediate_tensors_ptr, dma_inputs_ptr);

      if (enable_tensor_dump_) {
        DumpTensors(rv);
      }

      // Update the stack from the recipe itself
      UpdateOutputs(rv);
      ReturnCachedRecipe(rv);

      clear();
      PT_BRIDGE_END;
      return;
    } else {
      PT_BRIDGE_DEBUG(
          "HabanaOp recipe cache miss :: key ", spec_key->hashCode());
    }
  }
  // caching :: end

  AdjustInputLayout();

  //<Decription> This is the main function that
  //  a. creates the HabanaLaunchOp
  //  b. compiles and executes the same
  auto syn_graph =
      habana_helpers::create_graph(device.id(), GetSynapseGraphName());
  CompileAndExecuteHabanaFusedOpKernel(syn_graph);

  // clear the context
  // TODO : See if we need to add a contect to this object pointer or clearing
  // like this is good?

  // Fetch the output shape tensors for cache miss cases??

  clear();

  PT_BRIDGE_END;
}

void HabanaLaunchOpPT::run_pass() {
  PT_BRIDGE_BEGIN;
  auto& device = synapse_helpers::HPURegistrar::get_device();

  //
  // Run the compile and execute method to infer the shapes
  auto syn_graph =
      habana_helpers::create_graph(device.id(), GetSynapseGraphName(), true);
  syn_graph.set_dynamic_graph(true);
  AdjustInputLayout();
  CompileAndExecuteHabanaFusedOpKernel(syn_graph, true);
  //
  // clear the data that has been setup as part of the above
  // method
  clear(true);
  PT_BRIDGE_END;
}

void HabanaLaunchOpPT::run_shape_inference(
    const ShapeInfo::InferencePass& pass,
    DynamicShapeInfo& graph_input_info) {
  PT_BRIDGE_BEGIN;
  torch::jit::Stack new_stack;
  torch::jit::Stack* old_stack = nullptr;
  std::vector<IValPtrShared> old_pt_stack_sh;
  if ((pass == ShapeInfo::InferencePass::MIN_SHAPE) ||
      (pass == ShapeInfo::InferencePass::MAX_SHAPE)) {
    old_stack = pt_stack;
    old_pt_stack_sh = pt_stack_sh;
    pt_stack_sh.clear();
    if (pass == ShapeInfo::InferencePass::MIN_SHAPE) {
      new_stack = CreateStack(*pt_stack, graph_input_info.min_input_tshapes);
    } else {
      new_stack = CreateStack(*pt_stack, graph_input_info.max_input_tshapes);
    }
    pt_stack = &new_stack;

    size_t j = new_stack.size() - num_inputs;
    for (; j < new_stack.size(); j++) {
      IValPtrShared ivpsh = std::make_shared<IVal>(new_stack[j]);
      pt_stack_sh.push_back(ivpsh);
    }
  }
  m_map_shape.m_pass = pass;
  bool throw_exception = false;
  std::string error_str;
  try {
    run_pass();
  } catch (std::exception& e) {
    error_str = e.what();
    PT_DYNAMIC_SHAPE_WARN(
        "Exception occured in Pass = ", pass, " - Details :\n", error_str);
    throw_exception = true;
  }

  if (old_stack) {
    pt_stack_sh.clear();
    pt_stack = old_stack;
    pt_stack_sh = old_pt_stack_sh;
  }
  if (throw_exception == true) {
    throw PassException(pass, error_str);
  }
  PT_BRIDGE_END;
}

void HabanaLaunchOpPT::handle_pass_exception(
    DynamicShapeInfo& graph_input_info,
    const PassException& e) {
  PT_BRIDGE_BEGIN;
  PT_DYNAMIC_SHAPE_WARN("Handling the exception .. ");
  switch (e.Pass()) {
    // Min inference pass can have exception only in HISTORIC if exception is
    // in policy = CURRENT, it is unrecoverable, throw runtime error in this
    // case
    case ShapeInfo::InferencePass::MIN_SHAPE:
      switch (graph_input_info.min_policy) {
        case habana_helpers::DynamicDimsPolicy::HISTORIC:
          graph_input_info.min_policy =
              habana_helpers::DynamicDimsPolicy::CURRENT;
          break;
        default:
          PT_DYNAMIC_SHAPE_FATAL(
              "Unhandled Min Policy exiting .. ", graph_input_info.min_policy);
          throw std::runtime_error("Exception was not handled ..");
          break;
      }
      break;
    // Max inference pass can have exception only in CALCULATED if exception is
    // in policy = CURRENT, it is unrecoverable, throw runtime error in this
    // case
    case ShapeInfo::InferencePass::MAX_SHAPE:
      switch (graph_input_info.max_policy) {
        case habana_helpers::DynamicDimsPolicy::CALCULATED:
          graph_input_info.max_policy =
              habana_helpers::DynamicDimsPolicy::HISTORIC;
          break;
        case habana_helpers::DynamicDimsPolicy::HISTORIC:
          graph_input_info.max_policy =
              habana_helpers::DynamicDimsPolicy::CURRENT;
          break;
        default:
          PT_DYNAMIC_SHAPE_FATAL(
              "Unhandled Max Policy exiting .. ", graph_input_info.max_policy);
          throw std::runtime_error("Exception was not handled ..");
          break;
      }
      break;
    // In OUTPUT_SHAPE inference exception if min and max both was current,
    // meaning the failure is in static(fallback path), bail out execution by
    // throwing error. Otherwise the compilation error has occured and we need
    // to rerun with min and max policy as CURRENT.
    case ShapeInfo::InferencePass::OUTPUT_SHAPE:
      if (graph_input_info.min_policy ==
              habana_helpers::DynamicDimsPolicy::CURRENT &&
          graph_input_info.max_policy ==
              habana_helpers::DynamicDimsPolicy::CURRENT) {
        PT_DYNAMIC_SHAPE_FATAL("Unhandled exception exiting .. ");
        throw std::runtime_error("Exception was not handled ..");
        break;
      }
      PT_DYNAMIC_SHAPE_WARN(
          "Output/Compile exception changing min and max to CURRENT..");
      graph_input_info.min_policy = habana_helpers::DynamicDimsPolicy::CURRENT;
      graph_input_info.max_policy = habana_helpers::DynamicDimsPolicy::CURRENT;
      break;
    default:
      PT_DYNAMIC_SHAPE_FATAL("Unhandled exception exiting .. ");
      throw std::runtime_error("Exception was not handled ..");
      break;
  }

  // The above switch case changes the policy, get new ranges with changed
  // policy.
  current_dbipsh_->UpdateBucketWithPolicy(
      graph_input_info.current_bucket_id,
      graph_input_info.act_input_tshapes,
      graph_input_info.min_policy,
      graph_input_info.max_policy);
  auto fallback_ranges =
      current_dbipsh_->CalculateShapes(graph_input_info.current_bucket_id);

  // After calculating ranges reset bucket_info policy to default
  current_dbipsh_->SetDefaultPolicy();

  switch (e.Pass()) {
    // In reruning min pass, clear the min name-shape map and rerun
    case ShapeInfo::InferencePass::MIN_SHAPE:
      graph_input_info.min_input_tshapes.clear();
      graph_input_info.min_input_tshapes.insert(
          fallback_ranges.min_shapes.begin(), fallback_ranges.min_shapes.end());
      habana::ShapeInference::ResetMin();
      PT_DYNAMIC_SHAPE_DEBUG(
          "Rerun min shape inference pass with policy ",
          graph_input_info.min_policy);
      try_run_shape_inference(
          ShapeInfo::InferencePass::MIN_SHAPE, graph_input_info);
      break;
    // In reruning max pass, clear the max name-shape map and rerun
    case ShapeInfo::InferencePass::MAX_SHAPE:
      graph_input_info.max_input_tshapes.clear();
      graph_input_info.max_input_tshapes.insert(
          fallback_ranges.max_shapes.begin(), fallback_ranges.max_shapes.end());
      habana::ShapeInference::ResetMax();
      PT_DYNAMIC_SHAPE_DEBUG(
          "Rerun max shape inference pass with policy ",
          graph_input_info.max_policy);
      try_run_shape_inference(
          ShapeInfo::InferencePass::MAX_SHAPE, graph_input_info);
      break;
    // In reruning output pass, clear the min, max and actual name-shape map
    // populate the graph_input_info structure with ranges and call
    // CompileAndRunDynamicGraph to again try compilation. If the exception
    // occurs then we again call handle_pass_exception with both policy as
    // CURRENT and pass as OUTPUT_PASS which breaks the handling and throws
    // runtime error.
    case ShapeInfo::InferencePass::OUTPUT_SHAPE:
      PT_DYNAMIC_SHAPE_WARN(
          "Output/Compile Pass exception rerun with policy CURRENT ..");
      graph_input_info.min_input_tshapes.clear();
      graph_input_info.max_input_tshapes.clear();
      graph_input_info.max_input_tshapes.insert(
          fallback_ranges.max_shapes.begin(), fallback_ranges.max_shapes.end());
      graph_input_info.min_input_tshapes.insert(
          fallback_ranges.min_shapes.begin(), fallback_ranges.min_shapes.end());
      habana::ShapeInference::ResetMin();
      habana::ShapeInference::ResetMax();
      habana::ShapeInference::ResetActual();
      CompileAndRunDynamicGraph(graph_input_info);
      break;
    default:
      PT_DYNAMIC_SHAPE_FATAL("Unhandled exception exiting .. ");
      throw std::runtime_error("Exception was not handled ..");
      break;
  }
  PT_BRIDGE_END;
}

// Handle running passes and calls CompileAndExecute.
// Also handles fallback and failures.
void HabanaLaunchOpPT::CompileAndRunDynamicGraph(
    DynamicShapeInfo& graph_input_info) {
  auto& device = synapse_helpers::HPURegistrar::get_device();
  // If both min and max exists then the graph is dynamic
  bool is_dynamic_graph = (!graph_input_info.min_input_tshapes.empty()) &&
      (!graph_input_info.max_input_tshapes.empty());

  // In case of dynamic mode (cache miss & dynamic range exists), we need to
  // run shape inference for min and max passes
  // TODO: Once the bucket range issue is fixed, we need to run
  // the shape inference only for once for max shapes
  if (is_dynamic_graph) {
    // run min shape inference pass
    PT_DYNAMIC_SHAPE_DEBUG(
        "Running min shape inference pass with policy=",
        graph_input_info.min_policy);
    try_run_shape_inference(
        ShapeInfo::InferencePass::MIN_SHAPE, graph_input_info);
    // run max shape inference pass
    PT_DYNAMIC_SHAPE_DEBUG(
        "Running max shape inference pass with policy=",
        graph_input_info.max_policy);
    try_run_shape_inference(
        ShapeInfo::InferencePass::MAX_SHAPE, graph_input_info);
  }

  AdjustInputLayout();

  PT_DYNAMIC_SHAPE_DEBUG(
      "Running CompileAndExecuteHabanaFusedOpKernel with min{",
      graph_input_info.min_policy,
      "}:max{",
      graph_input_info.max_policy,
      "}");

  // Try running the CompileAndExecuteHabanaFusedOpKernel with min and max
  // infered above if the CompileAndExecuteHabanaFusedOpKernel fails, call
  // handle_pass_exception with pass type OUTPUT_SHAPE. In handling this
  // exception bucket ranges are recalculated as per min and max both as CURRENT
  // and again call CompileAndRunDynamicGraph with changed ranges and policy.
  // This is last resort if anything further fails bail out the execution. We
  // need not change anything in cache because exception either occurs in
  // compilation or launch and both happens before adding recipie to cache.
  try {
    auto syn_graph =
        habana_helpers::create_graph(device.id(), GetSynapseGraphName());
    syn_graph.set_dynamic_graph(is_dynamic_graph);
    CompileAndExecuteHabanaFusedOpKernel(syn_graph);
  } catch (std::exception& e) {
    PT_DYNAMIC_SHAPE_WARN(
        "Exception in CompileAndExecuteHabanaFusedOpKernel Details:\n",
        e.what());
    clear(true);
    PassException p(habana::ShapeInfo::InferencePass::OUTPUT_SHAPE, e.what());
    handle_pass_exception(graph_input_info, p);
  }
}
} // namespace habana
