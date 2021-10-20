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
#include "habana_kernels/random_gen_kernels.h"
#include "habana_kernels/unary_kernels.h"

#include "habana_lazy/hlexec.h"

#include "synapse_helpers/env_flags.h"

using namespace torch::jit;

namespace habana {

// static initializations
const std::unordered_set<std::string> HabanaMetaOpList::meta_ops = {
    // Add aten string here for ops to support
    // e.g  :: "aten::view"
    "aten::size",
    "prim::dtype"};

size_t HabanaLaunchOpPT::instance_count_ = 0;
std::unordered_set<std::string> HabanaLaunchOpPT::watchlist_ = {};
//--------------------------------------

habana_lazy::HbLazyTensorImpl* TryGetHbLazyImpl(const at::Tensor& tensor) {
  return dynamic_cast<habana_lazy::HbLazyTensorImpl*>(
      tensor.unsafeGetTensorImpl());
}

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

HabanaLaunchOpPT::HabanaLaunchOpPT(const torch::jit::Node* node, bool dbg)
    : HabanaLaunchOpPT(
          node->g(attr::Subgraph),
          dbg,
          node->kind().toQualString()) {}

HabanaLaunchOpPT::HabanaLaunchOpPT(
    std::shared_ptr<torch::jit::Graph> graph,
    bool dbg,
    const char* name)
    : jit_ir_graph{std::move(graph)}, debug(dbg) {
  refine_ds_enabled_ = GET_ENV_FLAG(PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES);
  op_name = (name ? std::string(name) : std::string("HabanaLaunchOp"));
  std::replace(op_name.begin(), op_name.end(), ':', '_');
  std::ostringstream oss;
  oss << op_name << '_' << instance_count_;
  instance_count_++;
  id_str = oss.str();

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
  enable_tensor_release_ = true;
  // Enable enable_tensor_release_ with lazy mode by default
  if (const auto envp = GET_ENV_FLAG(PT_HPU_LAZY_MODE)) {
    enable_tensor_release_ = (envp != 0);
  }

  // The caching as well as enable_tensor_release_ can be overridden
  // with HABANA_PGM_ENABLE_CACHE
  if (const auto envp = get2env("HABANA_PGM_ENABLE_CACHE")) {
    auto val = atoi(envp);
    if (val & 0x1) {
      enable_tensor_release_ = ((val & 0x3) == 0x3);
    } else {
      enable_caching_ = false;
    }
  }
  use_persistent_tensors = false;

  if (const auto envp = getenv("HABANA_USE_PERSISTENT_TENSOR")) {
    use_persistent_tensors = atoi(envp) == 1;
  }

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

LayoutFormat HabanaLaunchOpPT::getTensorChannelOrder(torch::jit::Value* val) {
  // The value of the node keeps the tensor physical layout memorized
  // We can update this later if we see any changes to the way layouts are
  // handled
  TORCH_CHECK(
      value_to_tensor_layout.find(val) != std::end(value_to_tensor_layout),
      "HabanaFusion : Channel order not updated");
  return value_to_tensor_layout[val].layout;
}

// See if we are in any leagally accepted channel orders
bool HabanaLaunchOpPT::isChannelOrderSupported(
    torch::jit::Value* val,
    const LayoutFormat& supported_channel_order) {
  return (supported_channel_order == LayoutFormat::ANY) ||
      (supported_channel_order == getTensorChannelOrder(val));
}

bool HabanaLaunchOpPT::IsOutputToRestride(torch::jit::Value* value) {
  auto uses = value->uses();
  for (auto u : uses) {
    auto restride_node = u.user;
    if (strcmp(restride_node->kind().toQualString(), "hpu::restride_cl") == 0) {
      return true;
    }
  }
  return false;
}

torch::jit::Value* HabanaLaunchOpPT::GetRestridedOutvalue(
    torch::jit::Value* val) {
  for (auto u : val->uses()) {
    auto restride_node = u.user;
    if (strcmp(restride_node->kind().toQualString(), "hpu::restride_cl") == 0) {
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
  auto impl = TryGetHbLazyImpl(pt_tensor);

  if (impl && impl->isShapeTensor()) {
    auto& syn_tensor =
        habana_op->AllocateSynapseInput(*syn_graph_ptr, pt_tensor, true, true);
    return syn_tensor;
  } else {
    habana_helpers::TensorShape min_shape, max_shape;

    void* pt_tensor_buffer_start = pt_tensor.storage().data_ptr().get();
    auto syn_tensor_it = buff_to_syn_tensor_map.find(pt_tensor_buffer_start);
    if (syn_tensor_it != buff_to_syn_tensor_map.end()) {
      synapse_helpers::tensor& st = syn_tensor_it->second;
      habana_op->set_is_duplicate_input_flag(true);
      habana_op->add_syn_input_tensor_orig(st);
    }

    auto& syn_tensor =
        habana_op->AllocateSynapseInput(*syn_graph_ptr, pt_tensor, true, false);

    if (syn_tensor_it != buff_to_syn_tensor_map.end()) {
      habana_op->set_is_duplicate_input_flag(false);
      habana_op->clear_syn_input_tensor_orig();
    }

    buff_to_syn_tensor_map.emplace(
        pt_tensor_buffer_start, tensor_or_ref(syn_tensor));
    return syn_tensor;
  }
}
void HabanaLaunchOpPT::HandleUnmappedTensor(
    CValPtr value_in,
    const HabanaOperatorPtr& habana_op,
    SharedSynTensorOrRefListPtr& tensorList) {
  std::vector<at::Tensor> pyTensorList;
  if (value_to_ivalue[value_in]->isTensor()) {
    pyTensorList.emplace_back(value_to_ivalue[value_in]->toTensor());
  } else {
    c10::List<at::Tensor> pytList = value_to_ivalue[value_in]->toTensorList();
    for (at::Tensor pyTensor : pytList) {
      pyTensorList.emplace_back(pyTensor);
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
        syn_tensor.tensor_type());
    tiv.push_back(ti);

    if (enable_caching_) {
      void* buffp = ti.get_buffer_start();
      buff_to_input_ivpsh_map.emplace(buffp, value_to_ivalue[value_in]);
    }
  }

  if (!tensorList->empty()) {
    auto it =
        pt_to_synapse_tensors.emplace(value_to_ivalue[value_in], tensorList);
    if (it.second == false) {
      pt_to_synapse_tensors[value_to_ivalue[value_in]] = tensorList;
    }

    if (enable_caching_) {
      input_tiv_map.emplace(value_to_ivalue[value_in], tiv);
      if (strcmp(value_in->node()->kind().toQualString(), "hpu::restride_cl") ==
          0) {
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
            SharedSynTensorOrRefListPtr tensor_ref_list_ptr_sh =
                std::make_shared<SynTensorOrRefList>();
            HABANA_ASSERT(value_to_ivalue[value_in]->isTensor());
            HandleMappedandUnmappedTensor(
                value_in, habana_op, tensor_ref_list_ptr_sh);
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
  //       With enable_tensor_release_, this is maintained in
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
  //       enable_tensor_release_
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
  //       enable_tensor_release_
  //    D: It is a duplicate of an existing output

  auto ti = PtTensorInfo(
      ivpsh,
      out_syntensor.name(),
      vp,
      watch_tensor_flag_,
      out_syntensor.tensor_type());
  void* buffp = ti.get_buffer_start();

  if (false == isInGraphOutputs(vp)) {
    if (buff_to_input_ivpsh_map.count(buffp)) {
      // Case 1.A: intermediate persistent tensor which an alias of an input
      PT_BRIDGE_DEBUG("Adding to duplicate_input_tivs ", buffp);
      duplicate_input_tivs.emplace_back(ti);
    } else {
      if (buff_to_output_ivpsh_map.count(buffp)) {
        duplicate_outtinfos.emplace_back(ti);
      } else {
        // Case 1.B: intermediate persistent tensor
        if (enable_tensor_release_ && ti.is_view_tensor()) {
          PT_BRIDGE_DEBUG(
              "Starting persistent intermediate is view tensor ",
              "with non zero offset ",
              ti.get_offset());
        }
        AddAtenIntermediate(ivpsh, ti);
      }
    }
  } else {
    if (!enable_tensor_release_) {
      // Case 2.A: graph output tensor
      output_tensorinfos.emplace_back(ti);
    } else {
      // Is this a duplicate tensor going to graph output?
      // See if this the buffer pointer matches any input, then -
      // Check whether it is an alias of any input
      if (buff_to_input_ivpsh_map.count(buffp)) {
        // Case 2.B: Graph output that is duplicate of input
        PT_BRIDGE_DEBUG(
            "Adding to duplicate_input_to_outtinfo_map ", ti.get_buffer());
        duplicate_input_to_outtinfo_map.emplace(ivpsh, ti);
      } else if (buff_to_intermediate_ivpsh_map.count(buffp)) {
        // Case 2.C: Graph output that is duplicate of a persistent
        // intermediate
        PT_BRIDGE_DEBUG(
            "Adding to duplicate_intermediate_to_outtinfo_map ",
            ti.get_buffer());
        duplicate_intermediate_to_outtinfo_map.emplace(ivpsh, ti);
      } else if (buff_to_output_ivpsh_map.count(buffp)) {
        // Case 2.D: Graph output that is duplicate of a previous output
        PT_BRIDGE_DEBUG(
            "Adding to duplicate_output_to_outtinfo_map ", ti.get_buffer());
        duplicate_output_to_outtinfo_map.emplace(ivpsh, ti);
      } else {
        // Case 2.A: graph output tensor, enable_tensor_release_
        PT_BRIDGE_DEBUG("Adding to output_tensorinfo_map ", ti.get_buffer());
        output_tensorinfo_map.emplace(ivpsh, ti);
        buff_to_output_ivpsh_map.emplace(buffp, ivpsh);
      }
    }
  }
}

void HabanaLaunchOpPT::ProcessSynapseOutputs(
    const HabanaOperatorPtr& habana_op,
    torch::jit::Node* node) {
  auto output_nodes = node->outputs();
  auto habana_kernel_meta_data = habana_op->GetKernelMetaData();
  LayoutFormat out_layout;

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

  // Note the input layout information for the node to pass on to output edge
  auto node_ins = node->inputs();
  LayoutFormat assigned_input_layout = LayoutFormat::NCHW;
  LayoutFormat origin_input_layout = LayoutFormat::NCHW;

  int node_idx = 0;
  for (auto value_in : node_ins) {
    if (value_to_ivalue[value_in] &&
        value_in->type()->kind() == c10::TypeKind::TensorType) {
      /* Get the input tensor layout information */
      assigned_input_layout =
          ((node_idx == 0) ||
           getTensorChannelOrder(value_in) == LayoutFormat::HWCK)
          ? getTensorChannelOrder(value_in)
          : assigned_input_layout;

      // Get the origin layout too, to pass it along..we see if any of the
      // inputs in NHWC origin then we mark the origin layout as NHWC We need
      // to make this more robust by having a tensor level memory of layout We
      // need to mark weight tensors by meta data so that we can recognize
      // them and not permute to NHWC at exit.
      origin_input_layout =
          value_to_tensor_layout[value_in].layout_at_graph_entry ==
              LayoutFormat::NHWC
          ? LayoutFormat::NHWC
          : origin_input_layout;
      node_idx++;
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
  size_t meta_size = habana_kernel_meta_data.output_layout.size();

  for (synapse_helpers::tensor& out_tensor_syn : habana_op->GetSynOutputs()) {
    out_layout = output_tensor_idx >= meta_size
        ? LayoutFormat::ANY
        : habana_kernel_meta_data.output_layout.at(output_tensor_idx);

    /* Pass down the layout information from input to output for layout
       agnostic output (only for single input ans single output op nodes) */
    value_to_tensor_layout[output_nodes[output_nodes_idx]].layout =
        out_layout == LayoutFormat::ANY ? assigned_input_layout : out_layout;
    value_to_tensor_layout[output_nodes[output_nodes_idx]]
        .layout_at_graph_entry = origin_input_layout;
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
    if (maybe_syn_shape_tensor.is_shape_tensor()) {
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

// For now, we permute tensors at graph leaves once
// THis function permutes a given tensor to desired layout and modifies
// input_tensor list to have the new tensor

at::Tensor HabanaLaunchOpPT::permuteTensor(
    torch::jit::Value* value_in,
    const at::Tensor& input,
    LayoutFormat permute_order) {
  auto& device = synapse_helpers::HPURegistrar::get_device();
  synDeviceId device_id = device.id();
  HabanaOperatorPtr permute_kernel = KernelRegistry().get(
      device_id, {"aten::permute", ""}, input.scalar_type());
  TORCH_CHECK(
      permute_kernel != nullptr,
      " \n Permute kernel isnt supported in graph mode ");

  TORCH_CHECK(
      value_to_ivalue[value_in]->isTensor(), "non tensor input for permute");

  habana_kernels.push_back(permute_kernel);
  // set input synapse tensors
  auto is_already_mapped =
      pt_to_synapse_tensors.find(value_to_ivalue[value_in]) !=
      std::end(pt_to_synapse_tensors);

  SharedSynTensorOrRefListPtr tensorList =
      std::make_shared<SynTensorOrRefList>();
  bool is_permin_interim_persistant = false;
  std::string permute_input_synname;
  if (is_already_mapped) {
    // value_in is not an input to the jit_ir_graph
    auto syn_tensor_input =
        pt_to_synapse_tensors.find(value_to_ivalue[value_in]);

    for (synapse_helpers::tensor& tensor : *(syn_tensor_input->second)) {
      synapse_helpers::tensor& syn_tensor =
          permute_kernel->SetSynapseInput(tensor);
      is_permin_interim_persistant = syn_tensor.is_persistent();
      permute_input_synname = std::string(syn_tensor.name());
      tensorList->emplace_back(tensor_or_ref(syn_tensor));
    }
    pt_to_synapse_tensors.erase(value_to_ivalue[value_in]);
    pt_to_synapse_tensors.emplace(value_to_ivalue[value_in], tensorList);
  } else {
    // value_in is an input to the jit_ir_graph
    auto pt_tensor = value_to_ivalue[value_in]->toTensor();

    auto& syn_tensor =
        permute_kernel->AllocateSynapseInput(*syn_graph_ptr, pt_tensor, true);
    tensorList->emplace_back(tensor_or_ref(syn_tensor));
    pt_to_synapse_tensors.emplace(value_to_ivalue[value_in], tensorList);

    PtTensorInfo ti(
        value_to_ivalue[value_in],
        syn_tensor.name(),
        value_in,
        watch_tensor_flag_,
        syn_tensor.tensor_type());
    if (enable_caching_) {
      input_tiv_map.emplace(value_to_ivalue[value_in], ti);
      buff_to_input_ivpsh_map.emplace(
          pt_tensor.data_ptr(), value_to_ivalue[value_in]);
    } else {
      input_tivs.emplace_back(ti);
    }
  }

  auto dims =
      getDimsForLayout(permute_order, value_to_tensor_layout[value_in].layout);

  torch::jit::Stack input_stack = {IValue(input), IValue(dims)};

  // setup the config params for the kernels
  bool is_perminput_persistent =
      isInGraphOutputs(value_in) || is_permin_interim_persistant;
  permute_kernel->AllocateAndAddSynapseNode(
      *syn_graph_ptr, input_stack, is_perminput_persistent);

  auto& ivpsh_in = value_to_ivalue[value_in];
  auto outputs_permute = permute_kernel->GetOutputs();

  // set output synapse tensor
  synapse_helpers::tensor& out_tensor_syn =
      permute_kernel->GetSynOutputs().at(0);
  {
    // make the output of permute the input for next synapse kernel
    // permute has a single output
    SharedSynTensorOrRefListPtr tensorList =
        std::make_shared<SynTensorOrRefList>();
    tensorList->emplace_back(tensor_or_ref(out_tensor_syn));
    if (is_perminput_persistent) {
      // The value_in can represent either of the following
      // A: persistent intermediate
      // B: a subgraph output
      // C: duplicate of an input
      // After adding the permute node, the pt_tensor corresponding to the
      // input of the permute needs to be added to aten_intermediates
      // if A or B is true.
      std::string perminput_type;
      if (!isInGraphOutputs(value_in)) {
        // Case A: The corresponding tinfo needs to be retained within
        // intermediate_tinfos.
        perminput_type = "persistent intermediate";
      } else {
        if (enable_tensor_release_) {
          if (output_tensorinfo_map.count(ivpsh_in)) {
            // Case B: ivpsh_in should be present in output_tensorinfo_map.
            // This used to be a graph output which has become an interim.
            // The corresponding tinfo needs to be moved from
            // output_tensorinfo_map to aten_intermediates.
            // Remove the corresponding tinfo from output_tensorinfo_map,
            output_tensorinfo_map.erase(ivpsh_in);
            perminput_type = "previous graph output";
          } else {
            perminput_type = "possible input duplicate";
          }
        } else {
          // Caching without tensor release will be depricated. Adding the
          // following code for keeping the flow consistent.
          void* buffp = ivpsh_in->toTensor().data_ptr();
          auto tinfo_it = std::find_if(
              output_tensorinfos.begin(),
              output_tensorinfos.end(),
              [&buffp](const PtTensorInfo& arg) {
                return (arg.get_buffer() == buffp);
              });
          if (tinfo_it != output_tensorinfos.end()) {
            output_tensorinfos.erase(tinfo_it);
            perminput_type = "previous graph output";
          } else {
            perminput_type = "possible input duplicate";
          }
        }
      }

      // Add the tinfo and pt_tensor for permute input as persistent
      // intermediate.
      AddAtenIntermediate(ivpsh_in, permute_input_synname, value_in);
      PT_BRIDGE_DEBUG(
          "Permute input is ",
          perminput_type,
          ". After adding tinfo ",
          intermediate_tinfos.back(),
          " to intermediate_tinfos #intermediates ",
          intermediate_tinfos.size());
    }

    value_to_ivalue[value_in] = std::make_shared<IVal>(outputs_permute[0]);
    value_to_tensor_layout[value_in].layout = permute_order;
    pt_to_synapse_tensors.erase(value_to_ivalue[value_in]);
    pt_to_synapse_tensors.emplace(value_to_ivalue[value_in], tensorList);

    if (is_perminput_persistent) {
      auto ti = PtTensorInfo(
          value_to_ivalue[value_in],
          out_tensor_syn.name(),
          value_in,
          watch_tensor_flag_,
          out_tensor_syn.tensor_type());
      if (!enable_tensor_release_) {
        output_tensorinfos.emplace_back(ti);
      } else {
        output_tensorinfo_map.emplace(value_to_ivalue[value_in], ti);
      }
    }
  }
  return outputs_permute[0];
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
        meta_syn_tensors.back().tensor_type());
    if (!isInGraphOutputs(value_in)) {
      duplicate_input_tivs.emplace_back(ti);
    } else {
      if (!enable_tensor_release_) {
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

void HabanaLaunchOpPT::processInputs(
    torch::jit::Node* node,
    const HabanaOperatorPtr& habana_kernel) {
  // Get the metadata for all inputs, used for preprocessing inputs
  auto& habana_kernel_meta_data = habana_kernel->GetKernelMetaData();
  // Check if its ok to change the input tensor in the graph attached to value
  auto node_ins = node->inputs();

  size_t tensor_idx = 0;
  LayoutFormat in_layout, prev_layout = LayoutFormat::ANY;
  size_t meta_size = habana_kernel_meta_data.input_layout.size();
  for (const auto value_in : node_ins) {
    if (value_to_ivalue[value_in] &&
        value_in->type()->kind() == c10::TypeKind::TensorType) {
      in_layout = tensor_idx >= meta_size
          ? LayoutFormat::ANY
          : habana_kernel_meta_data.input_layout.at(tensor_idx);

      if (GET_ENV_FLAG(PT_HPU_LAZY_MODE) != 0) {
        // For weight tensors we update the map before execution starts
        // through a pass If its marked HWCK in the map, we can override with
        // it
        auto tensor_layout = getTensorChannelOrder(value_in);
        if (tensor_layout == LayoutFormat::HWCK) {
          TORCH_CHECK(
              in_layout == LayoutFormat::HWCK || in_layout == LayoutFormat::ANY,
              "HabanaOp, got contradicting layout info from meta data and opt pass");
          in_layout = LayoutFormat::HWCK;
        }
      }
      if (in_layout == LayoutFormat::ANY && tensor_idx > 0) {
        // ATTENTION : We will support only homogeneous layouts for kernels
        // which dont pass meta data requirements for inputs
        // We make inputs homogeneous layouts in case kernel doesnt specify
        // any layout
        // TODO : Add a debug log heres
        in_layout = prev_layout;
      }

      auto tensor = value_to_ivalue[value_in]->toTensor();

      if (in_layout == LayoutFormat::HWCK) {
        in_layout = LayoutFormat::ANY;
        value_to_tensor_layout[value_in].layout = LayoutFormat::HWCK;
      }

      bool permute_required = !(isChannelOrderSupported(value_in, in_layout));
      LayoutFormat perm_layout = in_layout;
      // If the kernel changes dims of tensor, get it to original PT format
      // This is done as we cannot pass layout info for 4D tensors and it will
      // get lost in translation.
      if (habana_kernel_meta_data.changes_dims &&
          GET_ENV_FLAG(PT_HPU_LAZY_MODE) != 0) {
        if (getTensorChannelOrder(value_in) !=
                value_to_tensor_layout[value_in].layout_at_graph_entry &&
            getTensorChannelOrder(value_in) != LayoutFormat::HWCK) {
          permute_required = true;
          perm_layout = value_to_tensor_layout[value_in].layout_at_graph_entry;
        }
      }

      if (permute_required) {
        // We only support 4D tensors
        TORCH_CHECK(
            tensor.dim() <= 4,
            "WARNING: Kernel wants permute on non 4D tensor, not supproted");
        // permute
        if (tensor.dim() == 4) {
          permuteTensor(value_in, tensor, perm_layout);
        }
      }
      prev_layout =
          tensor_idx == 0 ? getTensorChannelOrder(value_in) : prev_layout;
      tensor_idx++;
    }
  }
  // TODO : add checks for doing flattening/slicing anything that is
  // required.
}

void HabanaLaunchOpPT::postProcessOutputs() {
  // Do we need a optimization pass here? What should we look for?
  for (auto node : jit_ir_graph->nodes()) {
    auto node_outs = node->outputs();
    for (const auto value_out : node_outs) {
      IValPtrShared ival = value_to_ivalue[value_out];
      if (!ival)
        continue;
      if (!(ival->isTensor()))
        continue;

      if (ival && value_out->type()->kind() == c10::TypeKind::TensorType &&
          isInGraphOutputs(value_out)) {
        auto tensor = ival->toTensor();
        // Add permutes only for 4D non weight tensors
        if (tensor.dim() == 4) {
          auto pre_layout =
              value_to_tensor_layout[value_out].layout_at_graph_entry;
          if (getTensorChannelOrder(value_out) == LayoutFormat::HWCK) {
            // Do Nothing
          } else if (getTensorChannelOrder(value_out) != pre_layout) {
            permuteTensor(value_out, tensor, pre_layout);
            if (pre_layout == LayoutFormat::NHWC) {
              // Make the shape according to NCHW again as PT maintains that
              // even for NHWC tensors Whereas we process internally as NHWC
              // shape only
              adjustSizesforPT(&tensor, true);
              auto ivpsh =
                  (value_to_ivalue.count(value_out) ? value_to_ivalue[value_out]
                                                    : nullptr);
              value_to_ivalue.erase(value_out);
              auto ivptrsh_updated = std::make_shared<IVal>(tensor);
              if (enable_tensor_release_ && ivpsh &&
                  output_tensorinfo_map.count(ivpsh)) {
                auto a = output_tensorinfo_map.find(ivpsh);
                auto ti = PtTensorInfo(
                    ivptrsh_updated,
                    a->second.get_syn_name(),
                    value_out,
                    watch_tensor_flag_,
                    a->second.tensor_type());
                output_tensorinfo_map.erase(ivpsh);
                output_tensorinfo_map.emplace(ivptrsh_updated, ti);
              }
              value_to_ivalue[value_out] = ivptrsh_updated;
            }
          } else {
            if (getTensorChannelOrder(value_out) == LayoutFormat::NHWC) {
              // Make the shape according to NCHW again as PT maintains that
              // even for NHWC tensors Whereas we process internally as NHWC
              // shape only
              adjustSizesforPT(&tensor, true);
              auto ivpsh =
                  (value_to_ivalue.count(value_out) ? value_to_ivalue[value_out]
                                                    : nullptr);
              value_to_ivalue.erase(value_out);
              auto ivptrsh_updated = std::make_shared<IVal>(tensor);
              if (enable_tensor_release_ && ivpsh &&
                  output_tensorinfo_map.count(ivpsh)) {
                auto a = output_tensorinfo_map.find(ivpsh);
                auto ti = PtTensorInfo(
                    ivptrsh_updated,
                    a->second.get_syn_name(),
                    value_out,
                    watch_tensor_flag_,
                    a->second.tensor_type());
                output_tensorinfo_map.erase(ivpsh);
                output_tensorinfo_map.emplace(ivptrsh_updated, ti);
              }
              value_to_ivalue[value_out] = ivptrsh_updated;
            }
          }
        }
      }
      // TODO : add checks for doing flattening/slicing anything that is
      // required.
    }
  }
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

void HabanaLaunchOpPT::handleRestrideNode(torch::jit::Node* node) {
  auto value_in = node->input(0);
  auto value_out = node->output(0);
  HABANA_ASSERT(value_to_ivalue.find(value_in) != std::end(value_to_ivalue));
  HABANA_ASSERT(value_to_ivalue[value_in]->isTensor());
  auto tensor = value_to_ivalue[value_in]->toTensor();
  auto is_5d_layout = tensor.dim() == 5 ? true : false;

  // for 0D and 1D tensors adjust sizes skipped
  if (tensor.dim() > 1) {
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

    if (isInGraphOutputs(value_out)) {
      auto format = is_5d_layout ? c10::MemoryFormat::ChannelsLast3d
                                 : c10::MemoryFormat::ChannelsLast;
      tensor.unsafeGetTensorImpl()->empty_tensor_restride(format);
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
        ivpsh_restrided, syn_tensor.name(), value_in, watch_tensor_flag_);

    value_to_ivalue.erase(value_in);

    if (enable_tensor_release_) {
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
      } else if (buff_to_intermediate_ivpsh_map.count(buffp)) {
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
      } else if (buff_to_input_ivpsh_map.count(buffp)) {
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
      } else if (buff_to_output_ivpsh_map.count(buffp)) {
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
            " not found in duplicate_intermediate_to_outtinfo_map");
      }
    }

    value_to_ivalue[value_in] = ivpsh_restrided;
    value_to_ivalue[value_out] = ivpsh_restrided;
  } else {
    PT_BRIDGE_DEBUG(
        "restride node output %", value_out->debugName(), " is non persistent");
    value_to_ivalue[value_out] = ivpsh_restrided;
    value_to_tensor_layout[value_out].layout = LayoutFormat::NHWC;
    value_to_tensor_layout[value_out].layout_at_graph_entry =
        LayoutFormat::NHWC;
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
        // Marking NCHW for now, for non 4D tensors layour doesnt matter
        // Marking default...can update it after ""first use" to correct
        // format
        value_to_tensor_layout[value].layout = LayoutFormat::NCHW;
        value_to_tensor_layout[value].layout_at_graph_entry =
            LayoutFormat::NCHW;
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
            meta_syn_tensors.back().tensor_type()));
        aten_intermediates.push_back(tensor);
      } else {
        value_to_ivalue[value] = ivptrsh;
      }
    }
  } else if (node->kind() == torch::jit::prim::ListConstruct) {
    auto node_ins = node->inputs();
    std::vector<at::Tensor> tensorVec;
    for (const auto value_in : node_ins) {
      auto ivptrsh = value_to_ivalue[value_in];
      if (ivptrsh->isTensor()) {
        tensorVec.push_back(ivptrsh->toTensor());
      }
    }

    // construct tensorList from tensorVec
    at::TensorList tensorList(tensorVec);

    // convert tensorList to Ivalue and update the stack
    IValPtrShared ivptrsh_tensor_list = std::make_shared<IVal>(tensorList);
    auto node_vals = node->outputs();
    HABANA_ASSERT(node_vals.size() == 1);
    value_to_ivalue[node_vals[0]] = ivptrsh_tensor_list;
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
    if (value_to_ivalue.count(input))
      stack_in.insert(stack_in.end(), *value_to_ivalue[input]);
    else
      stack_in.insert(stack_in.end(), IValue());
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
      /*out_layout = value_to_tensor_layout[value_in].layout;
      out_origin_layout =
          value_to_tensor_layout[value_in].layout_at_graph_entry;
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
  jit_op.getOperation()(&stack);

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
      value_to_tensor_layout[val_out].layout = out_layout;
      value_to_tensor_layout[val_out].layout_at_graph_entry =
    out_origin_layout; create_duplicate_syn_tensor(&tensor, val_out, true);
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
  size_t output_nontensor_cnt{0};
  for (auto output : jit_ir_graph->outputs()) {
    auto oit = value_to_ivalue.find(output);
    TORCH_CHECK(
        oit != value_to_ivalue.end(),
        "value_to_ivalue does not have an entry for %",
        output->debugName());

    IValPtrShared ivpsh = oit->second;
    TORCH_CHECK(nullptr != ivpsh, "IValPtrShared for subgraph output is null");

    // Checking where we can find the outputs
    if (!ivpsh->isTensor()) {
      PtTensorInfo ti(ivpsh);
      ti.set_output_index(output_idx);
      std::string irn = std::string("%") + output->debugName();
      ti.set_ir_name(irn);
      output_tensorinfos.emplace_back(ti);
      output_nontensor_cnt++;
    } else {
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
        PT_BRIDGE_FATAL(
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

  TORCH_CHECK(
      intermediate_tinfos.size() == rv.aten_intermediates.size(),
      "#intermediate tensor ",
      rv.aten_intermediates.size(),
      " does not match with #interim_tinfo ",
      intermediate_tinfos.size());

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
  if (GET_ENV_FLAG(PT_HPU_LAZY_MODE) != 0) {
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

    if (habana_lazy::exec::OptPassCfg::GetInstance()->IsEnabledPermutePass()) {
      if (strcmp(node->kind().toQualString(), "hpu::restride_cl") == 0) {
        handleRestrideNode(node);
        continue;
      }
    }

    // Get kernel context
    const auto& op = node->schema().operator_name();
    HabanaOperatorPtr HabanaKernel =
        KernelRegistry().get(device_id, op, getNodeScalarType(node));

    TORCH_CHECK(HabanaKernel, op, " isn't registered in KernelRegistry!");

    PT_BRIDGE_DEBUG("Going to add ", *node);
    // See if we need to modify/permute tesnors
    if (!habana_lazy::exec::OptPassCfg::GetInstance()->IsEnabledPermutePass())
      processInputs(node, HabanaKernel);

    // clear the accumulated synapse node indices corresponding to permute.
    // Otherwise this results in spurious control edges
    syn_graph_ptr->clear_node_indices();

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
        void* buffp = p.second.data_ptr();

        // Check whether it is an alias of any input
        auto it = buff_to_input_ivpsh_map.find(buffp);
        if (it != buff_to_input_ivpsh_map.end()) {
          std::string irn{"%appended_indup_"};
          irn += std::to_string(appended_index);
          appended_index++;

          PtTensorInfo ti(p.second, p.first, irn, watch_tensor_flag_);
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

          IValPtrShared ivpsh = std::make_shared<IVal>(p.second);
          AddAtenIntermediate(ivpsh, p.first, irn);
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

  if (!habana_lazy::exec::OptPassCfg::GetInstance()->IsEnabledPermutePass())
    postProcessOutputs();

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
  if (!enable_tensor_release_) {
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

    rv.aten_intermediates = std::move(aten_intermediates);
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
    rv.aten_intermediates = std::move(aten_intermediates);

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
    // Initiate recipe run time collection
    if (current_dbipsh_->NeedRunTimeSlot(current_bucket_id_)) {
      auto& syn_device = synapse_helpers::HPURegistrar::get_device();
      rv.time_slot_ = std::make_shared<synapse_helpers::TimeSlot>(
          syn_device.get_cached_time_event_handle(),
          syn_device.get_cached_time_event_handle(),
          static_cast<synStreamHandle>(syn_device.get_compute_stream()));
      current_dbipsh_->RegisterTimeSlot(rv.time_slot_, current_bucket_id_);
    }
    // Add the jit_ir_graph to current_dbipsh_
    current_dbipsh_->SetJitIRGraphPtr(jit_ir_graph);
    current_dbipsh_->UpdateCompileTime(t_compile_ns, current_bucket_id_);
  }

  PT_BRIDGE_DEBUG(
      "HabanaOp recipe cache :: launching new recipe", rv.header_str());

  rv.launch(input_refs);
  rv.update_hit_count();

  if (enable_tensor_dump_) {
    DumpTensors(rv);
  }

  if (enable_caching_) {
    // Add the <key,value> pair to the map
    // If we enable_tensor_release_, then the we don't match to a specific
    // graph instance, hence id_str matching is not required.
    if (false == refine_ds_enabled_) {
      std::shared_ptr<RecipeArgumentSpec> rargpsh =
          std::make_shared<RecipeArgumentSpec>(
              false,
              input_refs,
              jit_ir_graph,
              enable_tensor_release_ ? "" : id_str);
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
  PT_BRIDGE_DEBUG(rv.digest_str());

  UpdateOutputs(rv);
}

void HabanaLaunchOpPT::CreateDynamicBucketInputShapes(
    habana_helpers::DynamicBucketInfo::InpTensorShapes& shape_map) {
  for (size_t i = 0; i < input_refs.size(); i++) {
    auto input = input_refs[i];
    if (input.isTensor()) {
      input_tensor_indices.push_back(i);
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
    value_to_tensor_layout[value_input].layout = LayoutFormat::NCHW;
    value_to_tensor_layout[value_input].layout_at_graph_entry =
        LayoutFormat::NCHW;

    if (pt_stack_sh[j]->isTensor()) {
      // Taking alias as that allows us to detach it from PT and do metadata
      // changes It gives us more control over tensor changes, but caution
      // is needed. Its might be a bit dangerous, but only way to
      // communicate layour changes PT doesnt allow any stride changes we
      // want, we can review it with PT folks

      auto tensor = at::alias(pt_stack_sh[j]->toTensor());

      // WE dont support 0D tensors internally, so convert to 1D internally
      if (tensor.dim() == 0) {
        tensor.unsafeGetTensorImpl()->set_sizes_contiguous({1});
      }

      // Get  the logical layout from PT tensor
      // We dont touch this, even while doing permutes, the PT logical
      // tensor is retained For us all tensors are contiguous PT doesnt let
      // us mark logical layout directly so we dont change them
      value_to_tensor_layout[value_input].layout = getPTTensorLayout(tensor);
      value_to_tensor_layout[value_input].layout_at_graph_entry =
          getPTTensorLayout(tensor);

      if (getPTTensorLayout(tensor) == LayoutFormat::NHWC) {
        // Make the sizes according to NCHW as PT maintains
        // NCHW shapes even for NHWC tensors(It doesnt change shape)
        if (!habana_lazy::exec::OptPassCfg::GetInstance()
                 ->IsEnabledPermutePass())
          adjustSizesforPT(&tensor, false);
        IValPtrShared ivptrsh = std::make_shared<IVal>(tensor);
        value_to_ivalue[value_input] = ivptrsh;
        pt_stack_sh[j] = ivptrsh;
      } else {
        IValPtrShared ivptrsh = std::make_shared<IVal>(tensor);
        value_to_ivalue[value_input] = ivptrsh;
        pt_stack_sh[j] = ivptrsh;
      }
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
      auto tensor = stack[i].toTensor();

      //
      // TODO: When creating a new stack, we need to look, if this
      // can be done using storage less pytorch tensor, need to fix
      // this
      auto new_tensor = at::empty(
          dynamic_shapes.at(i).get_dims(),
          tensor.options(),
          tensor.suggest_memory_format());
      new_stack.push_back(torch::jit::IValue(new_tensor));
    } else {
      new_stack.push_back(stack[i]);
    }
  }
  PT_BRIDGE_END;
  return new_stack;
}

void HabanaLaunchOpPT::ProcessHabanaFusedOpWithDS() {
  PT_BRIDGE_BEGIN;
  auto& device = synapse_helpers::HPURegistrar::get_device();

  std::shared_ptr<RecipeArgumentSpec> rargpsh =
      std::make_shared<RecipeArgumentSpec>(jit_ir_graph, input_refs);
  PT_DYNAMIC_SHAPE_DEBUG(
      "====\n",
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

  CreateDynamicBucketInputShapes(act_input_tshapes);

  current_dbipsh_->CollectDynamicDims(act_input_tshapes);
  current_bucket_id_ = current_dbipsh_->GetBucketId(act_input_tshapes);
  cur_ds_token_ = current_dbipsh_->GetTokenForBucketId(current_bucket_id_);

  PT_DYNAMIC_SHAPE_DEBUG(
      jit_ir_graph->toString(),
      current_dbipsh_->digest_str(),
      "current bucket id : ",
      current_bucket_id_);

  auto ranges = current_dbipsh_->CalculateShapes(current_bucket_id_);
  if (ranges.empty()) {
    PT_DYNAMIC_SHAPE_DEBUG(
        "exact graph with token : ", cur_ds_token_, "\n----");
  } else {
    PT_DYNAMIC_SHAPE_DEBUG(
        "dynamic graph with token : ",
        cur_ds_token_,
        '\n',
        "current range ::",
        ranges,
        "----");

    min_input_tshapes.insert(
        ranges.min_shapes.begin(), ranges.min_shapes.end());
    max_input_tshapes.insert(
        ranges.max_shapes.begin(), ranges.max_shapes.end());
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
      if (current_dbipsh_->NeedRunTimeSlot(current_bucket_id_)) {
        auto& syn_device = synapse_helpers::HPURegistrar::get_device();
        rv.time_slot_ = std::make_shared<synapse_helpers::TimeSlot>(
            syn_device.get_cached_time_event_handle(),
            syn_device.get_cached_time_event_handle(),
            static_cast<synStreamHandle>(syn_device.get_compute_stream()));
        current_dbipsh_->RegisterTimeSlot(rv.time_slot_, current_bucket_id_);
      }

      if (rv.dynamic_graph) {
        // For Dynamic shapes in case of cache hit, we need to run
        // shape inference for determining the output shape and
        // persistent intermediates
        PT_BRIDGE_DEBUG("run output shape inference pass");
        run_shape_inference(ShapeInfo::InferencePass::OUTPUT_SHAPE);
      }


      std::shared_ptr<std::vector<IValPtrShared>> dma_inputs =
          std::make_shared<std::vector<IValPtrShared>>(
              std::vector<IValPtrShared>());

      rv.update_patching_table(
          input_refs, dma_inputs, m_map_shape.m_actual_shapes);

      if (enable_tensor_dump_) {
        DumpTensors_pre(rv);
      }
      rv.launch(input_refs, dma_inputs);

      if (enable_tensor_dump_) {
        DumpTensors(rv);
      }

      // Update the stack from the recipe itself
      UpdateOutputs(rv);
      ReturnCachedRecipe(rv);
      PT_DYNAMIC_SHAPE_DEBUG("HabanaOp recipe cache hit ::", rv.header_str());
      PT_DYNAMIC_SHAPE_DEBUG(rv.digest_str());

      clear();
      PT_BRIDGE_END;
      return;
    } else {
      PT_DYNAMIC_SHAPE_DEBUG(
          "HabanaOp recipe cache miss :: key ", spec_key->hashCode());
    }
  }

  // In case of dynamic mode (cache miss & dynamic range exists), we need to
  // run shape inference for min and max passes
  // TODO: Once the bucket range issue is fixed, we need to run
  // the shape inference only for once for max shapes
  if (ranges.empty() == false) {
    // run min shape inference pass
    PT_BRIDGE_DEBUG("run min shape inference pass");
    run_shape_inference(ShapeInfo::InferencePass::MIN_SHAPE);
    // run max shape inference pass
    PT_BRIDGE_DEBUG("run max shape inference pass");
    run_shape_inference(ShapeInfo::InferencePass::MAX_SHAPE);
  }

  std::stringstream ss;
  ss << op_name << "_" << instance_count_;
  auto syn_graph = habana_helpers::create_graph(device.id(), ss.str());
  syn_graph.set_dynamic_graph(!ranges.empty());
  AdjustInputLayout();
  PT_BRIDGE_DEBUG("run CompileAndExecuteHabanaFusedOpKernel");
  CompileAndExecuteHabanaFusedOpKernel(syn_graph);
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

  if (enable_tensor_release_) {
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
  value_to_tensor_layout.clear();
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

  if (refine_ds_enabled_) {
    ProcessHabanaFusedOpWithDS();
    return;
  }

  // caching :: begin
  if (enable_caching_) {
    // If we enable_tensor_release_, then the we don't match to a specific
    // graph instance, hence id_str matching is not required.

    std::shared_ptr<RecipeArgumentSpec> spec_key =
        std::make_shared<RecipeArgumentSpec>(
            false,
            input_refs,
            jit_ir_graph,
            enable_tensor_release_ ? "" : id_str);

    std::shared_ptr<RecipeValueSpec> rvpsh = GetCachedRecipe(spec_key);

    if (ABSL_PREDICT_TRUE(rvpsh)) {
      RecipeValueSpec& rv = *rvpsh;
      rv.update_hit_count();

      PT_BRIDGE_DEBUG(
          "HabanaOp recipe cache hit ::",
          rv.header_str(),
          "\n",
          rv.digest_str());

      std::shared_ptr<std::vector<IValPtrShared>> dma_inputs =
          std::make_shared<std::vector<IValPtrShared>>(
              std::vector<IValPtrShared>());

      rv.update_patching_table(
          input_refs, dma_inputs, m_map_shape.m_actual_shapes);

      if (enable_tensor_dump_) {
        DumpTensors_pre(rv);
      }
      rv.launch(input_refs, dma_inputs);

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
  std::stringstream ss;
  ss << op_name << "_" << instance_count_;
  auto syn_graph = habana_helpers::create_graph(device.id(), ss.str());
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
  std::stringstream ss;
  ss << op_name << "_" << instance_count_;
  auto syn_graph = habana_helpers::create_graph(device.id(), ss.str(), true);
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
    const ShapeInfo::InferencePass& pass) {
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
      new_stack = CreateStack(*pt_stack, min_input_tshapes);
    } else {
      new_stack = CreateStack(*pt_stack, max_input_tshapes);
    }
    pt_stack = &new_stack;

    size_t j = new_stack.size() - num_inputs;
    for (; j < new_stack.size(); j++) {
      IValPtrShared ivpsh = std::make_shared<IVal>(new_stack[j]);
      pt_stack_sh.push_back(ivpsh);
    }
  }
  m_map_shape.m_pass = pass;
  run_pass();

  if (old_stack) {
    pt_stack_sh.clear();
    pt_stack = old_stack;
    pt_stack_sh = old_pt_stack_sh;
  }
  PT_BRIDGE_END;
}
} // namespace habana
