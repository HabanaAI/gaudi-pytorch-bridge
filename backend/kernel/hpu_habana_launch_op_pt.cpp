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

#include "backend/kernel/hpu_habana_launch_op_pt.h"

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

#include "backend/backend_meta.h"
#include "backend/habana_device/HPUAllocator.h"
#include "backend/habana_device/tensor_builder.h"
#include "habana_helpers/logging.h"

#include "backend/kernel/ds_graph_recompile.h"
#include "backend/kernel/hpu_shape_inference.h"
#include "backend/kernel/refinement_engine.h"
#include "backend/passes/hpu_habana_persistence_marker_pass.h"

#include "backend/helpers/create_tensor.h"
#include "backend/helpers/event_dispatcher.h"
#include "backend/helpers/graph.h"
#include "backend/helpers/tensor_utils.h"
#include "habana_helpers/logging_pt.h"
#include "habana_helpers/misc_utils.h"

#include "habana_kernels/hccl_kernels.h"
#include "habana_kernels/kernel_utils.h"
#include "habana_kernels/lazy_kernels_declarations.h"
#include "habana_kernels/random_gen_kernels.h"
#include "habana_kernels/unary_kernels.h"

#include "habana_lazy/aten_lazy_bridge.h"
#include "habana_lazy/hlexec.h"
#include "habana_lazy/hpu_lazy_tensors.h"

#include "hpu_ops/hpu_op_helper.h"

#include "backend/jitgraph_utils.h"
#include "backend/synapse_helpers/env_flags.h"
#include "backend/synapse_helpers/tcmalloc_helper.h"
#include "pytorch_helpers/habana_helpers/dtype_helpers.h"

using namespace torch::jit;
using namespace jitgraph_utils;
namespace habana {

std::future<void> Singleton_CompileThreadPool::m_compile_thread_handle;
std::future<void> Singleton_ExecThreadPool::m_exec_thread_handle;

// static initializations
const std::unordered_set<std::string> HabanaMetaOpList::meta_ops = {
    // Add aten string here for ops to support
    // e.g  :: "aten::view"
    "aten::size",
    "prim::dtype"};

std::unordered_set<std::string> HabanaLaunchOpPT::watchlist_ = {};
std::unordered_set<std::string> HabanaLaunchOpPT::disabled_jit_ir_ops_ = {};
std::unordered_map<size_t, habana_helpers::InpTensorShapes>
    HabanaLaunchOpPT::ref_input_shape_map = {};
//--------------------------------------

void HabanaLaunchOpPT::cleanUp() {
  ref_input_shape_map = {};
  DynamicBucketInfoMap::get_instance().clear();
  RecipeCacheLRU::get_cache().clear();
}

bool dropCachedRecipe_LRU(size_t& recipe_count) {
  bool dropped{false};
  dropped = RecipeCacheLRU::get_cache().drop_lru(recipe_count);
  return dropped;
}

void emitCacheEvent(
    habana_helpers::EventDispatcher::EventDispatcher::Topic topic,
    std::string cache_name) {
  if (GET_ENV_FLAG_NEW(PT_HPU_ENABLE_CACHE_METRICS, true)) {
    habana_helpers::EmitEvent(
        topic,
        habana_helpers::EventDispatcher::EventParams(
            {{"recipe_id", cache_name}}));
  }
}

std::string makeIdStr(const std::string& name, size_t graph_index) {
  std::ostringstream oss;
  oss << name << '_' << graph_index;
  return oss.str();
}

std::string& HabanaLaunchOpPT::SetAndGetSynapseGraphName(
    const std::string& name,
    size_t g_index) {
  if (id_str == std::string()) {
    if (IS_BRIDGE_DEBUG_ENABLED ||
        GET_ENV_FLAG_NEW(PT_COMPILATION_STATS_PATH)) {
      id_str = makeIdStr(name, g_index);
    } else {
      id_str = name;
    }
  }
  return id_str;
}

void HabanaLaunchOpPT::SetSynapseGraphName(
    const std::string& name,
    size_t g_index) {
  if (id_str == std::string()) {
    id_str = makeIdStr(name, g_index);
  }
}

void HabanaLaunchOpPT::SetOpName(const std::string& name) {
  op_name = name;
}

HabanaLaunchOpPT::HabanaLaunchOpPT(
    std::shared_ptr<habana::OptimizedJITGraphAndMetaData>
        optimized_jit_graph_and_meta_data)
    : name(optimized_jit_graph_and_meta_data->GetOpName()),
      graph_index(optimized_jit_graph_and_meta_data->GetGraphIndex()),
      jit_ir_graph(optimized_jit_graph_and_meta_data->get_cached_graph()),
      debug(optimized_jit_graph_and_meta_data->GetDbgFlag()) {
  refine_ds_enabled_ = habana_helpers::GetRefineDynamicShapeStatus();
  enable_fast_shape_inf_ =
      GET_ENV_FLAG_NEW(PT_HPU_ENABLE_FAST_SHAPE_INFERENCE) &&
      refine_ds_enabled_;
  op_strs = optimized_jit_graph_and_meta_data->get_cached_opstrs();
  graph_key = optimized_jit_graph_and_meta_data->get_cached_graph_key();
  hpu_stream = optimized_jit_graph_and_meta_data->GetHPUStream();
  bool is_optimized_lazy_eager =
      optimized_jit_graph_and_meta_data->GetOptimizedLazyEagerFlag();
  jit_graph_and_meta_data = optimized_jit_graph_and_meta_data;

  SetOpName(name);

  PT_BRIDGE_DEBUG("Creating : ", SetAndGetSynapseGraphName(name, graph_index));

  tensor_dump_numel_ = -2;

  char* snumel = nullptr;
  if (!is_optimized_lazy_eager) {
    snumel = getenv("HABANA_PGM_DUMP_TENSOR_NUMEL");
  }
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

  execution_mode_ = jit_graph_and_meta_data->GetFrontendType();

  // To support old lazy eager mode
  // This must be removed once lazy eager mode is deprecated
  if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) == 2) {
    execution_mode_ = habana_helpers::HabanaFrontendTypes::EAGER;
  }

  // used for controlling recipe caching in non-eager backends
  enable_graph_caching_ =
      (execution_mode_ != habana_helpers::HabanaFrontendTypes::EAGER) &&
      GET_ENV_FLAG_NEW(PT_HPU_PGM_ENABLE_CACHE);

  // used for controlling recipe caching in eager backends
  // combined with PT_HPU_PGM_ENABLE_CACHE to allow debugging
  enable_eager_caching_ =
      (execution_mode_ == habana_helpers::HabanaFrontendTypes::EAGER) &&
          (!jit_graph_and_meta_data->get_is_eager_compiler_supported() &&
           GET_ENV_FLAG_NEW(PT_HPU_PGM_ENABLE_CACHE)) ||
      GET_ENV_FLAG_NEW(PT_HPU_ENABLE_EAGER_CACHE);

  enable_caching_ = enable_graph_caching_ || enable_eager_caching_;

  enable_shape_agnostic_caching_ =
      (execution_mode_ == habana_helpers::HabanaFrontendTypes::EAGER) &&
      GET_ENV_FLAG_NEW(PT_HPU_LAZY_EAGER_SHAPE_AGNOSTIC_GRAPH) &&
      !enable_caching_;

  HABANA_ASSERT(
      !(enable_caching_ && enable_shape_agnostic_caching_),
      "Both recipe and shape agnostic cache can not be enabled together!",
      " enable_caching_ : ",
      enable_caching_,
      ", enable_shape_agnostic_caching_ : ",
      enable_shape_agnostic_caching_);

  if (enable_shape_agnostic_caching_) {
    out_shapes = optimized_jit_graph_and_meta_data->get_output_shapes();
    HABANA_ASSERT(
        out_shapes.size() == jit_ir_graph->outputs().size(),
        "number of output shapes for patching ",
        out_shapes.size(),
        " is not equal to #outputs in jit graph ",
        jit_ir_graph->outputs().size());
  }

  use_persistent_tensors = GET_ENV_FLAG_NEW(HABANA_USE_PERSISTENT_TENSOR);

  if (enable_tensor_dump_) {
    std::string& idstrs = SetAndGetSynapseGraphName(name, graph_index);
    struct stat st = {};
    std::string dir_name{"./tensor_dumps"};
    mode_t dir_mode{0755};

    if (stat(dir_name.c_str(), &st) == -1) {
      auto ret = mkdir(dir_name.c_str(), dir_mode);
      TORCH_CHECK(0 == ret, std::string("failed to create " + dir_name));
    }

    dir_name += std::string("/") + idstrs;

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
      tensor_file << "---- id_str : " << idstrs << '\n'
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
      tensor_file << "---- id_str : " << idstrs << '\n'
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
  PT_BRIDGE_DEBUG("Destroying : ", GetSynapseGraphName());
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
  } else {
    is_persistent = persistence_marker_pass_data_ptr_.get()
        ? persistence_marker_pass_data_ptr_->IsPersistentNode(value_out)
        : false;
    if (is_persistent) {
      PT_BRIDGE_DEBUG(
          "Persistent tensor for ",
          node->kind().toQualString(),
          " for value %",
          value_out->debugName(),
          " created for an in-place op");
    }
  }

  return is_persistent;
}

bool HabanaLaunchOpPT::IsValueExternal(torch::jit::Value* value) {
  return persistence_marker_pass_data_ptr_.get()
      ? persistence_marker_pass_data_ptr_->IsExternalNode(value)
      : false;
}

OutputMetaDataVector HabanaLaunchOpPT::nodeOutputMetaData(
    torch::jit::Node* node) {
  auto node_outs = node->outputs();
  OutputMetaDataVector output_metadata{};
  // If node output is tensor list
  // tensorList and Unpack pair is supported
  if (*node->output(0)->type() == *torch::ListType::ofTensors() &&
      node->outputs().size() == 1) {
    auto unpack_node = GetUnpackNodeFromTensorList(node->output(0));
    HABANA_ASSERT(
        unpack_node != nullptr,
        "TensorList is not input to ListUnpack node. Node: ",
        node->kind().toQualString());
    for (auto value_out : unpack_node->outputs()) {
      OutputMetaData md(*value_out);
      md.persistent = nodeOutputPersistencePerValue(unpack_node, value_out);
      if (md.persistent) {
        md.external = IsValueExternal(value_out);
      }
      auto out_ptr = value_out->type()->cast<c10::TensorType>();
      if (out_ptr->scalarType().has_value()) {
        md.dtype = *out_ptr->scalarType();
      }
      output_metadata.emplace_back(md);
    }
  } else {
    for (auto value_out : node_outs) {
      OutputMetaData md(*value_out);
      md.persistent = nodeOutputPersistencePerValue(node, value_out);
      if (md.persistent) {
        md.external = IsValueExternal(value_out);
      }
      auto out_ptr = value_out->type()->cast<c10::TensorType>();

      if (out_ptr->scalarType().has_value()) {
        md.dtype = *out_ptr->scalarType();
      }
      output_metadata.emplace_back(md);
    }
  }
  return output_metadata;
}

void HabanaLaunchOpPT::HandleMappedTensor(
    CValPtr value_in,
    const HabanaOperatorPtr& habana_op,
    SharedSynTensorOrRefListPtr& tensorList) {
  PT_BRIDGE_TRACE;
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
    at::Tensor& pt_tensor,
    std::string idx) {
  PT_BRIDGE_TRACE;
  auto tmeta = get_tensor_extra_meta(pt_tensor, true);

  if (tmeta && tmeta->is_shape_tensor()) {
    void* host_ptr = tmeta->get_compile_host_ptr();
    auto& syn_tensor = habana_op->AllocateSynapseInput(
        *syn_graph_ptr, pt_tensor, true, tmeta->get_tensor_type(), host_ptr);
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
    auto& syn_tensor = habana_op->AllocateSynapseInput(
        *syn_graph_ptr, pt_tensor, true, DATA_TENSOR, nullptr, idx);

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
    SharedSynTensorOrRefListPtr& tensorList,
    std::string idx) {
  PT_BRIDGE_TRACE;
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

  std::vector<PtTensorInfoShared> tiv;
  for (auto& pt_tensor : pyTensorList) {
    if (!pt_tensor.defined()) {
      continue;
    }
    auto& syn_tensor = AllocateSynapseTensor(habana_op, pt_tensor, idx);
    PT_BRIDGE_DEBUG(
        "Allocated synpase tensor for input tensor: ", syn_tensor.id());

    tensorList->emplace_back(tensor_or_ref(syn_tensor));

    std::string irn = "%" + value_in->debugName();
    PtTensorInfoShared ti = std::make_shared<PtTensorInfo>(
        pt_tensor,
        syn_tensor.name(),
        irn,
        watch_tensor_flag_,
        syn_tensor.id(),
        syn_tensor.get(),
        syn_tensor.tensor_type());

    auto tmeta{get_tensor_extra_meta(pt_tensor)};
    if (tmeta) {
      ti->set_host_ptr(tmeta->get_host_ptr());
    }
    tiv.push_back(ti);
    ivalue_to_tensor_info_map[ivalue] = ti;

    if (enable_caching_ || enable_shape_agnostic_caching_) {
      void* buffp = ti->get_buffer_start();
      if (ti->is_ZST() == false) {
        buff_to_input_ivpsh_map.emplace(buffp, ivalue);
      }
    }
  }

  if (!tensorList->empty()) {
    auto it = pt_to_synapse_tensors.emplace(ivalue, tensorList);
    if (it.second == false) {
      pt_to_synapse_tensors[ivalue] = tensorList;
    }

    if (enable_caching_ || enable_shape_agnostic_caching_) {
      input_tiv_map.emplace(value_to_ivalue[value_in], tiv);
      auto node_qual_str = value_in->node()->kind().toQualString();
      if ((strcmp(node_qual_str, "hpu::restride_cl") == 0) ||
          (strcmp(node_qual_str, "hpu::restride") == 0)) {
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
    SharedSynTensorOrRefListPtr& tensorList,
    std::string idx) {
  auto is_already_mapped =
      pt_to_synapse_tensors.find(value_to_ivalue[value_in]) !=
      std::end(pt_to_synapse_tensors);
  if (is_already_mapped) {
    HandleMappedTensor(value_in, habana_op, tensorList);
  } else {
    HandleUnmappedTensor(value_in, habana_op, tensorList, idx);
  }
}

void HabanaLaunchOpPT::GetSynapseInputs(
    const HabanaOperatorPtr& habana_op,
    torch::jit::Node* node) {
  auto node_ins = node->inputs();
  int input_idx = 0;

  for (const auto value_in : node_ins) {
    auto value_exists = value_to_ivalue.find(value_in);
    HABANA_ASSERT(value_exists != std::end(value_to_ivalue));
    auto ivalue = value_exists->second;
    std::string scope_string;
    if (GET_ENV_FLAG_NEW(PT_HPU_INFERENCE_MODE)) {
      scope_string = std::string(node->scope()->name().toUnqualString());
      scope_string = !scope_string.empty()
          ? scope_string.substr(1, scope_string.length() - 1)
          : scope_string;
      std::replace(scope_string.begin(), scope_string.end(), '/', '.');
    }
    if ((ivalue->isTensor() || ivalue->isTensorList())) {
      // Find if an input tensor is already mapped
      // NB: It seems Habana doesn't support shared input to
      // different nodes in graph
      // note: else path is only of listcontruct is fused with another op like
      // cat. This case occurs in lazy eval but not in torch trace mode
      if (ivalue->isTensor() ||
          (value_in->node()->kind() != torch::jit::prim::ListConstruct)) {
        SharedSynTensorOrRefListPtr tensor_ref_list_ptr_sh =
            std::make_shared<SynTensorOrRefList>();
        HandleMappedandUnmappedTensor(
            value_in,
            habana_op,
            tensor_ref_list_ptr_sh,
            scope_string + ".placeholder." + std::to_string(input_idx));
      } else {
        // tensorlist
        auto prev_node = value_in->node();
        if (prev_node->kind() == torch::jit::prim::ListConstruct) {
          for (auto& value_in : prev_node->inputs()) {
            if (value_to_ivalue[value_in]->isTensor()) {
              SharedSynTensorOrRefListPtr tensor_ref_list_ptr_sh =
                  std::make_shared<SynTensorOrRefList>();
              HandleMappedandUnmappedTensor(
                  value_in,
                  habana_op,
                  tensor_ref_list_ptr_sh,
                  scope_string + ".placeholder." + std::to_string(input_idx));
            }
          }
        }
      } // else
      input_idx++;
    } // if (value_to_ivalue[value_in] && ..
  } // for (const auto value_in : node_ins)

  bool populate_seed = false;
  switch (node->kind()) {
    case torch::jit::aten::bernoulli:
      populate_seed = true;
      break;
  }

  if (populate_seed) {
    int seed = get_seed_hpu(c10::nullopt);
    at::Tensor seed_tensor = at::tensor(seed).to(at::kHPU);
    auto& syn_tensor = habana_op->AllocateSeed(*syn_graph_ptr, seed_tensor);
    PtTensorInfoShared ti = std::make_shared<PtTensorInfo>(
        seed_tensor,
        syn_tensor.name(),
        "%seed_input",
        watch_tensor_flag_,
        syn_tensor.id(),
        syn_tensor.get(),
        syn_tensor.tensor_type(),
        DMAInputGeneratorType::SEEDTENSOR);
    ti->set_dma_tensor_idx(aten_intermediates.size());
    dma_input_tensorinfos.emplace_back(ti);
    aten_intermediates.emplace_back(seed_tensor);
  }
}

PtTensorInfoShared HabanaLaunchOpPT::ProcessPersistentNodeOutput(
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

  PtTensorInfoShared ti = std::make_shared<PtTensorInfo>(
      ivpsh,
      out_syntensor.name(),
      vp,
      watch_tensor_flag_,
      out_syntensor.id(),
      out_syntensor.get(),
      out_syntensor.tensor_type());
  ti->set_external(out_syntensor.is_external());
  ivalue_to_tensor_info_map[ivpsh] = ti;
  void* buffp = ti->get_buffer_start();

  if (false == isInGraphOutputs(vp)) {
    if (ti->is_ZST() == false && buff_to_input_ivpsh_map.count(buffp)) {
      // Case 1.A: intermediate persistent tensor which an alias of an input
      PT_BRIDGE_DEBUG("Adding to duplicate_input_tivs ", *ti);
      duplicate_input_tivs.emplace_back(ti);
    } else {
      if (ti->is_ZST() == false && buff_to_output_ivpsh_map.count(buffp)) {
        duplicate_outtinfos.emplace_back(ti);
      } else {
        // Case 1.B: intermediate persistent tensor
        if (ti->is_view_tensor()) {
          PT_BRIDGE_DEBUG(
              "Starting persistent intermediate is view tensor ",
              "with non zero offset ",
              ti->get_offset());
        }
        AddAtenIntermediate(ivpsh, ti);
      }
    }
  } else {
    if (!enable_caching_ && !enable_shape_agnostic_caching_) {
      // Case 2.A: graph output tensor
      // needs to be added to enable layout handling for lazy eager
      PT_BRIDGE_DEBUG("Adding to output_tensorinfo_map ", *ti);
      output_tensorinfo_map.emplace(ivpsh, ti);
    } else {
      // Is this a duplicate tensor going to graph output?
      // See if this the buffer pointer matches any input, then -
      // Check whether it is an alias of any input
      if (ti->is_ZST() == false && buff_to_input_ivpsh_map.count(buffp)) {
        // Case 2.B: Graph output that is duplicate of input
        PT_BRIDGE_DEBUG("Adding to duplicate_input_to_outtinfo_map ", *ti);
        duplicate_input_to_outtinfo_map.emplace(ivpsh, ti);
      } else if (
          ti->is_ZST() == false &&
          buff_to_intermediate_ivpsh_map.count(buffp)) {
        // Case 2.C: Graph output that is duplicate of a persistent
        // intermediate
        PT_BRIDGE_DEBUG(
            "Adding to duplicate_intermediate_to_outtinfo_map ", *ti);
        duplicate_intermediate_to_outtinfo_map.emplace(ivpsh, ti);
      } else if (
          ti->is_ZST() == false && buff_to_output_ivpsh_map.count(buffp)) {
        // Case 2.D: Graph output that is duplicate of a previous output
        PT_BRIDGE_DEBUG("Adding to duplicate_output_to_outtinfo_map ", *ti);
        duplicate_output_to_outtinfo_map.emplace(ivpsh, ti);
      } else {
        // Case 2.A: graph output tensor, enable_tensor_release_
        PT_BRIDGE_DEBUG("Adding to output_tensorinfo_map ", *ti);
        output_tensorinfo_map.emplace(ivpsh, ti);
        if (ti->is_ZST() == false) {
          buff_to_output_ivpsh_map.emplace(buffp, ivpsh);
        }
      }
    }
  }
  return ti;
}

int64_t HabanaLaunchOpPT::ProcessSynapseOutputs(
    const HabanaOperatorPtr& habana_op,
    torch::jit::Node* node,
    OutputShapeInfRetType& op_output_shape) {
  auto output_nodes = node->outputs();
  auto habana_kernel_meta_data = habana_op->GetKernelMetaData();

  bool shape_inf_flag = enable_fast_shape_inf_ &&
      syn_graph_ptr->is_dynamic_graph() &&
      !(m_map_shape.m_pass == ShapeInfo::InferencePass::MIN_SHAPE ||
        m_map_shape.m_pass == ShapeInfo::InferencePass::MAX_SHAPE);

  if ((shape_inf_flag || enable_shape_agnostic_caching_) &&
      !op_output_shape.empty()) {
    HABANA_ASSERT(
        habana_op->GetSynOutputs().size() ==
            op_output_shape.GetOutputTensor().size(),
        "For node ",
        node->kind().toQualString(),
        "GetSynOutputs().size()=",
        habana_op->GetSynOutputs().size(),
        ", whereas GetOutputTensor().size()=",
        op_output_shape.GetOutputTensor().size());
  }

  if (*node->output(0)->type() == *torch::ListType::ofTensors() &&
      node->outputs().size() == 1) {
    auto unpack_node = GetUnpackNodeFromTensorList(node->output(0));
    HABANA_ASSERT(
        unpack_node != nullptr,
        "TensorList is not input to ListUnpack node. Node: ",
        node->kind().toQualString());
    output_nodes = unpack_node->outputs();
  }

  const auto& output_tensors_pt = habana_op->GetOutputs();
  const auto& excluded_out_indices =
      habana_op->GetSynOutputIndicesExcludedInNode();

  TORCH_CHECK(
      output_nodes.size() ==
          output_tensors_pt.size() - excluded_out_indices.size(),
      "HabanaFusionOp Lowering of node : ",
      node->kind().toQualString(),
      " Number of output nodes ",
      output_nodes.size(),
      " doesnt match the generated ",
      output_tensors_pt.size() - excluded_out_indices.size());

  size_t output_nodes_idx = 0, output_tensor_idx = 0;

  auto cur_sif_tidx = habana::ShapeInference::GetSifTensorId();

  auto handle_permutes = [&](PtTensorInfoShared ti,
                             synapse_helpers::tensor& sh_t,
                             IValPtrShared ivpsh) {
    // set permutation flag for persistent tensors
    if (GET_ENV_FLAG_NEW(PT_HPU_ENABLE_SYNAPSE_OUTPUT_PERMUTE) &&
        !is_hccl_send_mark_step()) {
      if (!ti->is_ZST()) {
        setSynapsePermuteFlag(sh_t, ti, ivpsh);
        if (pt_to_synapse_tensors.count(ivpsh)) {
          PT_BRIDGE_DEBUG(
              habana_helpers::DebugString(ivpsh),
              " already exists in pt_to_synapse_tensors map, ",
              *ti);
        }
      }
    }
  };

  auto handle_shape_inf = [&](PtTensorInfoShared ti,
                              bool use_output_shape,
                              bool shape_agn_flag) {
    if (shape_inf_flag || shape_agn_flag) {
      if (use_output_shape && !op_output_shape.empty()) {
        auto output = op_output_shape.GetOutputTensor().at(output_tensor_idx);
        auto output_sif_tidx{std::get<0>(output)};
        auto ret = sif_tidx_to_tinfo_map.insert({output_sif_tidx, ti});
        if (ret.second) {
          PT_DYNAMIC_SHAPE_DEBUG(
              "Output tensor cs: adding to sif_tidx_to_tinfo_map : ",
              output_sif_tidx,
              " -> ",
              *ti);
        } else {
          PT_DYNAMIC_SHAPE_DEBUG(
              "Output tensor cs: failed adding to sif_tidx_to_tinfo_map : ",
              output_sif_tidx,
              " -> ",
              *ti);
        }
      } else {
        // op_output_shape is empty
        auto ret = sif_tidx_to_tinfo_map.insert({cur_sif_tidx, ti});
        if (ret.second) {
          PT_DYNAMIC_SHAPE_DEBUG(
              "Output tensor manual: adding to sif_tidx_to_tinfo_map : ",
              cur_sif_tidx,
              " -> ",
              *ti);
        } else {
          PT_DYNAMIC_SHAPE_DEBUG(
              "Output tensor manual: failed adding to sif_tidx_to_tinfo_map : ",
              cur_sif_tidx,
              " -> ",
              *ti);
        }
      }
    }
  };

  auto handle_postprocess = [&](const auto& nodes,
                                int node_output_idx,
                                int tensor_idx,
                                synapse_helpers::tensor& sh_t) {
    SharedSynTensorOrRefListPtr tensorList =
        std::make_shared<SynTensorOrRefList>();
    tensorList->emplace_back(tensor_or_ref(sh_t));
    pt_to_synapse_tensors.emplace(
        value_to_ivalue[nodes[node_output_idx]], tensorList);

    // Validate external flag was set correctly
    const auto& value = nodes.at(tensor_idx);
    bool required_external = persistence_marker_pass_data_ptr_.get()
        ? persistence_marker_pass_data_ptr_->IsExternalNode(value)
        : false;
    if (required_external) {
      HABANA_ASSERT(
          sh_t.is_external() == required_external,
          "Output ",
          tensor_idx,
          " of node ",
          node->kind().toQualString(),
          " is not external");
    }
  };

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
        auto ti = ProcessPersistentNodeOutput(
            ivpsh, output_nodes[output_nodes_idx], out_tensor_syn);

        handle_permutes(ti, out_tensor_syn, ivpsh);

        constexpr bool use_output_shape = true;
        constexpr bool shape_agn_flag = false;
        handle_shape_inf(ti, use_output_shape, shape_agn_flag);
      } else if (enable_shape_agnostic_caching_) {
        // For shape agnostic flow for eager we need non-persistent info as well
        // Try maintaing it in another struct other than dtensor info struct
        PtTensorInfoShared ti = std::make_shared<PtTensorInfo>(
            out_tensor_syn.name(),
            watch_tensor_flag_,
            out_tensor_syn.id(),
            out_tensor_syn.get(),
            out_tensor_syn.tensor_type());
        constexpr bool use_output_shape = true;
        handle_shape_inf(ti, use_output_shape, enable_shape_agnostic_caching_);
        non_persistent_intermediate_tinfos.emplace_back(ti);
      }

      handle_postprocess(
          output_nodes, output_nodes_idx, output_tensor_idx, out_tensor_syn);

      output_nodes_idx++;
    }
    output_tensor_idx++;
    cur_sif_tidx++;
  }

  // Handle implicit syn outputs - these are input tensors that are being
  // updated inplace, but are not returned as outputs, so they can't be
  // treated as _out or common inplace input tensors.
  auto input_nodes = node->inputs();
  for (auto& syn_impl_op : habana_op->GetSynImplicitOutputs()) {
    const auto& pt_input_idx = syn_impl_op.pt_input_idx;
    const auto& syn_input_idx = syn_impl_op.syn_input_idx;
    synapse_helpers::tensor& sh_t = syn_impl_op.sh_t;
    IValPtrShared ivpsh = value_to_ivalue[input_nodes[pt_input_idx]];

    // For some kernels, like the inplace ones, the kernel output is always
    // created as persistent. Patching table needs to be updated accordingly
    // for such tensors.
    if (use_persistent_tensors || sh_t.is_persistent()) {
      PtTensorInfoShared ti = std::make_shared<PtTensorInfo>(
          ivpsh,
          sh_t.name(),
          input_nodes[pt_input_idx],
          watch_tensor_flag_,
          sh_t.id(),
          sh_t.get(),
          sh_t.tensor_type());

      ti->set_external(sh_t.is_external());
      duplicate_input_tivs.emplace_back(ti);

      // persistent tensor which an alias of an input
      PT_BRIDGE_DEBUG("Adding to duplicate_input_tivs ", *ti);

      handle_permutes(ti, sh_t, ivpsh);
      constexpr bool use_output_shape = false;
      constexpr bool shape_agn_flag = false;
      handle_shape_inf(ti, use_output_shape, shape_agn_flag);
    }

    handle_postprocess(input_nodes, syn_input_idx, syn_input_idx, sh_t);

    cur_sif_tidx++;
  }

  return cur_sif_tidx;
}

void HabanaLaunchOpPT::ProcessShapeTensorsCS(
    const OutputShapeInfRetType& output,
    std::vector<IdxTensorTup>& intermediate_shape_tensor_cs) {
  auto shape_tensors = output.GetShapeTensor();

  for (auto& st : shape_tensors) {
    intermediate_shape_tensor_cs.emplace_back(st);
  }

  for (auto& kernel : output.GetKernelOutputs()) {
    ProcessShapeTensorsCS(*kernel.get(), intermediate_shape_tensor_cs);
  }
}

void HabanaLaunchOpPT::ProcessSynapseShapeTensors(
    const HabanaOperatorPtr& habanaOp,
    std::vector<size_t>& intermediate_shape_tensors,
    std::vector<size_t>& inputs_shape_tensors,
    bool isRecursiveCall) {
  bool shape_inf_flag = enable_fast_shape_inf_ &&
      syn_graph_ptr->is_dynamic_graph() &&
      !(m_map_shape.m_pass == ShapeInfo::InferencePass::MIN_SHAPE ||
        m_map_shape.m_pass == ShapeInfo::InferencePass::MAX_SHAPE);

  if (auto op = std::dynamic_pointer_cast<OpBackend>(habanaOp)) {
    for (const auto& st : op->GetShapeTensors()) {
      auto irn = "%shapeInput_" + std::to_string(shape_index++);
      PtTensorInfoShared ti = std::make_shared<PtTensorInfo>(st, irn);
      if (shape_inf_flag) {
        if (st.is_intermediate_shape_tensor()) {
          intermediate_shape_tensors.emplace_back(shape_tensor_tinfos.size());
          PT_DYNAMIC_SHAPE_DEBUG(
              "auto_gen path: intermediate shape tensor : ", *ti);
        }
      }
      shape_tensor_tinfos.emplace_back(ti);
    }
  }

  for (synapse_helpers::tensor& maybe_syn_shape_tensor :
       habanaOp->GetSynInputs()) {
    if (maybe_syn_shape_tensor.is_shape_tensor() ||
        maybe_syn_shape_tensor.is_input_shape_tensor()) {
      std::string irn{"%shapeInput_"};
      irn += std::to_string(shape_index);
      shape_index++;
      PtTensorInfoShared ti =
          std::make_shared<PtTensorInfo>(maybe_syn_shape_tensor, irn);
      if (shape_inf_flag) {
        if (maybe_syn_shape_tensor.is_intermediate_shape_tensor()) {
          intermediate_shape_tensors.emplace_back(shape_tensor_tinfos.size());
          PT_DYNAMIC_SHAPE_DEBUG(
              "manual path: adding intermediate shape tensor for index = ",
              intermediate_shape_tensors.back(),
              " : ",
              *ti);
        } else {
          // Add input shape tensors for top level op only
          // skip adding for child kernels by check recursive call
          if (true == isRecursiveCall)
            continue;

          inputs_shape_tensors.emplace_back(shape_tensor_tinfos.size());
          PT_DYNAMIC_SHAPE_DEBUG(
              "manual path: adding input shape tensor for index = ",
              inputs_shape_tensors.back(),
              " : ",
              *ti);
        }
      }
      shape_tensor_tinfos.emplace_back(ti);
    }
  }

  // Add shape tensor for all Operator created inside habanaOp
  std::vector<HabanaOperatorPtr> habana_kernels = habanaOp->GetKernels();
  for (auto& habana_op : habana_kernels) {
    ProcessSynapseShapeTensors(
        habana_op, intermediate_shape_tensors, inputs_shape_tensors, true);
  }
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
    auto variant =
        synapse_helpers::tensor_builder(
            tensor->sizes(),
            tensor->strides(),
            habana_helpers::pytorch_to_synapse_type(dtype))
            .mark_persistence(true)
            .with_memory_section(syn_tensor_input.memorysection())
            .build(
                HPURegistrar::get_device(tensor->device().index()).syn_device(),
                syn_tensor_input.graph());

    meta_syn_tensors.push_back(
        absl::get<synapse_helpers::tensor>(std::move(variant)));

    PtTensorInfoShared ti = std::make_shared<PtTensorInfo>(
        value_to_ivalue[value_in],
        meta_syn_tensors.back().name(),
        value_in,
        watch_tensor_flag_,
        meta_syn_tensors.back().id(),
        meta_syn_tensors.back().get(),
        meta_syn_tensors.back().tensor_type());
    ivalue_to_tensor_info_map[value_to_ivalue[value_in]] = ti;
    if (!isInGraphOutputs(value_in)) {
      duplicate_input_tivs.emplace_back(ti);
    } else {
      if (!enable_caching_ && !enable_shape_agnostic_caching_) {
        output_tensorinfos.emplace_back(ti);
      } else {
        duplicate_outtinfos.emplace_back(ti);
      }
    }
  } else {
    pt_to_synapse_tensors.erase(value_to_ivalue[value_in]);
    auto variant = habana_helpers::create_tensor(
        *tensor, *syn_graph_ptr, persistence, false);
    meta_syn_tensors.push_back((std::move(variant)));
  }

  auto& syn_tensor = meta_syn_tensors.back();
  pt_to_synapse_tensors.erase(value_to_ivalue[value_in]);
  SharedSynTensorOrRefListPtr tensorList =
      std::make_shared<SynTensorOrRefList>();
  tensorList->emplace_back(tensor_or_ref(syn_tensor));
  pt_to_synapse_tensors.emplace(value_to_ivalue[value_in], tensorList);
}

IValPtrShared castConstantTensor(IValPtrShared ival) {
  auto tensor = ival->toTensor();
  auto dtype = tensor.scalar_type();

  const bool cast = habana_helpers::is_downcast_to_int_needed(dtype) ||
      dtype == c10::ScalarType::Double;
  if (cast) {
    const auto dst_type = dtype == c10::ScalarType::Long
        ? c10::ScalarType::Int
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
  bool is_jit_cached_graph_info_available =
      jit_graph_and_meta_data->get_jit_cached_graph_info_available_flag();

  if (is_jit_cached_graph_info_available == false) {
    bool is_in_graph_outputs = isInGraphOutputs(value_out);
    jit_graph_and_meta_data->set_is_in_graph_outputs(is_in_graph_outputs);
  }
  auto is_in_graph_outputs = jit_graph_and_meta_data->get_is_in_graph_outputs(
      restride_node_out_val_counter);
  restride_node_out_val_counter++;

  if ((tensor.dim() == 4) || (tensor.dim() == 5)) {
    if (is_jit_cached_graph_info_available == false) {
      auto new_pos = toIValue(node->input(1))->toIntVector();
      jit_graph_and_meta_data->set_new_pos(new_pos);
    }
    std::vector<int64_t>& new_pos =
        jit_graph_and_meta_data->get_new_pos(restride_node_swap_counter);
    restride_node_swap_counter++;

    auto sizes = tensor.sizes().vec();
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

    if (is_in_graph_outputs) {
      auto format = is_5d_layout ? c10::MemoryFormat::ChannelsLast3d
                                 : c10::MemoryFormat::ChannelsLast;
      if (!is_restride_cl) {
        if (tensor.dim() == 4 || tensor.dim() == 5) {
          auto hb_grad_weight{get_tensor_extra_meta(tensor)};
          hb_grad_weight->set_tensor_layout(habana::LayoutFormat::HWCK);
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

  if (is_in_graph_outputs) {
    TORCH_CHECK(
        pt_to_synapse_tensors.count(ivpsh),
        " Could not find the syn tensor corresponding to %",
        value_in->debugName());

    auto& syn_tensor_vec = pt_to_synapse_tensors[ivpsh];
    synapse_helpers::tensor& syn_tensor = syn_tensor_vec->at(0);
    habana::ShapeInference::UpdateShapeInfo(
        *syn_graph_ptr, syn_tensor.id(), tensor.sizes().vec());
    PtTensorInfoShared ti = std::make_shared<PtTensorInfo>(
        ivpsh_restrided,
        syn_tensor.name(),
        value_in,
        watch_tensor_flag_,
        syn_tensor.id(),
        syn_tensor.get(),
        syn_tensor.tensor_type());
    ti->set_restrided(true);

    value_to_ivalue.erase(value_in);

    if (enable_caching_ || enable_shape_agnostic_caching_) {
      void* buffp = ti->get_buffer_start();
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
          ti->is_ZST() == false &&
          buff_to_intermediate_ivpsh_map.count(buffp)) {
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
      } else if (
          ti->is_ZST() == false && buff_to_input_ivpsh_map.count(buffp)) {
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
          ti->is_ZST() == false && buff_to_output_ivpsh_map.count(buffp)) {
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
            (ti->is_ZST() ? " is ZST" : " is non ZST"),
            ", not found in any duplicate detection or output map");
      }
    }

    value_to_ivalue[value_in] = ivpsh_restrided;
    value_to_ivalue[value_out] = ivpsh_restrided;
    ivalue_to_tensor_info_map[value_to_ivalue[value_in]] = ti;
    ivalue_to_tensor_info_map[value_to_ivalue[value_out]] = ti;
  } else {
    PT_BRIDGE_DEBUG(
        "restride node output %", value_out->debugName(), " is non persistent");
    value_to_ivalue[value_out] = ivpsh_restrided;
  }
}

void HabanaLaunchOpPT::handlePrimNodes(torch::jit::Node* node) {
  PT_BRIDGE_TRACE;
  if (node->kind() == torch::jit::prim::Constant) {
    handlePrimConstantNode(node);
  } else if (node->kind() == torch::jit::prim::ListConstruct) {
    handlePrimListConstructNode(node);
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

void HabanaLaunchOpPT::handlePrimListConstructNode(torch::jit::Node* node) {
  const auto& node_ins = node->inputs();
  auto node_vals = node->outputs();
  HABANA_ASSERT(node_vals.size() == 1);

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
    value_to_ivalue[node_vals[0]] = std::make_shared<IVal>(opttensorList);
    return;
  }

  auto ivptrsh = value_to_ivalue[node_ins[0]];

  // Handle construction of list consisting tensor only
  if (ivptrsh->isTensor()) {
    c10::List<at::Tensor> tensorList;
    for (const auto& value_in : node_ins) {
      ivptrsh = value_to_ivalue[value_in];
      // Constructed list should be homogenous
      HABANA_ASSERT(ivptrsh->isTensor());
      tensorList.emplace_back(ivptrsh->toTensor());
    }
    value_to_ivalue[node_vals[0]] = std::make_shared<IVal>(tensorList);
    return;
  }

  //  Handle construction of list consisting ints only
  if (ivptrsh->isInt()) {
    c10::List<int64_t> intList;
    for (const auto& value_in : node_ins) {
      ivptrsh = value_to_ivalue[value_in];
      // Constructed list should be homogenous
      HABANA_ASSERT(ivptrsh->isInt());
      intList.emplace_back(ivptrsh->toInt());
    }
    value_to_ivalue[node_vals[0]] = std::make_shared<IVal>(intList);
    return;
  }

  //  Handle construction of list consisting bools only
  if (ivptrsh->isBool()) {
    c10::List<bool> boolList;
    for (const auto& value_in : node_ins) {
      ivptrsh = value_to_ivalue[value_in];
      // Constructed list should be homogenous
      HABANA_ASSERT(ivptrsh->isBool());
      boolList.emplace_back(ivptrsh->toBool());
    }
    value_to_ivalue[node_vals[0]] = std::make_shared<IVal>(boolList);
    return;
  }

  HABANA_ASSERT(false, "Unsupported list type in prim::ListConstruct");
}

void HabanaLaunchOpPT::handlePrimConstantNode(torch::jit::Node* node) {
  auto node_vals = node->outputs();
  bool is_jit_cached_graph_info_available =
      jit_graph_and_meta_data->get_jit_cached_graph_info_available_flag();
  for (const auto value : node_vals) {
    IValPtrShared ivptrsh = nullptr;
    if (is_jit_cached_graph_info_available == false) {
      ivptrsh = std::make_shared<IVal>(toIValue(value).value());
    }
    if (value->type()->kind() == c10::TypeKind::TensorType) {
      if (is_jit_cached_graph_info_available == false) {
        auto ivptrshUpdated = castConstantTensor(ivptrsh);
        jit_graph_and_meta_data->set_prim_nodes_ival(ivptrshUpdated);
      }
      auto ivptrsh_updated =
          jit_graph_and_meta_data->get_prim_nodes_ival(prim_nodes_ival_counter);
      value_to_ivalue[value] = ivptrsh_updated;
      std::string irn{"%intermediate_"};
      irn += std::to_string(intermediate_index);
      intermediate_index++;

      auto tensor = ivptrsh_updated->toTensor();
      meta_syn_tensors.push_back(habana_helpers::create_tensor(
          tensor, *syn_graph_ptr, true, false, tensor.scalar_type()));
      SharedSynTensorOrRefListPtr tensorList =
          std::make_shared<SynTensorOrRefList>();
      tensorList->emplace_back(tensor_or_ref(meta_syn_tensors.back()));
      pt_to_synapse_tensors.emplace(value_to_ivalue[value], tensorList);
      PtTensorInfoShared ti = std::make_shared<PtTensorInfo>(
          tensor,
          meta_syn_tensors.back().name(),
          irn,
          watch_tensor_flag_,
          meta_syn_tensors.back().id(),
          meta_syn_tensors.back().get(),
          meta_syn_tensors.back().tensor_type());

      ivalue_to_tensor_info_map[ivptrsh_updated] = ti;
      aten_intermediates.push_back(tensor);
    } else {
      if (is_jit_cached_graph_info_available == false) {
        jit_graph_and_meta_data->set_prim_nodes_ival(ivptrsh);
      } else {
        ivptrsh = jit_graph_and_meta_data->get_prim_nodes_ival(
            prim_nodes_ival_counter);
      }
      value_to_ivalue[value] = ivptrsh;
    }
    prim_nodes_ival_counter++;
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
  PT_BRIDGE_TRACE;
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

std::string HabanaLaunchOpPT::DumpNodeInputs(torch::jit::Node* node) {
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
  return str;
}
std::string HabanaLaunchOpPT::DumpNodeOutputs(torch::jit::Node* node) {
  std::ostringstream o;
  node->print(o, 0, nullptr);
  auto str = o.str();
  if (*node->output(0)->type() != *torch::ListType::ofTensors()) {
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

void HabanaLaunchOpPT::validateOutputShapeDynamic(
    const HabanaOperatorPtr& HabanaKernel,
    const OutputShapeInfRetType& output_shape_handle,
    const std::string& opname) {
  auto lowering_kernels = HabanaKernel->GetKernels();
  auto output_shape_kernels = output_shape_handle.GetKernels();
  HABANA_ASSERT(
      std::dynamic_pointer_cast<OpBackend>(HabanaKernel) or
          lowering_kernels.size() == output_shape_kernels.size(),
      "Node: ",
      opname,
      " number of sub kernels mismatch in shape ineference, expected: ",
      lowering_kernels.size(),
      " but got: ",
      output_shape_kernels.size());

  std::deque<synapse_helpers::tensor_or_ref>& syn_outputs =
      HabanaKernel->GetSynOutputs();
  std::deque<synapse_helpers::tensor_or_ref>& syn_inputs =
      HabanaKernel->GetSynInputs();
  int intermediate_shape_tensor_count = 0;
  // Auto gen op intermediate shape tensors
  if (auto op = std::dynamic_pointer_cast<OpBackend>(HabanaKernel)) {
    for (const auto& st : op->GetShapeTensors()) {
      if (st.is_intermediate_shape_tensor()) {
        HABANA_ASSERT(st.is_shape_tensor());
        intermediate_shape_tensor_count++;
      }
    }
  }
  // Manual op intermediate shape tensors
  for (synapse_helpers::tensor& in_tensor_syn : syn_inputs) {
    if (in_tensor_syn.is_intermediate_shape_tensor()) {
      HABANA_ASSERT(in_tensor_syn.is_shape_tensor());
      intermediate_shape_tensor_count++;
    }
  }
  auto output_vec = output_shape_handle.GetOutputTensor();
  auto output_shape_vec = output_shape_handle.GetShapeTensor();
  auto output_size = output_vec.size() + output_shape_vec.size();

  HABANA_ASSERT(
      (syn_outputs.size() + intermediate_shape_tensor_count) == output_size,
      "Node: ",
      opname,
      " number of output mismatch, expected: ",
      (syn_outputs.size() + intermediate_shape_tensor_count),
      " but got: ",
      output_size);
  // compare output shape
  size_t i = 0, j = 0;
  std::vector<int64_t> t;
  for (synapse_helpers::tensor& out_tensor_syn : syn_outputs) {
    if (out_tensor_syn.is_shape_tensor()) {
      t = std::get<at::Tensor>(output_shape_vec.at(i++)).sizes().vec();
    } else {
      t = std::get<at::Tensor>(output_vec.at(j++)).sizes().vec();
    }
    HABANA_ASSERT(
        out_tensor_syn.pt_shape() == t,
        "Node: ",
        opname,
        " shape validation failed",
        " expected: ",
        out_tensor_syn.pt_shape(),
        " got: ",
        t);
  }
  // for validation of shape tensor we rely on output shape tensor added
  // before intermediate shape tensor in ComputeOutputShape
  // Auto gen op shape tensors
  if (auto op = std::dynamic_pointer_cast<OpBackend>(HabanaKernel)) {
    for (const auto& st : op->GetShapeTensors()) {
      if (st.is_intermediate_shape_tensor()) {
        HABANA_ASSERT(st.is_shape_tensor());
        t = std::get<at::Tensor>(output_shape_vec.at(i++)).sizes().vec();

        HABANA_ASSERT(
            st.pt_shape() == t,
            "Node: ",
            opname,
            " shape tensor validation failed",
            " expected: ",
            st.pt_shape(),
            " but got: ",
            t);
      }
    }
  }
  // Manual op shape tensors
  for (synapse_helpers::tensor& in_tensor_syn : syn_inputs) {
    if (in_tensor_syn.is_intermediate_shape_tensor()) {
      HABANA_ASSERT(in_tensor_syn.is_shape_tensor());
      t = std::get<at::Tensor>(output_shape_vec.at(i++)).sizes().vec();
      HABANA_ASSERT(
          in_tensor_syn.pt_shape() == t,
          "Node: ",
          opname,
          " shape tensor validation failed",
          " expected: ",
          in_tensor_syn.pt_shape(),
          " but got: ",
          t);
    }
  }

  // check for the child kernels output and shape tensor
  for (size_t i = 0; i < lowering_kernels.size(); i++) {
    validateOutputShapeDynamic(
        lowering_kernels[i], *output_shape_kernels[i], opname);
  }
}

void HabanaLaunchOpPT::validateOutputShapeNonDynamic(
    const HabanaOperatorPtr& HabanaKernel,
    const OutputShapeInfRetType& output_shape_handle,
    const std::string& opname) {
  auto lowering_kernels = HabanaKernel->GetKernels();
  auto output_shape_kernels = output_shape_handle.GetKernels();
  HABANA_ASSERT(
      std::dynamic_pointer_cast<OpBackend>(HabanaKernel) or
          lowering_kernels.size() == output_shape_kernels.size(),
      "Node: ",
      opname,
      " number of sub kernels mismatch in shape ineference, expected: ",
      lowering_kernels.size(),
      " but got: ",
      output_shape_kernels.size());

  std::deque<synapse_helpers::tensor_or_ref>& syn_outputs =
      HabanaKernel->GetSynOutputs();
  auto output_vec = output_shape_handle.GetOutputTensor();
  auto output_size = output_vec.size();

  HABANA_ASSERT(
      syn_outputs.size() == output_size,
      "Node: ",
      opname,
      " number of output mismatch, expected: ",
      syn_outputs.size(),
      " but got: ",
      output_size);
  // compare output shape
  size_t j = 0;
  std::vector<int64_t> t;
  for (synapse_helpers::tensor& out_tensor_syn : syn_outputs) {
    t = std::get<at::Tensor>(output_vec.at(j++)).sizes().vec();
    HABANA_ASSERT(
        out_tensor_syn.pt_shape() == t,
        "Node: ",
        opname,
        " shape validation failed",
        " expected: ",
        out_tensor_syn.pt_shape(),
        " got: ",
        t);
  }
  // check for the child kernels output
  for (size_t i = 0; i < lowering_kernels.size(); i++) {
    validateOutputShapeNonDynamic(
        lowering_kernels[i], *output_shape_kernels[i], opname);
  }
}

void HabanaLaunchOpPT::validateOutputShape(
    const HabanaOperatorPtr& HabanaKernel,
    const OutputShapeInfRetType& output_shape_handle,
    const synapse_helpers::graph& syn_graph,
    const std::string& opname) {
  auto lowering_kernels = HabanaKernel->GetKernels();

  if (syn_graph.is_dynamic_graph()) {
    validateOutputShapeDynamic(HabanaKernel, output_shape_handle, opname);
  } else {
    validateOutputShapeNonDynamic(HabanaKernel, output_shape_handle, opname);
  }
}

void HabanaLaunchOpPT::setSynapsePermuteFlag(
    synapse_helpers::tensor& out_syntensor,
    PtTensorInfoShared& ti,
    IValPtrShared ivpsh) {
  if (out_syntensor.get() == nullptr || out_syntensor.is_dont_allow_permute()) {
    PT_BRIDGE_DEBUG(
        "Not setting synapse allow permutation on tensor: ",
        out_syntensor.id(),
        ", Name:",
        out_syntensor.name(),
        " because of nullptr tensor or specific tensor set with dont_allow_permute");
    return;
  }

  auto rank = out_syntensor.pt_shape().size();
  auto is_partial_view =
      (ivpsh->toTensor().nbytes() != ivpsh->toTensor().storage().nbytes());
  if ((rank >= 2) & !is_partial_view) {
    PT_BRIDGE_DEBUG(
        "Setting synapse allow permutation on tensor: ",
        out_syntensor.id(),
        ", Name:",
        out_syntensor.name());
    synTensorSetAllowPermutation(out_syntensor.get(), 1);
    ti->set_allow_permutation(true);
  } else {
    PT_BRIDGE_DEBUG(
        "Not setting synapse allow permutation on tensor: ",
        out_syntensor.id(),
        ", Name:",
        out_syntensor.name(),
        " because the PT tensor rank is 0D/1D. current rank: ",
        rank);
  }
}

void HabanaLaunchOpPT::ProcessGraphForConstantTensors() {
  PT_BRIDGE_BEGIN;
  for (auto value_input : jit_ir_graph->inputs()) {
    if (value_to_ivalue.find(value_input) == value_to_ivalue.end()) {
      continue;
    }
    if (!value_to_ivalue[value_input]->isTensor()) {
      continue;
    }
    auto tensor = value_to_ivalue[value_input]->toTensor();
    auto is_const_tensor = habana::is_tensor_const(tensor);
    if (is_const_tensor) {
      TensorExtraMeta::set_const_tensor(tensor, true);
    } else {
      continue;
    }
  }
  PT_BRIDGE_END;
}

void HabanaLaunchOpPT::BuildSynapseGraph(
    synapse_helpers::graph& syn_graph,
    bool is_shape_inference) {
  PT_BRIDGE_BEGIN;
  // figure out the right device id
  auto& device = HPURegistrar::get_device();
  synDeviceId device_id = device.id();

  synapse_helpers::detail::tensor_name_generator::reset();

  syn_graph_ptr = &syn_graph;

  if (current_dbipsh_) {
    jit_graph_and_meta_data->clear_cached_graph_info();
    prim_nodes_ival_counter = 0;
    restride_node_swap_counter = 0;
    restride_node_out_val_counter = 0;
  }

  // for each node in IR graph, at this point the graph is a list with nodes
  // topoloically sorted
  // TODO : check if we need to reorder nodes in any case
  torch::jit::graph_node_list graph_nodes = jit_ir_graph->nodes();

  // This is an optimization pass to mark all the nodes with sepcial layout
  // like weights which have HWCK Only activated in lazy mode for now
  bool is_jit_cached_graph_info_available =
      jit_graph_and_meta_data->get_jit_cached_graph_info_available_flag();
  if (is_jit_cached_graph_info_available == false) {
    persistence_marker_pass_data_ptr_ =
        std::move(PersistenceMarkerPass(this).VisitGraph(jit_ir_graph));
  }

  habana::ShapeInference::ResetSifTensorId();

  size_t outputs_metadata_index = 0;
  // Collect inputs shape tensors accross all nodes
  std::vector<size_t> inputs_shape_tensors_vec;
  // Collect intermediate shape tensors accross all nodes for not supporting
  // ComputeOutputShape
  std::vector<size_t> intermediate_shape_tensors_vec;
  int inx = 0;
  for (auto* node : graph_nodes) {
    std::vector<IdxTensorTup> intermediate_shape_tensor_cs;
    watch_tensor_flag_ = false;
    auto node_qual_str = node->kind().toQualString();
    std::string opname(node_qual_str);

    PT_BRIDGE_DEBUG("Working on ", node_qual_str);

    if (watchlist_.empty() || watchlist_.find(opname) != watchlist_.end()) {
      watch_tensor_flag_ = true;
    }

    // TODO: SW-68593 if node is collective add validation that outputs or
    // output duplicates are not used in the graph

    // If its a meta op we need to call the CPU impl and capture changes
    // Only valid for single tensor ops
    // Can we avoid the string match here?
    if (HabanaMetaOpList::isHabanaMetaOp(opname)) {
      handleMetaOps(node);
      continue;
    }

    // Prim nodes require special handling and are a special case
    if (node->kind().is_prim()) {
      handlePrimNodes(node);
      continue;
    }

    if ((strcmp(node_qual_str, "hpu::restride_cl") == 0) ||
        (strcmp(node_qual_str, "hpu::restride") == 0)) {
      bool is_restride_cl =
          (strcmp(node_qual_str, "hpu::restride_cl") == 0) ? true : false;
      handleRestrideNode(node, is_restride_cl);
      continue;
    }

    // Get kernel context
    const auto& op = node->schema().operator_name();
    HabanaOperatorPtr HabanaKernel =
        KernelRegistry().get(device_id, op, getNodeScalarType(node));

    TORCH_CHECK(HabanaKernel, op, " isn't registered in KernelRegistry!");

    if (GET_ENV_FLAG_NEW(PT_HPU_DETERMINISTIC_ENABLE)) {
      // Set the deterministic val
      auto one = torch::jit::attr::alpha;
      PT_BRIDGE_DEBUG("Deterministic value in BuildGraph: ", node->i(one));
      HabanaKernel->setDeterministic(node->i(one));
    }

    // Set kernel execution mode
    HabanaKernel->SetExecutionMode(execution_mode_);

    PT_BRIDGE_DEBUG("Going to add ", *node);

    static std::unordered_set<std::string> jit_ir_ops_;
    if (jit_ir_ops_.count(opname) == 0) {
      PT_DYNAMIC_SHAPE_DEBUG("Invoked_JIT_IR_OP: ", opname);
      jit_ir_ops_.insert(opname);
    }

    static std::unordered_set<std::string> auto_gen_jit_ir_ops_;
    static std::unordered_set<std::string> manual_jit_ir_ops_;
    if (std::dynamic_pointer_cast<OpBackend>(HabanaKernel)) {
      if (auto_gen_jit_ir_ops_.count(opname) == 0) {
        PT_DYNAMIC_SHAPE_DEBUG("Auto_gen_JIT_IR_OP: ", opname);
        auto_gen_jit_ir_ops_.insert(opname);
      }
    } else {
      if (manual_jit_ir_ops_.count(opname) == 0) {
        PT_DYNAMIC_SHAPE_DEBUG("Manual_JIT_IR_OP: ", opname);
        manual_jit_ir_ops_.insert(opname);
      }
    }

    // clear the accumulated synapse node indices corresponding to permute.
    // Otherwise this results in spurious control edges
    syn_graph_ptr->clear_node_indices();
    // set op name in synapse graph
    std::unique_ptr<synapse_helpers::graph::OpNameContext> op_name_context;
    const auto scope = node->scope();
    if (!scope->isBlank()) {
      op_name_context = std::make_unique<synapse_helpers::graph::OpNameContext>(
          syn_graph, scope->name().toUnqualString());
    }

    torch::jit::Stack input_stack = getStackForNode(node);

    // Create/attach the synapse inputs from aten tensors
    GetSynapseInputs(HabanaKernel, node);

    // setup the config params for the kernels
    if (is_jit_cached_graph_info_available == false) {
      auto outputs_metadata = nodeOutputMetaData(node);
      jit_graph_and_meta_data->set_outputs_metadata(outputs_metadata);
    }
    PT_BRIDGE_DEBUG(DumpNodeInputs(node));
    OutputMetaDataVector& outputs_metadata =
        jit_graph_and_meta_data->get_outputs_metadata(outputs_metadata_index);
    outputs_metadata_index++;

    if (node == *graph_nodes.rbegin() && allocated_outputs_.has_value()) {
      HABANA_ASSERT(outputs_metadata.size() == allocated_outputs_->size());
      for (auto [itm, ita] =
               std::tuple{
                   outputs_metadata.begin(), allocated_outputs_->begin()};
           itm != outputs_metadata.end();
           ++itm, ++ita)
        itm->allocated_tensor = *ita;
    }

    std::string module_name = node->scope()->name().toUnqualString();
    if (GET_ENV_FLAG_NEW(PT_HPU_INFERENCE_MODE) &&
        (strcmp(node->kind().toQualString(), "aten::view") == 0)) {
      auto val_ins = node->inputs();
      module_name = val_ins[0]->node()->scope()->name().toUnqualString();
    }
    if (GET_ENV_FLAG_NEW(PT_HPU_INFERENCE_MODE) && module_name.size() > 0) {
      if (((strcmp(node->kind().toQualString(), "aten::add") == 0) ||
           (strcmp(node->kind().toQualString(), "hpu::add") == 0)) &&
          std::string(node->scope()->name().toUnqualString()).find("add") ==
              std::string::npos) {
        module_name = inx == 0 ? std::string(".add")
                               : std::string(".add_") + std::to_string(inx);
        inx++;
      }
      if ((strcmp(node->kind().toQualString(), "hpu::cast") == 0))
        module_name = std::string(node->scope()->name().toUnqualString()) +
            ".placeholder";
      std::replace(module_name.begin(), module_name.end(), '/', '.');
      outputs_metadata.at(0).module_name = module_name;
    }
    // applicable for both persistent strided view and strided insert tensors
    if (((outputs_metadata.size() == 1) && (!is_shape_inference) &&
         (outputs_metadata.at(0).persistent == true)) &&
        ((opname.find("strided_insert") != std::string::npos) ||
         (opname.find("slice_insert") != std::string::npos) ||
         (opname.find("strided_view_out") != std::string::npos))) {
      ProcessStridedInsertAtOutput(
          node, HabanaKernel, input_stack, syn_graph, outputs_metadata);
    } else {
      std::vector<std::tuple<std::vector<int64_t>, std::vector<int64_t>>>
          minmax_list;
      // Currently max update which is less than bucket range issue exists for
      // slice. If other node needs this, can be added here.
      bool needUpdateMinMax =
          ((habana::ShapeInference::GetCurrentPass() ==
            habana::ShapeInfo::InferencePass::MAX_SHAPE) &&
           (habana::ShapeInference::GetMaxPolicyInUse() ==
            habana_helpers::DynamicDimsPolicy::CALCULATED) &&
           strcmp(node->kind().toQualString(), "hpu::slice") == 0);
      if (needUpdateMinMax) {
        minmax_list.resize(
            input_stack.size(),
            std::make_tuple(std::vector<int64_t>(), std::vector<int64_t>()));
        for (int i = 0; i < input_stack.size(); i++) {
          auto& inp = input_stack[i];
          if (inp.isTensor()) {
            minmax_list[i] = habana::ShapeInference::GetMinMaxShape(
                HabanaKernel->SynInput(i).ref().id());
          }
        }
      }

      // Check Compute output shapes for mismatch else raise exception
      if (syn_graph.is_dynamic_graph()) {
        if (auto op = std::dynamic_pointer_cast<OpBackend>(HabanaKernel))
          op->ComputeOutputShapes(input_stack);
      }

      HabanaKernel->AllocateAndAddSynapseNode(
          syn_graph, input_stack, outputs_metadata);
      if (needUpdateMinMax) {
        for (int i = 0; i < input_stack.size(); i++) {
          auto& inp = input_stack[i];
          if (inp.isTensor()) {
            // Check only for max, bucket mismatch issue happens for max only.
            auto new_max = std::get<1>(habana::ShapeInference::GetMinMaxShape(
                HabanaKernel->SynInput(i).ref().id()));
            auto max_from_list = std::get<1>(minmax_list[i]);
            HABANA_ASSERT(new_max.size() == max_from_list.size());
            for (auto j = 0; j < new_max.size(); j++) {
              if (new_max[j] != max_from_list[j]) {
                auto ivalHash = inp.hash().toInt();
                auto input_idx = 0;
                if (m_ival_hash_to_input_index_map.count(ivalHash)) {
                  input_idx = m_ival_hash_to_input_index_map[ivalHash];
                } else {
                  HABANA_ASSERT(
                      0,
                      "NOT found the entry in m_ival_hash_to_input_index_map index:",
                      ivalHash,
                      " total-entries:",
                      m_ival_hash_to_input_index_map.size());
                }
                PT_DYNAMIC_SHAPE_DEBUG(
                    "Need update bucket ",
                    current_bucket_id_,
                    "  oldval:",
                    max_from_list[j],
                    " newval:",
                    new_max[j],
                    " inputIdx:",
                    input_idx,
                    " dimIdx:",
                    j,
                    " current policy:",
                    habana::ShapeInference::GetMaxPolicyInUse());
                {
                  // Update with new shapes
                  std::lock_guard<std::mutex> lg(
                      current_dbipsh_->get_refine_mutex());
                  current_dbipsh_->UpdateShapes(
                      current_bucket_id_, input_idx, j, new_max[j]);
                }
              }
            }
          }
        }
      }
    }

    static std::unordered_set<std::string> cs_jit_ir_ops_;
    static std::unordered_set<std::string> empty_cs_jit_ir_ops_;

    habana::OutputShapeInfRetType kernel_output_cs(true);
    if (!disabled_jit_ir_ops_.count(node_qual_str)) {
      // Either the ComputeOutputShape flow is getting validated or
      // fast shape inference is running for dynamic shapes or
      // shape agnostic flow is enabled for eager.
      if (GET_ENV_FLAG_NEW(PT_HPU_VALIDATE_COMPUTE_SHAPE) ||
          (enable_fast_shape_inf_ && syn_graph_ptr->is_dynamic_graph()) ||
          enable_shape_agnostic_caching_) {
        PT_DYNAMIC_SHAPE_DEBUG(
            "Current sif tensor id = ",
            habana::ShapeInference::GetSifTensorId());
        HabanaOperatorPtr csHabanaKernel =
            KernelRegistry().get(device_id, op, getNodeScalarType(node));

        if (GET_ENV_FLAG_NEW(PT_HPU_DETERMINISTIC_ENABLE)) {
          auto one = torch::jit::attr::alpha;
          PT_BRIDGE_DEBUG("Deterministic value in BuildGraph: ", node->i(one));
          HabanaKernel->setDeterministic(node->i(one));
        }

        // Set output meta data if auto-gen op
        if (auto op = std::dynamic_pointer_cast<OpBackend>(csHabanaKernel)) {
          op->SetOutputMetadata(outputs_metadata);
        }
        kernel_output_cs = csHabanaKernel->ComputeOutputShape(input_stack);
        if (!kernel_output_cs.empty()) {
          // Output shape info based flow
          PT_DYNAMIC_SHAPE_DEBUG(
              "After ComputeOutputShape for ",
              node_qual_str,
              ": sif tensor id = ",
              habana::ShapeInference::GetSifTensorId());
          if (enable_fast_shape_inf_ && !is_shape_inference &&
              syn_graph.is_dynamic_graph()) {
            ProcessShapeTensorsCS(
                kernel_output_cs, intermediate_shape_tensor_cs);
          }
          try {
            validateOutputShape(
                HabanaKernel, kernel_output_cs, syn_graph, opname);
            if (cs_jit_ir_ops_.count(node_qual_str) == 0) {
              PT_DYNAMIC_SHAPE_DEBUG(
                  "ComputeOutputShape_JIT_IR_OP: ", node_qual_str);
              cs_jit_ir_ops_.insert(node_qual_str);
            }
          } catch (std::exception& e) {
            kernel_output_cs.set_empty();
            if (disabled_jit_ir_ops_.count(node_qual_str) == 0) {
              PT_DYNAMIC_SHAPE_DEBUG(
                  "DISABLED_ComputeOutputShape_JIT_IR_OP: ", node_qual_str);
              disabled_jit_ir_ops_.insert(node_qual_str);
            }
            TORCH_CHECK(
                false == GET_ENV_FLAG_NEW(PT_HPU_VALIDATE_COMPUTE_SHAPE),
                "ComputeOutputShape validation failed for op ",
                node_qual_str,
                " what(): ",
                e.what());
          }
        } else {
          if (empty_cs_jit_ir_ops_.count(node_qual_str) == 0) {
            PT_DYNAMIC_SHAPE_DEBUG(
                "Empty_ComputeOutputShape_JIT_IR_OP: ", node_qual_str);
            empty_cs_jit_ir_ops_.insert(node_qual_str);
          }
          TORCH_CHECK(
              false == GET_ENV_FLAG_NEW(PT_HPU_VALIDATE_COMPUTE_SHAPE),
              "ComputeOutputShape method not available for validation of op ",
              node_qual_str);
        }
      }
    }

    jit_to_synapse_node_idx_map.emplace(
        node, syn_graph_ptr->get_node_indices());
    syn_graph_ptr->clear_node_indices();

    if (refine_ds_enabled_ && (!is_shape_inference)) {
      // Process both synapse input and intermediate shape tensors
      std::vector<size_t> intermediate_shape_tensors;
      ProcessSynapseShapeTensors(
          HabanaKernel, intermediate_shape_tensors, inputs_shape_tensors_vec);
      if (enable_fast_shape_inf_ && syn_graph.is_dynamic_graph()) {
        if (!kernel_output_cs.empty()) {
          size_t index = 0;
          HABANA_ASSERT(
              intermediate_shape_tensors.size() ==
                  intermediate_shape_tensor_cs.size(),
              "intermediate_shape_tensors.size=",
              intermediate_shape_tensors.size(),
              " not matching with intermediate_shape_tensor_cs.size=",
              intermediate_shape_tensor_cs.size());
          for (auto& idx : intermediate_shape_tensors) {
            auto tensor_idx =
                std::get<0>(intermediate_shape_tensor_cs[index++]);
            auto ti{shape_tensor_tinfos[idx]};
            auto ret = sif_tidx_to_tinfo_map.insert({tensor_idx, ti});
            if (ret.second) {
              PT_DYNAMIC_SHAPE_DEBUG(
                  "Intermediate shape tensor: adding to sif_tidx_to_tinfo_map : ",
                  tensor_idx,
                  " -> ",
                  *ti);
            } else {
              PT_DYNAMIC_SHAPE_DEBUG(
                  "Intermediate shape tensor: failed adding to sif_tidx_to_tinfo_map : ",
                  tensor_idx,
                  " -> ",
                  *ti);
            }
          }
        } else {
          for (auto& idx : intermediate_shape_tensors) {
            auto ti{shape_tensor_tinfos[idx]};
            PT_DYNAMIC_SHAPE_DEBUG(
                "Intermediate shape tensor: delayed adding to sif_tidx_to_tinfo_map : ",
                *ti);
            intermediate_shape_tensors_vec.emplace_back(idx);
          }
        }
      }
    }

    // Get the output tensors created back from the kernel and do the
    // subsequent processing.
    // We set type so that the created tensor is propagated throughout graph
    auto cur_sif_tidx =
        ProcessSynapseOutputs(HabanaKernel, node, kernel_output_cs);

    // HybridSif specific
    const auto dynamic_compile_graph = refine_ds_enabled_ &&
        enable_fast_shape_inf_ && syn_graph.is_dynamic_graph() &&
        !is_shape_inference;
    if ((dynamic_compile_graph || enable_shape_agnostic_caching_) &&
        kernel_output_cs.empty()) {
      // Increment the sif tensor id
      auto output_count = get_output_tensors_count(HabanaKernel, syn_graph);
      habana::ShapeInference::IncrementSifTensorId(output_count);
      PT_DYNAMIC_SHAPE_DEBUG(
          "After increment: sif tensor id = ",
          habana::ShapeInference::GetSifTensorId(),
          " should match with ProcessSynapseOutputs return value = ",
          cur_sif_tidx);
    }

    PT_BRIDGE_DEBUG(DumpNodeOutputs(node));
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

          PtTensorInfoShared ti = std::make_shared<PtTensorInfo>(
              tensor, tensor_name, irn, watch_tensor_flag_, tensor_id);
          auto& ivpsh = it->second;
          ivalue_to_tensor_info_map[ivpsh] = ti;

          if (enable_caching_ || enable_shape_agnostic_caching_) {
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
              opname,
              " as persistent intermediate");
        }
      }
    }

    if (!is_shape_inference && habana_helpers::IsCollective(node->kind())) {
      // save indexes of kernel input stack in graph input stack
      // when launching provide new input stack to RunCollective
      std::shared_ptr<habana_helpers::collective_kernel_info> kernel_info =
          std::make_shared<habana_helpers::collective_kernel_info>();

      auto node_inputs = node->inputs();
      for (auto input : node_inputs) {
        auto ivalptr = value_to_ivalue.at(input);
        if (ivalptr->isTensor()) {
          PtTensorInfoShared ti = ivalue_to_tensor_info_map.at(ivalptr);
          kernel_info->input_tensor_infos.push_back(ti);
        } else {
          kernel_info->input_tensor_infos.push_back(nullptr);
        }
      }

      auto node_outputs = node->outputs();
      for (auto output : node_outputs) {
        auto ivalptr = value_to_ivalue.at(output);
        if (ivalptr->isTensor()) {
          PtTensorInfoShared ti = ivalue_to_tensor_info_map.at(ivalptr);
          kernel_info->output_tensor_infos.push_back(ti);
        } else {
          kernel_info->output_tensor_infos.push_back(nullptr);
        }
      }
      kernel_info->kernel =
          std::dynamic_pointer_cast<CollectiveOperator>(HabanaKernel);
      collective_kernels_info.push_back(kernel_info);
    }

    // Adding to a vector as we share context through shared pointers and we
    // dont want to call delete untill we are done with whole graph
    habana_kernels.push_back(HabanaKernel);
  }

  // Generate patching info for graph inputs during fast sif
  if ((refine_ds_enabled_ && enable_fast_shape_inf_ &&
       syn_graph.is_dynamic_graph() && !is_shape_inference)) {
    for (size_t i = 0; i < jit_ir_graph->inputs().size(); ++i) {
      auto input = jit_ir_graph->inputs().at(i);
      HABANA_ASSERT(value_to_ivalue.count(input));
      auto input_ivalue = value_to_ivalue[input];
      HABANA_ASSERT(ivalue_to_tensor_info_map.count(input_ivalue));
      auto tensor_idx = habana::ShapeInference::ReadAndIncrementSifTensorId();
      auto ret = sif_tidx_to_tinfo_map.insert(
          {tensor_idx, ivalue_to_tensor_info_map[input_ivalue]});
      if (ret.second) {
        PT_DYNAMIC_SHAPE_DEBUG(
            "Input tensor: adding to sif_tidx_to_tinfo_map : ",
            tensor_idx,
            " -> ",
            *ivalue_to_tensor_info_map[input_ivalue]);
      } else {
        PT_DYNAMIC_SHAPE_DEBUG(
            "Input tensor: failed adding to sif_tidx_to_tinfo_map : ",
            tensor_idx,
            " -> ",
            *ivalue_to_tensor_info_map[input_ivalue]);
      }
    }

    // Generate patching info for inputs shape tensors during fast sif
    for (auto const& idx : inputs_shape_tensors_vec) {
      auto tensor_idx = habana::ShapeInference::ReadAndIncrementSifTensorId();
      auto ret =
          sif_tidx_to_tinfo_map.insert({tensor_idx, shape_tensor_tinfos[idx]});
      if (ret.second) {
        PT_DYNAMIC_SHAPE_DEBUG(
            "Input shape tensor: adding to sif_tidx_to_tinfo_map : ",
            tensor_idx,
            " -> ",
            *shape_tensor_tinfos[idx]);
      } else {
        PT_DYNAMIC_SHAPE_DEBUG(
            "Input shape tensor: failed adding to sif_tidx_to_tinfo_map : ",
            tensor_idx,
            " -> ",
            *shape_tensor_tinfos[idx]);
      }
    }

    // Generate patching info for intermediate shape tensors for nodes not
    // supporting ComputeOutputShape during fast sif
    for (auto const& idx : intermediate_shape_tensors_vec) {
      auto tensor_idx = habana::ShapeInference::ReadAndIncrementSifTensorId();
      auto ret =
          sif_tidx_to_tinfo_map.insert({tensor_idx, shape_tensor_tinfos[idx]});
      if (ret.second) {
        PT_DYNAMIC_SHAPE_DEBUG(
            "Intermediate shape tensor: adding to sif_tidx_to_tinfo_map : ",
            tensor_idx,
            " -> ",
            *shape_tensor_tinfos[idx]);
      } else {
        PT_DYNAMIC_SHAPE_DEBUG(
            "Intermediate shape tensor: failed adding to sif_tidx_to_tinfo_map : ",
            tensor_idx,
            " -> ",
            *shape_tensor_tinfos[idx]);
      }
    }
  }

  // allow permutation only for output tensors
  if (GET_ENV_FLAG_NEW(PT_HPU_ENABLE_SYNAPSE_OUTPUT_PERMUTE) &&
      !is_hccl_send_mark_step()) {
    for (auto ti : output_tensorinfo_map) {
      auto ival = ti.first;
      auto iter = pt_to_synapse_tensors.find(ival);
      HABANA_ASSERT(pt_to_synapse_tensors.count(ival));
      if (iter != pt_to_synapse_tensors.end()) {
        auto syn_vec = (iter->second);
        auto& out_syntensor = (*syn_vec)[0];
        if (out_syntensor.ref().get() == nullptr) {
          PT_BRIDGE_DEBUG(
              "Not setting synapse allow permutation on tensor: ",
              out_syntensor.ref().id(),
              " because of nullptr tensor");
          continue;
        }
        if (iter->second->size() != 1) {
          PT_BRIDGE_DEBUG(
              "Not setting synapse allow permutation on tensor: ",
              out_syntensor.ref().id(),
              " because the PT tensor is mapped to multiple synapse tensors");
          continue;
        }
        if (out_syntensor.ref().pt_shape().size() < 2) {
          PT_BRIDGE_DEBUG(
              "Not setting synapse allow permutation on tensor: ",
              out_syntensor.ref().id(),
              " because it is 0D/1D");
          continue;
        }
        auto& t = ival->toTensor();
        if (t.nbytes() != t.storage().nbytes()) {
          PT_BRIDGE_DEBUG(
              "Not setting synapse allow permutation on tensor: ",
              out_syntensor.ref().id(),
              " because it is a view");
          continue;
        }
        if (out_syntensor.ref().is_dont_allow_permute()) {
          PT_BRIDGE_DEBUG(
              "Not setting synapse allow permutation on tensor: ",
              out_syntensor.ref().id(),
              " because tensor specific set with dont_allow_permute");
          continue;
        }
        PT_BRIDGE_DEBUG(
            "Setting synapse allow permutation on tensor: ",
            out_syntensor.ref().id());
        synTensorSetAllowPermutation(out_syntensor.ref().get(), 1);
        ti.second->set_allow_permutation(true);
      }
    }
    for (auto ti : duplicate_input_to_outtinfo_map) {
      auto ival = ti.first;
      auto iter = pt_to_synapse_tensors.find(ival);
      HABANA_ASSERT(pt_to_synapse_tensors.count(ival));
      if (iter != pt_to_synapse_tensors.end()) {
        // TODO refactor the below code to remove code duplication
        auto syn_vec = (iter->second);
        auto& out_syntensor = (*syn_vec)[0];
        if (out_syntensor.ref().get() == nullptr) {
          PT_BRIDGE_DEBUG(
              "Not setting synapse allow permutation on tensor: ",
              out_syntensor.ref().id(),
              " because of nullptr tensor");
          continue;
        }
        if (iter->second->size() != 1) {
          PT_BRIDGE_DEBUG(
              "Not setting synapse allow permutation on tensor: ",
              out_syntensor.ref().id(),
              " because the PT tensor is mapped to multiple synapse tensors");
          continue;
        }
        if (out_syntensor.ref().pt_shape().size() < 2) {
          PT_BRIDGE_DEBUG(
              "Not setting synapse allow permutation on tensor: ",
              out_syntensor.ref().id(),
              " because it is 0D/1D");
          continue;
        }
        auto& t = ival->toTensor();
        if (t.nbytes() != t.storage().nbytes()) {
          PT_BRIDGE_DEBUG(
              "Not setting synapse allow permutation on tensor: ",
              out_syntensor.ref().id(),
              " because it is a view");
          continue;
        }
        if (out_syntensor.ref().is_dont_allow_permute()) {
          PT_BRIDGE_DEBUG(
              "Not setting synapse allow permutation on tensor: ",
              out_syntensor.ref().id(),
              " because tensor specific set with dont_allow_permute");
          continue;
        }
        PT_BRIDGE_DEBUG(
            "Setting synapse allow permutation on tensor: ",
            out_syntensor.ref().id());
        synTensorSetAllowPermutation(out_syntensor.ref().get(), 1);
        ti.second->set_allow_permutation(true);
      }
    }
  }
  PT_BRIDGE_END;
}

void HabanaLaunchOpPT::CreateDynamicBucketInputShapes(
    habana_helpers::InpTensorShapes& shape_map) {
  for (size_t i = 0; i < input_refs.size(); i++) {
    auto& input = input_refs[i];
    if (input.isTensor()) {
      at::Tensor pt_tensor = input.toTensor();
      habana_helpers::TensorShape shape(
          pt_tensor.sizes(), pt_tensor.scalar_type());
      auto tmeta = get_tensor_extra_meta(pt_tensor);
      shape.set_tensor_type(tmeta->get_tensor_type());
      shape_map[i] = shape;
    }
  }
}

void HabanaLaunchOpPT::ProcessDynamicBucketInputShapesWithH2D(
    habana_helpers::InpTensorShapes& shape_map) {
  for (size_t i = 0; i < input_refs.size(); i++) {
    auto input = input_refs[i];
    if (input.isTensor()) {
      at::Tensor pt_tensor = input.toTensor();

      auto tmeta = get_tensor_extra_meta(pt_tensor);
      if (tmeta->get_tensor_type() == HOST_TO_DEVICE_TENSOR &&
          tmeta->peek_H2D_data_for_bucketing()) {
        size_t h2d_size = tmeta->get_host_size();

        std::vector<int64_t> h2d_vec;
        habana::HostDataType h2d_dt_type = tmeta->get_host_dt_type();
        if (h2d_dt_type == habana::HostDataType::INT32_T) {
          int32_t* h2d_data = static_cast<int32_t*>(tmeta->get_host_ptr());
          for (size_t i = 0; i < h2d_size; i++) {
            h2d_vec.push_back(static_cast<int64_t>(*h2d_data++));
          }
        } else if (h2d_dt_type == habana::HostDataType::UINT32_T) {
          uint32_t* h2d_data = static_cast<uint32_t*>(tmeta->get_host_ptr());
          for (size_t i = 0; i < h2d_size; i++) {
            h2d_vec.push_back(static_cast<int64_t>(*h2d_data++));
          }
        } else if (h2d_dt_type == habana::HostDataType::UINT64_T) {
          uint64_t* h2d_data = static_cast<uint64_t*>(tmeta->get_host_ptr());
          for (size_t i = 0; i < h2d_size; i++) {
            uint64_t h2d_elem = *h2d_data++;
            TORCH_CHECK(
                h2d_elem < LONG_MAX,
                "H2D data ",
                h2d_elem,
                " exceeds the int64 limit");
            h2d_vec.push_back(static_cast<int64_t>(h2d_elem));
          }
        } else {
          PT_DYNAMIC_SHAPE_DEBUG(
              "Host datatype Not Supported while processing host data from bucketing");
        }

        habana_helpers::TensorShape shape(h2d_vec, pt_tensor.scalar_type());
        shape.set_tensor_type(tmeta->get_tensor_type());
        shape_map[i] = shape;
      }
    }
  }
}

void HabanaLaunchOpPT::CreateStaticComplationDBI(size_t graph_key_with_perm) {
  std::string path = GET_ENV_FLAG_NEW(PT_COMPILATION_STATS_PATH);

  if (!ref_input_shape_map.count(graph_key_with_perm)) {
    habana_helpers::InpTensorShapes input_tshapes;
    CreateDynamicBucketInputShapes(input_tshapes);
    ProcessDynamicBucketInputShapesWithH2D(input_tshapes);
    PT_BRIDGE_DEBUG(
        "JIT IR graph_hash_code : ",
        graph_key,
        ", hash_code with data layout : ",
        graph_key_with_perm,
        "\nRecording the reference input shapes::",
        input_tshapes,
        "\n--------------------");
    ref_input_shape_map.emplace(graph_key_with_perm, input_tshapes);
    if (path != "") {
      CreateFirstDynamicBucket();
      DumpStaticCompilationStatistics(graph_key_with_perm, true);
    }
  } else if (path != "") {
    DumpStaticCompilationStatistics(graph_key_with_perm);
  }
}

void HabanaLaunchOpPT::CreateValueToIvalueMapForInputs() {
  PT_BRIDGE_BEGIN;
  for (size_t j = 0; j < pt_stack_sh.size(); j++) {
    auto value_input = jit_ir_graph->inputs().at(j);
    auto ivpsh = pt_stack_sh[j];
    value_to_ivalue[value_input] = ivpsh;
    m_ival_hash_to_input_index_map[ivpsh->hash().toInt()] = j;
  }
  PT_BRIDGE_END;
}

void HabanaLaunchOpPT::SetH2DMinMaxData(
    const torch::jit::Stack& stack,
    habana_helpers::InpTensorShapes& dynamic_shapes,
    const ShapeInfo::InferencePass& pass) {
  PT_BRIDGE_BEGIN;
  for (size_t i = 0; i < stack.size(); ++i) {
    if (dynamic_shapes.count(i)) {
      auto& tensor = stack[i].toTensor();
      auto tmeta = get_tensor_extra_meta(tensor);
      if (tmeta->get_tensor_type() == HOST_TO_DEVICE_TENSOR &&
          tmeta->peek_H2D_data_for_bucketing()) {
        habana::HostDataType h2d_dtype = tmeta->get_host_dt_type();
        if (h2d_dtype == habana::HostDataType::UINT64_T) {
          std::vector<uint64_t> stride_data_vec;
          std::vector<int64_t> h2d_sif_data = dynamic_shapes.at(i).get_dims();
          for (auto it = h2d_sif_data.begin(); it != h2d_sif_data.end(); ++it) {
            stride_data_vec.push_back(static_cast<uint64_t>(*it));
          }
          if (pass == ShapeInfo::InferencePass::MIN_SHAPE) {
            tmeta->set_min<uint64_t>(stride_data_vec);
          } else {
            tmeta->set_max<uint64_t>(stride_data_vec);
          }
        } else if (h2d_dtype == habana::HostDataType::UINT32_T) {
          std::vector<uint32_t> data_vec;
          std::vector<int64_t> h2d_sif_data = dynamic_shapes.at(i).get_dims();
          for (auto it = h2d_sif_data.begin(); it != h2d_sif_data.end(); ++it) {
            data_vec.push_back(static_cast<uint32_t>(*it));
          }
          if (pass == ShapeInfo::InferencePass::MIN_SHAPE) {
            tmeta->set_min<uint32_t>(data_vec);
          } else {
            tmeta->set_max<uint32_t>(data_vec);
          }
        } else if (h2d_dtype == habana::HostDataType::INT32_T) {
          std::vector<int32_t> data_vec;
          std::vector<int64_t> h2d_sif_data = dynamic_shapes.at(i).get_dims();
          for (auto it = h2d_sif_data.begin(); it != h2d_sif_data.end(); ++it) {
            data_vec.push_back(static_cast<int32_t>(*it));
          }
          if (pass == ShapeInfo::InferencePass::MIN_SHAPE) {
            tmeta->set_min<int32_t>(data_vec);
          } else {
            tmeta->set_max<int32_t>(data_vec);
          }
        } else {
          PT_DYNAMIC_SHAPE_DEBUG(
              "Host datatype Not Supported while setting Min/Max data");
        }
      }
    }
  }
  PT_BRIDGE_END;
}

torch::jit::Stack HabanaLaunchOpPT::CreateStack(
    const torch::jit::Stack& stack,
    habana_helpers::InpTensorShapes& dynamic_shapes) {
  PT_BRIDGE_BEGIN;
  torch::jit::Stack new_stack;

  for (size_t i = 0; i < stack.size(); ++i) {
    if (dynamic_shapes.count(i)) {
      auto& tensor = stack[i].toTensor();
      auto tmeta = get_tensor_extra_meta(tensor, true);
      //
      // TODO: When creating a new stack, we need to look, if this
      // can be done using storage less pytorch tensor, need to fix
      // this
      synTensorType tensor_type = DATA_TENSOR;
      if (tmeta) {
        tensor_type = tmeta->get_tensor_type();
        // to empty_hpu_lazy
      }

      at::Tensor new_tensor;
      if (tensor_type == HOST_TO_DEVICE_TENSOR &&
          tmeta->peek_H2D_data_for_bucketing()) {
        new_tensor = habana_lazy::empty_hpu_lazy(
            tensor.sizes(),
            tensor.options(),
            tensor.suggest_memory_format(),
            true,
            tensor_type);
      } else {
        new_tensor = habana_lazy::empty_hpu_lazy(
            dynamic_shapes.at(i).get_dims(),
            tensor.options(),
            tensor.suggest_memory_format(),
            true,
            tensor_type);
      }
      /*
       * Every new tensor is created using Habana Tensor Implementer.
       * Ensure propogation of shape tensor information for the new
       * tensor created for the stack.
       */
      auto new_tmeta = get_tensor_extra_meta(new_tensor);
      if (tmeta->get_shape_struct().has_shape_tensor_data()) {
        new_tmeta->get_shape_struct() = tmeta->get_shape_struct();
      }
      new_tmeta->set_compile_host_ptr(tmeta);

      if (tmeta) {
        new_tmeta->set_tensor_type(tmeta->get_tensor_type());
      }
      if (tensor_type == HOST_TO_DEVICE_TENSOR &&
          tmeta->peek_H2D_data_for_bucketing()) {
        new_tmeta->set_H2D_data_for_bucketing();
        new_tmeta->set_host_data(
            tmeta->get_host_ptr(),
            tmeta->get_host_size(),
            tmeta->get_host_el_size(),
            tmeta->get_host_dt_type());
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
  PT_BRIDGE_BEGIN;
  // Initiate recipe execution time collection
  if (current_dbipsh_->NeedRunTimeSlot(current_bucket_id_)) {
    auto& syn_device = HPURegistrar::get_device();
    auto& time_event_handle_cache = syn_device.get_time_event_handle_cache();
    if (time_event_handle_cache.get_total_events_count() <
        synapse_helpers::event_handle_cache::get_num_events_high_watermark()) {
      rv.time_slot_ = std::make_shared<synapse_helpers::TimeSlot>(
          syn_device.get_cached_time_event_handle(),
          syn_device.get_cached_time_event_handle(),
          static_cast<synStreamHandle>(syn_device.get_stream(hpu_stream)));
      current_dbipsh_->RegisterTimeSlot(rv.time_slot_, current_bucket_id_);
    } else {
      PT_BRIDGE_WARN(
          "High water mark for synapse events ",
          synapse_helpers::event_handle_cache::get_num_events_high_watermark(),
          " reached, will not create any time event");
      rv.time_slot_ = nullptr;
    }
  }
  PT_BRIDGE_END;
}

void HabanaLaunchOpPT::EvictSynapseRecipe(size_t& dsi_bucket_id) {
  size_t num_recipes = 1;
  bool dropped{true};
  // Keep evicting recipes until the memory usage goes below threshold
  if (habana::IsHostMemoryThresholdReached()) {
    // Remove in chunks of 512MB
    int64_t eviction_threshold_left = 512 * 1024 * 1024;
    while (dropped && eviction_threshold_left > 0) {
      dropped = dropCachedRecipe_LRU(num_recipes);
      if (dropped) {
        auto dropped_arg = RecipeCacheLRU::get_cache().dropped_recipe.first;
        auto dropped_val = RecipeCacheLRU::get_cache().dropped_recipe.second;
        // Update the eviction threshold left after removing this recipe
        eviction_threshold_left -=
            dropped_val->recipe->get_recipe_host_mem_size();
        auto dropped_dbi =
            DynamicBucketInfoMap::get_instance().get(dropped_arg);
        if (dropped_dbi != nullptr) {
          static_cast<void>(dsi_bucket_id);
          dropped_dbi->ResetSynapseRecipePtr(dropped_val);
        }
      }
    }
    // Call TcMalloc extension to release memory
    synapse_helpers::ReleaseFreeMemory();
  }
}

void HabanaLaunchOpPT::CreateFirstDynamicBucket() {
  RecipeCacheLRU::SetHostMemoryThreshold();

  std::shared_ptr<RecipeArgumentSpec> rargpsh_graph =
      std::make_shared<RecipeArgumentSpec>(input_refs, graph_key, op_strs);

  current_dbipsh_ = DynamicBucketInfoMap::get_instance().get(rargpsh_graph);
  if (nullptr == current_dbipsh_) {
    PT_BRIDGE_DEBUG(
        "====================\n",
        "Creating first dynamic bucket info \n",
        "JIT IR graph_hash_code : ",
        rargpsh_graph->graphHashCode(),
        ", hash_code with data layout : ",
        rargpsh_graph->hashCode());
    PT_DYNAMIC_SHAPE_DEBUG("Creating new DynamicBucketInfo");
    current_dbipsh_ = std::make_shared<habana_helpers::DynamicBucketInfo>(
        rargpsh_graph->graphHashCode());
    DynamicBucketInfoMap::get_instance().add(rargpsh_graph, current_dbipsh_);
    current_dbipsh_->create_statistics(
        habana_helpers::CompilationStatistics::Create(
            GetSynapseGraphName(), current_dbipsh_->getCount()));

    // Create bucket 0
    DynamicShapeInfo graph_input_info;
    graph_input_info.act_input_tshapes =
        ref_input_shape_map.at(rargpsh_graph->hashCode());
    PT_DYNAMIC_SHAPE_DEBUG(
        "Reference input shapes::",
        graph_input_info.act_input_tshapes,
        "\n--------------------");

    habana_helpers::ResultShapes ranges;
    size_t ref_bucket_id{};
    {
      std::lock_guard<std::mutex> lg(current_dbipsh_->get_refine_mutex());
      current_dbipsh_->CollectDynamicDims(graph_input_info.act_input_tshapes);
      ref_bucket_id =
          current_dbipsh_->GetBucketId(graph_input_info.act_input_tshapes);
      ranges = current_dbipsh_->CalculateShapes(ref_bucket_id);
    }
    auto start_ds_token = current_dbipsh_->GetTokenForBucketId(ref_bucket_id);
    PT_DYNAMIC_SHAPE_DEBUG(
        "for reference shape creating bucket id : ",
        ref_bucket_id,
        " with token ",
        start_ds_token);
  }
}

void HabanaLaunchOpPT::ProcessHabanaFusedOpWithDS() {
  PT_BRIDGE_BEGIN;

  std::shared_ptr<RecipeArgumentSpec> rargpsh_graph =
      std::make_shared<RecipeArgumentSpec>(input_refs, graph_key, op_strs);

  PT_DYNAMIC_SHAPE_DEBUG(
      "====================\n",
      "Processing with dynamic shape enabled\n",
      "JIT IR graph_hash_code : ",
      rargpsh_graph->graphHashCode(),
      ", hash_code with data layout : ",
      rargpsh_graph->hashCode());

  CreateFirstDynamicBucket();

  DynamicShapeInfo graph_input_info;
  CreateDynamicBucketInputShapes(graph_input_info.act_input_tshapes);
  ProcessDynamicBucketInputShapesWithH2D(graph_input_info.act_input_tshapes);
  PT_DYNAMIC_SHAPE_DEBUG(
      "Input shapes::",
      graph_input_info.act_input_tshapes,
      "\n--------------------");

  habana_helpers::ResultShapes ranges;
  {
    std::lock_guard<std::mutex> lg(current_dbipsh_->get_refine_mutex());
    current_dbipsh_->CollectDynamicDims(graph_input_info.act_input_tshapes);
    current_bucket_id_ =
        current_dbipsh_->GetBucketId(graph_input_info.act_input_tshapes);
    ranges = current_dbipsh_->CalculateShapes(current_bucket_id_);
  }

  cur_ds_token_ = current_dbipsh_->GetTokenForBucketId(current_bucket_id_);

  PT_DYNAMIC_SHAPE_DEBUG(
      jit_ir_graph->toString(), "current bucket id : ", current_bucket_id_);
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
  }
  graph_input_info.current_bucket_id = current_bucket_id_;
  graph_input_info.min_policy = current_dbipsh_->GetMinPolicy();
  graph_input_info.max_policy = current_dbipsh_->GetMaxPolicy();
  habana::ShapeInference::SetMinMaxPolicyInUse(
      graph_input_info.min_policy, graph_input_info.max_policy);

  cur_rargpsh = std::make_shared<RecipeArgumentSpec>(
      input_refs, graph_key, op_strs, cur_ds_token_);
  PT_DYNAMIC_SHAPE_DEBUG("cur_rargpsh = ", *cur_rargpsh);
  DynamicBucketInfoMap::get_instance().add(cur_rargpsh, current_dbipsh_);
  // Used only for compilation statistics purpose now
  current_dbipsh_->SetLastUsedStepForBucket(
      current_bucket_id_, current_dbipsh_->get_statistics()->GetCurrentStep());

  // Check for cached recipe
  if (enable_graph_caching_) {
    current_dbipsh_->SetRecipeKeyForBucket(
        graph_input_info.current_bucket_id, cur_rargpsh->hashCode());
    cur_rvalpsh = GetCachedRecipe(cur_rargpsh);

    if (ABSL_PREDICT_TRUE(cur_rvalpsh)) {
      // Cache hit for a dynamic bucket
      // Steps:
      // 1. Infer shapes of all persistent tensors which are not input
      // 2. Patch using the exact shape
      // 3. Launch
      // 4. Update outputs
      current_dbipsh_->IncrementHitCount(current_bucket_id_);
      if (cur_rvalpsh->get_refined()) {
        habana_helpers::DynamicBucketInfo::inc_num_refined_recipe_hits();
        if (cur_rvalpsh->get_refined_wirt()) {
          habana_helpers::DynamicBucketInfo::inc_num_refined_recipe_wirt_hits();
        }
      } else {
        habana_helpers::DynamicBucketInfo::inc_num_original_recipe_hits();
      }

      std::unordered_map<int64_t, at::Tensor> tidx_to_tensor_map;
      RecipeValueSpec& rv = *cur_rvalpsh;
      rv.update_hit_count();

      if (rv.dynamic_graph) {
        // For Dynamic shapes in case of cache hit, we need to run
        // shape inference for determining the output shape and
        // persistent intermediates
        PT_DYNAMIC_SHAPE_DEBUG(
            "Graph: ",
            name,
            '_',
            graph_index,
            ", graph_key: ",
            rargpsh_graph->graphHashCode(),
            ", recipe cache hit, recipe_key: ",
            cur_rargpsh->hashCode());
        PT_DYNAMIC_SHAPE_DEBUG("Running output shape inference pass");
        if (enable_fast_shape_inf_ && GET_ENV_FLAG_NEW(PT_HPU_RUN_HYBRID_SIF)) {
          PT_DYNAMIC_SHAPE_DEBUG(
              "Graph: ",
              name,
              '_',
              graph_index,
              ", graph_key: ",
              rargpsh_graph->graphHashCode(),
              ", recipe cache hit, recipe_key: ",
              cur_rargpsh->hashCode(),
              "HybridSif_BEGIN");

          habana::ShapeInference::ResetSifTensorId();
          constexpr bool dynamic_shapes_true = true;
          RunHybridSif<dynamic_shapes_true>(tidx_to_tensor_map);
          PT_DYNAMIC_SHAPE_DEBUG("HybridSif_END");
        } else {
          PT_DYNAMIC_SHAPE_DEBUG("OutputSif_BEGIN");
          try_run_shape_inference(
              ShapeInfo::InferencePass::OUTPUT_SHAPE, graph_input_info);
          PT_DYNAMIC_SHAPE_DEBUG("OutputSif_END");
        }
        current_dbipsh_->SetInputMetaData(*pt_stack, current_bucket_id_);
        bool refine_candidate =
            (current_dbipsh_->GetMFUBucket() == current_bucket_id_);
        current_dbipsh_->get_statistics()->LogUsedBucket(
            current_bucket_id_, jit_ir_graph, ranges, refine_candidate);
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
          m_map_shape.m_actual_shapes,
          tidx_to_tensor_map,
          allocated_outputs_);

      if (enable_tensor_dump_) {
        DumpTensors_pre(rv);
      }

      {
        std::lock_guard<std::mutex> lg(current_dbipsh_->get_refine_mutex());
        // Initiate recipe execution time collection
        if (GET_ENV_FLAG_NEW(PT_ENABLE_SYNLAUNCH_TIME_CAPTURE)) {
          InitiateSynlaunchTimeCapture(rv);
        }
      }

      if (!dry_run_) {
        rv.launch(
            hpu_stream, input_refs, intermediate_tensors_ptr, dma_inputs_ptr);
      }
      if (enable_tensor_dump_) {
        DumpTensors(rv);
      }

      // Update the stack from the recipe itself
      UpdateOutputs(rv);
      ReturnCachedRecipe(rv);

      RefinementEngine::GetEngine().AddGraphKey(rargpsh_graph->graphHashCode());
      PT_DYNAMIC_SHAPE_DEBUG(
          current_dbipsh_->digest_str(), current_dbipsh_->history_str());
      PT_IRGRAPH_DEBUG("HabanaOp recipe cache hit :: dynamic shapes");

      current_dbipsh_->get_statistics()->LogSelectedRecipe(
          cur_rargpsh->hashCode(), 0);
      current_dbipsh_->get_statistics()->LogShapes(
          jit_ir_graph, graph_input_info.act_input_tshapes);

      auto t_ns_base{current_dbipsh_->GetTimeBase(current_bucket_id_)};
      auto t_ns{current_dbipsh_->GetTime(current_bucket_id_)};

      current_dbipsh_->get_statistics()->LogLaunchBase(t_ns_base, 0);
      current_dbipsh_->get_statistics()->LogLaunch(t_ns, 0);
      current_dbipsh_->get_statistics()->LogLaunchPerf(t_ns_base, t_ns, 0);
      if (t_ns && t_ns_base) {
        habana_helpers::DynamicBucketInfo::update_improvement_map(
            cur_rargpsh->hashCode(), (t_ns < t_ns_base));
      }
      current_dbipsh_->get_statistics()->GetDigest(
          cur_rargpsh->graphHashCode(),
          current_bucket_id_,
          cur_ds_token_,
          cur_rargpsh->hashCode(),
          true);

      current_dbipsh_->get_statistics()->DumpAndNextStep();
      ClearMembers();
      ClearStatics();

      PT_BRIDGE_END;
      return;
    } else {
      PT_DYNAMIC_SHAPE_DEBUG(
          "HabanaOp recipe cache miss :: key ", cur_rargpsh->hashCode());
      PT_IRGRAPH_DEBUG("HabanaOp recipe cache miss :: dynamic shapes");
    }
  }

  CompileAndRunDynamicGraph(graph_input_info);
  m_ival_hash_to_input_index_map.clear();

  habana_helpers::DynamicBucketInfo::inc_original_recipe_count();
  current_dbipsh_->get_statistics()->GetDigest(
      cur_rargpsh->graphHashCode(),
      current_bucket_id_,
      cur_ds_token_,
      cur_rargpsh->hashCode(),
      false);

  current_dbipsh_->get_statistics()->DumpAndNextStep();
  PT_BRIDGE_END;
}

void HabanaLaunchOpPT::PrintRecipeInputs() {
  std::ostream& O = std::cout;

  O << "aten_inputs #" << num_inputs << "::" << '\n';
  size_t idx{0};
  for (size_t i = pt_stack_sh.size() - num_inputs; i < pt_stack_sh.size();
       i++) {
    auto vp = jit_ir_graph->inputs().at(i);
    O << idx++ << " : %" << vp->debugName() << " : ";
    habana_helpers::DebugString(pt_stack_sh.at(i));
  }
}

void RecipeValueSpec::create_outdup(
    size_t ti_idx,
    std::unordered_map<size_t, IValPtrShared>& parent_ivpsh_map,
    std::string map_name,
    bool is_shape_agnostic_graph) {
  // The aten_output_num is the total number of outputs
  size_t aten_output_num = num_outputs + num_input_to_outduplicates +
      num_intermediate_to_outduplicates + num_output_to_outduplicates;
  PtTensorInfo& ti = *(dtensorinfos->at(ti_idx));
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

  at::Tensor pt_outdup;

  // To Do - to handle view along with new view implementation
  // for eager mode shape agnostic
  if (is_shape_agnostic_graph) {
    pt_outdup = parent_tensor;
  } else {
    if ((parent_tensor.sizes() == pt_sizes) &&
        (parent_tensor.strides() == pt_strides)) {
      // inplace op
      pt_outdup = parent_tensor;
    } else {
      // view
      pt_outdup =
          at::as_strided(parent_tensor, pt_sizes, pt_strides, pt_opt_offset);
    }
  }

  if (!ti.get_allow_permutation()) {
    habana_helpers::set_tensor_memory_permutations(pt_outdup, {});
    PT_BRIDGE_DEBUG(
        "Resetting tensor ",
        ti.get_tensor_id(),
        " permutation because it is not allowed permutation (cache hit flow)");
  } else {
    PT_BACKEND_DEBUG_TENSOR(
        pt_outdup,
        "Setting tensor %d "
        " permutation from the TensorInfo cache record: %s"
        " old permutation was: %s",
        ti.get_tensor_id(),
        VecToString(ti.getHbInternalPermute()),
        habana_helpers::FormatTokens::Permutations);
    habana_helpers::set_tensor_memory_permutations(
        pt_outdup, ti.getHbInternalPermute());
  }
  PT_BACKEND_DEBUG_TENSOR(
      pt_outdup,
      " duplicate output HbInternal address : %s  storage address : %s",
      habana_helpers::FormatTokens::ImplPtr,
      habana_helpers::FormatTokens::DataPtr);
  ti.patch(pt_outdup);

  IValPtrShared ivpsh = std::make_shared<IVal>(pt_outdup);
  aten_outputs->at(output_idx) = ivpsh;
}

void HabanaLaunchOpPT::ReturnCachedRecipe(RecipeValueSpec& rv) {
  PT_BRIDGE_BEGIN;
  rv.set_use_flag(false);
  PT_BRIDGE_END;
}

// shape agnostic : duplicate synapse graph
void HabanaLaunchOpPT::DuplicateSynapseGraph() {
  // first duplicate call to populate number of tensors and nodes in the
  // graph
  syn_graph_ptr->duplicate(nullptr, nullptr);

  std::vector<synTensorHandleMap> tensorsMap(
      syn_graph_ptr->get_num_of_tensors());
  std::vector<synNodeHandleMap> nodesMap(syn_graph_ptr->get_num_of_nodes());

  // second duplicate call to to get the tensor and nodes map
  syn_graph_ptr->duplicate(tensorsMap.data(), nodesMap.data());

  MaybePrintDuplicateGraphInformation(
      syn_graph_ptr, tensorsMap, nodesMap, "cache miss");
}

// shape agnostic : store shape agnostic graph
void HabanaLaunchOpPT::StoreShapeAgnosticGraph() {
  cur_rvalpsh->shape_agnostic_synapse_graph_ =
      std::make_shared<synapse_helpers::graph>(std::move(*syn_graph_ptr));
  // after std::move the is_valid_ becomes false which is causing the
  // original graph handle to not getting destroyed. we need the original
  // graph handle throughout the use case so that it is duplicated each
  // time.
  auto shape_agnostic_graph_ptr =
      cur_rvalpsh->shape_agnostic_synapse_graph_.get();
  shape_agnostic_graph_ptr->set_num_of_tensors(
      syn_graph_ptr->get_num_of_tensors());
  shape_agnostic_graph_ptr->set_num_of_nodes(syn_graph_ptr->get_num_of_nodes());
  shape_agnostic_graph_ptr->set_is_empty_value(false);
  shape_agnostic_graph_ptr->set_build_phase(true);
  shape_agnostic_graph_ptr->set_is_valid(true);
}

// shape agnostic : validate Inputs and Outputs and disable shape agnostic if
// not supported
void HabanaLaunchOpPT::ValidateInputsAndOutputsAndDisableSA(
    at::ArrayRef<torch::jit::IValue>& input_refs) {
  // Validate if output shapes are filled correctly otherwise we can not
  // support shape agnostic graph caching.
  for (auto shape : out_shapes) {
    for (auto size : shape) {
      if (size == 0) {
        jit_graph_and_meta_data->set_is_shape_agnostic_supported(false);
        PT_EAGER_DEBUG(
            "[SHAPE AGNOSTIC] shape agnostic not supported for this Op",
            " output shapes not ok!");
        break;
      }
    }
  }

  // Check if any of the inputs is a shape tensor
  for (auto const& input : input_refs) {
    if (input.isTensor()) {
      auto& tensor = input.toTensor();
      auto tmeta{get_tensor_extra_meta(tensor, true)};
      bool is_shape_tensor = tmeta && tmeta->is_shape_tensor();
      if (is_shape_tensor) {
        jit_graph_and_meta_data->set_is_shape_agnostic_supported(false);
        PT_EAGER_DEBUG(
            "[SHAPE AGNOSTIC] shape agnostic not supported for this Op",
            " input is a shape tensor ! ",
            " tmeta : ",
            tmeta);
        break;
      }
    }
  }
}

// shape agnostic : print duplicate graph information
void HabanaLaunchOpPT::MaybePrintDuplicateGraphInformation(
    synapse_helpers::graph* graph_ptr,
    std::vector<synTensorHandleMap>& tensors_map,
    std::vector<synNodeHandleMap>& nodes_map [[maybe_unused]],
    std::string cache_hit_or_miss) {
  PT_EAGER_DEBUG(
      "[SHAPE AGNOSTIC] === ",
      cache_hit_or_miss,
      " duplicate graph information ====");
  PT_EAGER_DEBUG(
      "[SHAPE AGNOSTIC] original graph handle : ",
      graph_ptr->get_graph_handle(),
      " duplicate graph handle : ",
      graph_ptr->get_duplicate_graph_handle(),
      " numTensors :",
      graph_ptr->get_num_of_tensors(),
      " numNodes :",
      graph_ptr->get_num_of_nodes());

  for (size_t i = 0; i < tensors_map.size(); i++) {
    PT_EAGER_DEBUG(
        "[SHAPE AGNOSTIC] org handle : ",
        tensors_map.at(i).origHandle,
        " new handle : ",
        tensors_map.at(i).newHandle);
  }
}

void habana::HabanaLaunchOpPT::CompileSynapse(
    HabanaLaunchOpPT* hbLaunchOp,
    synapse_helpers::graph* syn_graph,
    synapse_helpers::graph* syn_graph_ptr,
    std::shared_ptr<RecipeValueSpec> cur_rvalpsh,
    bool is_shape_agnostic_cache_miss) {
  PT_BRIDGE_BEGIN;
  PT_LAZY_EAGER_DEBUG(
      "[LAZY EAGER MT] syn graph compile thread ",
      syn_graph,
      "syn_graph_ptr: ",
      syn_graph_ptr);
  if (hbLaunchOp->enable_shape_agnostic_caching_ &&
      hbLaunchOp->jit_graph_and_meta_data->get_is_shape_agnostic_supported()) {
    if (is_shape_agnostic_cache_miss) {
      hbLaunchOp->CompileSynapseGraph();
      hbLaunchOp->StoreShapeAgnosticGraph();
      hbLaunchOp->ConstructPatchingTable();
      hbLaunchOp->UpdateSynapsePermutations();
    } else {
      RecipeValueSpec& rv = *cur_rvalpsh;
      hbLaunchOp->CompileSynapseGraph(false);
      if (hbLaunchOp->enable_tensor_dump_) {
        hbLaunchOp->DumpTensors_pre(rv);
      }
    }
  } else {
    hbLaunchOp->CompileSynapseGraph();
    hbLaunchOp->ConstructPatchingTable();
    hbLaunchOp->UpdateSynapsePermutations();
  }
  PT_BRIDGE_END;
}

// call this function for recipe caching (graph/eager)
void habana::HabanaLaunchOpPT::ExecuteSynapseCache(
    synapse_helpers::hpuStream_t hpu_stream,
    size_t graph_key_with_perm,
    at::ArrayRef<torch::jit::IValue> input_refs,
    HabanaLaunchOpPT* hbLaunchOp,
    std::shared_ptr<RecipeValueSpec> cur_rvalpsh,
    std::shared_ptr<RecipeArgumentSpec> cur_rargpsh,
    std::optional<std::vector<at::Tensor>> allocated_outputs_,
    bool dry_run) {
  PT_BRIDGE_BEGIN;
  RecipeValueSpec& rv = *cur_rvalpsh;
  rv.update_hit_count();

  PT_BRIDGE_DEBUG(
      hbLaunchOp->id_str,
      ": ",
      "HabanaOp recipe cache hit :: key ",
      cur_rargpsh->hashCode(),
      "\n",
      rv.header_str(),
      "\n",
      rv.digest_str());
  PT_IRGRAPH_DEBUG("HabanaOp recipe cache hit :: static shapes");
  PT_TEST_DEBUG("HabanaOp recipe cache hit :: static path");

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
      hbLaunchOp->m_map_shape.m_actual_shapes,
      std::nullopt,
      allocated_outputs_);

  if (hbLaunchOp->enable_tensor_dump_) {
    hbLaunchOp->DumpTensors_pre(rv);
  }
  if (!dry_run) {
    rv.launch(hpu_stream, input_refs, intermediate_tensors_ptr, dma_inputs_ptr);
  }

  if (hbLaunchOp->enable_tensor_dump_) {
    hbLaunchOp->DumpTensors(rv);
  }

  hbLaunchOp->CreateStaticComplationDBI(graph_key_with_perm);

  // Update the stack from the recipe itself
  hbLaunchOp->UpdateOutputs(rv);
  PT_BRIDGE_DEBUG("Returning cached recipe : ", cur_rargpsh->hashCode());
  hbLaunchOp->ReturnCachedRecipe(rv);

  hbLaunchOp->ClearStatics();
  PT_BRIDGE_END;
}

void habana::HabanaLaunchOpPT::ExecuteSynapse(
    synapse_helpers::hpuStream_t hpu_stream,
    std::shared_ptr<habana::OptimizedJITGraphAndMetaData>
        jit_graph_and_meta_data,
    at::ArrayRef<torch::jit::IValue> input_refs,
    std::shared_ptr<std::vector<IValPtrShared>> intermediate_tensors_ptr,
    std::shared_ptr<std::vector<IValPtrShared>> dma_inputs_ptr,
    HabanaLaunchOpPT* hbLaunchOp,
    std::shared_ptr<RecipeValueSpec> cur_rvalpsh,
    bool is_shape_agnostic_cache_miss,
    bool dry_run) {
  PT_BRIDGE_BEGIN;
  if (hbLaunchOp->enable_shape_agnostic_caching_ &&
      hbLaunchOp->jit_graph_and_meta_data->get_is_shape_agnostic_supported()) {
    if (is_shape_agnostic_cache_miss) {
      jit_graph_and_meta_data->set_shape_agnostic_recipe(cur_rvalpsh);
      hbLaunchOp->ExecuteSynapseGraph(hpu_stream);
    } else {
      RecipeValueSpec& rv = *cur_rvalpsh;
      if (jit_graph_and_meta_data->GetFrontendType() !=
          habana_helpers::HabanaFrontendTypes::EAGER) {
        rv.update_output_permutation();
      }
      if (!dry_run) {
        rv.launch(
            hpu_stream, input_refs, intermediate_tensors_ptr, dma_inputs_ptr);
      }

      if (hbLaunchOp->enable_tensor_dump_) {
        hbLaunchOp->DumpTensors(rv);
      }
    }
  } else {
    hbLaunchOp->ExecuteSynapseGraph(hpu_stream);

    auto is_jit_cached_graph_info_available =
        jit_graph_and_meta_data->get_jit_cached_graph_info_available_flag();
    if (is_jit_cached_graph_info_available == false) {
      jit_graph_and_meta_data->set_jit_cached_graph_info_available_flag(true);
    }
    hbLaunchOp->ClearStatics();
  }
  PT_BRIDGE_END;
}

void HabanaLaunchOpPT::DumpStaticCompilationStatistics(
    size_t graph_key_with_perm,
    bool is_compile) {
  habana_helpers::ResultShapes ranges;

  habana_helpers::InpTensorShapes input_tshapes =
      ref_input_shape_map.at(graph_key_with_perm);
  if (is_compile) {
    current_dbipsh_->get_statistics()->LogCompilation(
        jit_ir_graph->toString(),
        jit_ir_graph,
        current_dbipsh_->GetMinPolicy(),
        current_dbipsh_->GetMaxPolicy(),
        ranges,
        cur_rargpsh->hashCode(),
        "OK",
        habana_helpers::CompilationPass::STATIC);
    current_dbipsh_->get_statistics()->LogShapes(jit_ir_graph, input_tshapes);
    current_dbipsh_->get_statistics()->LogUsedBucket(
        0, jit_ir_graph, ranges, false);
    current_dbipsh_->get_statistics()->LogSelectedRecipe(
        cur_rargpsh->hashCode(), 0);
    // current_dbipsh_->get_statistics()->LogRecipeMemory(cur_rvalpsh);
    current_dbipsh_->get_statistics()->GetDigest(
        cur_rargpsh->graphHashCode(), 0, 0, cur_rargpsh->hashCode(), false);
  } else {
    std::shared_ptr<RecipeArgumentSpec> rargpsh_graph =
        std::make_shared<RecipeArgumentSpec>(input_refs, graph_key, op_strs);
    current_dbipsh_ = DynamicBucketInfoMap::get_instance().get(rargpsh_graph);
    HABANA_ASSERT(
        (current_dbipsh_ != nullptr),
        "Dynamic bucketinfo got NULL in static cache hit");
    current_dbipsh_->SetLastUsedStepForBucket(
        0, current_dbipsh_->get_statistics()->GetCurrentStep());

    current_dbipsh_->get_statistics()->LogSelectedRecipe(
        cur_rargpsh->hashCode(), 0);
    current_dbipsh_->get_statistics()->LogShapes(jit_ir_graph, input_tshapes);

    auto t_ns_base{current_dbipsh_->GetTimeBase(0)};
    auto t_ns{current_dbipsh_->GetTime(0)};

    current_dbipsh_->get_statistics()->LogLaunchBase(t_ns_base, 0);
    current_dbipsh_->get_statistics()->LogLaunch(t_ns, 0);
    current_dbipsh_->get_statistics()->LogLaunchPerf(t_ns_base, t_ns, 0);
    if (t_ns && t_ns_base) {
      habana_helpers::DynamicBucketInfo::update_improvement_map(
          cur_rargpsh->hashCode(), (t_ns < t_ns_base));
    }
    current_dbipsh_->get_statistics()->GetDigest(
        cur_rargpsh->graphHashCode(), 0, 0, cur_rargpsh->hashCode(), true);
  }

  current_dbipsh_->get_statistics()->DumpAndNextStep();
}

void HabanaLaunchOpPT::run(
    torch::jit::Stack& input_st,
    std::optional<std::vector<at::Tensor>> allocated_outputs,
    bool dry_run) {
  PT_BRIDGE_BEGIN;
  static int idx{1};
  ProcessInputStack(input_st);
  allocated_outputs_ = std::move(allocated_outputs);

  dry_run_ = dry_run;
  iteration_count_++;
  auto& device = HPURegistrar::get_device();

  // Check whether dynamic shape is needed
  size_t graph_key_with_perm = graph_key;
  size_t perm_hash_code = habana::ComputePermutationHashCode(input_refs);
  graph_key_with_perm = at::hash_combine(graph_key_with_perm, perm_hash_code);

  const auto eager_mode =
      (execution_mode_ == habana_helpers::HabanaFrontendTypes::EAGER);
  PT_BRIDGE_DEBUG(
      "Lowering:\n",
      "JIT_IR_Graph_BEGIN\n",
      "Graph ",
      idx,
      '\n',
      jit_ir_graph->toString(),
      "JIT_IR_Graph_END\n");

  PT_TEST_DEBUG(
      "Lowering:\n",
      "Graph ",
      idx,
      '\n',
      "JIT IR graph_hash_code : ",
      graph_key,
      ", hash_code with data layout : ",
      graph_key_with_perm);

  idx += 1;
  if (enable_caching_ || IS_BRIDGE_DEBUG_ENABLED) {
    cur_rargpsh = std::make_shared<RecipeArgumentSpec>(
        false, input_refs, jit_ir_graph, graph_key, op_strs);
  }

  // recipe caching :: begin
  if (enable_graph_caching_) {
    cur_rvalpsh = GetCachedRecipe(cur_rargpsh);

    if (ABSL_PREDICT_TRUE(cur_rvalpsh)) {
      emitCacheEvent(
          habana_helpers::EventDispatcher::Topic::CACHE_HIT,
          std::to_string(cur_rargpsh->hashCode()));
      ExecuteSynapseCache(
          hpu_stream,
          graph_key_with_perm,
          input_refs,
          this,
          cur_rvalpsh,
          cur_rargpsh,
          allocated_outputs_,
          dry_run);
      PT_BRIDGE_END;
      return;
    } else {
      emitCacheEvent(
          habana_helpers::EventDispatcher::Topic::CACHE_MISS,
          std::to_string(cur_rargpsh->hashCode()));
      PT_BRIDGE_DEBUG(
          id_str,
          ": ",
          "HabanaOp recipe cache miss :: key ",
          cur_rargpsh->hashCode());
      PT_IRGRAPH_DEBUG("HabanaOp recipe cache miss :: static shapes");
    }
  }
  // recipe caching :: end

  // currently only eager backend supports pipelining
  // can be merged once non-eager backends support pipelining
  // eager recipe caching :: begin
  if (enable_eager_caching_) {
    PT_BRIDGE_DEBUG("Getting cached recipe : ", cur_rargpsh->hashCode());
    cur_rvalpsh = GetCachedRecipe(cur_rargpsh);

    if (ABSL_PREDICT_TRUE(cur_rvalpsh)) {
      emitCacheEvent(
          habana_helpers::EventDispatcher::Topic::CACHE_HIT,
          std::to_string(cur_rargpsh->hashCode()));
      if ((GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) == 2) &&
          !GET_ENV_FLAG_NEW(PT_HPU_ENABLE_EXECUTION_THREAD_NO_WAIT) &&
          GET_ENV_FLAG_NEW(PT_HPU_ENABLE_LAZY_EAGER_LAUNCH_EXEC_THREAD)) {
        PT_LAZY_EAGER_DEBUG(
            "[LAZY EAGER MT] Enqueue new task to the Compile and Execute Thread");

        auto doNothingLambda = [] {};
        Singleton_CompileThreadPool::m_compile_thread_handle =
            Singleton_CompileThreadPool::getInstance().enqueue(doNothingLambda);

        Singleton_CompileThreadPool::JoinPendingExecuteThread();

        Singleton_ExecThreadPool::m_exec_thread_handle =
            Singleton_ExecThreadPool::getInstance().enqueue(
                ExecuteSynapseCache,
                hpu_stream,
                graph_key_with_perm,
                input_refs,
                this,
                cur_rvalpsh,
                cur_rargpsh,
                allocated_outputs_,
                dry_run);

        Singleton_ExecThreadPool::JoinPendingExecuteThread();
      } else {
        ExecuteSynapseCache(
            hpu_stream,
            graph_key_with_perm,
            input_refs,
            this,
            cur_rvalpsh,
            cur_rargpsh,
            allocated_outputs_,
            dry_run);
      }
      PT_BRIDGE_END;
      return;
    } else {
      emitCacheEvent(
          habana_helpers::EventDispatcher::Topic::CACHE_MISS,
          std::to_string(cur_rargpsh->hashCode()));
      PT_BRIDGE_DEBUG(
          id_str,
          ": ",
          "HabanaOp recipe cache miss :: key ",
          cur_rargpsh->hashCode());
      PT_IRGRAPH_DEBUG("HabanaOp recipe cache miss :: static shapes");
    }
  }
  // eager recipe caching :: end

  bool is_jit_cached_graph_info_available =
      jit_graph_and_meta_data->get_jit_cached_graph_info_available_flag();
  if (is_jit_cached_graph_info_available == false) {
    jit_graph_and_meta_data->clear_cached_graph_info();
  }

  CreateValueToIvalueMapForInputs();
  ProcessGraphForConstantTensors();

  if (enable_shape_agnostic_caching_) {
    ValidateInputsAndOutputsAndDisableSA(input_refs);
  }

  // shape agnostic caching :: begin
  if (enable_shape_agnostic_caching_ &&
      jit_graph_and_meta_data->get_is_shape_agnostic_supported()) {
    cur_rvalpsh = jit_graph_and_meta_data->get_shape_agnostic_recipe();
    if (cur_rvalpsh == nullptr) {
      PT_EAGER_DEBUG("[SHAPE AGNOSTIC] shape agnostic cache miss (begin)");
      HABANA_ASSERT(
          eager_mode == true,
          "eager_mode is expected true for supporting shape agnostic graph");
      constexpr bool dry_run__ = false;
      auto syn_graph = habana_helpers::create_graph(
          device.id(), GetSynapseGraphName(), dry_run__, eager_mode);
      constexpr bool is_shape_agnostic_graph = true;
      syn_graph.set_shape_agnostic_graph(is_shape_agnostic_graph);
      BuildSynapseGraph(syn_graph);

      if (syn_graph_ptr->is_empty()) {
        PT_LAZY_EAGER_DEBUG(
            "Empty synapse graph. Nothing to duplicate. will update outputs directly.");
        UpdateOutputs();
        return;
      }

      DuplicateSynapseGraph();

      if (syn_graph_ptr->get_num_of_shape_tensors() > 0) {
        jit_graph_and_meta_data->set_is_shape_agnostic_supported(false);
        PT_EAGER_DEBUG(
            "[SHAPE AGNOSTIC] Shape agnostic not supported for Op",
            " with intermediate shape tensors : ",
            syn_graph_ptr->get_num_of_shape_tensors());
      }

      if (habana_kernels.size() != syn_graph_ptr->get_num_of_nodes()) {
        jit_graph_and_meta_data->set_is_shape_agnostic_supported(false);
        PT_EAGER_DEBUG(
            "[SHAPE AGNOSTIC] Shape agnostic not supported for compound Op(s)",
            " number of kernels : ",
            habana_kernels.size(),
            " number of synapse nodes : ",
            syn_graph_ptr->get_num_of_nodes());
      }

      PT_EAGER_DEBUG(
          "[SHAPE AGNOSTIC] Shape agnostic SIF tinfo map size : ",
          sif_tidx_to_tinfo_map.size());
      syn_graph_ptr->set_num_of_inter_tensors(sif_tidx_to_tinfo_map.size());

      if (eager_mode &&
          !GET_ENV_FLAG_NEW(PT_HPU_ENABLE_EXECUTION_THREAD_NO_WAIT) &&
          GET_ENV_FLAG_NEW(PT_HPU_ENABLE_LAZY_EAGER_LAUNCH_EXEC_THREAD)) {
        PT_LAZY_EAGER_DEBUG(
            "[LAZY EAGER MT] Enqueue new task to the Compile and Execute Thread");
        Singleton_CompileThreadPool::m_compile_thread_handle =
            Singleton_CompileThreadPool::getInstance().enqueue(
                CompileSynapse,
                this,
                nullptr,
                syn_graph_ptr,
                cur_rvalpsh,
                true);

        Singleton_CompileThreadPool::JoinPendingExecuteThread();

        Singleton_ExecThreadPool::m_exec_thread_handle =
            Singleton_ExecThreadPool::getInstance().enqueue(
                ExecuteSynapse,
                hpu_stream,
                jit_graph_and_meta_data,
                input_refs,
                nullptr,
                nullptr,
                this,
                cur_rvalpsh,
                true,
                dry_run);
        Singleton_ExecThreadPool::JoinPendingExecuteThread();
      } else {
        CompileSynapseGraph();
        StoreShapeAgnosticGraph();
        ConstructPatchingTable();
        UpdateSynapsePermutations();
        jit_graph_and_meta_data->set_shape_agnostic_recipe(cur_rvalpsh);
        ExecuteSynapseGraph(hpu_stream);
      }

      synGraphDestroy(syn_graph_ptr->get_duplicate_graph_handle());
      PT_EAGER_DEBUG("[SHAPE AGNOSTIC] shape agnostic cache miss (end)");
    } else {
      PT_EAGER_DEBUG("[SHAPE AGNOSTIC] shape agnostic cache hit (begin)");
      syn_graph_ptr = cur_rvalpsh->shape_agnostic_synapse_graph_.get();

      std::vector<synTensorHandleMap> tensorsMap(
          syn_graph_ptr->get_num_of_tensors());
      std::vector<synNodeHandleMap> nodesMap(syn_graph_ptr->get_num_of_nodes());

      syn_graph_ptr->duplicate(tensorsMap.data(), nodesMap.data());

      MaybePrintDuplicateGraphInformation(
          syn_graph_ptr, tensorsMap, nodesMap, "cache hit");

      RecipeValueSpec& rv = *cur_rvalpsh;
      rv.update_hit_count();
      PT_EAGER_DEBUG(
          id_str,
          ": ",
          "HabanaOp shape agnostic graph cache hit :: key ",
          graph_key,
          "\n",
          rv.header_str(),
          "\n",
          rv.digest_str());
      PT_EAGER_DEBUG(
          "HabanaOp shape agnostic graph cache hit :: static shapes");

      std::shared_ptr<std::vector<IValPtrShared>> intermediate_tensors_ptr =
          std::make_shared<std::vector<IValPtrShared>>(
              std::vector<IValPtrShared>());

      std::shared_ptr<std::vector<IValPtrShared>> dma_inputs_ptr =
          std::make_shared<std::vector<IValPtrShared>>(
              std::vector<IValPtrShared>());

      std::unordered_map<synTensor, synTensor> synapse_orig_to_new_handle{};
      for (size_t i = 0; i < tensorsMap.size(); i++) {
        synapse_orig_to_new_handle.insert(
            {tensorsMap.at(i).origHandle, tensorsMap.at(i).newHandle});
      }

      auto new_eager_mode =
          (jit_graph_and_meta_data->GetFrontendType() ==
           habana_helpers::HabanaFrontendTypes::EAGER);

      /*
       * Hybrid SIF is used for shape inference for intermediate tensors
       * Inputs shape is retrieved from input refs.
       * Ouptut shape info is passed in the jit ir graph meta data.
       */
      std::unordered_map<int64_t, at::Tensor> local_tidx_to_tensor_map;
      if (syn_graph_ptr->get_num_of_inter_tensors() > 0) {
        habana::ShapeInference::ResetSifTensorId();
        constexpr bool dynamic_shapes_false = false;
        RunHybridSif<dynamic_shapes_false>(local_tidx_to_tensor_map);
      }

      constexpr bool is_shape_agnostic_graph = true;
      rv.update_patching_table(
          input_refs,
          intermediate_tensors_ptr,
          dma_inputs_ptr,
          m_map_shape.m_actual_shapes,
          local_tidx_to_tensor_map,
          allocated_outputs_,
          out_shapes,
          syn_graph_ptr,
          synapse_orig_to_new_handle,
          is_shape_agnostic_graph,
          new_eager_mode);

      // To check if any other members just like ntensorbytes also need to be
      // updated
      rv.ntensorbytes = 0;
      for (auto& ti : *rv.dtensorinfos) {
        if (!ti->is_duplicate()) {
          rv.ntensorbytes += ti->get_size();
        }
      }

      syn_graph_ptr->set_build_phase(true);

      if (eager_mode &&
          !GET_ENV_FLAG_NEW(PT_HPU_ENABLE_EXECUTION_THREAD_NO_WAIT) &&
          GET_ENV_FLAG_NEW(PT_HPU_ENABLE_LAZY_EAGER_LAUNCH_EXEC_THREAD)) {
        PT_LAZY_EAGER_DEBUG(
            "[LAZY EAGER MT] Enqueue new task to the Compile and Execute Thread");
        Singleton_CompileThreadPool::m_compile_thread_handle =
            Singleton_CompileThreadPool::getInstance().enqueue(
                CompileSynapse,
                this,
                nullptr,
                syn_graph_ptr,
                cur_rvalpsh,
                false);

        Singleton_CompileThreadPool::JoinPendingExecuteThread();

        Singleton_ExecThreadPool::m_exec_thread_handle =
            Singleton_ExecThreadPool::getInstance().enqueue(
                ExecuteSynapse,
                hpu_stream,
                jit_graph_and_meta_data,
                input_refs,
                intermediate_tensors_ptr,
                dma_inputs_ptr,
                this,
                cur_rvalpsh,
                false,
                dry_run);
        Singleton_ExecThreadPool::JoinPendingExecuteThread();
      } else {
        CompileSynapseGraph(false);

        if (enable_tensor_dump_) {
          DumpTensors_pre(rv);
        }

        if (jit_graph_and_meta_data->GetFrontendType() !=
            habana_helpers::HabanaFrontendTypes::EAGER) {
          rv.update_output_permutation();
        }

        if (!dry_run) {
          rv.launch(
              hpu_stream, input_refs, intermediate_tensors_ptr, dma_inputs_ptr);
        }

        if (enable_tensor_dump_) {
          DumpTensors(rv);
        }
      }

      // Update the stack from the recipe itself
      UpdateOutputs(rv);
      ReturnCachedRecipe(rv);

      synGraphDestroy(syn_graph_ptr->get_duplicate_graph_handle());

      PT_EAGER_DEBUG("[SHAPE AGNOSTIC] shape agnostic cache hit (end)");
    }

    is_jit_cached_graph_info_available =
        jit_graph_and_meta_data->get_jit_cached_graph_info_available_flag();
    if (is_jit_cached_graph_info_available == false) {
      jit_graph_and_meta_data->set_jit_cached_graph_info_available_flag(true);
    }

    ClearStatics();
    PT_BRIDGE_END;
    return;
  }
  // shape agnostic caching :: end
  if (!eager_mode && ref_input_shape_map.count(graph_key_with_perm) &&
      habana_helpers::GetRefineDynamicShapeStatus()) {
    PT_DYNAMIC_SHAPE_DEBUG(
        "JIT IR graph_hash_code : ",
        graph_key,
        ", hash_code with data layout : ",
        graph_key_with_perm,
        "\nStarting dynamic shape flow");

    jit_graph_and_meta_data->clear_cached_graph_info();
    jit_graph_and_meta_data->set_jit_cached_graph_info_available_flag(
        false); // Disable Optimized Lowering based on Cached precalculated
                // graph information.
    ProcessHabanaFusedOpWithDS();
    return;
  }

  // Remember the input shapes for creating dynamic bucket info structure later.
  // Note that this needs to be done before execution of graph, otherwise
  // input_refs will get overwritten by outputs and we will create bucket
  // with incorrect shapes.
  if (!eager_mode && habana_helpers::GetRefineDynamicShapeStatus()) {
    CreateStaticComplationDBI(graph_key_with_perm);
  }

  constexpr bool dry_run__ = false;
  const auto use_eager_compiler =
      eager_mode && jit_graph_and_meta_data->get_is_eager_compiler_supported();
  auto syn_graph = habana_helpers::create_graph(
      device.id(), GetSynapseGraphName(), dry_run__, use_eager_compiler);
  m_map_shape.m_pass = ShapeInfo::InferencePass::INVALID;
  BuildSynapseGraph(syn_graph);
  if (enable_shape_agnostic_caching_) {
    syn_graph_ptr->copy_graph_handle_to_duplicate();
  }
  if (eager_mode && !GET_ENV_FLAG_NEW(PT_HPU_ENABLE_EXECUTION_THREAD_NO_WAIT) &&
      GET_ENV_FLAG_NEW(PT_HPU_ENABLE_LAZY_EAGER_LAUNCH_EXEC_THREAD)) {
    PT_LAZY_EAGER_DEBUG(
        "[LAZY EAGER MT] Enqueue new task to the Compile and Execute Thread");
    Singleton_CompileThreadPool::m_compile_thread_handle =
        Singleton_CompileThreadPool::getInstance().enqueue(
            CompileSynapse, this, &syn_graph, syn_graph_ptr, nullptr, false);
    Singleton_CompileThreadPool::JoinPendingExecuteThread();

    Singleton_ExecThreadPool::m_exec_thread_handle =
        Singleton_ExecThreadPool::getInstance().enqueue(
            ExecuteSynapse,
            hpu_stream,
            jit_graph_and_meta_data,
            input_refs,
            nullptr,
            nullptr,
            this,
            nullptr,
            false,
            dry_run);
    Singleton_ExecThreadPool::JoinPendingExecuteThread();
  } else {
    CompileSynapseGraph();
    ConstructPatchingTable();
    UpdateSynapsePermutations();
    ExecuteSynapseGraph(hpu_stream);

    is_jit_cached_graph_info_available =
        jit_graph_and_meta_data->get_jit_cached_graph_info_available_flag();
    if (is_jit_cached_graph_info_available == false) {
      jit_graph_and_meta_data->set_jit_cached_graph_info_available_flag(true);
    }

    ClearStatics();
  }

  PT_BRIDGE_END;
}

void HabanaLaunchOpPT::CompileGraphWithRange(
    torch::jit::Stack& input_st,
    habana_helpers::ResultShapes& input_ranges,
    habana_helpers::Bucket& new_bucket,
    size_t& new_recipe_key,
    std::shared_ptr<habana_helpers::CompilationStatistics> statpsh,
    std::shared_ptr<habana_helpers::DynamicBucketInfo> dbipsh) {
  PT_BRIDGE_BEGIN;
  ProcessInputStack(input_st);

  PT_DYNAMIC_SHAPE_DEBUG(
      "Input range for new bucket:\n",
      "Min\n",
      input_ranges.min_shapes,
      "Max\n",
      input_ranges.max_shapes,
      "--------------------");

  current_dbipsh_ = dbipsh;
  CreateValueToIvalueMapForInputs();

  DynamicShapeInfo graph_input_info;
  CreateDynamicBucketInputShapes(graph_input_info.act_input_tshapes);
  ProcessDynamicBucketInputShapesWithH2D(graph_input_info.act_input_tshapes);

  graph_input_info.min_input_tshapes.insert(
      input_ranges.min_shapes.begin(), input_ranges.min_shapes.end());
  graph_input_info.max_input_tshapes.insert(
      input_ranges.max_shapes.begin(), input_ranges.max_shapes.end());

  PT_DYNAMIC_SHAPE_DEBUG(
      "Current graph_input_info",
      "\nact_input_tshapes",
      graph_input_info.act_input_tshapes,
      "\nmin_input_tshapes",
      graph_input_info.min_input_tshapes,
      "\nmax_input_tshapes",
      graph_input_info.max_input_tshapes);

  // Min shape inference
  {
    torch::jit::Stack new_stack;
    torch::jit::Stack* old_stack = nullptr;
    std::vector<IValPtrShared> old_pt_stack_sh;

    old_stack = pt_stack;
    old_pt_stack_sh = pt_stack_sh;
    pt_stack_sh.clear();

    new_stack = CreateStack(*pt_stack, graph_input_info.min_input_tshapes);
    SetH2DMinMaxData(
        *old_stack,
        graph_input_info.min_input_tshapes,
        ShapeInfo::InferencePass::MIN_SHAPE);
    pt_stack = &new_stack;

    for (size_t j{0}; j < new_stack.size(); j++) {
      IValPtrShared ivpsh = std::make_shared<IVal>(new_stack[j]);
      pt_stack_sh.push_back(ivpsh);
    }
    m_map_shape.m_pass = ShapeInfo::InferencePass::MIN_SHAPE;
    std::string error_str;

    try {
      run_pass();
    } catch (std::exception& e) {
      error_str = e.what();
      PT_DYNAMIC_SHAPE_DEBUG(
          "Exception occured in Pass = ",
          m_map_shape.m_pass,
          " - Details :\n",
          error_str);

      pt_stack_sh.clear();
      pt_stack = old_stack;
      pt_stack_sh = old_pt_stack_sh;
      throw;
    }

    pt_stack_sh.clear();
    pt_stack = old_stack;
    pt_stack_sh = old_pt_stack_sh;

    PT_DYNAMIC_SHAPE_DEBUG(
        "Pass = ",
        m_map_shape.m_pass,
        " completed. Shapes\n",
        m_map_shape.m_min_shapes);
  }

  // Max shape inference
  {
    torch::jit::Stack new_stack;
    torch::jit::Stack* old_stack = nullptr;
    std::vector<IValPtrShared> old_pt_stack_sh;

    old_stack = pt_stack;
    old_pt_stack_sh = pt_stack_sh;
    pt_stack_sh.clear();

    new_stack = CreateStack(*pt_stack, graph_input_info.max_input_tshapes);
    SetH2DMinMaxData(
        *old_stack,
        graph_input_info.max_input_tshapes,
        ShapeInfo::InferencePass::MAX_SHAPE);
    pt_stack = &new_stack;

    for (size_t j{0}; j < new_stack.size(); j++) {
      IValPtrShared ivpsh = std::make_shared<IVal>(new_stack[j]);
      pt_stack_sh.push_back(ivpsh);
    }
    m_map_shape.m_pass = ShapeInfo::InferencePass::MAX_SHAPE;
    std::string error_str;

    try {
      run_pass();
    } catch (std::exception& e) {
      error_str = e.what();
      PT_DYNAMIC_SHAPE_DEBUG(
          "Exception occured in Pass = ",
          m_map_shape.m_pass,
          " - Details :\n",
          error_str);

      pt_stack_sh.clear();
      pt_stack = old_stack;
      pt_stack_sh = old_pt_stack_sh;
      throw;
    }

    pt_stack_sh.clear();
    pt_stack = old_stack;
    pt_stack_sh = old_pt_stack_sh;

    PT_DYNAMIC_SHAPE_DEBUG(
        "Pass = ",
        m_map_shape.m_pass,
        " completed. Shapes\n",
        m_map_shape.m_max_shapes);
  }

  auto& device = HPURegistrar::get_device();
  std::string graphName{GetSynapseGraphName()};

  auto create_graph_for_refinement{[&]() -> synapse_helpers::graph {
    auto graph_or_error = synapse_helpers::graph::create_for_refinement(
        device.syn_device(), name);

    if (absl::holds_alternative<synapse_helpers::synapse_error>(
            graph_or_error)) {
      auto error = absl::get<synapse_helpers::synapse_error>(graph_or_error);
      TORCH_CHECK(error.status, error.error);
    }
    return absl::get<synapse_helpers::graph>(std::move(graph_or_error));
  }};

  auto syn_graph = create_graph_for_refinement();

  // Compile the graph
  {
    CreateValueToIvalueMapForInputs();

    syn_graph.set_dynamic_graph(true);

    std::string error_str;
    try {
      cur_ds_token_ = new_bucket.getToken();
      cur_rargpsh = std::make_shared<RecipeArgumentSpec>(
          input_refs, graph_key, op_strs, cur_ds_token_);
      new_recipe_key = cur_rargpsh->hashCode();

      m_map_shape.m_pass = ShapeInfo::InferencePass::INVALID;
      BuildSynapseGraph(syn_graph);
      CompileSynapseGraph();
      ConstructPatchingTable();
      UpdateSynapsePermutations();
    } catch (std::exception& e) {
      error_str = e.what();
      PT_DYNAMIC_SHAPE_DEBUG(
          "Exception occured in compilation - Details :\n", error_str);

      std::string result_str{"FAIL"};
      uint64_t current_step{statpsh->GetCurrentStep()};
      statpsh->LogRefineCompilation(
          input_ranges,
          jit_ir_graph,
          new_recipe_key,
          new_bucket.GetIndex(),
          result_str,
          current_step);

      throw;
    }
  }
  PT_DYNAMIC_SHAPE_DEBUG("Compilation completed");

  // Add the <key,value> pair to the map
  cur_rvalpsh->dynamic_graph = syn_graph.is_dynamic_graph();
  cur_rvalpsh->set_op_strs(cur_rargpsh->get_op_strs());
  RecipeCacheLRU::get_cache().add(cur_rargpsh, cur_rvalpsh);
  DynamicBucketInfoMap::get_instance().add(cur_rargpsh, current_dbipsh_);

  new_recipe_key = cur_rargpsh->hashCode();
  // Add the recipe to the corresponding bucket
  new_bucket.SetSynapseRecipePtr(cur_rvalpsh);

  PT_DYNAMIC_SHAPE_DEBUG(
      "HabanaOp recipe cache :: adding new recipe to cache ::",
      cur_rvalpsh->header_str(),
      "\n",
      cur_rvalpsh->digest_str(),
      "\n",
      "--------------------");

  ClearMembers();
  ClearStatics();

  PT_BRIDGE_END;
}

void HabanaLaunchOpPT::run_pass() {
  PT_BRIDGE_BEGIN;
  auto& device = HPURegistrar::get_device();

  //
  // Run the compile and execute method to infer the shapes
  static thread_local auto syn_graph =
      habana_helpers::create_graph(device.id(), GetSynapseGraphName(), true);
  syn_graph.set_dynamic_graph(true);
  CreateValueToIvalueMapForInputs();
  BuildSynapseGraph(syn_graph, true);
  //
  // clear the data that has been setup as part of the above
  // method
  ClearMembers(true);
  ClearStatics(true);
  PT_BRIDGE_END;
}

void HabanaLaunchOpPT::run_shape_inference(
    const ShapeInfo::InferencePass& pass,
    DynamicShapeInfo& graph_input_info) {
  PT_BRIDGE_BEGIN;
  torch::jit::Stack new_stack;
  torch::jit::Stack* old_stack = nullptr;
  std::vector<IValPtrShared> old_pt_stack_sh;
  m_map_shape.m_pass = pass;
  if ((pass == ShapeInfo::InferencePass::MIN_SHAPE) ||
      (pass == ShapeInfo::InferencePass::MAX_SHAPE)) {
    old_stack = pt_stack;
    old_pt_stack_sh = pt_stack_sh;
    pt_stack_sh.clear();
    if (pass == ShapeInfo::InferencePass::MIN_SHAPE) {
      new_stack = CreateStack(*pt_stack, graph_input_info.min_input_tshapes);
      SetH2DMinMaxData(
          *old_stack,
          graph_input_info.min_input_tshapes,
          ShapeInfo::InferencePass::MIN_SHAPE);
      SetH2DMinMaxData(
          new_stack,
          graph_input_info.min_input_tshapes,
          ShapeInfo::InferencePass::MIN_SHAPE);
    } else {
      new_stack = CreateStack(*pt_stack, graph_input_info.max_input_tshapes);
      SetH2DMinMaxData(
          *old_stack,
          graph_input_info.max_input_tshapes,
          ShapeInfo::InferencePass::MAX_SHAPE);
      SetH2DMinMaxData(
          new_stack,
          graph_input_info.max_input_tshapes,
          ShapeInfo::InferencePass::MAX_SHAPE);
    }
    pt_stack = &new_stack;

    size_t j = new_stack.size() - num_inputs;
    for (; j < new_stack.size(); j++) {
      IValPtrShared ivpsh = std::make_shared<IVal>(new_stack[j]);
      pt_stack_sh.push_back(ivpsh);
    }
  }
  bool throw_exception = false;
  std::string error_str;
  try {
    run_pass();
  } catch (std::exception& e) {
    std::string error = e.what();
    error_str = error.substr(0, error.find("\n"));
    PT_DYNAMIC_SHAPE_DEBUG("Exception occured in Pass = ", pass);
    PT_DYNAMIC_SHAPE_DEBUG("Exception Details : ", error_str);
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
  PT_DYNAMIC_SHAPE_DEBUG("Handling the exception ..");
  switch (e.Pass()) {
    // Min inference pass can have exception only in HISTORIC if exception is
    // in policy = CURRENT, it is unrecoverable, throw runtime error in this
    // case
    case ShapeInfo::InferencePass::MIN_SHAPE: {
      current_dbipsh_->get_statistics()->LogFallback(
          "MIN_PASS", graph_input_info.min_policy, e.what());
      // In case there is fallback for LOCAL_HISTORIC we need to discard the
      // current running min and reset running min to previous successfull one
      if (graph_input_info.min_policy ==
          habana_helpers::DynamicDimsPolicy::LOCAL_HISTORIC) {
        current_dbipsh_->RestoreLocalMinHistory();
      }
      if (graph_input_info.min_policy ==
          habana_helpers::DynamicDimsPolicy::LOCAL_HIST_PER_TSR) {
        current_dbipsh_->RestoreLocalHistoryPerTensor(true);
      }
      graph_input_info.set_next_min_policy();
      std::string min_policy_seq =
          GET_ENV_FLAG_NEW(PT_HPU_DYNAMIC_MIN_POLICY_ORDER);
      // If the size of fallback sequence is less than equal to the
      // index we calculte to get the next fallback policy, means there no more
      // policy to fallback. Exit the execution.
      if (min_policy_seq.size() > graph_input_info.min_fallback_seq_num) {
        graph_input_info.min_policy = habana_helpers::getPolicy(
            min_policy_seq.at(graph_input_info.min_fallback_seq_num) -
            habana_helpers::zero_offset);
      } else {
        throw std::runtime_error("No more fallback exiting ..");
      }
      break;
    }
    // Max inference pass can have exception only in CALCULATED if exception is
    // in policy = CURRENT, it is unrecoverable, throw runtime error in this
    // case
    case ShapeInfo::InferencePass::MAX_SHAPE: {
      current_dbipsh_->get_statistics()->LogFallback(
          "MAX_PASS", graph_input_info.max_policy, e.what());
      // In case there is fallback for LOCAL_HISTORIC we need to discard the
      // current running max and reset running max to previous successfult one
      if (graph_input_info.max_policy ==
          habana_helpers::DynamicDimsPolicy::LOCAL_HISTORIC) {
        current_dbipsh_->RestoreLocalMaxHistory();
      }
      if (graph_input_info.max_policy ==
          habana_helpers::DynamicDimsPolicy::LOCAL_HIST_PER_TSR) {
        current_dbipsh_->RestoreLocalHistoryPerTensor(false);
      }
      graph_input_info.set_next_max_policy();
      std::string max_policy_seq =
          GET_ENV_FLAG_NEW(PT_HPU_DYNAMIC_MAX_POLICY_ORDER);
      // If the size of fallback sequence is less than equal to the
      // index we calculte to get the next fallback policy, means there no more
      // policy to fallback. Exit the execution.
      if (max_policy_seq.size() > graph_input_info.max_fallback_seq_num) {
        graph_input_info.max_policy = habana_helpers::getPolicy(
            max_policy_seq.at(graph_input_info.max_fallback_seq_num) -
            habana_helpers::zero_offset);
      } else {
        throw std::runtime_error("No more fallback exiting ..");
      }
      break;
    }
    // The OUTPUT_SHAPE inference exception is actually compile exception.
    // if min and max both was current, meaning the failure is in
    // static(fallback path), bail out execution by throwing error.
    case ShapeInfo::InferencePass::OUTPUT_SHAPE: {
      if (graph_input_info.min_policy ==
              habana_helpers::DynamicDimsPolicy::CURRENT &&
          graph_input_info.max_policy ==
              habana_helpers::DynamicDimsPolicy::CURRENT) {
        PT_DYNAMIC_SHAPE_FATAL("Unhandled exception exiting ..");
        throw std::runtime_error("Exception was not handled ..");
      }
      graph_input_info.min_policy = habana_helpers::DynamicDimsPolicy::CURRENT;
      graph_input_info.max_policy = habana_helpers::DynamicDimsPolicy::CURRENT;
      break;
    }
    default:
      PT_DYNAMIC_SHAPE_FATAL("Unhandled exception exiting ..");
      throw std::runtime_error("Exception was not handled ..");
      break;
  }

  // The above switch case changes the policy, get new ranges with changed
  // policy.
  habana::ShapeInference::SetMinMaxPolicyInUse(
      graph_input_info.min_policy, graph_input_info.max_policy);
  current_dbipsh_->UpdateBucketWithPolicy(
      graph_input_info.current_bucket_id,
      graph_input_info.act_input_tshapes,
      graph_input_info.min_policy,
      graph_input_info.max_policy);
  auto fallback_ranges =
      current_dbipsh_->CalculateShapes(graph_input_info.current_bucket_id);
  PT_DYNAMIC_SHAPE_DEBUG(
      "After fallback\n",
      "min policy: ",
      graph_input_info.min_policy,
      '\n',
      "max policy: ",
      graph_input_info.max_policy,
      '\n',
      "Input shapes:",
      graph_input_info.act_input_tshapes,
      '\n',
      "Fallback range ::\n",
      fallback_ranges.DebugString(),
      "--------------------");
  // After calculating ranges set bucket_info policy to DEFAULT
  // so that for next bucket created the starting policy be started again
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
      PT_DYNAMIC_SHAPE_DEBUG("Rerun with policy CURRENT ..");
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
      PT_DYNAMIC_SHAPE_FATAL("Unhandled exception exiting ..");
      throw std::runtime_error("Exception was not handled ..");
      break;
  }
  PT_BRIDGE_END;
}

// Handle running passes and calls CompileAndExecute.
// Also handles fallback and failures.
void HabanaLaunchOpPT::CompileAndRunDynamicGraph(
    DynamicShapeInfo& graph_input_info) {
  auto& device = HPURegistrar::get_device();
  habana_helpers::CompilationPass last_compilation_pass =
      habana_helpers::CompilationPass::STATIC;
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

  CreateValueToIvalueMapForInputs();

  PT_DYNAMIC_SHAPE_DEBUG(
      "Running BuildSynapseGraph with min{",
      graph_input_info.min_policy,
      "}:max{",
      graph_input_info.max_policy,
      "}");

  std::string result = "OK";
  std::string jit_ir = "";
  bool try_catch_fail = false;
  if (graph_input_info.current_bucket_id == 0) {
    jit_ir = jit_ir_graph->toString();
  }
  auto ranges =
      current_dbipsh_->CalculateShapes(graph_input_info.current_bucket_id);

  if (ranges.empty()) {
    if (graph_input_info.max_policy ==
        habana_helpers::DynamicDimsPolicy::CURRENT) {
      last_compilation_pass = habana_helpers::CompilationPass::DYNAMIC_CURRENT;
    }
  } else {
    last_compilation_pass = habana_helpers::CompilationPass::DYNAMIC_MAX;
  }
  if (GET_ENV_FLAG_NEW(PT_HPU_ENABLE_DYNAMIC_LAUNCH_FALLBACK)) {
    // Try running the BuildSynapseGraph with min and max
    // infered above if the BuildSynapseGraph fails, call
    // handle_pass_exception with pass type OUTPUT_SHAPE. In handling this
    // exception bucket ranges are recalculated as per min and max both as
    // CURRENT and again call CompileAndRunDynamicGraph with changed ranges and
    // policy. This is last resort if anything further fails bail out the
    // execution. We need not change anything in cache because exception either
    // occurs in compilation or launch and both happens before adding recipie to
    // cache.
    try {
      m_map_shape.m_pass = ShapeInfo::InferencePass::INVALID;
      auto syn_graph =
          habana_helpers::create_graph(device.id(), GetSynapseGraphName());
      syn_graph.set_dynamic_graph(is_dynamic_graph);
      BuildSynapseGraph(syn_graph);
      CompileSynapseGraph();
      ConstructPatchingTable();
      UpdateSynapsePermutations();
      ExecuteSynapseGraph(hpu_stream);
    } catch (std::exception& e) {
      PT_DYNAMIC_SHAPE_DEBUG("Exception in BuildSynapseGraph");
      PT_DYNAMIC_SHAPE_DEBUG("Details:\n", e.what());
      ClearMembers(true);
      ClearStatics(true);
      PassException p(habana::ShapeInfo::InferencePass::OUTPUT_SHAPE, e.what());
      try_catch_fail = true;
      current_dbipsh_->get_statistics()->LogCompilation(
          jit_ir,
          jit_ir_graph,
          graph_input_info.min_policy,
          graph_input_info.max_policy,
          ranges,
          current_dbipsh_->GetRecipeKeyForBucket(
              graph_input_info.current_bucket_id),
          "dynamic compilation failed",
          last_compilation_pass);
      current_dbipsh_->get_statistics()->LogShapes(
          jit_ir_graph, graph_input_info.act_input_tshapes);
      current_dbipsh_->get_statistics()->LogSelectedRecipe(
          current_dbipsh_->GetRecipeKeyForBucket(
              graph_input_info.current_bucket_id),
          0);
      current_dbipsh_->get_statistics()->LogRecipeMemory(cur_rvalpsh);
      RestoreInputTensorMetadata();
      handle_pass_exception(graph_input_info, p);
    }
  } else {
    m_map_shape.m_pass = ShapeInfo::InferencePass::INVALID;
    auto syn_graph =
        habana_helpers::create_graph(device.id(), GetSynapseGraphName());
    syn_graph.set_dynamic_graph(is_dynamic_graph);
    EvictSynapseRecipe(graph_input_info.current_bucket_id);
    BuildSynapseGraph(syn_graph);
    CompileSynapseGraph();
    ConstructPatchingTable();
    UpdateSynapsePermutations();
    ExecuteSynapseGraph(hpu_stream);
  }
  if (!try_catch_fail) {
    current_dbipsh_->get_statistics()->LogCompilation(
        jit_ir,
        jit_ir_graph,
        graph_input_info.min_policy,
        graph_input_info.max_policy,
        ranges,
        current_dbipsh_->GetRecipeKeyForBucket(
            graph_input_info.current_bucket_id),
        result,
        last_compilation_pass);
    current_dbipsh_->get_statistics()->LogShapes(
        jit_ir_graph, graph_input_info.act_input_tshapes);
    bool refine_candidate = false;
    if (last_compilation_pass != habana_helpers::CompilationPass::STATIC) {
      refine_candidate =
          (current_dbipsh_->GetMFUBucket() ==
           graph_input_info.current_bucket_id);
    }
    current_dbipsh_->get_statistics()->LogUsedBucket(
        graph_input_info.current_bucket_id,
        jit_ir_graph,
        ranges,
        refine_candidate);
    current_dbipsh_->get_statistics()->LogSelectedRecipe(
        current_dbipsh_->GetRecipeKeyForBucket(
            graph_input_info.current_bucket_id),
        0);
    current_dbipsh_->get_statistics()->LogRecipeMemory(cur_rvalpsh);
  }
}

/*Optimizes the memory usage for chain of strided inserts by reusing the input
 * memory for the graph output. Such a use case is common in allreduce*/
void HabanaLaunchOpPT::ProcessStridedInsertAtOutput(
    torch::jit::Node* node,
    HabanaOperatorPtr HabanaKernel,
    torch::jit::Stack& input_stack,
    synapse_helpers::graph& syn_graph,
    const OutputMetaDataVector& outputs_metadata) {
  // strided insert as graph output (i.e. persistence set as true)
  bool is_reuse_input = false;
  auto val_ins = node->inputs();
  auto node_qual_str = node->kind().toQualString();

  // Check for unbroken chain of strided inserts from graph output to input
  torch::jit::Node* input_node = node;
  while ((strcmp(node_qual_str, "prim::Param") != 0)) {
    input_node = val_ins[0]->node();
    node_qual_str = input_node->kind().toQualString();

    std::string node_str(node_qual_str);

    // perform memory reuse if the chain has either strided/slice inserts or
    // inplace ops add control edges between consumers of inplace/ctrl edge
    // nodes and the last strided insert
    if (node_str.find("strided_insert") == std::string::npos &&
        node_str.find("slice_insert") == std::string::npos) {
      if (nodeRequiresControlEdge(input_node) ==
          ControlEdgeType::kCONTROL_EDGE_NONE) {
        break;
      } else {
        memory_reuse_pairs.emplace_back(
            std::make_pair(input_node->output(0), node));
      }
    }

    val_ins = input_node->inputs();
  }

  if (strcmp(node_qual_str, "prim::Param") == 0) {
    // reached input with unbroken chain of strided inserts
    is_reuse_input = true;
  }

  if (is_reuse_input == false) {
    OutputMetaDataVector md(1, outputs_metadata.at(0));
    md.at(0).persistent = true;
    HabanaKernel->AllocateAndAddSynapseNode(syn_graph, input_stack, md);
  } else {
    TORCH_CHECK(
        value_to_ivalue.count(val_ins[0]),
        "incorrect input for strided insert");
    const auto& ivalue = value_to_ivalue[val_ins[0]];
    TORCH_CHECK(
        pt_to_synapse_tensors.find(ivalue) != pt_to_synapse_tensors.end(),
        "incorrect ivalue for strided insert input");

    input_stack.insert(input_stack.end(), *ivalue);
    HabanaKernel->ReuseMemoryAndAddSynapseNode(
        syn_graph,
        input_stack,
        *pt_to_synapse_tensors[ivalue],
        outputs_metadata);

    /* Since memory is reused we need control edges between the consumers of the
    graph input (prim:param) and the strided insert at the graph output. Refer
    gtest LazyBasicKernelTest.allreducewithcontroledge*/
    // book keep the node pair that reuses same memory
    memory_reuse_pairs.emplace_back(std::make_pair(val_ins[0], node));
  }
}

void HabanaLaunchOpPT::set_lazy_front_end_info(
    std::shared_ptr<habana_lazy::HbLazyFrontEndInfoToBackend> info) {
  lazy_info = info;
}

bool HabanaLaunchOpPT::is_hccl_send_mark_step() {
  if (lazy_info == nullptr) {
    return false;
  }
  return lazy_info->get_is_hccl_send_mark_step();
}

} // namespace habana
