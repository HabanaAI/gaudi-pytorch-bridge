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
#include <ATen/native/Resize.h>
#include <ATen/record_function.h>
#include <absl/container/flat_hash_map.h>
#include <absl/container/flat_hash_set.h>
#include <absl/container/inlined_vector.h>
#include <absl/hash/hash.h>
#include <absl/memory/memory.h>
#include <absl/types/optional.h>
#include <torch/csrc/jit/ir/constants.h>
#include <torch/csrc/jit/runtime/interpreter.h>
#include <algorithm>
#include <iomanip>
#include <sstream>
#include <typeinfo>
#include <unordered_map>
#include "backend/backend_meta.h"
#include "backend/habana_device/HPUAllocator.h"
#include "backend/habana_device/tensor_builder.h"
#include "backend/helpers/compilation_statistics.h"
#include "backend/helpers/create_tensor.h"
#include "backend/helpers/eager_pipeline.h"
#include "backend/helpers/event_dispatcher.h"
#include "backend/helpers/graph.h"
#include "backend/helpers/tensor_utils.h"
#include "backend/jitgraph_utils.h"
#include "backend/kernel/control_edges_processing.h"
#include "backend/kernel/hpu_habana_compile_op_pt.h"
#include "backend/kernel/hpu_habana_meta_op_list.h"
#include "backend/kernel/hpu_shape_inference.h"
#include "backend/kernel/refinement_engine.h"
#include "backend/passes/hpu_habana_persistence_marker_pass.h"
#include "backend/synapse_helpers/env_flags.h"
#include "backend/synapse_helpers/tcmalloc_helper.h"
#include "habana_helpers/logging.h"
#include "habana_helpers/misc_utils.h"
#include "habana_kernels/hccl_kernels.h"
#include "habana_kernels/index_kernels.h"
#include "habana_kernels/lazy_kernels_declarations.h"
#include "habana_kernels/random_gen_kernels.h"
#include "habana_lazy/hpu_lazy_tensors.h"

using namespace torch::jit;
using namespace jitgraph_utils;
namespace habana {
std::unordered_set<std::string> HabanaLaunchOpPT::disabled_jit_ir_ops_ = {};
std::unordered_map<size_t, habana_helpers::InpTensorShapes>
    HabanaLaunchOpPT::ref_input_shape_map_ = {};
std::unordered_map<
    int,
    std::pair<size_t, std::vector<HabanaLaunchOpPT::constInfo_t>>>
    HabanaLaunchOpPT::m_const_checksum_map;
std::mutex HabanaLaunchOpPT::checksum_map_mtx;
//--------------------------------------

namespace HabanaLaunchOpPipeline {

class PipelineCallBase {
 public:
  virtual void operator()(bool) {
    PT_BRIDGE_FATAL("HabanaLaunchOpPT has been used without pipeline wrapper");
  }
};

PipelineCallBase NoPipeline;

class PipelineCall : public PipelineCallBase {
 public:
  virtual void operator()(bool sync_needed) override {
    sync_with_compile_stage_ = sync_needed;
  }
  bool is_called() {
    return sync_with_compile_stage_.has_value();
  }
  bool is_sync_needed() {
    return sync_with_compile_stage_.value();
  }

 private:
  std::optional<bool> sync_with_compile_stage_{};
};

void LoweringTask(
    std::unique_ptr<habana::HabanaLaunchOpPT>&& launch_op,
    torch::jit::Stack& stack,
    std::optional<std::vector<at::Tensor>> allocated_outputs) {
  PipelineCall pipeline_call;

  launch_op->run(stack, allocated_outputs, false, pipeline_call);

  if (!pipeline_call.is_called()) {
    PT_BRIDGE_DEBUG(
        "HabanaLaunchOpPT wraped by pipeline has been called but pipelined path haven't been chosen in run call");
    // TODO: we have to be sure that habana::eager::JoinPendingPipelineThreads()
    // has been called before
    return;
  }

  habana_helpers::Singleton_CompileThreadPool::getInstance()
      .ScheduleWorkAndUpdateThreadHandle(
          HabanaLaunchOpPipeline::CompileSynapseTask, std::move(launch_op));

  if (pipeline_call.is_sync_needed())
    habana_helpers::Singleton_CompileThreadPool::getInstance()
        .JoinPendingThread();
}
} // namespace HabanaLaunchOpPipeline

void HabanaLaunchOpPT::cleanUp() {
  ref_input_shape_map_ = {};
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
  if (id_str_ == std::string()) {
    if (IS_BRIDGE_DEBUG_ENABLED ||
        GET_ENV_FLAG_NEW(PT_COMPILATION_STATS_PATH)) {
      id_str_ = makeIdStr(name, g_index);
    } else {
      id_str_ = name;
    }
  }
  return id_str_;
}

HabanaLaunchOpPT::HabanaLaunchOpPT(
    std::shared_ptr<habana::OptimizedJITGraphAndMetaData>
        optimized_jit_graph_and_meta_data)
    : name_(optimized_jit_graph_and_meta_data->GetOpName()),
      graph_index_(optimized_jit_graph_and_meta_data->GetGraphIndex()),
      jit_ir_graph_(optimized_jit_graph_and_meta_data->get_cached_graph()),
      use_persistent_tensors{GET_ENV_FLAG_NEW(HABANA_USE_PERSISTENT_TENSOR)} {
  refine_ds_enabled_ = optimized_jit_graph_and_meta_data->GetDynamicGraph();
  enable_fast_shape_inf_ =
      GET_ENV_FLAG_NEW(PT_HPU_ENABLE_FAST_SHAPE_INFERENCE) &&
      refine_ds_enabled_;
  op_strs_ = optimized_jit_graph_and_meta_data->get_cached_opstrs();
  graph_key_ = optimized_jit_graph_and_meta_data->get_cached_graph_key();
  hpu_stream_ = optimized_jit_graph_and_meta_data->GetHPUStream();
  jit_graph_and_meta_data_ = optimized_jit_graph_and_meta_data;

  PT_BRIDGE_DEBUG(
      "Creating : ", SetAndGetSynapseGraphName(name_, graph_index_));

  auto front_end_type = jit_graph_and_meta_data_->GetFrontendType();
  execution_mode_ = front_end_type;

  // used for controlling recipe caching in non-eager backends
  enable_graph_caching_ =
      (execution_mode_ != habana_helpers::HabanaFrontendTypes::EAGER) &&
      GET_ENV_FLAG_NEW(PT_HPU_PGM_ENABLE_CACHE);

  // used for controlling recipe caching in eager backends
  // combined with PT_HPU_PGM_ENABLE_CACHE to allow debugging
  enable_eager_caching_ =
      ((execution_mode_ == habana_helpers::HabanaFrontendTypes::EAGER) &&
       (!jit_graph_and_meta_data_->get_is_eager_compiler_supported() &&
        GET_ENV_FLAG_NEW(PT_HPU_PGM_ENABLE_CACHE))) ||
      GET_ENV_FLAG_NEW(PT_HPU_ENABLE_EAGER_CACHE);

  enable_caching_ = enable_graph_caching_ || enable_eager_caching_;

  // Eager compiler is supported for Gaudi2 device.
  const auto& device = HPURegistrar::get_device();
  const bool is_eager_compiler_enabled = device.type() != synDeviceGaudi &&
      jit_graph_and_meta_data_->get_is_eager_compiler_supported() &&
      GET_ENV_FLAG_NEW(PT_HPU_ENABLE_EAGER_COMPILER);

  // Enable shape agnostic caching when in eager mode of execution and
  // eager compiler is supported and enabled and recipe cache is disabled
  enable_shape_agnostic_caching_ =
      (execution_mode_ == habana_helpers::HabanaFrontendTypes::EAGER) &&
      GET_ENV_FLAG_NEW(PT_HPU_EAGER_SHAPE_AGNOSTIC_GRAPH) &&
      is_eager_compiler_enabled && !enable_caching_;

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
        out_shapes.size() == jit_ir_graph_->outputs().size(),
        "number of output shapes for patching ",
        out_shapes.size(),
        " is not equal to #outputs in jit graph ",
        jit_ir_graph_->outputs().size());
  }

  auto frontend_type_eager_or_compile =
      ((front_end_type == habana_helpers::HabanaFrontendTypes::EAGER) ||
       (front_end_type == habana_helpers::HabanaFrontendTypes::COMPILE));
  enable_2stage_pipeline_ =
      jit_graph_and_meta_data_->get_is_pipeline_supported();
  // To Do - To also enable 4-stage pipeline for dynamic shapes
  enable_4stage_pipeline_ = enable_2stage_pipeline_ &&
      GET_ENV_FLAG_NEW(PT_HPU_EAGER_4_STAGE_PIPELINE_ENABLE) &&
      !refine_ds_enabled_ && frontend_type_eager_or_compile;
}

HabanaLaunchOpPT::~HabanaLaunchOpPT() {
  PT_BRIDGE_DEBUG("Destroying : ", GetSynapseGraphName());
}

void HabanaLaunchOpPT::clearRecipeCacheForConst() {
  m_const_checksum_map.clear();
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
    tensorList->emplace_back(synapse_helpers::tensor_or_ref(syn_tensor));
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
        *syn_graph_ptr_, pt_tensor, true, tmeta->get_tensor_type(), host_ptr);
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
        *syn_graph_ptr_, pt_tensor, true, DATA_TENSOR, nullptr, idx);

    if (is_duplicate_syn_tensor) {
      habana_op->set_is_duplicate_input_flag(false);
      habana_op->clear_syn_input_tensor_orig();
    }

    if (pt_tensor_buffer_start != nullptr) {
      buff_to_syn_tensor_map.emplace(
          pt_tensor_buffer_start, synapse_helpers::tensor_or_ref(syn_tensor));
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

    tensorList->emplace_back(synapse_helpers::tensor_or_ref(syn_tensor));

    std::string irn = "%" + value_in->debugName();
    PtTensorInfoShared ti = std::make_shared<PtTensorInfo>(
        pt_tensor,
        syn_tensor.name(),
        irn,
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
    if (habana_helpers::IsInferenceMode()) {
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
    case torch::jit::aten::exponential:
    case torch::jit::aten::_fused_dropout:
    case torch::jit::aten::normal:
      populate_seed = true;
      break;
  }

  if (populate_seed) {
    int seed = get_seed_hpu(c10::nullopt);

    at::Tensor seed_cpu_tensor = at::tensor(seed);
    at::Tensor seed_tensor = at::empty(
        seed_cpu_tensor.sizes(),
        seed_cpu_tensor.options().device(c10::DeviceType::HPU),
        c10::MemoryFormat::Contiguous);
    habana_helpers::copy_data_to_device(seed_cpu_tensor, seed_tensor, false);

    auto& syn_tensor = habana_op->AllocateSeed(*syn_graph_ptr_, seed_tensor);
    PtTensorInfoShared ti = std::make_shared<PtTensorInfo>(
        seed_tensor,
        syn_tensor.name(),
        "%seed_input",
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
    InferOutputMetaRetType& op_output_shape) {
  auto output_nodes = node->outputs();
  auto habana_kernel_meta_data = habana_op->GetKernelMetaData();

  bool shape_inf_flag = enable_fast_shape_inf_ &&
      syn_graph_ptr_->is_dynamic_graph() &&
      !(m_map_shape.m_pass == ShapeInfo::InferencePass::MIN_SHAPE ||
        m_map_shape.m_pass == ShapeInfo::InferencePass::MAX_SHAPE);

  if ((shape_inf_flag || enable_shape_agnostic_caching_) &&
      !op_output_shape.empty()) {
    size_t exclude_outputs = 0;
    if (auto op = std::dynamic_pointer_cast<OpBackend>(habana_op)) {
      exclude_outputs = op->GetSynImplicitOutputs().size();
    }
    HABANA_ASSERT(
        habana_op->GetSynOutputs().size() ==
            op_output_shape.GetOutputTensor().size() - exclude_outputs,
        "For node ",
        node->kind().toQualString(),
        "GetSynOutputs().size()=",
        habana_op->GetSynOutputs().size(),
        ", whereas GetOutputTensor().size()=",
        op_output_shape.GetOutputTensor().size(),
        ", GetSynImplicitOutputs().size()=",
        exclude_outputs);
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
    tensorList->emplace_back(synapse_helpers::tensor_or_ref(sh_t));
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
        const auto& out_val = output_nodes[output_nodes_idx];
        auto ti = ProcessPersistentNodeOutput(ivpsh, out_val, out_tensor_syn);

        handle_permutes(ti, out_tensor_syn, ivpsh);

        // persistent intermediate synapse tensor i.e. out_tensor_syn
        if (false == isInGraphOutputs(out_val)) {
          intermediate_syn_tensors_count_++;
        }
        // Add node output tinfo i.e. graph output for multiple nodes graph
        // to get shape via shape inference, for ex strided insert
        // ToDo: Fix output shape info from frontend for strided insert
        //       when adding node params patching support.
        constexpr bool use_output_shape = true;
        bool shape_agn_flag = enable_shape_agnostic_caching_ &&
            (intermediate_syn_tensors_count_ > 0);
        handle_shape_inf(ti, use_output_shape, shape_agn_flag);
      } else if (enable_shape_agnostic_caching_) {
        // For shape agnostic flow for eager we need non-persistent info as well
        // Try maintaing it in another struct other than dtensor info struct
        PtTensorInfoShared ti = std::make_shared<PtTensorInfo>(
            out_tensor_syn.name(),
            out_tensor_syn.id(),
            out_tensor_syn.get(),
            out_tensor_syn.tensor_type());
        constexpr bool use_output_shape = true;
        handle_shape_inf(ti, use_output_shape, enable_shape_agnostic_caching_);
        // non-persistent intermediate synapse tensor
        intermediate_syn_tensors_count_++;
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
          sh_t.id(),
          sh_t.get(),
          sh_t.tensor_type());

      ti->set_external(sh_t.is_external());
      duplicate_input_tivs.emplace_back(ti);

      // persistent tensor which an alias of an input
      PT_BRIDGE_DEBUG("Adding to duplicate_input_tivs ", *ti);
      implicit_syn_tensors_count_++;

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
    const InferOutputMetaRetType& output,
    std::vector<IdxTensorTuple>& intermediate_shape_tensor_cs) {
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
      syn_graph_ptr_->is_dynamic_graph() &&
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
    if (maybe_syn_shape_tensor.is_shape_tensor()) {
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
        *tensor, *syn_graph_ptr_, persistence, false);
    meta_syn_tensors.push_back((std::move(variant)));
  }

  auto& syn_tensor = meta_syn_tensors.back();
  pt_to_synapse_tensors.erase(value_to_ivalue[value_in]);
  SharedSynTensorOrRefListPtr tensorList =
      std::make_shared<SynTensorOrRefList>();
  tensorList->emplace_back(synapse_helpers::tensor_or_ref(syn_tensor));
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
      jit_graph_and_meta_data_->get_jit_cached_graph_info_available_flag();

  if (is_jit_cached_graph_info_available == false) {
    bool is_in_graph_outputs = isInGraphOutputs(value_out);
    jit_graph_and_meta_data_->set_is_in_graph_outputs(is_in_graph_outputs);
  }
  auto is_in_graph_outputs = jit_graph_and_meta_data_->get_is_in_graph_outputs(
      restride_node_out_val_counter);
  restride_node_out_val_counter++;

  if ((tensor.dim() == 4) || (tensor.dim() == 5)) {
    if (is_jit_cached_graph_info_available == false) {
      auto new_pos = toIValue(node->input(1))->toIntVector();
      jit_graph_and_meta_data_->set_new_pos(new_pos);
    }
    std::vector<int64_t>& new_pos =
        jit_graph_and_meta_data_->get_new_pos(restride_node_swap_counter);
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
        *syn_graph_ptr_, syn_tensor.id(), tensor.sizes().vec());
    PtTensorInfoShared ti = std::make_shared<PtTensorInfo>(
        ivpsh_restrided,
        syn_tensor.name(),
        value_in,
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

IValPtrShared GetPrimListConstructNodeOuputIValue(
    torch::jit::Node* node,
    CValuePtrToIValuePtrMap& value_to_ivalue) {
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
    IValPtrShared out_ival = std::make_shared<IVal>(opttensorList);
    return out_ival;
  }

  // Handle empty list
  if (node_ins.empty()) {
    return std::make_shared<IVal>(c10::List<int64_t>());
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
    IValPtrShared out_ival = std::make_shared<IVal>(tensorList);
    return out_ival;
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
    IValPtrShared out_ival = std::make_shared<IVal>(intList);
    return out_ival;
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
    IValPtrShared out_ival = std::make_shared<IVal>(boolList);
    return out_ival;
  }

  HABANA_ASSERT(false, "Unsupported list type in prim::ListConstruct");
}

at::Tensor createDynamicTensor(
    const std::vector<int64_t>& size,
    synTensorType type) {
  auto allocator = habana::getHABANADeviceAllocator();
  constexpr c10::DispatchKeySet hpu_ks(c10::DispatchKey::HPU);
  auto dtype = c10::ScalarType::Float;

  at::Tensor tensor = at::detail::empty_generic(
      at::asIntArrayRefUnchecked({0}), allocator, hpu_ks, dtype, c10::nullopt);

  auto tmeta{habana::get_tensor_extra_meta(tensor)};
  tmeta->set_tensor_type(type);
  tensor.unsafeGetTensorImpl()->set_sizes_contiguous(size);
  PT_EAGER_DEBUG(
      "Created dynamic tensor of type:", type, ", size:", tensor.sizes());
  return tensor;
}

void HabanaLaunchOpPT::handlePrimListConstructNode(torch::jit::Node* node) {
  auto node_vals = node->outputs();
  HABANA_ASSERT(node_vals.size() == 1);
  IValPtrShared ival =
      GetPrimListConstructNodeOuputIValue(node, value_to_ivalue);
  value_to_ivalue[node_vals[0]] = ival;
}

void HabanaLaunchOpPT::handlePrimConstantNode(torch::jit::Node* node) {
  auto node_vals = node->outputs();
  bool is_jit_cached_graph_info_available =
      jit_graph_and_meta_data_->get_jit_cached_graph_info_available_flag();
  for (const auto value : node_vals) {
    IValPtrShared ivptrsh = nullptr;
    if (is_jit_cached_graph_info_available == false) {
      ivptrsh = std::make_shared<IVal>(toIValue(value).value());
    }
    if (value->type()->kind() == c10::TypeKind::TensorType) {
      if (is_jit_cached_graph_info_available == false) {
        auto ivptrshUpdated = castConstantTensor(ivptrsh);
        jit_graph_and_meta_data_->set_prim_nodes_ival(ivptrshUpdated);
      }
      auto ivptrsh_updated = jit_graph_and_meta_data_->get_prim_nodes_ival(
          prim_nodes_ival_counter);
      value_to_ivalue[value] = ivptrsh_updated;
      std::string irn{"%intermediate_"};
      irn += std::to_string(intermediate_index);
      intermediate_index++;

      auto tensor = ivptrsh_updated->toTensor();
      meta_syn_tensors.push_back(habana_helpers::create_tensor(
          tensor, *syn_graph_ptr_, true, false, tensor.scalar_type()));
      SharedSynTensorOrRefListPtr tensorList =
          std::make_shared<SynTensorOrRefList>();
      tensorList->emplace_back(
          synapse_helpers::tensor_or_ref(meta_syn_tensors.back()));
      pt_to_synapse_tensors.emplace(value_to_ivalue[value], tensorList);
      PtTensorInfoShared ti = std::make_shared<PtTensorInfo>(
          tensor,
          meta_syn_tensors.back().name(),
          irn,
          meta_syn_tensors.back().id(),
          meta_syn_tensors.back().get(),
          meta_syn_tensors.back().tensor_type());

      ivalue_to_tensor_info_map[ivptrsh_updated] = ti;
      aten_intermediates.push_back(tensor);
    } else {
      if (is_jit_cached_graph_info_available == false) {
        jit_graph_and_meta_data_->set_prim_nodes_ival(ivptrsh);
      } else {
        ivptrsh = jit_graph_and_meta_data_->get_prim_nodes_ival(
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
        tensorList->emplace_back(synapse_helpers::tensor_or_ref(meta_syn_tensors.back()));
        pt_to_synapse_tensors.emplace(value_to_ivalue[value_in], tensorList);

        if (enable_caching_) {
          input_tiv_map.emplace(
              value_to_ivalue[value_in],
              PtTensorInfo(
                  value_to_ivalue[value_in],
                  meta_syn_tensors.back().name(),
                  value_in));
          buff_to_input_ivpsh_map.emplace(in_data, value_to_ivalue[value_in]);
        } else {
          input_tivs.emplace_back(PtTensorInfo(
              value_to_ivalue[value_in],
              meta_syn_tensors.back().name(),
              value_in));
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

namespace {
std::string DumpNodeInputs(
    const torch::jit::Node* const node,
    const CValuePtrToIValuePtrMap& value_to_ivalue) {
  std::ostringstream o;
  node->print(o, 0, nullptr);
  auto str = o.str();
  if (node->input(0)->type() != torch::ListType::ofTensors()) {
    for (auto value_in : node->inputs()) {
      const auto ivalue = value_to_ivalue.find(value_in);
      if (ivalue != value_to_ivalue.end() && ivalue->second->isTensor()) {
        const auto tensor = ivalue->second->toTensor();
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

std::string DumpNodeOutputs(
    const torch::jit::Node* const node,
    const CValuePtrToIValuePtrMap& value_to_ivalue) {
  std::ostringstream o;
  node->print(o, 0, nullptr);
  auto str = o.str();
  if (*node->output(0)->type() != *torch::ListType::ofTensors()) {
    for (auto value_out : node->outputs()) {
      const auto ivalue = value_to_ivalue.find(value_out);
      if (ivalue != value_to_ivalue.end() && ivalue->second->isTensor()) {
        const auto tensor = ivalue->second->toTensor();
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
} // namespace

void HabanaLaunchOpPT::validateOutputShapeDynamic(
    const HabanaOperatorPtr& HabanaKernel,
    const InferOutputMetaRetType& output_shape_handle,
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
  auto num_undefined_output_tensors =
      output_shape_handle.GetNumUndefinedOutputTensors();

  size_t exclude_outputs = 0;
  if (auto op = std::dynamic_pointer_cast<OpBackend>(HabanaKernel)) {
    exclude_outputs = HabanaKernel->GetSynImplicitOutputs().size();
  }

  auto num_outs_expected = syn_outputs.size() +
      intermediate_shape_tensor_count - num_undefined_output_tensors;
  auto num_outs_got = output_size - exclude_outputs;
  HABANA_ASSERT(
      num_outs_expected == num_outs_got,
      "Node: ",
      opname,
      " number of output mismatch, expected: ",
      num_outs_expected,
      " but got: ",
      num_outs_got);
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
  // before intermediate shape tensor in InferOutputMeta
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
    const InferOutputMetaRetType& output_shape_handle,
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
  auto num_undefined_output_tensors =
      output_shape_handle.GetNumUndefinedOutputTensors();

  size_t exclude_outputs = 0;
  if (auto op = std::dynamic_pointer_cast<OpBackend>(HabanaKernel)) {
    exclude_outputs = HabanaKernel->GetSynImplicitOutputs().size();
  }

  auto num_outs_expected = syn_outputs.size() - num_undefined_output_tensors;
  auto num_outs_got = output_size - exclude_outputs;
  HABANA_ASSERT(
      num_outs_expected == num_outs_got,
      "Node: ",
      opname,
      " number of output mismatch, expected: ",
      num_outs_expected,
      " but got: ",
      num_outs_got);
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
    const InferOutputMetaRetType& output_shape_handle,
    const synapse_helpers::graph& syn_graph,
    const std::string& opname) {
  auto lowering_kernels = HabanaKernel->GetKernels();

  if (syn_graph.is_dynamic_graph()) {
    validateOutputShapeDynamic(HabanaKernel, output_shape_handle, opname);
  } else {
    validateOutputShapeNonDynamic(HabanaKernel, output_shape_handle, opname);
  }
}

bool is_allow_view_output_permutation(const at::Tensor& t) {
  auto tmeta{habana::get_tensor_extra_meta(t)};
  if (!tmeta->is_view_tensor())
    return true;
  if (tmeta->is_maybe_grad_view()) {
    PT_BRIDGE_DEBUG("Allowed view output permutation. sizes: ", t.sizes())
    return true;
  } else
    return false;
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
  auto is_allow = is_allow_view_output_permutation(ivpsh->toTensor());
  if ((rank >= 2) && is_allow) {
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

namespace {

// Utilies for marking constant tensors in JIT graph as consts in
// Synapse graph. It works when parameter marking is done
// from the model
void ProcessGraphForConstantTensors(
    const torch::jit::Graph& jit_ir_graph,
    const CValuePtrToIValuePtrMap& value_to_ivalue) {
  PT_BRIDGE_BEGIN;
  if (!habana_helpers::IsInferenceMode()) {
    PT_BRIDGE_END;
    return;
  }
  for (const auto* const value_input : jit_ir_graph.inputs()) {
    auto ivalue = value_to_ivalue.find(value_input);
    if (ivalue == value_to_ivalue.end() || !ivalue->second->isTensor()) {
      continue;
    }
    auto tensor = ivalue->second->toTensor();
    if (habana::is_tensor_const(tensor)) {
      TensorExtraMeta::set_const_tensor(tensor, true);
    }
  }
  PT_BRIDGE_END;
}
} // namespace

void HabanaLaunchOpPT::FillMaxValues(
    const HabanaOperatorPtr& habana_op,
    const torch::jit::Stack& input_stack,
    std::unordered_map<int64_t, std::vector<int64_t>>& index2maxvalues) {
  for (size_t i = 0; i < input_stack.size(); ++i) {
    auto& input_tensor = input_stack[i];
    if (input_tensor.isTensor()) {
      auto tmeta = get_tensor_extra_meta(input_tensor.toTensor());
      if (tmeta->get_tensor_type() == HOST_TO_DEVICE_TENSOR &&
          tmeta->peek_H2D_data_for_bucketing()) {
        index2maxvalues[i] = SliceOperator::GetH2DTensorData(
            input_tensor.toTensor(), true, false);
      } else {
        index2maxvalues[i] = std::get<1>(habana::ShapeInference::GetMinMaxShape(
            habana_op->SynInput(i).ref().id()));
      }
    }
  }
}

void HabanaLaunchOpPT::UpdateMaxValues(
    const HabanaOperatorPtr& habana_op,
    const torch::jit::Stack& input_stack,
    std::unordered_map<int64_t, std::vector<int64_t>>& index2maxvalues) {
  for (size_t i = 0; i < input_stack.size(); ++i) {
    auto& input_tensor = input_stack[i];
    std::vector<int64_t> max_new, max_old;
    if (input_tensor.isTensor()) {
      auto tmeta = get_tensor_extra_meta(input_tensor.toTensor());
      auto ivalHash = input_tensor.hash().toInt();
      auto input_idx = ival_hash_to_input_index_map_[ivalHash];
      if (tmeta->get_tensor_type() == HOST_TO_DEVICE_TENSOR &&
          tmeta->peek_H2D_data_for_bucketing()) {
        max_old = index2maxvalues[i];
        max_new = SliceOperator::GetH2DTensorData(
            input_tensor.toTensor(), true, false);
      } else {
        max_new = std::get<1>(habana::ShapeInference::GetMinMaxShape(
            habana_op->SynInput(i).ref().id()));
        max_old = index2maxvalues[i];
      }
      HABANA_ASSERT(max_new.size() == max_old.size());
      for (size_t j = 0; j < max_new.size(); ++j) {
        if (max_new[j] != max_old[j]) {
          PT_DYNAMIC_SHAPE_DEBUG(
              "Need to update dynamic ranges of bucket id ",
              current_bucket_id_,
              " from ",
              max_old[j],
              " to ",
              max_new[j],
              " at input_idx ",
              input_idx,
              " dim ",
              j);
          {
            std::lock_guard<std::mutex> lg(current_dbipsh_->get_refine_mutex());
            current_dbipsh_->UpdateShapes(
                current_bucket_id_, input_idx, j, max_new[j]);
          }
        }
      }
    }
  }
  index2maxvalues.clear();
}

void HabanaLaunchOpPT::UpdatePTStack(DynamicShapeInfo& graph_input_info) {
  auto ranges =
      current_dbipsh_->CalculateShapes(graph_input_info.current_bucket_id);
  graph_input_info.max_input_tshapes.clear();
  graph_input_info.max_input_tshapes.insert(
      ranges.max_shapes.begin(), ranges.max_shapes.end());
  SetH2DMinMaxData(
      *pt_stack,
      graph_input_info.max_input_tshapes,
      ShapeInfo::InferencePass::MAX_SHAPE);
  updatemax_graph = false;
}

void HabanaLaunchOpPT::RevertH2DMinMaxData() {
  for (size_t i = 0; i < pt_stack->size(); ++i) {
    if (pt_stack->at(i).isTensor()) {
      auto& input_tensor = pt_stack->at(i);
      auto tmeta = get_tensor_extra_meta(pt_stack->at(i).toTensor());
      if (tmeta->get_tensor_type() == HOST_TO_DEVICE_TENSOR &&
          tmeta->peek_H2D_data_for_bucketing()) {
        auto ivalHash = input_tensor.hash().toInt();
        auto input_idx = ival_hash_to_input_index_map_[ivalHash];
        if (!current_dbipsh_->IsBucketMember(input_idx, current_bucket_id_)) {
          auto host_ptr = tmeta->get_host_ptr();
          size_t data_size = tmeta->get_host_size() * tmeta->get_host_el_size();
          auto compile_host_ptr = tmeta->get_compile_host_ptr();
          memcpy(compile_host_ptr, host_ptr, data_size);
        }
      }
    }
  }
  updatemax_graph = false;
}

void HabanaLaunchOpPT::BuildSynapseGraph(
    std::shared_ptr<synapse_helpers::graph>& syn_graph,
    bool is_shape_inference) {
  PT_BRIDGE_BEGIN;
  // figure out the right device id
  auto& device = HPURegistrar::get_device();
  synDeviceId device_id = device.id();

  synapse_helpers::detail::tensor_name_generator::reset();

  syn_graph_ptr_ = syn_graph;

  if (current_dbipsh_) {
    jit_graph_and_meta_data_->clear_cached_graph_info();
    prim_nodes_ival_counter = 0;
    restride_node_swap_counter = 0;
    restride_node_out_val_counter = 0;
  }

  // for each node in IR graph, at this point the graph is a list with nodes
  // topoloically sorted
  // TODO : check if we need to reorder nodes in any case
  torch::jit::graph_node_list graph_nodes = jit_ir_graph_->nodes();

  // This is an optimization pass to mark all the nodes with sepcial layout
  // like weights which have HWCK Only activated in lazy mode for now
  bool is_jit_cached_graph_info_available =
      jit_graph_and_meta_data_->get_jit_cached_graph_info_available_flag();
  if (is_jit_cached_graph_info_available == false) {
    persistence_marker_pass_data_ptr_ =
        PersistenceMarkerPass(this).VisitGraph(jit_ir_graph_);
  }

  habana::ShapeInference::ResetSifTensorId();

  std::optional<std::vector<at::Tensor>::iterator> allocated_outputs_iter;
  if (allocated_outputs_.has_value()) {
    allocated_outputs_iter = allocated_outputs_->begin();
  }

  size_t outputs_metadata_index = 0;
  // Collect inputs shape tensors accross all nodes
  std::vector<size_t> inputs_shape_tensors_vec;
  // Collect intermediate shape tensors accross all nodes for not supporting
  // InferOutputMeta
  std::vector<size_t> intermediate_shape_tensors_vec;
  std::vector<std::pair<torch::jit::Value*, torch::jit::Node*>>
      memory_reuse_pairs;
  int inx = 0;
  PT_OP_DEBUG("JIT Graph: ", jit_ir_graph_->toString());
  for (auto* node : graph_nodes) {
    std::vector<IdxTensorTuple> intermediate_shape_tensor_cs;
    auto node_qual_str = node->kind().toQualString();
    std::string opname(node_qual_str);

    PT_BRIDGE_DEBUG("Working on ", node_qual_str);

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

    // Set the deterministic val
    PT_BRIDGE_DEBUG(
        "Deterministic value in BuildGraph: ",
        node->i(torch::jit::attr::deterministic));
    HabanaKernel->setDeterministic(node->i(torch::jit::attr::deterministic));

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
    syn_graph_ptr_->clear_node_indices();
    // set op name in synapse graph
    std::unique_ptr<synapse_helpers::graph::OpNameContext> op_name_context;
    const auto scope = node->scope();
    if (!scope->isBlank()) {
      op_name_context = std::make_unique<synapse_helpers::graph::OpNameContext>(
          *syn_graph, scope->name().toUnqualString());
    }

    torch::jit::Stack input_stack = getStackForNode(node);

    // If there is a "meta attribute" marked with attr::arg1, add the meta attr
    // value to stack for the ops to work with. At this point, only StridedView
    // ops in eager mode uses it.
    auto meta = torch::jit::attr::arg1;
    if (node->hasAttribute(meta)) {
      HABANA_ASSERT(
          !strcmp("aten::as_strided", node_qual_str),
          "Meta op can only be marked for aten::as_strided, not supported in op ",
          node_qual_str);
      input_stack.insert(input_stack.end(), IValue(node->i(meta)));
      meta_attribute_nodes_count_++;
    }

    // Create/attach the synapse inputs from aten tensors
    GetSynapseInputs(HabanaKernel, node);

    // setup the config params for the kernels
    if (is_jit_cached_graph_info_available == false) {
      auto outputs_metadata = nodeOutputMetaData(node);
      jit_graph_and_meta_data_->set_outputs_metadata(outputs_metadata);
    }
    PT_BRIDGE_DEBUG(DumpNodeInputs(node, value_to_ivalue));
    OutputMetaDataVector& outputs_metadata =
        jit_graph_and_meta_data_->get_outputs_metadata(outputs_metadata_index);
    outputs_metadata_index++;

    if (!dry_run_ && allocated_outputs_iter.has_value()) {
      c10::ArrayRef<torch::jit::Value*> node_outputs = getNodeOutputs(node);
      for (auto [itm, itn] =
               std::tuple{outputs_metadata.begin(), node_outputs.begin()};
           itm != outputs_metadata.end();
           ++itm, ++itn) {
        if (jitgraph_utils::isInGraphOutputs(*itn)) {
          HABANA_ASSERT(
              allocated_outputs_iter.value() !=
                  allocated_outputs_.value().end(),
              "number of allocated_outputs_ is smaller than numer of outputs found in JIT graph");
          itm->allocated_tensor = *allocated_outputs_iter.value();
          allocated_outputs_iter.value()++;
        }
      }
    }

    std::string module_name = node->scope()->name().toUnqualString();
    if (habana_helpers::IsInferenceMode() &&
        (strcmp(node->kind().toQualString(), "aten::view") == 0)) {
      auto val_ins = node->inputs();
      module_name = val_ins[0]->node()->scope()->name().toUnqualString();
    }
    if (habana_helpers::IsInferenceMode() && module_name.size() > 0) {
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
    if ((outputs_metadata.size() == 1) && !is_shape_inference &&
        (outputs_metadata.at(0).persistent == true) &&
        (habana::control_edges::IsNodeStridedInsertOrSliceInsert(opname) ||
         (opname.find("strided_view_out") != std::string::npos))) {
      habana::control_edges::ProcessStridedInsertAtOutput(
          node,
          HabanaKernel,
          input_stack,
          syn_graph,
          outputs_metadata,
          memory_reuse_pairs,
          value_to_ivalue,
          pt_to_synapse_tensors);
    } else {
      std::unordered_map<int64_t, std::vector<int64_t>> index2maxvalues;
      // Currently max update which is less than bucket range issue exists for
      // slice. If other node needs this, can be added here.
      bool updatemax_node =
          ((habana::ShapeInference::GetCurrentPass() ==
            habana::ShapeInfo::InferencePass::MAX_SHAPE) &&
           (habana::ShapeInference::GetMaxPolicyInUse() ==
            habana_helpers::DynamicDimsPolicy::CALCULATED) &&
           ((strcmp(node->kind().toQualString(), "hpu::slice") == 0) ||
            strcmp(node->kind().toQualString(), "hpu::slice_ht") == 0));
      updatemax_graph |=
          ((habana::ShapeInference::GetCurrentPass() ==
            habana::ShapeInfo::InferencePass::MAX_SHAPE) &&
           (habana::ShapeInference::GetMaxPolicyInUse() ==
            habana_helpers::DynamicDimsPolicy::CALCULATED) &&
           strcmp(node->kind().toQualString(), "hpu::slice_ht") == 0);
      if (updatemax_node) {
        FillMaxValues(HabanaKernel, input_stack, index2maxvalues);
      }
      // Check Compute output shapes for mismatch else raise exception
      if (syn_graph->is_dynamic_graph()) {
        if (auto op = std::dynamic_pointer_cast<OpBackend>(HabanaKernel))
          op->ComputeOutputShapes(input_stack);
      }

      HabanaKernel->AllocateAndAddSynapseNode(
          *syn_graph, input_stack, outputs_metadata);
      HabanaKernel->dump(node, input_stack);
      if (updatemax_node) {
        UpdateMaxValues(HabanaKernel, input_stack, index2maxvalues);
      }
    }

    static std::unordered_set<std::string> cs_jit_ir_ops_;
    static std::unordered_set<std::string> empty_cs_jit_ir_ops_;

    habana::InferOutputMetaRetType kernel_output_cs(true);
    if (!disabled_jit_ir_ops_.count(node_qual_str)) {
      // Either the InferOutputMeta flow is getting validated or
      // fast shape inference is running for dynamic shapes or
      // shape agnostic flow is enabled for eager.
      if (GET_ENV_FLAG_NEW(PT_HPU_VALIDATE_COMPUTE_SHAPE) ||
          (enable_fast_shape_inf_ && syn_graph->is_dynamic_graph()) ||
          enable_shape_agnostic_caching_) {
        PT_DYNAMIC_SHAPE_DEBUG(
            "Current sif tensor id = ",
            habana::ShapeInference::GetSifTensorId());
        HabanaOperatorPtr csHabanaKernel =
            KernelRegistry().get(device_id, op, getNodeScalarType(node));

        PT_BRIDGE_DEBUG(
            "Deterministic value in BuildGraph: ",
            node->i(torch::jit::attr::deterministic));
        HabanaKernel->setDeterministic(
            node->i(torch::jit::attr::deterministic));

        // Set output meta data if auto-gen op
        if (auto op = std::dynamic_pointer_cast<OpBackend>(csHabanaKernel)) {
          op->SetOutputMetadata(outputs_metadata);
        }
        kernel_output_cs = csHabanaKernel->InferOutputMeta(input_stack);
        if (!kernel_output_cs.empty()) {
          // Output shape info based flow
          PT_DYNAMIC_SHAPE_DEBUG(
              "After InferOutputMeta for ",
              node_qual_str,
              ": sif tensor id = ",
              habana::ShapeInference::GetSifTensorId());
          if (enable_fast_shape_inf_ && !is_shape_inference &&
              syn_graph->is_dynamic_graph()) {
            ProcessShapeTensorsCS(
                kernel_output_cs, intermediate_shape_tensor_cs);
          }
          try {
            validateOutputShape(
                HabanaKernel, kernel_output_cs, *syn_graph, opname);
            if (cs_jit_ir_ops_.count(node_qual_str) == 0) {
              PT_DYNAMIC_SHAPE_DEBUG(
                  "InferOutputMeta_JIT_IR_OP: ", node_qual_str);
              cs_jit_ir_ops_.insert(node_qual_str);
            }
          } catch (std::exception& e) {
            kernel_output_cs.set_empty();
            if (disabled_jit_ir_ops_.count(node_qual_str) == 0) {
              PT_DYNAMIC_SHAPE_DEBUG(
                  "DISABLED_InferOutputMeta_JIT_IR_OP: ", node_qual_str);
              disabled_jit_ir_ops_.insert(node_qual_str);
            }
            TORCH_CHECK(
                false == GET_ENV_FLAG_NEW(PT_HPU_VALIDATE_COMPUTE_SHAPE),
                "InferOutputMeta validation failed for op ",
                node_qual_str,
                " what(): ",
                e.what());
          }
        } else {
          if (empty_cs_jit_ir_ops_.count(node_qual_str) == 0) {
            PT_DYNAMIC_SHAPE_DEBUG(
                "Empty_InferOutputMeta_JIT_IR_OP: ", node_qual_str);
            empty_cs_jit_ir_ops_.insert(node_qual_str);
          }
          TORCH_CHECK(
              false == GET_ENV_FLAG_NEW(PT_HPU_VALIDATE_COMPUTE_SHAPE),
              "InferOutputMeta method not available for validation of op ",
              node_qual_str);
        }
      }
    }

    jit_to_synapse_node_idx_map.emplace(
        node, syn_graph_ptr_->get_node_indices());
    syn_graph_ptr_->clear_node_indices();

    if (refine_ds_enabled_ && (!is_shape_inference)) {
      // Process both synapse input and intermediate shape tensors
      std::vector<size_t> intermediate_shape_tensors;
      ProcessSynapseShapeTensors(
          HabanaKernel, intermediate_shape_tensors, inputs_shape_tensors_vec);
      if (enable_fast_shape_inf_ && syn_graph->is_dynamic_graph()) {
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
        enable_fast_shape_inf_ && syn_graph->is_dynamic_graph() &&
        !is_shape_inference;
    if ((dynamic_compile_graph || enable_shape_agnostic_caching_) &&
        kernel_output_cs.empty()) {
      // Increment the sif tensor id
      auto output_count = get_output_tensors_count(HabanaKernel, *syn_graph);
      habana::ShapeInference::IncrementSifTensorId(output_count);
      PT_DYNAMIC_SHAPE_DEBUG(
          "After increment: sif tensor id = ",
          habana::ShapeInference::GetSifTensorId(),
          " should match with ProcessSynapseOutputs return value = ",
          cur_sif_tidx);
    }

    PT_BRIDGE_DEBUG(DumpNodeOutputs(node, value_to_ivalue));
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
              tensor, tensor_name, irn, tensor_id);
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
       syn_graph->is_dynamic_graph() && !is_shape_inference)) {
    for (size_t i = 0; i < jit_ir_graph_->inputs().size(); ++i) {
      auto input = jit_ir_graph_->inputs().at(i);
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
    // supporting InferOutputMeta during fast sif
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
        if (!is_allow_view_output_permutation(ival->toTensor())) {
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
        if (!is_allow_view_output_permutation(ival->toTensor())) {
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

  if (refine_ds_enabled_ ||
      (not jit_graph_and_meta_data_
               ->get_jit_cached_graph_info_available_flag()) ||
      jit_graph_and_meta_data_->get_is_control_edge_processing_required()) {
    // Process control edges
    control_edges::ProcessControlEdges(
        *jit_ir_graph_,
        *jit_graph_and_meta_data_,
        jit_to_synapse_node_idx_map,
        memory_reuse_pairs,
        syn_graph_ptr_.get());
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

void HabanaLaunchOpPT::CreateStaticCompilationDBI(size_t graph_key_with_perm) {
  std::string path = GET_ENV_FLAG_NEW(PT_COMPILATION_STATS_PATH);

  if (!ref_input_shape_map_.count(graph_key_with_perm)) {
    habana_helpers::InpTensorShapes input_tshapes;
    CreateDynamicBucketInputShapes(input_tshapes);
    ProcessDynamicBucketInputShapesWithH2D(input_tshapes);
    PT_BRIDGE_DEBUG(
        "JIT IR graph_hash_code : ",
        graph_key_,
        ", hash_code with data layout : ",
        graph_key_with_perm,
        "\nRecording the reference input shapes::",
        input_tshapes,
        "\n--------------------");
    ref_input_shape_map_.emplace(graph_key_with_perm, input_tshapes);
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
    auto value_input = jit_ir_graph_->inputs().at(j);
    auto ivpsh = pt_stack_sh[j];
    value_to_ivalue[value_input] = ivpsh;
    ival_hash_to_input_index_map_[ivpsh->hash().toInt()] = j;
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
        if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) == 0) {
          new_tensor = createDynamicTensor(tensor.sizes().vec(), tensor_type);
        } else {
          new_tensor = habana_lazy::empty_hpu_lazy(
              tensor.sizes(),
              tensor.options(),
              tensor.suggest_memory_format(),
              true,
              tensor_type);
        }
      } else if (tensor_type == SHAPE_TENSOR) {
        if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) == 0) {
          new_tensor =
              createDynamicTensor(dynamic_shapes.at(i).get_dims(), tensor_type);
        } else {
          new_tensor = habana_lazy::empty_hpu_lazy(
              dynamic_shapes.at(i).get_dims(),
              tensor.options(),
              tensor.suggest_memory_format(),
              true,
              tensor_type);
        }
      } else {
        if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) == 0) {
          new_tensor = at::empty(
                           dynamic_shapes.at(i).get_dims(),
                           tensor.options().dtype(),
                           tensor.suggest_memory_format())
                           .to(at::kHPU);
        } else {
          new_tensor = habana_lazy::empty_hpu_lazy(
              dynamic_shapes.at(i).get_dims(),
              tensor.options(),
              tensor.suggest_memory_format(),
              true,
              tensor_type);
        }
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
    rv.time_slot_ = HPURegistrar::get_device().create_time_slot(hpu_stream_);
    if (rv.time_slot_) {
      current_dbipsh_->RegisterTimeSlot(rv.time_slot_, current_bucket_id_);
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
      std::make_shared<RecipeArgumentSpec>(input_refs, graph_key_, op_strs_);

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
        ref_input_shape_map_.at(rargpsh_graph->hashCode());
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
      std::make_shared<RecipeArgumentSpec>(input_refs, graph_key_, op_strs_);

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
      jit_ir_graph_->toString(), "current bucket id : ", current_bucket_id_);
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
      input_refs, graph_key_, op_strs_, cur_ds_token_);
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
            name_,
            '_',
            graph_index_,
            ", graph_key: ",
            rargpsh_graph->graphHashCode(),
            ", recipe cache hit, recipe_key: ",
            cur_rargpsh->hashCode());
        PT_DYNAMIC_SHAPE_DEBUG("Running output shape inference pass");
        if (enable_fast_shape_inf_ && GET_ENV_FLAG_NEW(PT_HPU_RUN_HYBRID_SIF)) {
          PT_DYNAMIC_SHAPE_DEBUG(
              "Graph: ",
              name_,
              '_',
              graph_index_,
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
            current_bucket_id_, jit_ir_graph_, ranges, refine_candidate);
      }

      cur_rvalpsh = GetCachedRecipe(cur_rargpsh);
      UpdatePatchingInformation(true, tidx_to_tensor_map);

      {
        std::lock_guard<std::mutex> lg(current_dbipsh_->get_refine_mutex());
        // Initiate recipe execution time collection
        if (GET_ENV_FLAG_NEW(PT_ENABLE_SYNLAUNCH_TIME_CAPTURE)) {
          InitiateSynlaunchTimeCapture(rv);
        }
      }

      if (!dry_run_) {
        rv.launch(
            hpu_stream_,
            input_refs,
            intermediate_tensors_ptr_sh_,
            *aten_outputs_ptr_sh_,
            syn_launch_info_,
            external_tensor_info_indexes_,
            dma_inputs_);
      }

      // Update the stack from the recipe itself
      UpdateRecipeOutputs();
      ReturnCachedRecipe(rv);

      RefinementEngine::GetEngine().AddGraphKey(rargpsh_graph->graphHashCode());
      PT_DYNAMIC_SHAPE_DEBUG(
          current_dbipsh_->digest_str(), current_dbipsh_->history_str());
      PT_IRGRAPH_DEBUG("HabanaOp recipe cache hit :: dynamic shapes");

      current_dbipsh_->get_statistics()->LogSelectedRecipe(
          cur_rargpsh->hashCode(), 0);
      current_dbipsh_->get_statistics()->LogShapes(
          jit_ir_graph_, graph_input_info.act_input_tshapes);

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
  ival_hash_to_input_index_map_.clear();

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

void RecipeValueSpec::create_outdup(
    size_t ti_idx,
    std::unordered_map<size_t, IValPtrShared>& parent_ivpsh_map,
    std::string map_name,
    VecOfIValPtrSh& aten_outputs,
    bool is_shape_agnostic_graph) const {
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
      // view. avoid invoking torch/aten operators from lowering context
      // pt_outdup = at::as_strided(parent_tensor, pt_sizes, pt_strides,
      // pt_opt_offset);
      pt_outdup = at::detail::make_tensor<at::TensorImpl>(
          c10::TensorImpl::VIEW,
          c10::Storage(parent_tensor.storage()),
          parent_tensor.key_set(),
          parent_tensor.dtype());
      c10::IntArrayRef size_vec(pt_sizes);
      c10::IntArrayRef stride_vec(pt_strides);
      at::native::setStrided(
          pt_outdup, size_vec, stride_vec, pt_opt_offset.value());
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
        "Setting tensor {:d} "
        " permutation from the TensorInfo cache record: {:s}"
        " old permutation was: {:s}",
        ti.get_tensor_id(),
        VecToString(ti.getHbInternalPermute()),
        habana_helpers::FormatTokens::Permutations);
    habana_helpers::set_tensor_memory_permutations(
        pt_outdup, ti.getHbInternalPermute());
  }
  PT_BACKEND_DEBUG_TENSOR(
      pt_outdup,
      " duplicate output HbInternal address : {:s}  storage address : {:s}",
      habana_helpers::FormatTokens::ImplPtr,
      habana_helpers::FormatTokens::DataPtr);
  ti.patch(pt_outdup, is_shape_agnostic_graph);

  IValPtrShared ivpsh = std::make_shared<IVal>(pt_outdup);
  aten_outputs.at(output_idx) = ivpsh;
}

void HabanaLaunchOpPT::ReturnCachedRecipe(RecipeValueSpec& rv) {
  PT_BRIDGE_BEGIN;
  rv.decrement_use_count();
  PT_BRIDGE_END;
}

// shape agnostic : duplicate synapse graph
void HabanaLaunchOpPT::DuplicateSynapseGraph() {
  auto tensorsMap = syn_graph_ptr_->duplicate();
  MaybePrintDuplicateGraphInformation(syn_graph_ptr_, tensorsMap, false);
}

// shape agnostic : store shape agnostic graph
void HabanaLaunchOpPT::StoreShapeAgnosticGraph() {
  cur_rvalpsh->shape_agnostic_synapse_graph_ =
      std::make_unique<synapse_helpers::graph>(std::move(*syn_graph_ptr_));

  cur_rvalpsh->shape_agnostic_synapse_graph_->set_build_phase(true);
  HABANA_ASSERT(cur_rvalpsh->shape_agnostic_synapse_graph_->get_is_valid());
}

// shape agnostic : validate Inputs and Outputs and disable shape agnostic if
// not supported
void HabanaLaunchOpPT::ValidateInputsAndOutputsAndDisableSA(
    at::ArrayRef<torch::jit::IValue>& input_refs) {
  // Validate if output shapes are filled correctly otherwise we can not
  // support shape agnostic graph caching.
  for (auto shape : out_shapes) {
    // Check for ZST tensor, It is supported for SAG, To Do proper fix
    if (shape.size() == 1 && shape[0] == 0)
      continue;
    for (auto size : shape) {
      if (size == 0) {
        jit_graph_and_meta_data_->set_is_shape_agnostic_supported(false);
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
        jit_graph_and_meta_data_->set_is_shape_agnostic_supported(false);
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
    const std::shared_ptr<synapse_helpers::graph>& graph_ptr,
    const std::vector<synTensorHandleMap>& tensors_map,
    bool is_cache_hit) {
  PT_EAGER_DEBUG(
      "[SHAPE AGNOSTIC] === cache ",
      is_cache_hit ? "hit" : "miss",
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

void habana::HabanaLaunchOpPT::ExecuteSynapseCacheTask(
    size_t graph_key_with_perm,
    std::shared_ptr<HabanaLaunchOpPT> hbLaunchOp) {
  hbLaunchOp->ExecuteSynapseCache(graph_key_with_perm);
}

// call this function for recipe caching (graph/eager)
void HabanaLaunchOpPT::ExecuteSynapseCache(size_t graph_key_with_perm) {
  PT_BRIDGE_BEGIN;
  RecipeValueSpec& rv = *cur_rvalpsh;

  if (habana_helpers::IsInferenceMode()) {
    for (size_t j = 0; j < pt_stack_sh.size(); j++) {
      auto ivpsh = pt_stack_sh[j];
      if (ivpsh.get()->isTensor()) {
        auto pt_tensor = ivpsh.get()->toTensor();
        if (habana::is_tensor_const_with_valid_const_id(pt_tensor)) {
          auto const_id = habana::get_tensor_const_id(pt_tensor);
          HABANA_ASSERT(
              m_const_checksum_map.find(const_id) != m_const_checksum_map.end(),
              " No checksum exists for const_id: ",
              const_id,
              " in the map");
          auto recipe_checksum =
              GetConstCheckSumForRecipe(const_id, cur_rargpsh->hashCode());
          // PT_BRIDGE_DEBUG("[Cache hit] const_id:  ", const_id, " recipe
          // checksum:
          // ", recipe_checksum, " current checksum on device: ",
          // m_const_checksum_map[const_id].first)
          if (m_const_checksum_map[const_id].first != recipe_checksum) {
            GetConstPtrForRecipe(const_id, cur_rargpsh->hashCode(), pt_tensor);
            InsertConstantChecksum(const_id, recipe_checksum);
            // hbLaunchOp->ivalue_to_tensor_info_map[ivpsh]->set_buffer(
            //    (void*)(pt_tensor.storage().data_ptr().get()));
            PT_BRIDGE_DEBUG(
                "Tensor with const_id: ",
                const_id,
                " has moved data pointer for the data corresponding to checksum: ",
                recipe_checksum,
                " for cache hit on key ",
                cur_rargpsh->hashCode());
          }
        }
      }
    }
  }

  if (!dry_run_) {
    rv.launch(
        hpu_stream_,
        input_refs,
        intermediate_tensors_ptr_sh_,
        *aten_outputs_ptr_sh_,
        syn_launch_info_,
        external_tensor_info_indexes_,
        dma_inputs_);
  }

  if (habana_helpers::GetRefineDynamicShapeStatus()) {
    CreateStaticCompilationDBI(graph_key_with_perm);
  }

  if (!get_enable_2stage_pipeline()) {
    // Update the stack from the recipe itself
    UpdateRecipeOutputs();
  }
  PT_BRIDGE_DEBUG("Returning cached recipe : ", cur_rargpsh->hashCode());
  ReturnCachedRecipe(rv);

  ClearStatics();
  PT_BRIDGE_END;
}

void HabanaLaunchOpPT::DumpStaticCompilationStatistics(
    size_t graph_key_with_perm,
    bool is_compile) {
  habana_helpers::ResultShapes ranges;

  habana_helpers::InpTensorShapes input_tshapes =
      ref_input_shape_map_.at(graph_key_with_perm);
  if (is_compile) {
    current_dbipsh_->get_statistics()->LogCompilation(
        jit_ir_graph_->toString(),
        jit_ir_graph_,
        current_dbipsh_->GetMinPolicy(),
        current_dbipsh_->GetMaxPolicy(),
        ranges,
        cur_rargpsh->hashCode(),
        "OK",
        habana_helpers::CompilationPass::STATIC);
    current_dbipsh_->get_statistics()->LogShapes(jit_ir_graph_, input_tshapes);
    current_dbipsh_->get_statistics()->LogUsedBucket(
        0, jit_ir_graph_, ranges, false);
    current_dbipsh_->get_statistics()->LogSelectedRecipe(
        cur_rargpsh->hashCode(), 0);
    // current_dbipsh_->get_statistics()->LogRecipeMemory(cur_rvalpsh);
    current_dbipsh_->get_statistics()->GetDigest(
        cur_rargpsh->graphHashCode(), 0, 0, cur_rargpsh->hashCode(), false);
  } else {
    std::shared_ptr<RecipeArgumentSpec> rargpsh_graph =
        std::make_shared<RecipeArgumentSpec>(input_refs, graph_key_, op_strs_);
    current_dbipsh_ = DynamicBucketInfoMap::get_instance().get(rargpsh_graph);
    HABANA_ASSERT(
        (current_dbipsh_ != nullptr),
        "Dynamic bucketinfo got NULL in static cache hit");
    current_dbipsh_->SetLastUsedStepForBucket(
        0, current_dbipsh_->get_statistics()->GetCurrentStep());

    current_dbipsh_->get_statistics()->LogSelectedRecipe(
        cur_rargpsh->hashCode(), 0);
    current_dbipsh_->get_statistics()->LogShapes(jit_ir_graph_, input_tshapes);

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

void HabanaLaunchOpPT::UpdatePatchingInformation(
    bool is_ds_patching_update,
    std::optional<
        std::reference_wrapper<const std::unordered_map<int64_t, at::Tensor>>>
        local_tidx_to_tensor_map,
    const std::unordered_map<synTensor, synTensor>& synapse_orig_to_new_handle,
    const bool is_shape_agnostic_graph) {
  RecipeValueSpec& rv = *cur_rvalpsh;

  intermediate_tensors_ptr_sh_ = std::make_shared<VecOfIValPtrSh>();

  // The aten_output_num is the total number of outputs
  size_t aten_output_num = rv.get_aten_output_num();

  aten_outputs_ptr_sh_ = std::make_unique<VecOfIValPtrSh>(aten_output_num);
  if (!is_ds_patching_update) {
    rv.update_patching_table(
        input_refs,
        intermediate_tensors_ptr_sh_,
        dma_inputs_,
        *aten_outputs_ptr_sh_,
        m_map_shape.m_actual_shapes,
        local_tidx_to_tensor_map,
        allocated_outputs_,
        out_shapes,
        synapse_orig_to_new_handle,
        is_shape_agnostic_graph);
  } else {
    rv.update_patching_table(
        input_refs,
        intermediate_tensors_ptr_sh_,
        dma_inputs_,
        *aten_outputs_ptr_sh_,
        m_map_shape.m_actual_shapes,
        local_tidx_to_tensor_map,
        allocated_outputs_);
  }
  if (!dry_run_) {
    if (rv.recipe) {
      rv.patch_launch_info(syn_launch_info_, external_tensor_info_indexes_);
    } else {
      PT_BRIDGE_DEBUG("Skipping patch_launch_info for empty recipe");
    }
  }
}

void HabanaLaunchOpPT::run(
    torch::jit::Stack& stack,
    std::optional<std::vector<at::Tensor>> allocated_outputs,
    bool dry_run,
    HabanaLaunchOpPipeline::PipelineCallBase& pipeline_execution) {
  PT_BRIDGE_BEGIN;
  static int idx{1};
  ProcessInputStack(stack);
  allocated_outputs_ = std::move(allocated_outputs);

  dry_run_ = dry_run;
  auto& device = HPURegistrar::get_device();

  // Check whether dynamic shape is needed
  size_t graph_key_with_perm = graph_key_;
  size_t sym_hash_code = habana::ComputeSymSizeHashCode(input_refs);
  graph_key_with_perm = at::hash_combine(graph_key_with_perm, sym_hash_code);
  size_t perm_hash_code = habana::ComputePermutationHashCode(input_refs);
  graph_key_with_perm = at::hash_combine(graph_key_with_perm, perm_hash_code);

  const auto eager_mode =
      (execution_mode_ == habana_helpers::HabanaFrontendTypes::EAGER);
  const auto compile_mode =
      (execution_mode_ == habana_helpers::HabanaFrontendTypes::COMPILE);
  PT_BRIDGE_DEBUG(
      "Lowering:\n",
      "JIT_IR_Graph_BEGIN\n",
      "Graph ",
      idx,
      '\n',
      jit_ir_graph_->toString(),
      "JIT_IR_Graph_END\n");

  PT_TEST_DEBUG(
      "Lowering:\n",
      "Graph ",
      idx,
      '\n',
      "JIT IR graph_hash_code : ",
      graph_key_,
      ", hash_code with data layout : ",
      graph_key_with_perm,
      "is dynamic : ",
      refine_ds_enabled_);

  idx += 1;
  if (enable_caching_ || IS_BRIDGE_DEBUG_ENABLED) {
    cur_rargpsh = std::make_shared<RecipeArgumentSpec>(
        false, input_refs, jit_ir_graph_, graph_key_, op_strs_);
  }

  auto is_enable_4stage_pipeline = enable_4stage_pipeline_;

  // eager and graph recipe caching :: begin
  if (enable_caching_) {
    HABANA_ASSERT(
        enable_graph_caching_ || enable_eager_caching_,
        " something went wrong! either eager or graph recipe caching should be enabled");
    PT_BRIDGE_DEBUG("Getting cached recipe : ", cur_rargpsh->hashCode());
    cur_rvalpsh = GetCachedRecipe(cur_rargpsh);

    if (ABSL_PREDICT_TRUE(cur_rvalpsh)) {
      emitCacheEvent(
          habana_helpers::EventDispatcher::Topic::CACHE_HIT,
          std::to_string(cur_rargpsh->hashCode()));

      RecipeValueSpec& rv = *cur_rvalpsh;
      rv.update_hit_count();

      PT_BRIDGE_DEBUG(
          id_str_,
          ": ",
          "HabanaOp recipe cache hit :: key ",
          cur_rargpsh->hashCode(),
          "\n",
          rv.header_str(),
          "\n",
          rv.digest_str());
      PT_IRGRAPH_DEBUG("HabanaOp recipe cache hit :: static shapes");
      PT_TEST_DEBUG("HabanaOp recipe cache hit :: static path");

      UpdatePatchingInformation();

      // currently only eager backend supports pipelining
      // can be merged once non-eager backends support pipelining
      if (enable_graph_caching_ && !compile_mode) {
        ExecuteSynapseCache(graph_key_with_perm);
      } else {
        PT_LAZY_EAGER_DEBUG(
            "[LAZY EAGER MT] Enqueue new task to the Compile and Execute Thread");
        if (!is_enable_4stage_pipeline) {
          ExecuteSynapseCache(graph_key_with_perm);
        } else {
          execution_control_.cached_task(graph_key_with_perm);
          pipeline_execution(false);
        }
      }
      PT_BRIDGE_END;
      return;
    } else {
      emitCacheEvent(
          habana_helpers::EventDispatcher::Topic::CACHE_MISS,
          std::to_string(cur_rargpsh->hashCode()));
      PT_BRIDGE_DEBUG(
          id_str_,
          ": ",
          "HabanaOp recipe cache miss :: key ",
          cur_rargpsh->hashCode());
      PT_IRGRAPH_DEBUG("HabanaOp recipe cache miss :: static shapes");
    }
  }
  // eager and graph recipe caching :: end

  bool is_jit_cached_graph_info_available =
      jit_graph_and_meta_data_->get_jit_cached_graph_info_available_flag();
  if (is_jit_cached_graph_info_available == false) {
    jit_graph_and_meta_data_->clear_cached_graph_info();
  }

  CreateValueToIvalueMapForInputs();
  ProcessGraphForConstantTensors(*jit_ir_graph_, value_to_ivalue);

  if (enable_shape_agnostic_caching_) {
    ValidateInputsAndOutputsAndDisableSA(input_refs);
  }

  // shape agnostic caching :: begin
  if (enable_shape_agnostic_caching_ &&
      jit_graph_and_meta_data_->get_is_shape_agnostic_supported()) {
    cur_rvalpsh = jit_graph_and_meta_data_->get_shape_agnostic_recipe();
    if (cur_rvalpsh == nullptr) {
      PT_EAGER_DEBUG("[SHAPE AGNOSTIC] shape agnostic cache miss (begin)");
      HABANA_ASSERT(
          eager_mode == true,
          "eager_mode is expected true for supporting shape agnostic graph");
      is_shape_agnostic_supported_ =
          jit_graph_and_meta_data_->get_is_shape_agnostic_supported();
      constexpr bool dry_run__ = false;
      auto syn_graph =
          std::make_shared<synapse_helpers::graph>(habana_helpers::create_graph(
              device.id(), GetSynapseGraphName(), dry_run__, eager_mode));

      constexpr bool is_shape_agnostic_graph = true;
      syn_graph->set_shape_agnostic_graph(is_shape_agnostic_graph);
      BuildSynapseGraph(syn_graph);

      if (syn_graph_ptr_->is_empty()) {
        PT_LAZY_EAGER_DEBUG(
            "Empty synapse graph. Nothing to duplicate. will update outputs directly.");
        UpdateOutputs();
        return;
      }

      DuplicateSynapseGraph();

      if (syn_graph_ptr_->get_num_of_shape_tensors() > 0) {
        jit_graph_and_meta_data_->set_is_shape_agnostic_supported(false);
        PT_EAGER_DEBUG(
            "[SHAPE AGNOSTIC] Shape agnostic not supported for Op",
            " with intermediate shape tensors : ",
            syn_graph_ptr_->get_num_of_shape_tensors());
      }

      // ToDo: Refactor this logic later w.r.t synapse shape inference
      // When meta attribute is set JIT IR Op kernel does not add synapse node
      if ((habana_kernels.size() - meta_attribute_nodes_count_) !=
          syn_graph_ptr_->get_num_of_nodes()) {
        jit_graph_and_meta_data_->set_is_shape_agnostic_supported(false);
        PT_EAGER_DEBUG(
            "[SHAPE AGNOSTIC] Shape agnostic not supported for compound Op(s)",
            " number of kernels : ",
            habana_kernels.size(),
            " number of JIT IR nodes with meta attribute : ",
            meta_attribute_nodes_count_,
            " number of synapse nodes : ",
            syn_graph_ptr_->get_num_of_nodes());
      }

      /*
       * Check for non-persistent syn tensors (i.e. added with in habana kernel)
       * Example Gelu Pytorch Op is unary kernel i.e. 1 input and 1 input
       * But TPC kernel has two outputs second output is internal to synapse
       * i.e. non-persitent and is reserved for backward pass calculation.
       *
       * Adding fallback for such cases.
       * This fallback should not occur with views as of today sizes/strides
       * are cached. Need to fix once params agnostic support is added.
       * To remove this fallback once shape inference is supported for
       * non-persistent synapse tensors.
       */
      if (jit_graph_and_meta_data_->get_is_shape_agnostic_supported() &&
          ((syn_graph_ptr_->get_num_of_tensors() -
            syn_graph_ptr_->get_num_of_const_tensors()) !=
           (pt_to_synapse_tensors.size() + implicit_syn_tensors_count_))) {
        jit_graph_and_meta_data_->set_is_shape_agnostic_supported(false);
        PT_EAGER_DEBUG(
            "[SHAPE AGNOSTIC] Shape agnostic not supported for non-persistent"
            " total number of syn tensors : ",
            syn_graph_ptr_->get_num_of_tensors(),
            " number of const tensors : ",
            syn_graph_ptr_->get_num_of_const_tensors(),
            " number of persistent tensors : ",
            pt_to_synapse_tensors.size(),
            " number implicit tensors : ",
            implicit_syn_tensors_count_,
            " number of syn nodes : ",
            syn_graph_ptr_->get_num_of_nodes());
      }

      PT_EAGER_DEBUG(
          "[SHAPE AGNOSTIC] Shape agnostic SIF tinfo map size : ",
          sif_tidx_to_tinfo_map.size(),
          " intermediate tensors size : ",
          intermediate_syn_tensors_count_);
      syn_graph_ptr_->set_num_of_inter_tensors(intermediate_syn_tensors_count_);

      jit_graph_and_meta_data_->set_jit_cached_graph_info_available_flag(true);

      aten_outputs_ptr_sh_ = std::make_unique<VecOfIValPtrSh>();

      PT_LAZY_EAGER_DEBUG(
          "[LAZY EAGER MT] Enqueue new task to the Compile and Execute Thread");
      execution_control_.sag_cache_miss();
      // In case of SAG cache miss we have to wait till a compile thread sets
      // permutation for outputs (In case of SAG cache hit, that info is taken
      // from SAG recipe (from dtensorinfos))
      pipeline_execution(true);
      return;
    } else {
      PT_EAGER_DEBUG("[SHAPE AGNOSTIC] shape agnostic cache hit (begin)");
      is_shape_agnostic_supported_ = true;
      syn_graph_ptr_ = std::make_shared<synapse_helpers::graph>(
          *(cur_rvalpsh->shape_agnostic_synapse_graph_.get()));

      auto tensorsMap = syn_graph_ptr_->duplicate();
      MaybePrintDuplicateGraphInformation(syn_graph_ptr_, tensorsMap, true);

      RecipeValueSpec& rv = *cur_rvalpsh;
      rv.update_hit_count();
      PT_EAGER_DEBUG(
          id_str_,
          ": ",
          "HabanaOp shape agnostic graph cache hit :: key ",
          graph_key_,
          "\n",
          rv.header_str(),
          "\n",
          rv.digest_str());
      PT_EAGER_DEBUG(
          "HabanaOp shape agnostic graph cache hit :: static shapes");

      std::unordered_map<synTensor, synTensor> synapse_orig_to_new_handle{};
      for (size_t i = 0; i < tensorsMap.size(); i++) {
        synapse_orig_to_new_handle.insert(
            {tensorsMap.at(i).origHandle, tensorsMap.at(i).newHandle});
      }

      for (const auto& out_shape : out_shapes) {
        PT_EAGER_DEBUG("[SHAPE AGNOSTIC] output shape - ", out_shape);
      }

      /*
       * Hybrid SIF is used for shape inference for intermediate tensors
       * Inputs shape is retrieved from input refs.
       * Ouptut shape info is passed in the jit ir graph meta data.
       */
      std::unordered_map<int64_t, at::Tensor> local_tidx_to_tensor_map;
      if (syn_graph_ptr_->get_num_of_inter_tensors() > 0) {
        habana::ShapeInference::ResetSifTensorId();
        constexpr bool dynamic_shapes_false = false;
        RunHybridSif<dynamic_shapes_false>(local_tidx_to_tensor_map);
      }

      constexpr bool is_shape_agnostic_graph = true;
      UpdatePatchingInformation(
          false,
          local_tidx_to_tensor_map,
          synapse_orig_to_new_handle,
          is_shape_agnostic_graph);

      // To check if any other members just like ntensorbytes also need to be
      // updated
      // SAG cache hit case - to avoid race condition with execute thread
      hpu_op_ntensorbytes_ = 0;
      for (auto& ti : *rv.dtensorinfos) {
        if (!ti->is_duplicate()) {
          hpu_op_ntensorbytes_ += ti->get_size();
        }
      }

      if (refine_ds_enabled_ ||
          (not jit_graph_and_meta_data_
                   ->get_jit_cached_graph_info_available_flag()) ||
          jit_graph_and_meta_data_->get_is_control_edge_processing_required()) {
        std::vector<std::pair<torch::jit::Value*, torch::jit::Node*>>
            memory_reuse_pairs;
        control_edges::ProcessControlEdges(
            *jit_ir_graph_,
            *jit_graph_and_meta_data_,
            jit_to_synapse_node_idx_map,
            memory_reuse_pairs,
            syn_graph_ptr_.get());
      }

      syn_graph_ptr_->set_build_phase(true);
      jit_graph_and_meta_data_->set_jit_cached_graph_info_available_flag(true);

      PT_LAZY_EAGER_DEBUG(
          "[LAZY EAGER MT] Enqueue new task to the Compile and Execute Thread");
      pipeline_execution(!is_enable_4stage_pipeline);
    }
    PT_BRIDGE_END;
    return;
  }
  // shape agnostic caching :: end
  if (!eager_mode && ref_input_shape_map_.count(graph_key_with_perm) &&
      refine_ds_enabled_) {
    PT_DYNAMIC_SHAPE_DEBUG(
        "JIT IR graph_hash_code : ",
        graph_key_,
        ", hash_code with data layout : ",
        graph_key_with_perm,
        "\nStarting dynamic shape flow");

    jit_graph_and_meta_data_->clear_cached_graph_info();
    ProcessHabanaFusedOpWithDS();
    PT_BRIDGE_END;
    return;
  }

  // Remember the input shapes for creating dynamic bucket info structure later.
  // Note that this needs to be done before execution of graph, otherwise
  // input_refs will get overwritten by outputs and we will create bucket
  // with incorrect shapes.

  if (!eager_mode && habana_helpers::GetRefineDynamicShapeStatus()) {
    CreateStaticCompilationDBI(graph_key_with_perm);
  }

  constexpr bool dry_run__ = false;
  const auto use_eager_compiler =
      eager_mode && jit_graph_and_meta_data_->get_is_eager_compiler_supported();
  auto syn_graph =
      std::make_shared<synapse_helpers::graph>(habana_helpers::create_graph(
          device.id(), GetSynapseGraphName(), dry_run__, use_eager_compiler));
  m_map_shape.m_pass = ShapeInfo::InferencePass::INVALID;
  BuildSynapseGraph(syn_graph);
  if (enable_shape_agnostic_caching_) {
    syn_graph_ptr_->copy_graph_handle_to_duplicate();
  }

  aten_outputs_ptr_sh_ = std::make_unique<VecOfIValPtrSh>();

  if ((eager_mode || compile_mode) &&
      !GET_ENV_FLAG_NEW(PT_HPU_ENABLE_EXECUTION_THREAD_NO_WAIT) &&
      jit_graph_and_meta_data_->get_is_pipeline_supported()) {
    PT_LAZY_EAGER_DEBUG(
        "[LAZY EAGER MT] Enqueue new task to the Compile and Execute Thread");

    bool is_permute_data_cached = jit_graph_and_meta_data_->is_permute_set();
    if (is_permute_data_cached) {
      ApplyOutputPermutationsFromCache();
    } else {
      permutation_info_saver_ =
          std::make_unique<PermutationInfoSaver>(jit_graph_and_meta_data_);
    }
    jit_graph_and_meta_data_->set_jit_cached_graph_info_available_flag(true);

    pipeline_execution(
        !is_permute_data_cached || enable_caching_ ||
        !is_enable_4stage_pipeline);
  } else {
    CompileSynapseGraph();
    ConstructPatchingTableAndAtenOutputs();
    UpdateSynapsePermutations();
    StoreCompiledInformation();
    ExecuteSynapseGraph();

    jit_graph_and_meta_data_->set_jit_cached_graph_info_available_flag(true);
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
    VecOfIValPtrSh old_pt_stack_sh;

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
    VecOfIValPtrSh old_pt_stack_sh;

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

  auto syn_graph = std::make_shared<synapse_helpers::graph>(
      synapse_helpers::graph::create_for_refinement(
          device.syn_device(), name_));

  // Compile the graph
  {
    CreateValueToIvalueMapForInputs();

    syn_graph->set_dynamic_graph(true);

    std::string error_str;
    try {
      cur_ds_token_ = new_bucket.getToken();
      cur_rargpsh = std::make_shared<RecipeArgumentSpec>(
          input_refs, graph_key_, op_strs_, cur_ds_token_);
      new_recipe_key = cur_rargpsh->hashCode();

      m_map_shape.m_pass = ShapeInfo::InferencePass::INVALID;
      BuildSynapseGraph(syn_graph);
      CompileSynapseGraph();
      aten_outputs_ptr_sh_ = std::make_unique<VecOfIValPtrSh>();
      ConstructPatchingTableAndAtenOutputs();
      UpdateSynapsePermutations();
    } catch (std::exception& e) {
      error_str = e.what();
      PT_DYNAMIC_SHAPE_DEBUG(
          "Exception occured in compilation - Details :\n", error_str);

      std::string result_str{"FAIL"};
      uint64_t current_step{statpsh->GetCurrentStep()};
      statpsh->LogRefineCompilation(
          input_ranges,
          jit_ir_graph_,
          new_recipe_key,
          new_bucket.GetIndex(),
          result_str,
          current_step);

      throw;
    }
  }
  PT_DYNAMIC_SHAPE_DEBUG("Compilation completed");

  // Add the <key,value> pair to the map
  cur_rvalpsh->dynamic_graph = syn_graph->is_dynamic_graph();
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
  auto syn_graph = std::make_shared<synapse_helpers::graph>(
      habana_helpers::create_graph(device.id(), GetSynapseGraphName(), true));
  syn_graph->set_dynamic_graph(true);
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
  VecOfIValPtrSh old_pt_stack_sh;
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
    RevertH2DMinMaxData();
  }

  if (old_stack) {
    pt_stack_sh.clear();
    pt_stack = old_stack;
    pt_stack_sh = old_pt_stack_sh;
  }

  if (updatemax_graph) {
    UpdatePTStack(graph_input_info);
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
  if (graph_input_info.current_bucket_id == 0) {
    jit_ir = jit_ir_graph_->toString();
  }
  auto ranges =
      current_dbipsh_->CalculateShapes(graph_input_info.current_bucket_id);
  auto new_ds_token = current_dbipsh_->GetTokenForBucketId(current_bucket_id_);

  if (new_ds_token != cur_ds_token_) {
    cur_rargpsh = std::make_shared<RecipeArgumentSpec>(
        input_refs, graph_key_, op_strs_, new_ds_token);
    current_dbipsh_->SetRecipeKeyForBucket(
        current_bucket_id_, cur_rargpsh->hashCode());
    DynamicBucketInfoMap::get_instance().add(cur_rargpsh, current_dbipsh_);
  }

  if (ranges.empty()) {
    if (graph_input_info.max_policy ==
        habana_helpers::DynamicDimsPolicy::CURRENT) {
      last_compilation_pass = habana_helpers::CompilationPass::DYNAMIC_CURRENT;
    }
  } else {
    last_compilation_pass = habana_helpers::CompilationPass::DYNAMIC_MAX;
  }
  m_map_shape.m_pass = ShapeInfo::InferencePass::INVALID;
  auto syn_graph = std::make_shared<synapse_helpers::graph>(
      habana_helpers::create_graph(device.id(), GetSynapseGraphName()));
  syn_graph->set_dynamic_graph(is_dynamic_graph);
  EvictSynapseRecipe(graph_input_info.current_bucket_id);
  BuildSynapseGraph(syn_graph);
  CompileSynapseGraph();
  aten_outputs_ptr_sh_ = std::make_unique<VecOfIValPtrSh>();
  ConstructPatchingTableAndAtenOutputs();
  UpdateSynapsePermutations();
  StoreCompiledInformation();
  ExecuteSynapseGraph();

  current_dbipsh_->get_statistics()->LogCompilation(
      jit_ir,
      jit_ir_graph_,
      graph_input_info.min_policy,
      graph_input_info.max_policy,
      ranges,
      current_dbipsh_->GetRecipeKeyForBucket(
          graph_input_info.current_bucket_id),
      result,
      last_compilation_pass);
  current_dbipsh_->get_statistics()->LogShapes(
      jit_ir_graph_, graph_input_info.act_input_tshapes);
  bool refine_candidate = false;
  if (last_compilation_pass != habana_helpers::CompilationPass::STATIC) {
    refine_candidate =
        (current_dbipsh_->GetMFUBucket() == graph_input_info.current_bucket_id);
  }
  current_dbipsh_->get_statistics()->LogUsedBucket(
      graph_input_info.current_bucket_id,
      jit_ir_graph_,
      ranges,
      refine_candidate);
  current_dbipsh_->get_statistics()->LogSelectedRecipe(
      current_dbipsh_->GetRecipeKeyForBucket(
          graph_input_info.current_bucket_id),
      0);
  if (!syn_graph_ptr_->is_empty()) {
    current_dbipsh_->get_statistics()->LogRecipeMemory(cur_rvalpsh);
  }
}

void HabanaLaunchOpPT::set_lazy_front_end_info(
    std::shared_ptr<habana_lazy::HbLazyFrontEndInfoToBackend> info) {
  lazy_info_ = info;
}

bool HabanaLaunchOpPT::is_hccl_send_mark_step() {
  if (lazy_info_ == nullptr) {
    return false;
  }
  return lazy_info_->get_is_hccl_send_mark_step();
}
} // namespace habana
