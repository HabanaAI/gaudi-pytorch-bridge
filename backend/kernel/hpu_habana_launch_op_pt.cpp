/**
* Copyright (c) 2021-2024 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
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
#include <sstream>
#include <unordered_map>
#include "backend/backend_meta.h"
#include "backend/habana_device/hpu_cached_devices.h"
#include "backend/habana_device/tensor_builder.h"
#include "backend/helpers/compilation_statistics.h"
#include "backend/helpers/create_tensor.h"
#include "backend/helpers/event_dispatcher.h"
#include "backend/helpers/graph.h"
#include "backend/helpers/symbolic_expression.h"
#include "backend/helpers/tensor_utils.h"
#include "backend/jitgraph_utils.h"
#include "backend/kernel/constant_information.h"
#include "backend/kernel/control_edges_processing.h"
#include "backend/kernel/hpu_habana_compile_op_pt.h"
#include "backend/kernel/hpu_habana_meta_op_list.h"
#include "backend/kernel/hpu_shape_inference.h"
#include "backend/kernel/refinement_engine.h"
#include "backend/passes/hpu_habana_persistence_marker_pass.h"
#include "backend/random.h"
#include "backend/synapse_helpers/env_flags.h" // IWYU pragma: keep // NOLINT
#include "backend/synapse_helpers/tcmalloc_helper.h"
#include "habana_helpers/logging.h"
#include "habana_helpers/misc_utils.h"
#include "habana_kernels/hccl_kernels.h"
#include "habana_kernels/index_kernels.h"
#include "habana_kernels/lazy_kernels_declarations.h"
#include "habana_lazy/hpu_lazy_tensors.h"
#include "habana_lazy/lazy_executor.h"
#include "hpu_ops/op_backend.h"
#include "hpu_ops/op_logger.h"
using namespace torch::jit;
using namespace jitgraph_utils;
namespace habana {
//--------------------------------------

namespace HabanaLaunchOpPipeline {

class PipelineCallBase {
 public:
  virtual void operator()() {
    PT_BRIDGE_FATAL("HabanaLaunchOpPT has been used without pipeline wrapper");
  }
  virtual void compile_sync() {
    PT_BRIDGE_FATAL("HabanaLaunchOpPT has been used without pipeline wrapper");
  }
};

PipelineCallBase NoPipeline;

class PipelineCall : public PipelineCallBase {
 public:
  virtual void operator()() override {
    HABANA_ASSERT(!is_called_);
    is_called_ = true;
  }
  bool is_called() {
    return is_called_;
  }
  void compile_sync() override {
    HPUDeviceContext::compile_thread().waitWorkComplete();
  }

 private:
  bool is_called_ = false;
};

void LoweringTask(
    std::unique_ptr<habana::HabanaLaunchOpPT>&& launch_op,
    torch::jit::Stack& stack,
    std::optional<std::vector<at::Tensor>> allocated_outputs,
    std::optional<std::vector<std::vector<int64_t>>> output_shapes) {
  PipelineCall pipeline_call;

  bool sync_with_compile_stage = !launch_op->get_enable_4stage_pipeline();

  launch_op->run(
      stack, nullptr, allocated_outputs, output_shapes, false, pipeline_call);

  if (!pipeline_call.is_called()) {
    PT_BRIDGE_DEBUG(
        "HabanaLaunchOpPT wraped by pipeline has been called but pipelined path haven't been chosen in run call");
    // TODO: we have to be sure that habana::eager::JoinPendingPipelineThreads()
    // has been called before
    return;
  }

  HPUDeviceContext::compile_thread().enqueue(
      HabanaLaunchOpPipeline::CompileSynapseTask, std::move(launch_op));

  if (sync_with_compile_stage)
    HPUDeviceContext::compile_thread().waitWorkComplete();
}
} // namespace HabanaLaunchOpPipeline

namespace HabanaLaunchOpUtils {
std::unordered_map<size_t, habana_helpers::InpTensorShapes>&
ref_input_shape_map() {
  static std::unordered_map<size_t, habana_helpers::InpTensorShapes> map;
  return map;
};

std::unordered_set<std::string>& disabled_jit_ir_ops() {
  static std::unordered_set<std::string> set;
  return set;
};

void cleanUp() {
  ref_input_shape_map() = {};
  DynamicBucketInfoMap::get_instance().clear();
  HPUDeviceContext::recipe_cache_clear();
}
} // namespace HabanaLaunchOpUtils

bool dropCachedRecipe_LRU(size_t& recipe_count) {
  bool dropped{false};
  dropped = HPUDeviceContext::recipe_cache().drop_lru(recipe_count);
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
      use_persistent_tensors_{GET_ENV_FLAG_NEW(HABANA_USE_PERSISTENT_TENSOR)} {
  refine_ds_enabled_ = optimized_jit_graph_and_meta_data->GetDynamicGraph();
  enable_fast_shape_inf_ =
      GET_ENV_FLAG_NEW(PT_HPU_ENABLE_FAST_SHAPE_INFERENCE) &&
      refine_ds_enabled_;
  op_strs_ = optimized_jit_graph_and_meta_data->get_cached_opstrs();
  graph_key_ = optimized_jit_graph_and_meta_data->get_cached_graph_key();
  jit_graph_cache_hit_count_ =
      optimized_jit_graph_and_meta_data->get_jit_cache_hit_count();
  hpu_stream_ = optimized_jit_graph_and_meta_data->GetHPUStream();
  jit_graph_and_meta_data_ = optimized_jit_graph_and_meta_data;
  enable_user_dynamic_ranges_ =
      optimized_jit_graph_and_meta_data->IsUserMarkDynamic();
  optimized_jit_graph_and_meta_data->SetUserMarkDynamic(false);
  range_infos_ = optimized_jit_graph_and_meta_data->GetUserRangesDynamic();

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
  const auto& device = HPUDeviceContext::get_device();
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

  auto frontend_type_eager_or_compile =
      ((front_end_type == habana_helpers::HabanaFrontendTypes::EAGER) ||
       (front_end_type == habana_helpers::HabanaFrontendTypes::COMPILE));
  enable_2stage_pipeline_ =
      jit_graph_and_meta_data_->get_is_pipeline_supported();
  enable_4stage_pipeline_ = enable_2stage_pipeline_ &&
      GET_ENV_FLAG_NEW(PT_HPU_EAGER_4_STAGE_PIPELINE_ENABLE) &&
      frontend_type_eager_or_compile;
  PT_DYNAMIC_SHAPE_DEBUG("Enable 4 stage pipeline = ", enable_4stage_pipeline_);

  graph_symint_hash_ =
      optimized_jit_graph_and_meta_data->get_graph_symint_hash();
  graph_perm_hash_ = optimized_jit_graph_and_meta_data->get_graph_perm_hash();
  graph_key_with_perm_ =
      optimized_jit_graph_and_meta_data->get_graph_key_with_perm();

  // Symbolic expression hash is checked for -1 to avoid a scenario where
  // FX graph have symbolic inputs but the symbols in FX nodes output_shape
  // meta has replaced with actual values.
  sym_expr_hash_ = optimized_jit_graph_and_meta_data->get_sym_expr_hash();
  enable_optim_output_sif_ =
      optimized_jit_graph_and_meta_data->get_enable_optim_output_sif();
  PT_DYNAMIC_SHAPE_DEBUG(
      "Enable dynamic shape symbolic output sif = ", enable_optim_output_sif_);

  if (enable_optim_output_sif_) {
    maybe_static_recipe_ =
        optimized_jit_graph_and_meta_data->get_maybe_static_recipe();
    curr_symval_hash_ =
        optimized_jit_graph_and_meta_data->get_curr_symval_hash();
  }
}

HabanaLaunchOpPT::~HabanaLaunchOpPT() {
  PT_BRIDGE_DEBUG("Destroying : ", GetSynapseGraphName());
}

bool HabanaLaunchOpPT::nodeOutputPersistencePerValue(
    torch::jit::Node* node,
    torch::jit::Value* value_out) {
  bool is_persistent = false;
  if (use_persistent_tensors_ || isInGraphOutputs(value_out)) {
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
  bool sfg_enable = false;
  if (node->hasAttribute(c10::Symbol::attr("sfg"))) {
    sfg_enable = true;
  }
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
        if (sfg_enable) {
          md.external = true;
        }
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
        if (sfg_enable) {
          md.external = true;
        }
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
  auto syn_tensor_input =
      pt_to_synapse_tensors_.find(value_to_ivalue_[value_in]);

  for (synapse_helpers::tensor& tensor : *(syn_tensor_input->second)) {
    synapse_helpers::tensor& syn_tensor = habana_op->SetSynapseInput(tensor);
    tensorList->emplace_back(synapse_helpers::tensor_or_ref(syn_tensor));
  }

  pt_to_synapse_tensors_.erase(value_to_ivalue_[value_in]);
  pt_to_synapse_tensors_.emplace(value_to_ivalue_[value_in], tensorList);
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
         buff_to_syn_tensor_map_.count(pt_tensor_buffer_start))};
    if (is_duplicate_syn_tensor) {
      auto syn_tensor_it = buff_to_syn_tensor_map_.find(pt_tensor_buffer_start);
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
      buff_to_syn_tensor_map_.emplace(
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
  const auto& ivalue = value_to_ivalue_[value_in];
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
    ivalue_to_tensor_info_map_[ivalue] = ti;

    if (enable_caching_ || enable_shape_agnostic_caching_) {
      void* buffp = ti->get_buffer_start();
      if (ti->is_ZST() == false) {
        buff_to_input_ivpsh_map_.emplace(buffp, ivalue);
      }
    }
  }

  if (!tensorList->empty()) {
    auto it = pt_to_synapse_tensors_.emplace(ivalue, tensorList);
    if (it.second == false) {
      pt_to_synapse_tensors_[ivalue] = tensorList;
    }

    if (enable_caching_ || enable_shape_agnostic_caching_) {
      input_tiv_map_.emplace(value_to_ivalue_[value_in], tiv);
      auto node_qual_str = value_in->node()->kind().toQualString();
      if ((strcmp(node_qual_str, "hpu::restride_cl") == 0) ||
          (strcmp(node_qual_str, "hpu::restride") == 0)) {
        auto restride_node = value_in->node();
        auto restride_value_in = restride_node->input(0);
        if (isInGraphInputs(restride_value_in) != -1) {
          input_tiv_map_.emplace(value_to_ivalue_[restride_value_in], tiv);
        }
      }
    } else {
      input_tivs_.emplace_back(tiv);
    }
  }
}

void HabanaLaunchOpPT::HandleMappedandUnmappedTensor(
    CValPtr value_in,
    const HabanaOperatorPtr& habana_op,
    SharedSynTensorOrRefListPtr& tensorList,
    std::string idx) {
  auto is_already_mapped =
      pt_to_synapse_tensors_.find(value_to_ivalue_[value_in]) !=
      std::end(pt_to_synapse_tensors_);
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
    auto value_exists = value_to_ivalue_.find(value_in);
    HABANA_ASSERT(value_exists != std::end(value_to_ivalue_));
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
            if (value_to_ivalue_[value_in]->isTensor()) {
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
    } // if (value_to_ivalue_[value_in] && ..
  } // for (const auto value_in : node_ins)

  bool populate_seed = false;
  switch (node->kind()) {
    case torch::jit::aten::bernoulli:
    case torch::jit::aten::exponential:
    case torch::jit::aten::_fused_dropout:
    case torch::jit::aten::native_dropout:
    case torch::jit::aten::normal:
    case torch::jit::aten::randperm:
    case torch::jit::aten::rrelu_with_noise:
      populate_seed = true;
      break;
  }

  if (populate_seed) {
    int seed = 0;
    if (!syn_graph_ptr_->is_dry_run()) {
      seed = get_seed_hpu(c10::nullopt);
    }
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
    ti->set_dma_tensor_idx(aten_intermediates_.size());
    dma_input_tensorinfos_.emplace_back(ti);
    aten_intermediates_.emplace_back(seed_tensor);
  }
}

PtTensorInfoShared HabanaLaunchOpPT::ProcessPersistentNodeOutput(
    const IValPtrShared& ivpsh,
    const ValPtr& vp,
    const synapse_helpers::tensor& out_syntensor) {
  // If the node output is persistent, there are the following possibilities
  // 1. The output is not in graph output, hence it is an intermediate
  //    which is persistent.
  //    A: Add it to duplicate_input_tivs_ as this would be a duplicate of a
  //       persistent input tensor if a parent exists
  //    B: Add it to intermediate tensor list if no parent available.
  //       Right now thats the only data structure that supports adding it
  //       to patching
  //
  // 2. The output is in graph output. In this scenario, it could be -
  //    A: It is an output created by the PT kernel that goes to the
  //       graph output.
  //       Add it to output_tensorinfos_ as it is to be counted as an
  //       output tensor of the recipe.
  //       With enable_caching_, this is maintained in
  //       output_tensorinfo_map_
  //    B: It is a duplicate of an input. An example:
  //           graph(%id:0 : Float(*),
  //                 ..
  //             %1 : FLoat(*) = aten::add_(%id:0, ...)
  //                 ..
  //             return (%1, ...)
  //       Here, the aten::add_ creates a duplicate for output from the
  //       input, hence %1 is a duplicate of input %id:0. The duplicate
  //       output also goes to graph output.
  //       Add it to duplicate_input_to_outtinfo_map_ with
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
  //       Add it to duplicate_intermediate_to_outtinfo_map_ with
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
  ivalue_to_tensor_info_map_[ivpsh] = ti;
  void* buffp = ti->get_buffer_start();

  if (false == isInGraphOutputs(vp)) {
    if (ti->is_ZST() == false && buff_to_input_ivpsh_map_.count(buffp)) {
      // Case 1.A: intermediate persistent tensor which an alias of an input
      PT_BRIDGE_DEBUG("Adding to duplicate_input_tivs_ ", *ti);
      duplicate_input_tivs_.emplace_back(ti);
    } else {
      if (ti->is_ZST() == false && buff_to_output_ivpsh_map_.count(buffp)) {
        duplicate_outtinfos_.emplace_back(ti);
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
      PT_BRIDGE_DEBUG("Adding to output_tensorinfo_map_ ", *ti);
      output_tensorinfo_map_.emplace(ivpsh, ti);
    } else {
      // Is this a duplicate tensor going to graph output?
      // See if this the buffer pointer matches any input, then -
      // Check whether it is an alias of any input
      if (ti->is_ZST() == false && buff_to_input_ivpsh_map_.count(buffp)) {
        // Case 2.B: Graph output that is duplicate of input
        PT_BRIDGE_DEBUG("Adding to duplicate_input_to_outtinfo_map_ ", *ti);
        duplicate_input_to_outtinfo_map_.emplace(ivpsh, ti);
      } else if (
          ti->is_ZST() == false &&
          buff_to_intermediate_ivpsh_map_.count(buffp)) {
        // Case 2.C: Graph output that is duplicate of a persistent
        // intermediate
        PT_BRIDGE_DEBUG(
            "Adding to duplicate_intermediate_to_outtinfo_map_ ", *ti);
        duplicate_intermediate_to_outtinfo_map_.emplace(ivpsh, ti);
      } else if (
          ti->is_ZST() == false && buff_to_output_ivpsh_map_.count(buffp)) {
        // Case 2.D: Graph output that is duplicate of a previous output
        PT_BRIDGE_DEBUG("Adding to duplicate_output_to_outtinfo_map_ ", *ti);
        duplicate_output_to_outtinfo_map_.emplace(ivpsh, ti);
      } else {
        // Case 2.A: graph output tensor, enable_tensor_release_
        PT_BRIDGE_DEBUG("Adding to output_tensorinfo_map_ ", *ti);
        output_tensorinfo_map_.emplace(ivpsh, ti);
        if (ti->is_ZST() == false) {
          buff_to_output_ivpsh_map_.emplace(buffp, ivpsh);
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

  bool shape_inf_flag = enable_fast_shape_inf_ &&
      syn_graph_ptr_->is_dynamic_graph() &&
      !(map_shape_.m_pass == ShapeInfo::InferencePass::MIN_SHAPE ||
        map_shape_.m_pass == ShapeInfo::InferencePass::MAX_SHAPE);

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
        if (pt_to_synapse_tensors_.count(ivpsh)) {
          PT_BRIDGE_DEBUG(
              habana_helpers::DebugString(ivpsh),
              " already exists in pt_to_synapse_tensors_ map, ",
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
        auto ret = sif_tidx_to_tinfo_map_.insert({output_sif_tidx, ti});
        if (ret.second) {
          PT_DYNAMIC_SHAPE_DEBUG(
              "Output tensor cs: adding to sif_tidx_to_tinfo_map_ : ",
              output_sif_tidx,
              " -> ",
              *ti);
        } else {
          PT_DYNAMIC_SHAPE_DEBUG(
              "Output tensor cs: failed adding to sif_tidx_to_tinfo_map_ : ",
              output_sif_tidx,
              " -> ",
              *ti);
        }
      } else {
        // op_output_shape is empty
        auto ret = sif_tidx_to_tinfo_map_.insert({cur_sif_tidx, ti});
        if (ret.second) {
          PT_DYNAMIC_SHAPE_DEBUG(
              "Output tensor manual: adding to sif_tidx_to_tinfo_map_ : ",
              cur_sif_tidx,
              " -> ",
              *ti);
        } else {
          PT_DYNAMIC_SHAPE_DEBUG(
              "Output tensor manual: failed adding to sif_tidx_to_tinfo_map_ : ",
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
    pt_to_synapse_tensors_.emplace(
        value_to_ivalue_[nodes[node_output_idx]], tensorList);

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
      value_to_ivalue_[output_nodes[output_nodes_idx]] = ivpsh;

      // For some kernels, like the inplace ones, the kernel output is always
      // created as persistent. Patching table needs to be updated accordingly
      // for such tensors.
      if (use_persistent_tensors_ || out_tensor_syn.is_persistent()) {
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
            out_tensor_syn.tensor_type(),
            out_tensor_syn.pt_shape());
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
    IValPtrShared ivpsh = value_to_ivalue_[input_nodes[pt_input_idx]];

    // For some kernels, like the inplace ones, the kernel output is always
    // created as persistent. Patching table needs to be updated accordingly
    // for such tensors.
    if (use_persistent_tensors_ || sh_t.is_persistent()) {
      PtTensorInfoShared ti = std::make_shared<PtTensorInfo>(
          ivpsh,
          sh_t.name(),
          input_nodes[pt_input_idx],
          sh_t.id(),
          sh_t.get(),
          sh_t.tensor_type());

      ti->set_external(sh_t.is_external());
      duplicate_input_tivs_.emplace_back(ti);

      // persistent tensor which an alias of an input
      PT_BRIDGE_DEBUG("Adding to duplicate_input_tivs_ ", *ti);
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
      !(map_shape_.m_pass == ShapeInfo::InferencePass::MIN_SHAPE ||
        map_shape_.m_pass == ShapeInfo::InferencePass::MAX_SHAPE);

  auto backend_ST_TIDs = ShapeInference::GetBackendStTidxList();

  if (auto op = std::dynamic_pointer_cast<OpBackend>(habanaOp)) {
    for (const auto& st : op->GetShapeTensors()) {
      if (enable_optim_output_sif_ &&
          (backend_ST_TIDs.find(st.id()) == backend_ST_TIDs.end())) {
        continue;
      }

      auto irn = "%shapeInput_" + std::to_string(shape_index_++);
      PtTensorInfoShared ti = std::make_shared<PtTensorInfo>(st, irn);
      if (shape_inf_flag) {
        if (st.is_intermediate_shape_tensor()) {
          intermediate_shape_tensors.emplace_back(shape_tensor_tinfos_.size());
          PT_DYNAMIC_SHAPE_DEBUG(
              "auto_gen path: intermediate shape tensor : ", *ti);
        }
      }
      shape_tensor_tinfos_.emplace_back(ti);
    }
  }

  for (synapse_helpers::tensor& maybe_syn_shape_tensor :
       habanaOp->GetSynInputs()) {
    if (maybe_syn_shape_tensor.is_shape_tensor()) {
      if (enable_optim_output_sif_ &&
          (backend_ST_TIDs.find(maybe_syn_shape_tensor.id()) ==
           backend_ST_TIDs.end())) {
        continue;
      }

      std::string irn{"%shapeInput_"};
      irn += std::to_string(shape_index_);
      shape_index_++;
      PtTensorInfoShared ti =
          std::make_shared<PtTensorInfo>(maybe_syn_shape_tensor, irn);
      if (shape_inf_flag) {
        if (maybe_syn_shape_tensor.is_intermediate_shape_tensor()) {
          intermediate_shape_tensors.emplace_back(shape_tensor_tinfos_.size());
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

          inputs_shape_tensors.emplace_back(shape_tensor_tinfos_.size());
          PT_DYNAMIC_SHAPE_DEBUG(
              "manual path: adding input shape tensor for index = ",
              inputs_shape_tensors.back(),
              " : ",
              *ti);
        }
      }
      shape_tensor_tinfos_.emplace_back(ti);
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
      pt_to_synapse_tensors_.find(value_to_ivalue_[value_in]);
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
                HPUDeviceContext::get_device(tensor->device().index()),
                syn_tensor_input.graph());

    meta_syn_tensors_.push_back(
        absl::get<synapse_helpers::tensor>(std::move(variant)));

    PtTensorInfoShared ti = std::make_shared<PtTensorInfo>(
        value_to_ivalue_[value_in],
        meta_syn_tensors_.back().name(),
        value_in,
        meta_syn_tensors_.back().id(),
        meta_syn_tensors_.back().get(),
        meta_syn_tensors_.back().tensor_type());
    ivalue_to_tensor_info_map_[value_to_ivalue_[value_in]] = ti;
    if (!isInGraphOutputs(value_in)) {
      duplicate_input_tivs_.emplace_back(ti);
    } else {
      if (!enable_caching_ && !enable_shape_agnostic_caching_) {
        output_tensorinfos_.emplace_back(ti);
      } else {
        duplicate_outtinfos_.emplace_back(ti);
      }
    }
  } else {
    pt_to_synapse_tensors_.erase(value_to_ivalue_[value_in]);
    auto variant = habana_helpers::create_tensor(
        *tensor, *syn_graph_ptr_, persistence, false);
    meta_syn_tensors_.push_back((std::move(variant)));
  }

  auto& syn_tensor = meta_syn_tensors_.back();
  pt_to_synapse_tensors_.erase(value_to_ivalue_[value_in]);
  SharedSynTensorOrRefListPtr tensorList =
      std::make_shared<SynTensorOrRefList>();
  tensorList->emplace_back(synapse_helpers::tensor_or_ref(syn_tensor));
  pt_to_synapse_tensors_.emplace(value_to_ivalue_[value_in], tensorList);
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
    SynBuildCache& syn_build_cache,
    bool is_restride_cl) {
  auto value_in = node->input(0);
  auto value_out = node->output(0);
  HABANA_ASSERT(value_to_ivalue_.find(value_in) != std::end(value_to_ivalue_));
  HABANA_ASSERT(value_to_ivalue_[value_in]->isTensor());
  auto tensor = value_to_ivalue_[value_in]->toTensor();
  auto is_5d_layout = tensor.dim() == 5 ? true : false;

  auto is_in_graph_outputs =
      syn_build_cache.get_or_compute<&SynBuildCache::is_in_graph_outputs>(
          [value_out]() { return isInGraphOutputs(value_out); },
          restride_node_out_val_counter_);

  restride_node_out_val_counter_++;

  if ((tensor.dim() == 4) || (tensor.dim() == 5)) {
    std::vector<int64_t>& new_pos =
        syn_build_cache.get_or_compute_ref<&SynBuildCache::new_positions>(
            [node]() { return toIValue(node->input(1))->toIntVector(); },
            restride_node_swap_counter_);

    restride_node_swap_counter_++;

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
  auto ivpsh = value_to_ivalue_[value_in];
  auto ivpsh_restrided = std::make_shared<IVal>(tensor);
  PT_BRIDGE_DEBUG(
      "processing restride node, input %",
      value_in->debugName(),
      " output",
      value_out->debugName());

  if (is_in_graph_outputs) {
    TORCH_CHECK(
        pt_to_synapse_tensors_.count(ivpsh),
        " Could not find the syn tensor corresponding to %",
        value_in->debugName());

    auto& syn_tensor_vec = pt_to_synapse_tensors_[ivpsh];
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

    value_to_ivalue_.erase(value_in);

    if (enable_caching_ || enable_shape_agnostic_caching_) {
      void* buffp = ti->get_buffer_start();
      if (output_tensorinfo_map_.count(ivpsh)) {
        // Case 2.A: graph output tensor
        PT_BRIDGE_DEBUG(
            "removing ivalue for restride input %",
            value_in->debugName(),
            " from output_tensorinfo_map_");
        output_tensorinfo_map_.erase(ivpsh);

        PT_BRIDGE_DEBUG(
            "adding ivalue for restride output %",
            value_out->debugName(),
            " to output_tensorinfo_map_");
        output_tensorinfo_map_.emplace(ivpsh_restrided, ti);
      } else if (
          ti->is_ZST() == false &&
          buff_to_intermediate_ivpsh_map_.count(buffp)) {
        // Case 2.C: Graph output that is duplicate of a persistent
        // intermediate
        PT_BRIDGE_DEBUG(
            "updating buff_to_intermediate_ivpsh_map_ entry for ",
            buffp,
            " with ivalue for restride output %",
            value_out->debugName());

        buff_to_intermediate_ivpsh_map_.erase(buffp);
        buff_to_intermediate_ivpsh_map_.emplace(buffp, ivpsh_restrided);
        TORCH_CHECK(
            duplicate_intermediate_to_outtinfo_map_.count(ivpsh),
            " entry for restride input %",
            value_in->debugName(),
            " not found in duplicate_intermediate_to_outtinfo_map_");

        PT_BRIDGE_DEBUG(
            "updating duplicate_intermediate_to_outtinfo_map_ entry for ",
            buffp,
            " with ivalue for restride output %",
            value_out->debugName());
        duplicate_intermediate_to_outtinfo_map_.erase(ivpsh);
        duplicate_intermediate_to_outtinfo_map_.emplace(ivpsh_restrided, ti);
      } else if (
          ti->is_ZST() == false && buff_to_input_ivpsh_map_.count(buffp)) {
        // Case 2.B: Graph output that is duplicate of input
        PT_BRIDGE_DEBUG(
            "updating buff_to_input_ivpsh_map_ entry for ",
            buffp,
            " with ivalue for restride output %",
            value_out->debugName());

        buff_to_input_ivpsh_map_.erase(buffp);
        buff_to_input_ivpsh_map_.emplace(buffp, ivpsh_restrided);

        TORCH_CHECK(
            duplicate_input_to_outtinfo_map_.count(ivpsh),
            " entry for restride input %",
            value_in->debugName(),
            " not found in duplicate_input_to_outtinfo_map_");

        PT_BRIDGE_DEBUG(
            "updating duplicate_input_to_outtinfo_map_ entry for ",
            buffp,
            " with ivalue for restride output %",
            value_out->debugName());
        duplicate_input_to_outtinfo_map_.erase(ivpsh);
        duplicate_input_to_outtinfo_map_.emplace(ivpsh_restrided, ti);
      } else if (
          ti->is_ZST() == false && buff_to_output_ivpsh_map_.count(buffp)) {
        // Case 2.D: Graph output that is duplicate of a previous output
        PT_BRIDGE_DEBUG(
            "updating buff_to_output_ivpsh_map_ entry for ",
            buffp,
            " with ivalue for restride output %",
            value_out->debugName());

        buff_to_output_ivpsh_map_.erase(buffp);
        buff_to_output_ivpsh_map_.emplace(buffp, ivpsh_restrided);

        TORCH_CHECK(
            duplicate_output_to_outtinfo_map_.count(ivpsh),
            " entry for restride input %",
            value_in->debugName(),
            " not found in duplicate_output_to_outtinfo_map_");

        PT_BRIDGE_DEBUG(
            "updating duplicate_output_to_outtinfo_map_ entry for ",
            buffp,
            " with ivalue for restride output %",
            value_out->debugName());
        duplicate_output_to_outtinfo_map_.erase(ivpsh);
        duplicate_output_to_outtinfo_map_.emplace(ivpsh_restrided, ti);
      } else {
        TORCH_CHECK(
            false,
            " unhandled scenario for restride input %",
            value_in->debugName(),
            (ti->is_ZST() ? " is ZST" : " is non ZST"),
            ", not found in any duplicate detection or output map");
      }
    }

    value_to_ivalue_[value_in] = ivpsh_restrided;
    value_to_ivalue_[value_out] = ivpsh_restrided;
    ivalue_to_tensor_info_map_[value_to_ivalue_[value_in]] = ti;
    ivalue_to_tensor_info_map_[value_to_ivalue_[value_out]] = ti;
  } else {
    PT_BRIDGE_DEBUG(
        "restride node output %", value_out->debugName(), " is non persistent");
    value_to_ivalue_[value_out] = ivpsh_restrided;
  }
}

void HabanaLaunchOpPT::handlePrimNodes(
    torch::jit::Node* node,
    SynBuildCache& syn_build_cache) {
  PT_BRIDGE_TRACE;
  if (node->kind() == torch::jit::prim::Constant) {
    handlePrimConstantNode(node, syn_build_cache);
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

#define HANDLE_LIST_OF(T, isFn, toFn)                      \
  if (ivptrsh->isFn()) {                                   \
    c10::List<T> list;                                     \
    for (const auto& value_in : node_ins) {                \
      ivptrsh = value_to_ivalue[value_in];                 \
      HABANA_ASSERT(ivptrsh->isFn());                      \
      list.emplace_back(ivptrsh->toFn());                  \
    }                                                      \
    IValPtrShared out_ival = std::make_shared<IVal>(list); \
    return out_ival;                                       \
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

  HANDLE_LIST_OF(at::Tensor, isTensor, toTensor)
  HANDLE_LIST_OF(int64_t, isInt, toInt)
  HANDLE_LIST_OF(double, isDouble, toDouble)
  HANDLE_LIST_OF(bool, isBool, toBool)

  HABANA_ASSERT(false, "Unsupported list type in prim::ListConstruct");
}

void UpdateIshapeForPrimListConstructNode(
    torch::jit::Node* node,
    habana_helpers::DynamicSIFInfo& dsi) {
  const auto& node_ins = node->inputs();
  const auto& node_vals = node->outputs();
  HABANA_ASSERT(node_vals.size() == 1);
  for (const auto& value_in : node_ins) {
    const auto& value_name = value_in->debugName();
    if (!dsi.value_to_ishape[value_name].IsUpdated()) {
      if (dsi.value_to_ishape[value_name].isScalar()) {
        continue;
      } else if (dsi.value_to_ishape[value_name].isTensor()) {
        auto& size_expr_list = dsi.value_to_sizeexpr[value_name];
        HABANA_ASSERT(size_expr_list.size() == 1);
        auto& size_expr = size_expr_list[0];
        if (size_expr != nullptr) {
          SymExprFactory& expr_factory = SymExprFactory::getInstance();
          std::vector<int64_t> concrete_size =
              expr_factory.evaluate_symsize(size_expr);
          dsi.value_to_ishape[value_name].UpdateTensor(concrete_size);
        }
      }
    }
  }
}

IValPtrShared MapPrimListConstructNodeInputIShape(
    torch::jit::Node* node,
    habana_helpers::DynamicSIFInfo* dsi,
    CValuePtrToIValuePtrMap& value_to_ivalue) {
  const auto& node_ins = node->inputs();
  auto node_outs = node->outputs();
  HABANA_ASSERT(node_outs.size() == 1);
  for (const auto& value_in : node_ins) {
    if (value_to_ivalue.count(value_in) == 0) {
      auto value_name = value_in->debugName();
      torch::jit::Node* producer_node = value_in->node();
      if (producer_node->kind() == torch::jit::prim::Constant) {
        // The Scalar as stack input already got updated as input part and
        // Rest is constant so no need to update, we can use previous values
        torch::jit::IValue const_ivalue =
            torch::jit::toIValue(value_in).value();
        auto ivptrsh = std::make_shared<IVal>(const_ivalue);
        value_to_ivalue[value_in] = ivptrsh;
      } else if (dsi->value_to_ishape[value_name].isScalar()) {
        auto ival =
            torch::jit::IValue(dsi->value_to_ishape[value_name].getScalar());
        auto ivptrsh = std::make_shared<IVal>(ival);
        value_to_ivalue[value_in] = ivptrsh;
      } else if (dsi->value_to_ishape[value_name].isTensor()) {
        auto& size_expr_list = dsi->value_to_sizeexpr[value_name];
        HABANA_ASSERT(size_expr_list.size() == 1);
        auto& size_expr = size_expr_list[0];
        if (size_expr == nullptr) {
          HABANA_ASSERT(
              dsi->value_to_ishape.count(value_name),
              "Node not added to value to ishape map:",
              value_in->debugName());
          std::vector<int64_t> concrete_size;
          if (dsi->value_to_ishape[value_name].IsUpdated()) {
            concrete_size = dsi->value_to_ishape[value_name].getTensorShape();
          }
          at::Tensor dummy_t = createDynamicTensor(
              concrete_size,
              DATA_TENSOR,
              dsi->value_to_ishape[value_name].getScalarType());
          auto iv_tensor = torch::jit::IValue(dummy_t);
          auto ivptrsh = std::make_shared<IVal>(iv_tensor);
          value_to_ivalue[value_in] = ivptrsh;
          PT_DYNAMIC_SHAPE_DEBUG(
              "Creating ivalue for node:", value_name, " nullptr size_expr:");
        } else {
          SymExprFactory& expr_factory = SymExprFactory::getInstance();
          std::vector<int64_t> concrete_size =
              expr_factory.evaluate_symsize(size_expr);
          if (dsi->value_to_ishape[value_name].isTensor()) {
            dsi->value_to_ishape[value_name].UpdateTensor(concrete_size);
          }
          at::Tensor dummy_t = createDynamicTensor(
              concrete_size,
              DATA_TENSOR,
              dsi->value_to_ishape[value_name].getScalarType());
          auto iv_tensor = torch::jit::IValue(dummy_t);
          auto ivptrsh = std::make_shared<IVal>(iv_tensor);
          PT_DYNAMIC_SHAPE_DEBUG(
              "Creating ivalue for node:",
              value_name,
              " with concrete_size = ",
              concrete_size);
          value_to_ivalue[value_in] = ivptrsh;
        }
      } else {
        HABANA_ASSERT(
            0,
            "Unable to create ivalue for the ListConstruct input:",
            value_name);
      }
    }
  }

  // ListConstruct can have optional and non-optional tensors as item types
  if (node->output()->type()->containedTypes()[0]->kind() ==
      OptionalType::Kind) {
    c10::List<c10::optional<at::Tensor>> opttensorList;
    for (const auto& value_in : node_ins) {
      static_cast<void>(value_in);
      opttensorList.emplace_back(c10::nullopt);
    }

    IValPtrShared out_ival = std::make_shared<IVal>(opttensorList);
    return out_ival;
  }
  // Handle empty list
  if (node_ins.empty()) {
    return std::make_shared<IVal>(c10::List<int64_t>());
  }

  auto ivptrsh = value_to_ivalue[node_ins[0]];
  HANDLE_LIST_OF(at::Tensor, isTensor, toTensor)
  HANDLE_LIST_OF(int64_t, isInt, toInt)
  HANDLE_LIST_OF(double, isDouble, toDouble)
  HANDLE_LIST_OF(bool, isBool, toBool)
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

at::Tensor createDynamicTensor(
    const std::vector<int64_t>& size,
    synTensorType type,
    c10::ScalarType dtype) {
  auto allocator = habana::getHABANADeviceAllocator();
  constexpr c10::DispatchKeySet hpu_ks(c10::DispatchKey::HPU);

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
      GetPrimListConstructNodeOuputIValue(node, value_to_ivalue_);
  value_to_ivalue_[node_vals[0]] = ival;
}

void HabanaLaunchOpPT::handlePrimConstantNode(
    torch::jit::Node* node,
    SynBuildCache& syn_build_cache) {
  auto node_vals = node->outputs();
  bool is_jit_cached_graph_info_available = syn_build_cache.is_complete();
  for (const auto value : node_vals) {
    IValPtrShared ivptrsh = nullptr;
    if (is_jit_cached_graph_info_available == false) {
      ivptrsh = std::make_shared<IVal>(toIValue(value).value());
    }
    if (value->type()->kind() == c10::TypeKind::TensorType) {
      auto ivptrsh_updated =
          syn_build_cache.get_or_compute<&SynBuildCache::prim_nodes_ivals>(
              [&ivptrsh]() { return castConstantTensor(ivptrsh); },
              prim_nodes_ival_counter_);
      value_to_ivalue_[value] = ivptrsh_updated;
      std::string irn{"%intermediate_"};
      irn += std::to_string(intermediate_index_);
      intermediate_index_++;

      auto tensor = ivptrsh_updated->toTensor();
      meta_syn_tensors_.push_back(habana_helpers::create_tensor(
          tensor, *syn_graph_ptr_, true, false, tensor.scalar_type()));
      SharedSynTensorOrRefListPtr tensorList =
          std::make_shared<SynTensorOrRefList>();
      tensorList->emplace_back(
          synapse_helpers::tensor_or_ref(meta_syn_tensors_.back()));
      pt_to_synapse_tensors_.emplace(value_to_ivalue_[value], tensorList);
      PtTensorInfoShared ti = std::make_shared<PtTensorInfo>(
          tensor,
          meta_syn_tensors_.back().name(),
          irn,
          meta_syn_tensors_.back().id(),
          meta_syn_tensors_.back().get(),
          meta_syn_tensors_.back().tensor_type());

      ivalue_to_tensor_info_map_[ivptrsh_updated] = ti;
      aten_intermediates_.push_back(tensor);
    } else {
      ivptrsh =
          syn_build_cache.get_or_compute<&SynBuildCache::prim_nodes_ivals>(
              [&ivptrsh]() { return ivptrsh; }, prim_nodes_ival_counter_);
      value_to_ivalue_[value] = ivptrsh;
    }
    const CValPtrtoIValueMap& param_val_to_ival_map =
        jit_graph_and_meta_data_->get_param_jit_val_to_ivalue_map();
    if (param_val_to_ival_map.count(value)) {
      auto ivalue = param_val_to_ival_map.at(value);
      value_to_ivalue_[value] = std::make_shared<IVal>(ivalue);
      PT_BRIDGE_DEBUG(
          "For %",
          value->debugName(),
          " updating to ivalue: ",
          habana_helpers::DebugString(ivalue));
    }
    prim_nodes_ival_counter_++;
  }
}

torch::jit::Stack HabanaLaunchOpPT::getStackForNode(torch::jit::Node* node) {
  torch::jit::Stack stack_in;
  auto node_inputs = node->inputs();
  for (auto input : node_inputs) {
    if (value_to_ivalue_.count(input)) {
      stack_in.insert(stack_in.end(), *value_to_ivalue_[input]);
    } else {
      stack_in.insert(stack_in.end(), IValue());
    }
  }
  return stack_in;
}

habana_helpers::IShapeList HabanaLaunchOpPT::getInputIShapesForNode(
    torch::jit::Node* node,
    RecipeValueSpec& rv) {
  auto& dsi = rv.ds_sifinfo_map[sym_expr_hash_];
  const auto& node_inputs = node->inputs();
  habana_helpers::IShapeList ishape_list;
  ishape_list.reserve(node_inputs.size());
  for (auto& input : node_inputs) {
    const auto& value_name = input->debugName();
    if (dsi.value_to_ishape.count(value_name)) {
      auto ishape = dsi.value_to_ishape[value_name];
      if (ishape.isTensor() || ishape.isScalar()) {
        ishape_list.push_back(ishape);
      } else {
        PT_DYNAMIC_SHAPE_DEBUG(
            "Value = ", value_name, " Ishape is not a supported type !!!");
      }
    } else {
      ishape_list.push_back(habana_helpers::IShape());
      PT_DYNAMIC_SHAPE_DEBUG("Dummy IShape added for node = ", value_name);
    }
  }
  return ishape_list;
}

habana_helpers::IShapeList HabanaLaunchOpPT::getOutputIShapesForNode(
    torch::jit::Node* node,
    RecipeValueSpec& rv) {
  auto& dsi = rv.ds_sifinfo_map[sym_expr_hash_];
  const auto& node_outputs = node->outputs();
  habana_helpers::IShapeList ishape_list;
  ishape_list.reserve(node_outputs.size());
  for (auto& node_val : node_outputs) {
    const auto& value_name = node_val->debugName();
    if (dsi.value_to_ishape.count(value_name)) {
      auto& ishape = dsi.value_to_ishape[value_name];
      if (ishape.isTensor()) {
        ishape_list.push_back(ishape);
      } else {
        PT_BRIDGE_DEBUG(
            "Node = ",
            value_name,
            " Ishape is not a tensor type and not supported !!!");
      }
    } else {
      ishape_list.push_back(habana_helpers::IShape());
      PT_BRIDGE_DEBUG(
          "Node = ",
          value_name,
          " Ishape is missing in the value_to_ishape map !!!");
    }
  }

  return ishape_list;
}

c10::ScalarType HabanaLaunchOpPT::getNodeScalarType(torch::jit::Node* node) {
  // return the data type of first input tensor
  for (auto input : node->inputs()) {
    if (value_to_ivalue_.count(input) && value_to_ivalue_[input]->isTensor()) {
      return value_to_ivalue_[input]->toTensor().scalar_type();
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
    stack.insert(stack.end(), *value_to_ivalue_[value_in]);
    if (value_to_ivalue_[value_in]->isTensor()) {
      auto tensor = value_to_ivalue_[value_in]->toTensor();
      HABANA_ASSERT(
          pt_to_synapse_tensors_.find(value_to_ivalue_[value_in]) !=
          std::end(pt_to_synapse_tensors_))

      // Below code is commented for now, since we dont handle
      // any meta ops that would create a new pytorch/syanpse tensor
      // If we need view to be handled as a meta op, need to enable
      // the below code
      /*
      if (pt_to_synapse_tensors_.find(value_to_ivalue_[value_in]) ==
          std::end(pt_to_synapse_tensors_)) {
        in_data = tensor.data_ptr();
        input_ptr = value_to_ivalue_[value_in];
        auto dtype = tensor.scalar_type();
        meta_syn_tensors_.push_back(habana_helpers::create_tensor(
            tensor, *syn_graph_ptr, true, dtype));
        SharedSynTensorOrRefListPtr tensorList =
            std::make_shared<SynTensorOrRefList>();
        tensorList->emplace_back(synapse_helpers::tensor_or_ref(meta_syn_tensors_.back()));
        pt_to_synapse_tensors_.emplace(value_to_ivalue_[value_in], tensorList);

        if (enable_caching_) {
          input_tiv_map_.emplace(
              value_to_ivalue_[value_in],
              PtTensorInfo(
                  value_to_ivalue_[value_in],
                  meta_syn_tensors_.back().name(),
                  value_in));
          buff_to_input_ivpsh_map_.emplace(in_data, value_to_ivalue_[value_in]);
        } else {
          input_tivs_.emplace_back(PtTensorInfo(
              value_to_ivalue_[value_in],
              meta_syn_tensors_.back().name(),
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
    value_to_ivalue_[val_out] = ival;
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
        o << " scalarType ";
        o << tensor.scalar_type();
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
        o << " scalarType ";
        o << tensor.scalar_type();
        str.append(o.str());
        str.append("\n");
      }
    }
  }
  return str;
}
} // namespace

namespace OpInfo {
std::string DumpPassInfo(
    synapse_helpers::graph& graph,
    const ShapeInfo::InferencePass& pass) {
  if (graph.is_dynamic_graph()) {
    if (pass == ShapeInfo::InferencePass::MIN_SHAPE) {
      return "SIF_MIN ";
    } else if (pass == ShapeInfo::InferencePass::MAX_SHAPE) {
      return "SIF_MAX ";
    } else if (pass == ShapeInfo::InferencePass::OUTPUT_SHAPE) {
      return "SIF_OUTPUT ";
    }
  }
  return "";
}

std::string DumpOpInfo(
    const at::OperatorName& opname,
    const torch::jit::Stack& input_stack) {
  std::ostringstream out;
  out << opname << ":";
  int arg_id = 1;
  for (size_t i = 0; i < input_stack.size(); i++) {
    auto& input = input_stack[i];
    out << " %" << arg_id++ << "=";
    if (input.isTensor()) {
      out << to_string(input.toTensor());
    } else if (input.isTensorList()) {
      out << to_string(input.toTensorList());
    } else if (input.isOptionalTensorList()) {
      out << to_string(input.toOptionalTensorList());
    } else {
      out << input;
    }
  }
  return out.str();
}
} // namespace OpInfo

void DumpSymbolValueMap(InputSymbolMap& symbol_value_map) {
  std::for_each(
      symbol_value_map.begin(),
      symbol_value_map.end(),
      [&](const std::pair<std::string, std::shared_ptr<double>>& p) {
        PT_TEST_DEBUG(
            "Dump Input symbol->value map, key:",
            p.first,
            ", value:",
            *p.second);
      });
}

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
  const auto is_allow = is_allow_view_output_permutation(ivpsh->toTensor());
  const auto skip_permute =
      jit_graph_and_meta_data_->is_skip_tensor_permutation();
  if ((rank >= 2) && is_allow && !skip_permute) {
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
      *pt_stack_,
      graph_input_info.max_input_tshapes,
      ShapeInfo::InferencePass::MAX_SHAPE);
  updatemax_graph_ = false;
}

void HabanaLaunchOpPT::RevertH2DMinMaxData() {
  for (size_t i = 0; i < pt_stack_->size(); ++i) {
    if (pt_stack_->at(i).isTensor()) {
      auto& input_tensor = pt_stack_->at(i);
      auto tmeta = get_tensor_extra_meta(pt_stack_->at(i).toTensor());
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
  updatemax_graph_ = false;
}

void HabanaLaunchOpPT::UpdateValueIShapeMapForListUnpack(
    torch::jit::Node* node,
    RecipeValueSpec& rv) {
  auto& dsi = rv.ds_sifinfo_map[sym_expr_hash_];
  auto node_input = node->inputs()[0];
  auto input_name = node_input->debugName();
  for (auto& output_val : node->outputs()) {
    auto output_name = output_val->debugName();
    HABANA_ASSERT(
        dsi.value_to_ishape.count(output_name),
        "ListUnpack %s is not part of value_to_ishape",
        output_name)
    auto size_expr_list = dsi.value_to_sizeexpr[output_name];
    HABANA_ASSERT(
        size_expr_list.size() == 1,
        "ListUnpack %s is not part of value_to_sizeexpr",
        output_name);
    auto& size_expr = size_expr_list[0];
    SymExprFactory& expr_factory = SymExprFactory::getInstance();
    std::vector<int64_t> concrete_size =
        expr_factory.evaluate_symsize(size_expr);
    dsi.value_to_ishape[output_name].UpdateTensor(concrete_size);
    PT_DYNAMIC_SHAPE_DEBUG(
        "ListUnpack output_name:%s",
        output_name,
        ", size_expr:",
        size_expr->get_size_expr_str(),
        ", concrete_size:",
        concrete_size);
  }
}

void HabanaLaunchOpPT::CreateValueIShapeMapForNode(
    torch::jit::Node* node,
    torch::jit::Node* rv_node,
    const torch::jit::Stack& input_stack,
    OutputMetaDataVector& output_meta_vec) {
  HABANA_ASSERT(
      node->inputs().size() == rv_node->inputs().size(),
      "Input size missmatch while creating IShape");
  PT_DYNAMIC_SHAPE_DEBUG("Performing IShape creation");
  auto node_inputs = node->inputs();
  int64_t input_count = 0;
  auto& dsi = ds_sif_info_;
  for (auto node_val : node_inputs) {
    auto rv_node_val = rv_node->input(input_count);
    if (value_to_ivalue_.count(node_val) &&
        (!dsi.value_to_ishape.count(rv_node_val->debugName()))) {
      auto rv_value_name = rv_node_val->debugName();
      torch::jit::Node* producer_node = node_val->node();
      torch::jit::Node* rv_producer_node = rv_node_val->node();
      auto ivalue = value_to_ivalue_[node_val];
      if (producer_node->kind() == torch::jit::prim::Constant) {
        if (ivalue->isTensor()) {
          auto tensor = ivalue->toTensor();
          habana_helpers::IShape ishape(
              tensor.sizes().vec(), tensor.scalar_type());
          dsi.value_to_ishape.insert({rv_value_name, ishape});
          PT_DYNAMIC_SHAPE_DEBUG(
              "Create Ishape Tensor ",
              rv_value_name,
              " Scalar type ",
              tensor.scalar_type());
        } else if (node_val->type() != torch::jit::NoneType::get()) {
          // TODO Albin check if any other condition need to check here and
          // check the value
          torch::jit::IValue const_ivalue =
              torch::jit::toIValue(node_val).value();
          if (const_ivalue.isScalar()) {
            // TODO: Check Ishape has to be created with actual scalar type
            habana_helpers::IShape ishape(
                const_ivalue.toScalar(), const_ivalue.toScalar().type());
            dsi.value_to_ishape.insert({rv_value_name, ishape});
            PT_DYNAMIC_SHAPE_DEBUG(
                "Create Ishape Scalar ",
                rv_value_name,
                " Scalar type ",
                ishape.getScalarType());
          } else {
            PT_DYNAMIC_SHAPE_DEBUG(
                "Input node of non-scalar type is added to value_to_ishape map, type:",
                rv_node_val->type());
          }
        }
      } else if (producer_node->kind() == torch::jit::prim::ListConstruct) {
        CreateValueIShapeMapForNode(
            producer_node, rv_producer_node, input_stack, output_meta_vec);
      } else if (ivalue->isTensor()) {
        auto input_stack_dtype =
            input_stack.at(input_count).toTensor().scalar_type();
        auto tensor = ivalue->toTensor();
        habana_helpers::IShape ishape(tensor.sizes().vec(), input_stack_dtype);
        dsi.value_to_ishape.insert({rv_value_name, ishape});
        PT_DYNAMIC_SHAPE_DEBUG(
            "Create Ishape Tensor ",
            rv_value_name,
            " Scalar type ",
            tensor.scalar_type());
      } else if (ivalue->isScalar()) {
        auto scalar = ivalue->toScalar();
        // TODO: Check Ishape has to be created with actual scalar type
        habana_helpers::IShape ishape(scalar, scalar.type());
        dsi.value_to_ishape.insert({rv_value_name, ishape});
        PT_DYNAMIC_SHAPE_DEBUG(
            "Create Ishape Scalar ",
            rv_value_name,
            " Scalar type ",
            ishape.getScalarType());
      } else {
        PT_DYNAMIC_SHAPE_DEBUG(
            "Input node %s is not tensor nor scalar. Skip !!!!", rv_value_name);
      }
    }
    input_count++;
  }

  auto node_outputs = node->outputs();
  auto rv_node_outputs = rv_node->outputs();
  // Handle prim::ListUnpack with tensor list inputs.
  // Current Jit node
  if (*node->output(0)->type() == *torch::ListType::ofTensors() &&
      node->outputs().size() == 1 &&
      !(node->kind() == torch::jit::prim::ListConstruct)) {
    auto unpack_node = GetUnpackNodeFromTensorList(node->output(0));
    HABANA_ASSERT(
        unpack_node != nullptr,
        "TensorList is not input to ListUnpack node. Node: ",
        node->kind().toQualString());
    node_outputs = unpack_node->outputs();
  }
  // Cached Jit rv_node
  if (*rv_node->output(0)->type() == *torch::ListType::ofTensors() &&
      rv_node->outputs().size() == 1 &&
      !(rv_node->kind() == torch::jit::prim::ListConstruct)) {
    auto unpack_node = GetUnpackNodeFromTensorList(rv_node->output(0));
    HABANA_ASSERT(
        unpack_node != nullptr,
        "TensorList is not input to ListUnpack node. Node: ",
        node->kind().toQualString());
    rv_node_outputs = unpack_node->outputs();
  }

  HABANA_ASSERT(
      rv_node_outputs.size() == node_outputs.size(),
      "Output size missmatch while creating IShape");
  auto rv_node_output_it = rv_node_outputs.begin();
  for (auto node_out_val : node_outputs) {
    auto rv_value_name = (*rv_node_output_it)->debugName();
    if (value_to_ivalue_.count(node_out_val)) {
      auto ivalue = value_to_ivalue_[node_out_val];
      if (ivalue->isTensor()) {
        auto tensor = ivalue->toTensor();
        habana_helpers::IShape ishape(
            tensor.sizes().vec(), tensor.scalar_type());
        dsi.value_to_ishape.insert({rv_value_name, ishape});
        PT_DYNAMIC_SHAPE_DEBUG(
            "Create Ishape Tensor ",
            rv_value_name,
            " Scalar type ",
            tensor.scalar_type());
      } else if (ivalue->isScalar()) {
        auto scalar = ivalue->toScalar();
        habana_helpers::IShape ishape(scalar, scalar.type());
        dsi.value_to_ishape.insert({rv_value_name, ishape});
        PT_DYNAMIC_SHAPE_DEBUG(
            "Create Ishape Scalar ",
            rv_value_name,
            " Scalar type ",
            ishape.getScalarType());
      } else {
        PT_DYNAMIC_SHAPE_DEBUG(
            "Ouput node is not tensor nor scalar. Skip !!!!");
      }
    } else {
      PT_DYNAMIC_SHAPE_DEBUG(
          "Ouput node %s is not presented in the value_to_ivalue_",
          rv_value_name);
    }
    ++rv_node_output_it;
  }
}

void HabanaLaunchOpPT::UpdateIshapeForNodeInputs(
    torch::jit::Node* node,
    RecipeValueSpec& rv) {
  auto& dsi = rv.ds_sifinfo_map[sym_expr_hash_];
  const auto& node_inputs = node->inputs();
  for (auto* node_val : node_inputs) {
    const auto& value_name = node_val->debugName();
    if (!dsi.value_to_ishape[value_name].IsUpdated()) {
      torch::jit::Node* producer_node = node_val->node();
      PT_DYNAMIC_SHAPE_DEBUG("Updating IShape for input node = ", value_name);
      if (producer_node->kind() == torch::jit::prim::ListConstruct) {
        UpdateIshapeForPrimListConstructNode(producer_node, dsi);
      } else if (dsi.value_to_ishape[value_name].isTensor()) {
        auto& size_expr_list = dsi.value_to_sizeexpr[value_name];
        HABANA_ASSERT(size_expr_list.size() == 1);
        auto& size_expr = size_expr_list[0];
        if (size_expr != nullptr) {
          SymExprFactory& expr_factory = SymExprFactory::getInstance();
          std::vector<int64_t> concrete_size =
              expr_factory.evaluate_symsize(size_expr);
          PT_DYNAMIC_SHAPE_DEBUG(
              "Input Node: ",
              value_name,
              ", concrete_size:",
              concrete_size,
              ", dtype:",
              dsi.value_to_ishape[value_name].getScalarType());
          dsi.value_to_ishape[value_name].UpdateTensor(concrete_size);
        } else {
          PT_DYNAMIC_SHAPE_DEBUG(
              "Cannot evaluate Input shape for node ",
              value_name,
              " expression is NULL");
        }
      }
    }
  }
}

void HabanaLaunchOpPT::CreateIValueForNodeInputs(
    torch::jit::Node* node,
    habana_helpers::DynamicSIFInfo* dsi) {
  auto node_inputs = node->inputs();
  for (auto* node_val : node_inputs) {
    if (value_to_ivalue_.count(node_val) == 0) {
      torch::jit::Node* producer_node = node_val->node();
      auto value_name = node_val->debugName();
      PT_DYNAMIC_SHAPE_DEBUG(
          "Processing node inputs for Ivalue creation:", value_name);
      if (producer_node->kind() == torch::jit::prim::Constant) {
        // The Scalar as stack input already got updated as input part and
        // Rest is constant so no need to update, we can use previous values
        torch::jit::IValue const_ivalue =
            torch::jit::toIValue(node_val).value();
        auto ivptrsh = std::make_shared<IVal>(const_ivalue);
        value_to_ivalue_[node_val] = ivptrsh;
      } else if (producer_node->kind() == torch::jit::prim::ListConstruct) {
        IValPtrShared ival = MapPrimListConstructNodeInputIShape(
            producer_node, dsi, value_to_ivalue_);
        value_to_ivalue_[node_val] = ival;
      } else {
        auto& size_expr_list = dsi->value_to_sizeexpr[value_name];
        HABANA_ASSERT(size_expr_list.size() == 1);
        auto& size_expr = size_expr_list[0];
        if (size_expr == nullptr) {
          std::vector<int64_t> concrete_size;
          if (dsi->value_to_ishape.count(value_name) &&
              dsi->value_to_ishape[value_name].IsUpdated()) {
            concrete_size = dsi->value_to_ishape[value_name].getTensorShape();
          }
          at::Tensor dummy_t = createDynamicTensor(
              concrete_size,
              DATA_TENSOR,
              dsi->value_to_ishape[value_name].getScalarType());
          auto iv_tensor = torch::jit::IValue(dummy_t);
          auto ivptrsh = std::make_shared<IVal>(iv_tensor);
          value_to_ivalue_[node_val] = ivptrsh;
        } else {
          SymExprFactory& expr_factory = SymExprFactory::getInstance();
          std::vector<int64_t> concrete_size =
              expr_factory.evaluate_symsize(size_expr);
          PT_BRIDGE_DEBUG(
              "Input Node: ", value_name, ", concrete_size:", concrete_size);
          at::Tensor dummy_t = createDynamicTensor(
              concrete_size,
              DATA_TENSOR,
              dsi->value_to_ishape[value_name].getScalarType());
          auto iv_tensor = torch::jit::IValue(dummy_t);
          auto ivptrsh = std::make_shared<IVal>(iv_tensor);
          value_to_ivalue_[node_val] = ivptrsh;
        }
      }
    }
  }
}

void HabanaLaunchOpPT::UpdateIshapeForNodeOuputs(
    torch::jit::Node* node,
    RecipeValueSpec& rv) {
  auto& dsi = rv.ds_sifinfo_map[sym_expr_hash_];
  const auto& node_outputs = node->outputs();
  for (auto* node_val : node_outputs) {
    const auto& value_name = node_val->debugName();
    PT_DYNAMIC_SHAPE_DEBUG("Updating Ishape for output node = ", value_name);
    if (!dsi.value_to_ishape[value_name].IsUpdated()) {
      if (node->kind() == torch::jit::prim::Constant) {
        HABANA_ASSERT(0, "Invalid Constant call for node:", value_name);
      } else if (node->kind() == torch::jit::prim::ListConstruct) {
        HABANA_ASSERT(0, "Invalid ListConstruct call for node:", value_name);
      } else if (
          dsi.value_to_ishape[value_name].isTensor() &&
          dsi.value_to_sizeexpr[value_name].size() != 0) {
        auto& size_expr_list = dsi.value_to_sizeexpr[value_name];
        HABANA_ASSERT(size_expr_list.size() == 1);
        auto& size_expr = size_expr_list[0];
        if (size_expr != nullptr) {
          SymExprFactory& expr_factory = SymExprFactory::getInstance();
          std::vector<int64_t> concrete_size =
              expr_factory.evaluate_symsize(size_expr);
          PT_DYNAMIC_SHAPE_DEBUG(
              "Output Node: ",
              value_name,
              ", concrete_size:",
              concrete_size,
              ", dtype:",
              dsi.value_to_ishape[value_name].getScalarType());
          dsi.value_to_ishape[value_name].UpdateTensor(concrete_size);
        } else {
          PT_DYNAMIC_SHAPE_DEBUG(
              "Cannot evaluate out shape for node ",
              value_name,
              " expression is NULL");
        }
      }
    }
  }
}

void HabanaLaunchOpPT::CreateORUpdateExprSymbolicTable(RecipeValueSpec* rv) {
  PT_BRIDGE_BEGIN;
  if (rv != nullptr && rv->ds_sifinfo_map.count(sym_expr_hash_) > 0) {
    auto& dsi = rv->ds_sifinfo_map[sym_expr_hash_];
    std::for_each(
        in_symbol_value_map_.begin(),
        in_symbol_value_map_.end(),
        [&](const std::pair<std::string, std::shared_ptr<double>>& p) {
          if (dsi.expr_symbolic_table[p.first])
            *(dsi.expr_symbolic_table[p.first]) = *(p.second);
        });
  } else {
    auto& dsi = ds_sif_info_;
    std::for_each(
        in_symbol_value_map_.begin(),
        in_symbol_value_map_.end(),
        [&](const std::pair<std::string, std::shared_ptr<double>>& p) {
          dsi.expr_symbolic_table[p.first] = std::move(p.second);
        });
  }
}

void HabanaLaunchOpPT::ProcessIntermediateSymbolicShapes(
    std::shared_ptr<torch::jit::Graph>& jit_graph) {
  auto& dsi = ds_sif_info_;
  CreateORUpdateExprSymbolicTable();

  // sym_lookup_Jit and jit_graph will be same as jit_ir_graph_ in case of
  // compilation.
  std::shared_ptr<torch::jit::Graph> sym_lookup_Jit = jit_ir_graph_;
  torch::jit::graph_node_list graph_nodes = jit_graph->nodes();
  torch::jit::graph_node_list sym_graph_nodes = sym_lookup_Jit->nodes();
  auto itr_node = sym_graph_nodes.begin();

  for (auto* node : graph_nodes) {
    auto node_qual_str = node->kind().toQualString();
    std::string opname(node_qual_str);
    PT_BRIDGE_DEBUG("Symbolic Working on ", node_qual_str);

    if ((node->kind() != torch::jit::prim::Constant) &&
        (node->kind() != torch::jit::prim::ListConstruct) &&
        (node->kind() != torch::jit::prim::ListUnpack)) {
      auto node_val = node->output(0);
      auto value_name = node_val->debugName();
      auto outputshapes_attr = c10::Symbol::attr("output_shapes");
      std::string outputshape_str;
      if (node->hasAttribute(outputshapes_attr)) {
        auto sym_node = *itr_node;
        TORCH_CHECK(
            node->kind() == sym_node->kind(),
            "Nodes not matching!!!",
            "\nsym_node:",
            sym_node->output(0)->debugName(),
            ",\ncurr_node:",
            node->output(0)->debugName());
        outputshape_str = sym_node->s(outputshapes_attr);
      } else {
        TORCH_CHECK(
            0,
            "Attr 'output_shapes' is missing for node:",
            node->output(0)->debugName());
      }

      // Porcess multiple output shapes
      std::string shape_strs =
          outputshape_str.substr(1, outputshape_str.size() - 2);
      std::vector<std::string> shape_str_list;
      std::stringstream ss(shape_strs);
      std::string shape_str;
      while (!ss.eof()) {
        std::getline(ss, shape_str, ';');
        shape_str_list.push_back(shape_str);
      }

      PT_BRIDGE_DEBUG("Node output shapes:", shape_str_list);
      std::vector<std::shared_ptr<habana::SizeExpression>> size_expr_vec;
      for (auto shape_str : shape_str_list) {
        if (shape_str == "[]") {
          size_expr_vec.push_back(nullptr);
        } else {
          SymExprFactory& expr_factory = SymExprFactory::getInstance();
          auto size_expr = std::make_shared<SizeExpression>(
              shape_str, dsi.expr_symbolic_table);
          size_expr_vec.push_back(size_expr);
          std::vector<int64_t> concrete_size =
              expr_factory.evaluate_symsize(size_expr);
        }
      }

      if (node->outputs().size() == 1) {
        auto value_name = node->output(0)->debugName();
        dsi.value_to_sizeexpr[value_name] = size_expr_vec;
        // Handle prim::ListUnpack nodes
        if (*node->output(0)->type() == *torch::ListType::ofTensors()) {
          auto unpack_node = GetUnpackNodeFromTensorList(node->output(0));
          HABANA_ASSERT(
              unpack_node != nullptr,
              "TensorList is not input to ListUnpack node. Node: ",
              node->kind().toQualString());

          auto node_outputs = unpack_node->outputs();
          size_t output_idx = 0;
          for (auto node_out_val : node_outputs) {
            auto unpack_value_name = node_out_val->debugName();
            dsi.value_to_sizeexpr[unpack_value_name] = {
                size_expr_vec[output_idx]};
            output_idx++;
          }
        }
      } else {
        size_t output_idx = 0;
        for (auto& output_val : node->outputs()) {
          auto value_name = output_val->debugName();
          if (output_val->hasUses()) {
            dsi.value_to_sizeexpr[value_name] = {size_expr_vec[output_idx]};
            output_idx++;
          }
        }
      }
    }
    ++itr_node;
  }
}

void HabanaLaunchOpPT::HandleOutputSIFException(
    torch::jit::Node* node,
    const HabanaOperatorPtr& habana_op,
    RecipeValueSpec& rv,
    size_t& outputs_meta_index,
    SynBuildCache& syn_build_cache) {
  PT_BRIDGE_BEGIN
  bool ds_sif_info_cached =
      (rv.ds_sifinfo_map.count(sym_expr_hash_) > 0) ? true : false;
  habana_helpers::DynamicSIFInfo* dsi = (ds_sif_info_cached)
      ? &(rv.ds_sifinfo_map[sym_expr_hash_])
      : &ds_sif_info_;

  CreateIValueForNodeInputs(node, dsi);

  // Handle exception with a fallback
  OutputMetaDataVector& outputs_metadata =
      syn_build_cache.get_or_compute_ref<&SynBuildCache::outputs_metadata>(
          [this, node]() { return nodeOutputMetaData(node); },
          outputs_meta_index);
  outputs_meta_index++;
  torch::jit::Stack input_stack = getStackForNode(node);
  GetSynapseInputs(habana_op, node);
  habana_op->AllocateAndAddSynapseNode(
      *syn_graph_ptr_, input_stack, outputs_metadata);
  if (!ds_sif_info_cached) {
    habana::InferOutputMetaRetType kernel_output_cs(true);
    ProcessSynapseOutputs(habana_op, node, kernel_output_cs);
    CreateValueIShapeMapForNode(node, node, input_stack, outputs_metadata);
  }
}

void HabanaLaunchOpPT::ResetIShapeUpdateStatus(RecipeValueSpec& rv) {
  auto& dsi = rv.ds_sifinfo_map[sym_expr_hash_];
  for (auto& pair : dsi.value_to_ishape) {
    PT_DYNAMIC_SHAPE_DEBUG("Reset Ishape update status for node :", pair.first);
    pair.second.ResetIshapeUpdate();
  }
}

void HabanaLaunchOpPT::ReCreateValueToIvalueMapForInputs(
    std::shared_ptr<torch::jit::Graph>& jit_graph) {
  PT_BRIDGE_BEGIN;
  value_to_ivalue_.clear();

  for (size_t j = 0; j < pt_stack_sh_.size(); j++) {
    auto value_input = jit_graph->inputs().at(j);
    auto ivpsh = pt_stack_sh_[j];
    value_to_ivalue_[value_input] = ivpsh;
  }
  PT_BRIDGE_END;
}

void HabanaLaunchOpPT::HandleOutputExprUnMappedJITGraph(
    RecipeValueSpec& rv,
    std::shared_ptr<synapse_helpers::graph>& syn_graph,
    SynBuildCache& syn_build_cache) {
  PT_BRIDGE_BEGIN
  PT_DYNAMIC_SHAPE_DEBUG("Running HandleOutputExprUnMappedJITGraph");
  BuildSynapseGraph(syn_graph, syn_build_cache, true);
  rv.ds_sifinfo_map[sym_expr_hash_] = std::move(ds_sif_info_);
}

void HabanaLaunchOpPT::HandleOutputExprMappedJITGraph(
    std::shared_ptr<torch::jit::Graph>& rv_jit_graph,
    RecipeValueSpec& rv,
    SynBuildCache& syn_build_cache) {
  PT_DYNAMIC_SHAPE_DEBUG("Running HandleOutputExprMappedJITGraph");
  auto& device = HPUDeviceContext::get_device();
  synDeviceId device_id = device.id();
  CreateORUpdateExprSymbolicTable(&rv);
  ResetIShapeUpdateStatus(rv);
  UpdateValueToIShapeMapForInputs(rv_jit_graph, rv);

  if (current_dbipsh_) {
    syn_build_cache.clear_cached_graph_info();
  }

  size_t outputs_metadata_index = 0;
  uint32_t node_idx = static_cast<uint32_t>(-1);
  torch::jit::graph_node_list graph_nodes = rv_jit_graph->nodes();
  for (auto* node : graph_nodes) {
    ++node_idx;
    // Skip for static backend STs and all frontend STs
    if (rv.dynamic_nodes_with_backend_STs.find(node_idx) ==
        rv.dynamic_nodes_with_backend_STs.end()) {
      continue;
    }

    auto node_qual_str = node->kind().toQualString();
    std::string opname(node_qual_str);

    PT_DYNAMIC_SHAPE_DEBUG("Working on ", node_qual_str);
    if ((node->kind() == torch::jit::prim::Constant) ||
        (node->kind() == torch::jit::prim::ListConstruct) ||
        (node->kind() == torch::jit::prim::ListUnpack)) {
      continue;
    }

    // Handle prim::ListUnpack nodes
    if (*node->output(0)->type() == *torch::ListType::ofTensors() &&
        node->outputs().size() == 1) {
      auto unpack_node = GetUnpackNodeFromTensorList(node->output(0));
      HABANA_ASSERT(
          unpack_node != nullptr,
          "TensorList is not input to ListUnpack node. Node: ",
          node->kind().toQualString());
      UpdateValueIShapeMapForListUnpack(unpack_node, rv);
    }

    const auto& op = node->schema().operator_name();
    HabanaOperatorPtr HabanaKernel =
        KernelRegistry().get(device_id, op, getNodeScalarType(node));

    TORCH_CHECK(HabanaKernel, op, " isn't registered in KernelRegistry!");

    UpdateIshapeForNodeInputs(node, rv);
    UpdateIshapeForNodeOuputs(node, rv);
    habana_helpers::IShapeList input_ishape = getInputIShapesForNode(node, rv);
    habana_helpers::IShapeList output_ishape =
        getOutputIShapesForNode(node, rv);
    bool t_meta_exists = false;
    PT_DYNAMIC_SHAPE_DEBUG("STMeta calling");
    if (auto op_auto = std::dynamic_pointer_cast<OpBackend>(HabanaKernel)) {
      t_meta_exists = op_auto->STMeta(input_ishape, output_ishape);
    } else {
      t_meta_exists = HabanaKernel->STMeta(input_ishape, output_ishape);
    }
    if (!t_meta_exists) {
      PT_DYNAMIC_SHAPE_DEBUG("ST Meta not found for op:", node_qual_str);
      HandleOutputSIFException(
          node, HabanaKernel, rv, outputs_metadata_index, syn_build_cache);
    }
  }
}

void HabanaLaunchOpPT::BuildSynapseGraphLite(
    std::shared_ptr<synapse_helpers::graph>& syn_graph,
    SynBuildCache& syn_build_cache) {
  PT_BRIDGE_BEGIN;
  PT_DYNAMIC_SHAPE_DEBUG("Running BuildSynapseGraphLite");
  auto recipe_holder = GetCachedRecipe(cur_rargpsh_);
  RecipeValueSpec& rv = *recipe_holder->rvs_;
  SymExprFactory::getInstance().clear_expr_cache();
  if (rv.ds_sifinfo_map.count(sym_expr_hash_) > 0) {
    synapse_helpers::detail::tensor_name_generator::reset();
    habana::ShapeInference::ResetShapeTensorId();
    ShapeInference::SetTensorMapping(rv.st_to_tensor_idx_map);
    syn_graph_ptr_ = syn_graph;
    auto rv_jit_graph = rv.jit_graph_;
    ReCreateValueToIvalueMapForInputs(rv_jit_graph);

    PT_DYNAMIC_SHAPE_DEBUG(
        "Recipe JIT graph:",
        rv_jit_graph->toString(),
        "Current JIT graph:",
        "\nkey:",
        graph_key_with_perm_);
    HandleOutputExprMappedJITGraph(rv_jit_graph, rv, syn_build_cache);
  } else {
    HandleOutputExprUnMappedJITGraph(rv, syn_graph, syn_build_cache);
  }
}

void HabanaLaunchOpPT::BuildSynapseGraph(
    std::shared_ptr<synapse_helpers::graph>& syn_graph,
    SynBuildCache& syn_build_cache,
    bool is_shape_inference) {
  PT_BRIDGE_BEGIN;

  // Clear cache if build graph failed
  CallFinally clear_cache_if_incomplete([&syn_build_cache] {
    if (!syn_build_cache.is_complete())
      syn_build_cache.clear_cached_graph_info();
  });

  auto& device = HPUDeviceContext::get_device();
  synDeviceId device_id = device.id();
  synapse_helpers::detail::tensor_name_generator::reset();
  syn_graph_ptr_ = syn_graph;

  // This is used to create mapping between shape Tensor to tensor idx
  habana::ShapeInference::ResetShapeTensorId();
  habana::ShapeInference::ResetTensorMapping();
  habana::ShapeInference::ResetBackendStTidxList();

  std::shared_ptr<torch::jit::Graph> rv_jit_graph = nullptr;
  torch::jit::graph_node_list::iterator itr_rv_node;
  if (syn_graph->is_dynamic_graph() && enable_optim_output_sif_) {
    SymExprFactory::getInstance().clear_expr_cache();
    if (map_shape_.m_pass == ShapeInfo::InferencePass::INVALID) {
      ProcessIntermediateSymbolicShapes(jit_ir_graph_);
      CreateValueToIShapeMapForInputs(jit_ir_graph_);
    } else if (map_shape_.m_pass == ShapeInfo::InferencePass::OUTPUT_SHAPE) {
      auto recipe_holder = GetCachedRecipe(cur_rargpsh_);
      RecipeValueSpec& rv = *recipe_holder->rvs_;
      ShapeInference::SetTensorMapping(rv.st_to_tensor_idx_map);
      rv_jit_graph = rv.jit_graph_;
      torch::jit::graph_node_list rv_graph_nodes = rv_jit_graph->nodes();
      itr_rv_node = rv_graph_nodes.begin();
      PT_DYNAMIC_SHAPE_DEBUG(
          "Recipe JIT graph:",
          rv_jit_graph->toString(),
          "Current JIT graph:",
          "\nkey:",
          graph_key_with_perm_);
      ProcessIntermediateSymbolicShapes(rv_jit_graph);
      CreateValueToIShapeMapForInputs(rv_jit_graph);
    }
  }

  if (current_dbipsh_) {
    syn_build_cache.clear_cached_graph_info();
    prim_nodes_ival_counter_ = 0;
    restride_node_swap_counter_ = 0;
    restride_node_out_val_counter_ = 0;
  }

  // for each node in IR graph, at this point the graph is a list with nodes
  // topoloically sorted
  // TODO : check if we need to reorder nodes in any case
  torch::jit::graph_node_list graph_nodes = jit_ir_graph_->nodes();

  // This is an optimization pass to mark all the nodes with sepcial layout
  // like weights which have HWCK Only activated in lazy mode for now
  if (!syn_build_cache.is_complete()) {
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
  uint32_t node_idx = static_cast<uint32_t>(-1);
  PT_OP_DEBUG("JIT Graph: ", jit_ir_graph_->toString());
  for (auto* node : graph_nodes) {
    ++node_idx;
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
      handlePrimNodes(node, syn_build_cache);
      continue;
    }

    if ((strcmp(node_qual_str, "hpu::restride_cl") == 0) ||
        (strcmp(node_qual_str, "hpu::restride") == 0)) {
      bool is_restride_cl =
          (strcmp(node_qual_str, "hpu::restride_cl") == 0) ? true : false;
      handleRestrideNode(node, syn_build_cache, is_restride_cl);
      continue;
    }

    // Get kernel context
    const auto& op = node->schema().operator_name();
    HabanaOperatorPtr HabanaKernel =
        KernelRegistry().get(device_id, op, getNodeScalarType(node));

    TORCH_CHECK(HabanaKernel, op, " isn't registered in KernelRegistry!");

    if (enable_optim_output_sif_) {
      bool is_node_dynamic = habana_helpers::isNodeDynamic(
          node, org_stack_index_map, value_to_ivalue_);
      HabanaKernel->SetOpDynamicity(is_node_dynamic);
      PT_DYNAMIC_SHAPE_DEBUG(
          "For op = ", opname, ", is current node dynamic = ", is_node_dynamic);
    }

    // Set the deterministic val
    HabanaKernel->setDeterministic(node->i(torch::jit::attr::deterministic));

    // Set kernel execution mode
    HabanaKernel->SetExecutionMode(execution_mode_);

    // Set node hints
    // firstly extract hints from node
    if (node->hasAttribute(c10::Symbol::attr("hints")))
      HabanaKernel->setContextHints(node->s(c10::Symbol::attr("hints")));

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
    // This log line is used by the logging analysis tool. Please be cautious
    // when changing.
    PT_OP_INFO(
        "JIT_OP ",
        OpInfo::DumpPassInfo(*syn_graph, map_shape_.m_pass),
        OpInfo::DumpOpInfo(op, input_stack));

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

    PT_BRIDGE_DEBUG(DumpNodeInputs(node, value_to_ivalue_));
    OutputMetaDataVector& outputs_metadata =
        syn_build_cache.get_or_compute_ref<&SynBuildCache::outputs_metadata>(
            [this, node]() { return nodeOutputMetaData(node); },
            outputs_metadata_index);
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
          value_to_ivalue_,
          pt_to_synapse_tensors_);
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
      updatemax_graph_ |=
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

      uint64_t curr_st_id = habana::ShapeInference::GetShapeTensorId();
      HabanaKernel->AllocateAndAddSynapseNode(
          *syn_graph, input_stack, outputs_metadata);
      // ST_Id is incremented only when the node is dynamic and it creates ST(s)
      uint64_t changed_st_id = habana::ShapeInference::GetShapeTensorId();

      if (enable_optim_output_sif_ &&
          (map_shape_.m_pass == ShapeInfo::InferencePass::INVALID) &&
          curr_st_id != changed_st_id) {
        dynamic_nodes_with_backend_STs.insert(node_idx);
        PT_DYNAMIC_SHAPE_DEBUG(
            "Dynamic Node with op", opname, "is creating ST(s) at the backend");
      }

      HabanaKernel->dump(node, input_stack);
      if (updatemax_node) {
        UpdateMaxValues(HabanaKernel, input_stack, index2maxvalues);
      }
    }

    static std::unordered_set<std::string> cs_jit_ir_ops_;
    static std::unordered_set<std::string> empty_cs_jit_ir_ops_;

    habana::InferOutputMetaRetType kernel_output_cs(true);
    if (!HabanaLaunchOpUtils::disabled_jit_ir_ops().count(node_qual_str)) {
      // Either the InferOutputMeta flow is getting validated or
      // fast shape inference is running for dynamic shapes or
      // shape agnostic flow is enabled for eager.
      if (GET_ENV_FLAG_NEW(PT_HPU_VALIDATE_COMPUTE_SHAPE) ||
          (enable_fast_shape_inf_ && syn_graph->is_dynamic_graph()) ||
          enable_shape_agnostic_caching_) {
        auto cur_sif_tid = habana::ShapeInference::GetSifTensorId();
        PT_DYNAMIC_SHAPE_DEBUG("Current sif tensor id = ", cur_sif_tid);
        HabanaOperatorPtr csHabanaKernel =
            KernelRegistry().get(device_id, op, getNodeScalarType(node));

        HabanaKernel->setDeterministic(
            node->i(torch::jit::attr::deterministic));

        // Set kernel execution mode
        csHabanaKernel->SetExecutionMode(execution_mode_);

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
            // Restore the sif tensor id
            habana::ShapeInference::SetSifTensorId(cur_sif_tid);
            if (HabanaLaunchOpUtils::disabled_jit_ir_ops().count(
                    node_qual_str) == 0) {
              PT_DYNAMIC_SHAPE_DEBUG(
                  "DISABLED_InferOutputMeta_JIT_IR_OP: ", node_qual_str);
              HabanaLaunchOpUtils::disabled_jit_ir_ops().insert(node_qual_str);
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

    jit_to_synapse_node_idx_map_.emplace(
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
            auto ti{shape_tensor_tinfos_[idx]};
            auto ret = sif_tidx_to_tinfo_map_.insert({tensor_idx, ti});
            if (ret.second) {
              PT_DYNAMIC_SHAPE_DEBUG(
                  "Intermediate shape tensor: adding to sif_tidx_to_tinfo_map_ : ",
                  tensor_idx,
                  " -> ",
                  *ti);
            } else {
              PT_DYNAMIC_SHAPE_DEBUG(
                  "Intermediate shape tensor: failed adding to sif_tidx_to_tinfo_map_ : ",
                  tensor_idx,
                  " -> ",
                  *ti);
            }
          }
        } else {
          for (auto& idx : intermediate_shape_tensors) {
            auto ti{shape_tensor_tinfos_[idx]};
            PT_DYNAMIC_SHAPE_DEBUG(
                "Intermediate shape tensor: delayed adding to sif_tidx_to_tinfo_map_ : ",
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

    if (syn_graph->is_dynamic_graph() && enable_optim_output_sif_ &&
        (map_shape_.m_pass == ShapeInfo::InferencePass::INVALID ||
         map_shape_.m_pass == ShapeInfo::InferencePass::OUTPUT_SHAPE)) {
      if (map_shape_.m_pass == ShapeInfo::InferencePass::INVALID) {
        CreateValueIShapeMapForNode(node, node, input_stack, outputs_metadata);
      } else {
        while ((*itr_rv_node)->kind() != node->kind()) {
          ++itr_rv_node;
        }
        torch::jit::Node* rv_node = *itr_rv_node;
        CreateValueIShapeMapForNode(
            node, rv_node, input_stack, outputs_metadata);
        ++itr_rv_node;
      }
    }

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

    PT_BRIDGE_DEBUG(DumpNodeOutputs(node, value_to_ivalue_));

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
        auto it = buff_to_input_ivpsh_map_.find(buffp);
        if (it != buff_to_input_ivpsh_map_.end()) {
          std::string irn{"%appended_indup_"};
          irn += std::to_string(appended_index_);
          appended_index_++;

          PtTensorInfoShared ti = std::make_shared<PtTensorInfo>(
              tensor, tensor_name, irn, tensor_id);
          auto& ivpsh = it->second;
          ivalue_to_tensor_info_map_[ivpsh] = ti;

          if (enable_caching_ || enable_shape_agnostic_caching_) {
            auto mit = input_tiv_map_.find(ivpsh);
            TORCH_CHECK(input_tiv_map_.end() != mit, "tinfo missing for input");

            TORCH_CHECK(ivpsh->isTensor(), "non tensor parent found");
          }

          duplicate_input_tivs_.emplace_back(ti);
        } else {
          std::string irn{"%intermediate_"};
          irn += std::to_string(intermediate_index_);
          intermediate_index_++;

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
      habana_helpers::CollectiveKernelInfos::Info kernel_info;

      auto node_inputs = node->inputs();
      for (auto input : node_inputs) {
        auto ivalptr = value_to_ivalue_.at(input);
        if (ivalptr->isTensor()) {
          HABANA_ASSERT(ivalue_to_tensor_info_map_.count(ivalptr));
          PtTensorInfoShared ti = ivalue_to_tensor_info_map_.at(ivalptr);
          kernel_info.input_tensor_infos.push_back(ti);
        } else {
          kernel_info.input_tensor_infos.push_back(nullptr);
        }
      }

      auto node_outputs = node->outputs();
      for (auto output : node_outputs) {
        auto ivalptr = value_to_ivalue_.at(output);
        if (ivalptr->isTensor()) {
          HABANA_ASSERT(ivalue_to_tensor_info_map_.count(ivalptr));
          PtTensorInfoShared ti = ivalue_to_tensor_info_map_.at(ivalptr);
          kernel_info.output_tensor_infos.push_back(ti);
        } else {
          kernel_info.output_tensor_infos.push_back(nullptr);
        }
      }
      kernel_info.kernel =
          std::dynamic_pointer_cast<CollectiveOperator>(HabanaKernel);
      collective_kernels_info_.AddKernel(std::move(kernel_info));
    }

    // Adding to a vector as we share context through shared pointers and we
    // dont want to call delete untill we are done with whole graph
    habana_kernels_.push_back(HabanaKernel);
  }


  // Generate patching info for graph inputs during fast sif
  if ((refine_ds_enabled_ && enable_fast_shape_inf_ &&
       syn_graph->is_dynamic_graph() && !is_shape_inference)) {
    for (size_t i = 0; i < jit_ir_graph_->inputs().size(); ++i) {
      auto input = jit_ir_graph_->inputs().at(i);
      HABANA_ASSERT(value_to_ivalue_.count(input));
      auto input_ivalue = value_to_ivalue_[input];
      HABANA_ASSERT(ivalue_to_tensor_info_map_.count(input_ivalue));
      auto tensor_idx = habana::ShapeInference::ReadAndIncrementSifTensorId();
      auto ret = sif_tidx_to_tinfo_map_.insert(
          {tensor_idx, ivalue_to_tensor_info_map_[input_ivalue]});
      if (ret.second) {
        PT_DYNAMIC_SHAPE_DEBUG(
            "Input tensor: adding to sif_tidx_to_tinfo_map_ : ",
            tensor_idx,
            " -> ",
            *ivalue_to_tensor_info_map_[input_ivalue]);
      } else {
        PT_DYNAMIC_SHAPE_DEBUG(
            "Input tensor: failed adding to sif_tidx_to_tinfo_map_ : ",
            tensor_idx,
            " -> ",
            *ivalue_to_tensor_info_map_[input_ivalue]);
      }
    }

    // Generate patching info for inputs shape tensors during fast sif
    for (auto const& idx : inputs_shape_tensors_vec) {
      auto tensor_idx = habana::ShapeInference::ReadAndIncrementSifTensorId();
      auto ret = sif_tidx_to_tinfo_map_.insert(
          {tensor_idx, shape_tensor_tinfos_[idx]});
      if (ret.second) {
        PT_DYNAMIC_SHAPE_DEBUG(
            "Input shape tensor: adding to sif_tidx_to_tinfo_map_ : ",
            tensor_idx,
            " -> ",
            *shape_tensor_tinfos_[idx]);
      } else {
        PT_DYNAMIC_SHAPE_DEBUG(
            "Input shape tensor: failed adding to sif_tidx_to_tinfo_map_ : ",
            tensor_idx,
            " -> ",
            *shape_tensor_tinfos_[idx]);
      }
    }

    // Generate patching info for intermediate shape tensors for nodes not
    // supporting InferOutputMeta during fast sif
    for (auto const& idx : intermediate_shape_tensors_vec) {
      auto tensor_idx = habana::ShapeInference::ReadAndIncrementSifTensorId();
      auto ret = sif_tidx_to_tinfo_map_.insert(
          {tensor_idx, shape_tensor_tinfos_[idx]});
      if (ret.second) {
        PT_DYNAMIC_SHAPE_DEBUG(
            "Intermediate shape tensor: adding to sif_tidx_to_tinfo_map_ : ",
            tensor_idx,
            " -> ",
            *shape_tensor_tinfos_[idx]);
      } else {
        PT_DYNAMIC_SHAPE_DEBUG(
            "Intermediate shape tensor: failed adding to sif_tidx_to_tinfo_map_ : ",
            tensor_idx,
            " -> ",
            *shape_tensor_tinfos_[idx]);
      }
    }
  }

  // allow permutation only for output tensors
  if (GET_ENV_FLAG_NEW(PT_HPU_ENABLE_SYNAPSE_OUTPUT_PERMUTE) &&
      !is_hccl_send_mark_step() &&
      !(jit_graph_and_meta_data_->is_skip_tensor_permutation())) {
    for (auto ti : output_tensorinfo_map_) {
      auto ival = ti.first;
      auto iter = pt_to_synapse_tensors_.find(ival);
      HABANA_ASSERT(pt_to_synapse_tensors_.count(ival));
      if (iter != pt_to_synapse_tensors_.end()) {
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
    for (auto ti : duplicate_input_to_outtinfo_map_) {
      auto ival = ti.first;
      auto iter = pt_to_synapse_tensors_.find(ival);
      HABANA_ASSERT(pt_to_synapse_tensors_.count(ival));
      if (iter != pt_to_synapse_tensors_.end()) {
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

  if (refine_ds_enabled_ || !syn_build_cache.is_complete() ||
      syn_build_cache.get_is_control_edge_processing_required()) {
    // Process control edges
    bool control_edge_processing_required = control_edges::ProcessControlEdges(
        *jit_ir_graph_,
        jit_to_synapse_node_idx_map_,
        memory_reuse_pairs,
        syn_graph_ptr_.get());
    if (control_edge_processing_required)
      syn_build_cache.set_is_control_edge_processing_required();
  }
  syn_build_cache.complete();
  PT_BRIDGE_END;
}

void HabanaLaunchOpPT::CreateDynamicBucketInputShapes(
    habana_helpers::InpTensorShapes& shape_map) {
  for (size_t i = 0; i < input_refs_.size(); i++) {
    auto& input = input_refs_[i];
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
  for (size_t i = 0; i < input_refs_.size(); i++) {
    auto input = input_refs_[i];
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

  if (!HabanaLaunchOpUtils::ref_input_shape_map().count(graph_key_with_perm)) {
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
    HabanaLaunchOpUtils::ref_input_shape_map().emplace(
        graph_key_with_perm, input_tshapes);
    if (path != "") {
      CreateFirstDynamicBucket();
      DumpStaticCompilationStatistics(graph_key_with_perm, true);
    }
  } else if (path != "") {
    DumpStaticCompilationStatistics(graph_key_with_perm);
  }
}

void HabanaLaunchOpPT::CreateDynamicDBI(size_t graph_key_with_perm) {
  if (!HabanaLaunchOpUtils::ref_input_shape_map().count(graph_key_with_perm)) {
    habana_helpers::InpTensorShapes input_tshapes;
    CreateDynamicBucketInputShapes(input_tshapes);
    ProcessDynamicBucketInputShapesWithH2D(input_tshapes);
    PT_BRIDGE_DEBUG(
        "Recording the reference input shapes::",
        input_tshapes,
        "\n--------------------");
    HabanaLaunchOpUtils::ref_input_shape_map().emplace(
        graph_key_with_perm, input_tshapes);
  }

  std::shared_ptr<RecipeArgumentSpec> rargpsh_graph =
      std::make_shared<RecipeArgumentSpec>(input_refs_, graph_key_, op_strs_);

  current_dbipsh_ = DynamicBucketInfoMap::get_instance().get(rargpsh_graph);
  if (nullptr == current_dbipsh_) {
    PT_DYNAMIC_SHAPE_DEBUG("Initilizing DynamicBucketInfo for mark_dynamic");
    current_dbipsh_ = std::make_shared<habana_helpers::DynamicBucketInfo>(
        rargpsh_graph->graphHashCode());
    DynamicBucketInfoMap::get_instance().add(rargpsh_graph, current_dbipsh_);
    current_dbipsh_->create_statistics(
        habana_helpers::CompilationStatistics::Create(
            GetSynapseGraphName(),
            current_dbipsh_->getCount(),
            rargpsh_graph->hashCode()));
  }
}

void HabanaLaunchOpPT::CreateValueToIvalueMapForInputs() {
  PT_BRIDGE_BEGIN;
  for (size_t j = 0; j < pt_stack_sh_.size(); j++) {
    auto value_input = jit_ir_graph_->inputs().at(j);
    auto ivpsh = pt_stack_sh_[j];
    value_to_ivalue_[value_input] = ivpsh;
    ival_hash_to_input_index_map_[ivpsh->hash().toInt()] = j;
  }
  PT_BRIDGE_END;
}

void HabanaLaunchOpPT::CreateValueToIShapeMapForInputs(
    std::shared_ptr<torch::jit::Graph>& jit_graph) {
  PT_BRIDGE_BEGIN;
  for (size_t j = 0; j < pt_stack_sh_.size(); j++) {
    auto& value_input = jit_graph->inputs().at(j);
    auto& ivalue_input = pt_stack_sh_[j];
    const auto& value_name = value_input->debugName();
    auto& dsi = ds_sif_info_;
    if (ivalue_input->isTensor()) {
      auto tensor = ivalue_input->toTensor();
      habana_helpers::IShape ishape(tensor.sizes().vec(), tensor.scalar_type());
      dsi.value_to_ishape.insert({value_name, ishape});
      PT_DYNAMIC_SHAPE_DEBUG(
          "Create Ishape Tensor ",
          value_name,
          " Scalar type ",
          tensor.scalar_type());
    } else if (ivalue_input->isScalar()) {
      auto scalar = ivalue_input->toScalar();
      habana_helpers::IShape ishape(scalar, scalar.type());
      dsi.value_to_ishape.insert({value_name, ishape});
      PT_DYNAMIC_SHAPE_DEBUG(
          "Create Ishape Scalar ", value_name, " Scalar type ", scalar.type());
    } else {
      habana_helpers::IShape ishape;
      dsi.value_to_ishape.insert({value_name, ishape});
    }
  }
  PT_BRIDGE_END;
}

void HabanaLaunchOpPT::UpdateValueToIShapeMapForInputs(
    std::shared_ptr<torch::jit::Graph>& jit_graph,
    RecipeValueSpec& rv) {
  PT_BRIDGE_BEGIN;
  auto& dsi = rv.ds_sifinfo_map[sym_expr_hash_];
  for (size_t j = 0; j < pt_stack_sh_.size(); j++) {
    auto& value_input = jit_graph->inputs().at(j);
    auto& ivalue_input = pt_stack_sh_[j];
    const auto& value_name = value_input->debugName();
    if (ivalue_input->isTensor()) {
      auto tensor = ivalue_input->toTensor();
      dsi.value_to_ishape[value_name].UpdateTensor(tensor.sizes().vec());
    } else if (ivalue_input->isScalar()) {
      auto scalar = ivalue_input->toScalar();
      dsi.value_to_ishape[value_name].UpdateScalar(scalar);
    }
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
          auto original_dtype =
              c10::typeMetaToScalarType(tensor.options().dtype());
          new_tensor = createDynamicTensor(
              dynamic_shapes.at(i).get_dims(), tensor_type, original_dtype);
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

void HabanaLaunchOpPT::InitiateSynlaunchTimeCapture(RecipeLauncher& rv) {
  PT_BRIDGE_BEGIN;
  // Initiate recipe execution time collection
  if (current_dbipsh_->NeedRunTimeSlot(current_bucket_id_)) {
    rv.time_slot_ = HPUDeviceContext::create_time_slot(hpu_stream_);
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
        auto dropped_arg =
            HPUDeviceContext::recipe_cache().dropped_recipe.first;
        auto dropped_val =
            HPUDeviceContext::recipe_cache().dropped_recipe.second;
        // Update the eviction threshold left after removing this recipe
        eviction_threshold_left -=
            dropped_val->rl_->recipe_->get_recipe_host_mem_size();
        auto dropped_dbi =
            DynamicBucketInfoMap::get_instance().get(dropped_arg);
        if (dropped_dbi != nullptr) {
          static_cast<void>(dsi_bucket_id);
          dropped_dbi->ResetSynapseRecipePtr(dropped_val->rvs_);
        }
      }
    }
    // Call TcMalloc extension to release memory
    synapse_helpers::ReleaseFreeMemory();
  }
}

void HabanaLaunchOpPT::CreateFirstDynamicBucket(
    std::shared_ptr<RecipeArgumentSpec> rargpsh_graph) {
  PT_BRIDGE_BEGIN
  RecipeCacheLRU::SetHostMemoryThreshold();

  if (nullptr == rargpsh_graph) {
    rargpsh_graph = std::make_shared<RecipeArgumentSpec>(
        input_refs_,
        graph_key_,
        graph_symint_hash_,
        graph_perm_hash_,
        op_strs_);
  }

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
            GetSynapseGraphName(),
            current_dbipsh_->getCount(),
            rargpsh_graph->hashCode()));
    // Create bucket 0
    DynamicShapeInfo graph_input_info;
    graph_input_info.act_input_tshapes =
        HabanaLaunchOpUtils::ref_input_shape_map().at(
            rargpsh_graph->hashCode());
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

void HabanaLaunchOpPT::ProcessHabanaFusedOpWithDS(
    HabanaLaunchOpPipeline::PipelineCallBase& pipeline_execution) {
  PT_BRIDGE_BEGIN;

  std::shared_ptr<RecipeArgumentSpec> rargpsh_graph =
      std::make_shared<RecipeArgumentSpec>(
          input_refs_,
          graph_key_,
          graph_symint_hash_,
          graph_perm_hash_,
          op_strs_);

  PT_DYNAMIC_SHAPE_DEBUG(
      "====================\n",
      "Processing with dynamic shape enabled\n",
      "JIT IR graph_hash_code : ",
      rargpsh_graph->graphHashCode(),
      ", hash_code with data layout : ",
      rargpsh_graph->hashCode());

  if (enable_user_dynamic_ranges_)
    CreateDynamicDBI(graph_key_with_perm_);
  CreateFirstDynamicBucket(rargpsh_graph);

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
    // Only for 1 launch the bucket is created, rest follows normal flow
    if (enable_user_dynamic_ranges_) {
      current_bucket_id_ = current_dbipsh_->GetUserBucketId(
          graph_input_info.act_input_tshapes, range_infos_);
      enable_user_dynamic_ranges_ = false;
    } else {
      current_bucket_id_ =
          current_dbipsh_->GetBucketId(graph_input_info.act_input_tshapes);
    }
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

  cur_rargpsh_ = std::make_shared<RecipeArgumentSpec>(
      input_refs_,
      graph_key_,
      graph_symint_hash_,
      graph_perm_hash_,
      op_strs_,
      cur_ds_token_);
  PT_DYNAMIC_SHAPE_DEBUG("cur_rargpsh_ = ", *cur_rargpsh_);
  DynamicBucketInfoMap::get_instance().add(cur_rargpsh_, current_dbipsh_);
  // Used only for compilation statistics purpose now
  current_dbipsh_->SetLastUsedStepForBucket(
      current_bucket_id_, current_dbipsh_->get_statistics()->GetCurrentStep());

  // Check for cached recipe
  if (enable_graph_caching_) {
    current_dbipsh_->SetRecipeKeyForBucket(
        graph_input_info.current_bucket_id, cur_rargpsh_->hashCode());
    auto recipe_holder = GetCachedRecipe(cur_rargpsh_);

    if (ABSL_PREDICT_TRUE(recipe_holder)) {
      recipe_launcher_ = recipe_holder->rl_;
      // Cache hit for a dynamic bucket
      // Steps:
      // 1. Infer shapes of all persistent tensors which are not input
      // 2. Patch using the exact shape
      // 3. Launch
      // 4. Update outputs
      current_dbipsh_->IncrementHitCount(current_bucket_id_);
      RecipeValueSpec& rv = *recipe_holder->rvs_;
      if (rv.get_refined()) {
        habana_helpers::DynamicBucketInfo::inc_num_refined_recipe_hits();
        if (rv.get_refined_wirt()) {
          habana_helpers::DynamicBucketInfo::inc_num_refined_recipe_wirt_hits();
        }
      } else {
        habana_helpers::DynamicBucketInfo::inc_num_original_recipe_hits();
      }

      std::unordered_map<int64_t, at::Tensor> tidx_to_tensor_map;
      rv.update_hit_count();

      if (rv.curr_symval_hash_ == curr_symval_hash_) {
        is_symval_changed_from_prev_ = false;
      } else {
        rv.curr_symval_hash_ = curr_symval_hash_;
      }

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
            cur_rargpsh_->hashCode());

        // Updtating enable_optim_output_sif with RecipeValueSpec's saved value
        // so that it's consistent with compilation time's shape inference flow.
        bool cached_enable_optim_output_sif = rv.get_optim_output_sif_value();
        enable_optim_output_sif_ = cached_enable_optim_output_sif;
        PT_DYNAMIC_SHAPE_DEBUG(
            "Cached recipe's dynamic shape enable symbolic output sif value: ",
            cached_enable_optim_output_sif)

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
              cur_rargpsh_->hashCode(),
              "HybridSif_BEGIN");

          habana::ShapeInference::ResetSifTensorId();
          constexpr bool dynamic_shapes_true = true;
          if (rv.disabled_jit_ir_ops_.size()) {
            HabanaLaunchOpUtils::disabled_jit_ir_ops() =
                rv.disabled_jit_ir_ops_;
          } else {
            HabanaLaunchOpUtils::disabled_jit_ir_ops().insert(
                rv.disabled_jit_ir_ops_.begin(), rv.disabled_jit_ir_ops_.end());
          }
          RunHybridSif<dynamic_shapes_true>(tidx_to_tensor_map);
          PT_DYNAMIC_SHAPE_DEBUG("HybridSif_END");
        } else {
          PT_DYNAMIC_SHAPE_DEBUG("OutputSif_BEGIN");
          try_run_shape_inference(
              ShapeInfo::InferencePass::OUTPUT_SHAPE, graph_input_info);
          PT_DYNAMIC_SHAPE_DEBUG("OutputSif_END");
        }
        current_dbipsh_->SetInputMetaData(*pt_stack_, current_bucket_id_);
        bool refine_candidate =
            (current_dbipsh_->GetMFUBucket() == current_bucket_id_);
        current_dbipsh_->get_statistics()->LogUsedBucket(
            current_bucket_id_, jit_ir_graph_, ranges, refine_candidate);
      }

      UpdatePatchingInformation(
          rv, !recipe_launcher_->recipe_, true, tidx_to_tensor_map);

      {
        std::lock_guard<std::mutex> lg(current_dbipsh_->get_refine_mutex());
        // Initiate recipe execution time collection
        if (GET_ENV_FLAG_NEW(PT_ENABLE_SYNLAUNCH_TIME_CAPTURE)) {
          InitiateSynlaunchTimeCapture(*recipe_launcher_);
        }
      }

      if (!enable_4stage_pipeline_) {
        if (!dry_run_) {
          recipe_launcher_->Launch(
              hpu_stream_,
              input_refs_,
              intermediate_tensors_ptr_sh_,
              aten_outputs_,
              syn_launch_info_,
              external_tensor_info_indexes_,
              dma_inputs_);
        }

        // Update the stack from the recipe itself
        UpdateRecipeOutputs();

        if (execution_mode_ != habana_helpers::HabanaFrontendTypes::COMPILE) {
          RefinementEngine::GetEngine().AddGraphKey(
              rargpsh_graph->graphHashCode());
        }

        PT_DYNAMIC_SHAPE_DEBUG(
            current_dbipsh_->digest_str(), current_dbipsh_->history_str());
        PT_IRGRAPH_DEBUG("HabanaOp recipe cache hit :: dynamic shapes");

        current_dbipsh_->get_statistics()->LogSelectedRecipe(
            cur_rargpsh_->hashCode(), 0);
        current_dbipsh_->get_statistics()->LogShapes(
            jit_ir_graph_, graph_input_info.act_input_tshapes);

        auto t_ns_base{current_dbipsh_->GetTimeBase(current_bucket_id_)};
        auto t_ns{current_dbipsh_->GetTime(current_bucket_id_)};

        current_dbipsh_->get_statistics()->LogLaunchBase(t_ns_base, 0);
        current_dbipsh_->get_statistics()->LogLaunch(t_ns, 0);
        current_dbipsh_->get_statistics()->LogLaunchPerf(t_ns_base, t_ns, 0);
        if (t_ns && t_ns_base) {
          habana_helpers::DynamicBucketInfo::update_improvement_map(
              cur_rargpsh_->hashCode(), (t_ns < t_ns_base));
        }
        current_dbipsh_->get_statistics()->GetDigest(
            cur_rargpsh_->graphHashCode(),
            current_bucket_id_,
            cur_ds_token_,
            cur_rargpsh_->hashCode(),
            true);

        current_dbipsh_->get_statistics()->DumpAndNextStep();
        ClearMembers();
        ClearStatics();
      } else {
        PT_DYNAMIC_SHAPE_DEBUG("Cache hit pipeline flow");
        current_dbipsh_->get_statistics()->LogSelectedRecipe(
            cur_rargpsh_->hashCode(), 0);
        current_dbipsh_->get_statistics()->LogShapes(
            jit_ir_graph_, graph_input_info.act_input_tshapes);
        current_dbipsh_->get_statistics()->GetDigest(
            cur_rargpsh_->graphHashCode(),
            current_bucket_id_,
            cur_ds_token_,
            cur_rargpsh_->hashCode(),
            true);
        current_dbipsh_->get_statistics()->DumpAndNextStep();
        execution_control_.cached_task(graph_key_with_perm_);
        pipeline_execution();
      }
      PT_BRIDGE_END;
      return;
    } else {
      PT_DYNAMIC_SHAPE_DEBUG(
          "HabanaOp recipe cache miss :: key ", cur_rargpsh_->hashCode());
      PT_DYNAMIC_SHAPE_DEBUG("HabanaOp recipe cache miss :: dynamic shapes");
    }
  }

  CompileAndRunDynamicGraph(graph_input_info, pipeline_execution);
  ival_hash_to_input_index_map_.clear();

  habana_helpers::DynamicBucketInfo::inc_original_recipe_count();
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
  PtTensorInfo& ti = *(dtensorinfos.at(ti_idx));
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

// shape agnostic : duplicate synapse graph
void HabanaLaunchOpPT::ConstructDuplicateShapeMap(
    const std::vector<synTensorHandleMap>& tensorsMap,
    std::vector<std::pair<synTensor, std::vector<int64_t>>>&
        duplicate_tensors_shape_map) {
  // Get all persistent tensors info i.e. graph inputs/outputs
  std::vector<PtTensorInfoShared> tensors_info;
  for (size_t i = 0; i < jit_ir_graph_->inputs().size(); ++i) {
    auto input = jit_ir_graph_->inputs().at(i);
    HABANA_ASSERT(value_to_ivalue_.count(input));
    auto input_ivalue = value_to_ivalue_[input];
    if (!input_ivalue->isTensor())
      continue;
    HABANA_ASSERT(ivalue_to_tensor_info_map_.count(input_ivalue));
    tensors_info.emplace_back(ivalue_to_tensor_info_map_[input_ivalue]);
  }

  for (size_t i = 0; i < jit_ir_graph_->outputs().size(); ++i) {
    auto output = jit_ir_graph_->outputs().at(i);
    HABANA_ASSERT(value_to_ivalue_.count(output));
    auto output_ivalue = value_to_ivalue_[output];
    HABANA_ASSERT(ivalue_to_tensor_info_map_.count(output_ivalue));
    tensors_info.emplace_back(ivalue_to_tensor_info_map_[output_ivalue]);
  }

  // Get all non-persistent tensors info supported by hybrid sif
  for (auto& tinfo_map : sif_tidx_to_tinfo_map_) {
    tensors_info.emplace_back(tinfo_map.second);
  }

  // Get map for orig handle -> <duplicate handle and boolean set bit>
  std::unordered_map<synTensor, std::pair<synTensor, bool>>
      synapse_orig_to_new_handle_info{};
  for (const auto& tensorMap : tensorsMap) {
    synapse_orig_to_new_handle_info.insert(
        {tensorMap.origHandle, std::make_pair(tensorMap.newHandle, false)});
  }

  // Set tensor geometry on duplicate tensors for known shapes
  for (const auto& tinfo : tensors_info) {
    const auto& orig_handle = tinfo->get_orig_syn_handle();
    const auto& tensor_map = synapse_orig_to_new_handle_info.find(orig_handle);
    if (tensor_map != synapse_orig_to_new_handle_info.end()) {
      const auto& new_handle = tensor_map->second.first;
      auto shape = tinfo->get_shape();
      if (shape.size() == 0 && !tinfo->is_ZST()) {
        shape = {1};
      }
      syn_graph_ptr_->setTensorGeometry(new_handle, shape);
      PT_EAGER_DEBUG(
          "[SHAPE AGNOSTIC] new handle : ",
          new_handle,
          " set geometry : ",
          shape);
      // Mark set bit and add to duplicate tensors shapes map
      tensor_map->second.second = true;
      duplicate_tensors_shape_map.emplace_back(
          std::make_pair(new_handle, std::move(shape)));
    } else {
      PT_EAGER_DEBUG(
          "[SHAPE AGNOSTIC] orig handle : ",
          orig_handle,
          " not present in the synapse_orig_to_new_handle map");
    }
  }

  // Collect org graph non-persistent shapes not aware to bridge SIF
  // to reset to the correct shapes if synapse Infer shapes failed
  for (const auto& tensor_map : synapse_orig_to_new_handle_info) {
    // check if set bit not set
    const auto& orig_handle = tensor_map.first;
    const auto& new_handle_info = tensor_map.second;
    if (new_handle_info.second == false) {
      // Get org tensor shapes and add it to duplicate tensors shape maps
      std::vector<int64_t> shape;
      syn_graph_ptr_->getTensorGeometry(orig_handle, shape);
      const auto& new_handle = new_handle_info.first;
      PT_EAGER_DEBUG(
          "[SHAPE AGNOSTIC] org handle : ",
          orig_handle,
          " new handle : ",
          new_handle,
          " get geomtery org shapes : ",
          shape);
      // Add to duplicate tensors shapes map
      duplicate_tensors_shape_map.emplace_back(
          std::make_pair(new_handle, std::move(shape)));
    }
  }
}

static void ResetDuplicateTensorShapes(
    std::shared_ptr<synapse_helpers::graph> syn_graph_ptr,
    std::vector<std::pair<synTensor, std::vector<int64_t>>>&
        duplicate_tensors_shape_map) {
  for (const auto& map : duplicate_tensors_shape_map) {
    const auto& new_handle = map.first;
    const auto& shape = map.second;
    syn_graph_ptr->setTensorGeometry(new_handle, shape);
    PT_EAGER_DEBUG(
        "[SHAPE AGNOSTIC] new handle : ",
        new_handle,
        " reset geomtery : ",
        shape);
  }
}

// shape agnostic : validate Inputs and Outputs and disable shape agnostic if
// not supported
void HabanaLaunchOpPT::ValidateInputsAndOutputsAndDisableSA(
    at::ArrayRef<torch::jit::IValue>& input_refs) {
  // Validate if output shapes are filled correctly otherwise we can not
  // support shape agnostic graph caching.
  for (auto shape : out_shapes_) {
    // Check for ZST tensor, It is supported for SAG, To Do proper fix
    if (shape.size() == 1 && shape[0] == 0)
      continue;
    for (auto size : shape) {
      if (size == 0) {
        jit_graph_and_meta_data_->set_is_shape_agnostic_supported(false);
        PT_EAGER_DEBUG(
            "[SHAPE AGNOSTIC] Shape agnostic not supported for op ",
            name_,
            ", output shapes not ok!");
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
            "[SHAPE AGNOSTIC] Shape agnostic not supported for op ",
            name_,
            ", input is a shape tensor ! ",
            " tmeta : ",
            tmeta);
        break;
      }
    }
  }
}

// shape agnostic : print duplicate graph information
void HabanaLaunchOpPT::MaybePrintDuplicateGraphInformation(
    const synapse_helpers::graph& graph_ptr,
    const std::vector<synTensorHandleMap>& tensors_map,
    const std::vector<synNodeHandleMap>& nodes_map,
    bool is_cache_hit) {
  PT_EAGER_DEBUG(
      "[SHAPE AGNOSTIC] === cache ",
      is_cache_hit ? "hit" : "miss",
      " duplicate graph information ====");
  PT_EAGER_DEBUG(
      "[SHAPE AGNOSTIC] original graph handle : ",
      graph_ptr.get_graph_handle(),
      " numTensors :",
      graph_ptr.get_num_of_tensors(),
      " numNodes :",
      graph_ptr.get_num_of_nodes());

  for (size_t i = 0; i < tensors_map.size(); i++) {
    PT_EAGER_DEBUG(
        "[SHAPE AGNOSTIC] org tensor handle : ",
        tensors_map.at(i).origHandle,
        " new tensor handle : ",
        tensors_map.at(i).newHandle);
  }

  for (size_t i = 0; i < nodes_map.size(); i++) {
    PT_EAGER_DEBUG(
        "[SHAPE AGNOSTIC] org node handle : ",
        nodes_map.at(i).origHandle,
        " new node handle : ",
        nodes_map.at(i).newHandle);
  }
}

void HabanaLaunchOpPT::update_syn_launch_info(
    uint64_t oldAddress,
    uint64_t newAddress) {
  for (auto& info : syn_launch_info_) {
    if (info.pTensorAddress == oldAddress) {
      info.pTensorAddress = newAddress;
      return;
    }
  }
  PT_BRIDGE_WARN(
      "[UpdateSynLaunchInfo] No such tensor address found in launch info");
}

// call this function for recipe caching (graph/eager)
void HabanaLaunchOpPT::ExecuteSynapseCache(size_t graph_key_with_perm) {
  PT_BRIDGE_BEGIN;

  if (habana_helpers::IsInferenceMode()) {
    ConstantInformation::checksum_t zero_checksum{0};
    for (size_t j = 0; j < pt_stack_sh_.size(); j++) {
      auto ivpsh = pt_stack_sh_[j];
      if (ivpsh.get()->isTensor()) {
        auto pt_tensor = ivpsh.get()->toTensor();
        if (habana::is_tensor_const_with_valid_const_id(pt_tensor)) {
          ConstantInformation::id_t const_id{
              habana::get_tensor_const_id(pt_tensor)};
          auto& constant_information = ConstantInformationValue();
          auto info_exists = constant_information.DoesConstInfoExist(
              const_id, ConstantInformation::key_t{cur_rargpsh_->hashCode()});
          if (!info_exists) {
            uint64_t oldAddress = reinterpret_cast<uint64_t>(
                pt_tensor.storage().data_ptr().get());
            DeserializeConstSection(pt_tensor, cur_rargpsh_->hashCode());
            uint64_t newAddress = reinterpret_cast<uint64_t>(
                pt_tensor.storage().data_ptr().get());
            update_syn_launch_info(oldAddress, newAddress);
          }
          auto checksum_and_recipe_checksum =
              constant_information.GetConstCheckSumForRecipe(
                  const_id,
                  ConstantInformation::key_t{cur_rargpsh_->hashCode()});
          PT_BRIDGE_DEBUG(
              "Execute Synapse cache, key: ",
              cur_rargpsh_->hashCode(),
              " const_id: ",
              const_id,
              " checksum on device: ",
              checksum_and_recipe_checksum.const_checksum_,
              " checksum for recipe: ",
              checksum_and_recipe_checksum.const_checksum_for_recipe_);
          // If constant checksum for recipe is 0, then there will be no pointer
          // assosciated with the constant info the constant has been folded in
          // the recipe and we can leave how it is on the device
          if (checksum_and_recipe_checksum.const_checksum_for_recipe_ ==
              zero_checksum) {
            continue;
          }
          if (checksum_and_recipe_checksum.const_checksum_ !=
              checksum_and_recipe_checksum.const_checksum_for_recipe_) {
            uint64_t oldAddress = reinterpret_cast<uint64_t>(
                pt_tensor.storage().data_ptr().get());
            constant_information.GetConstPtrForRecipe(
                const_id,
                ConstantInformation::key_t{cur_rargpsh_->hashCode()},
                pt_tensor);
            constant_information.Insert(
                const_id,
                checksum_and_recipe_checksum.const_checksum_for_recipe_);
            uint64_t newAddress = reinterpret_cast<uint64_t>(
                pt_tensor.storage().data_ptr().get());
            update_syn_launch_info(oldAddress, newAddress);
          }
        }
      }
    }
  }

  if (!dry_run_) {
    recipe_launcher_->Launch(
        hpu_stream_,
        input_refs_,
        intermediate_tensors_ptr_sh_,
        aten_outputs_,
        syn_launch_info_,
        external_tensor_info_indexes_,
        dma_inputs_);
  }

  if (habana_helpers::GetRefineDynamicShapeStatus() || refine_ds_enabled_) {
    CreateStaticCompilationDBI(graph_key_with_perm);
  }

  if (!get_enable_2stage_pipeline()) {
    // Update the stack from the recipe itself
    UpdateRecipeOutputs();
  }
  PT_BRIDGE_DEBUG("Returning cached recipe : ", cur_rargpsh_->hashCode());

  ClearStatics();
  PT_BRIDGE_END;
}

void HabanaLaunchOpPT::DumpStaticCompilationStatistics(
    size_t graph_key_with_perm,
    bool is_compile) {
  habana_helpers::ResultShapes ranges;

  habana_helpers::InpTensorShapes input_tshapes =
      HabanaLaunchOpUtils::ref_input_shape_map().at(graph_key_with_perm);
  if (is_compile) {
    current_dbipsh_->get_statistics()->LogCompilation(
        jit_ir_graph_->toString(),
        jit_ir_graph_,
        current_dbipsh_->GetMinPolicy(),
        current_dbipsh_->GetMaxPolicy(),
        ranges,
        cur_rargpsh_->hashCode(),
        "OK",
        habana_helpers::CompilationPass::STATIC);
    current_dbipsh_->get_statistics()->LogShapes(jit_ir_graph_, input_tshapes);
    current_dbipsh_->get_statistics()->LogUsedBucket(
        0, jit_ir_graph_, ranges, false);
    current_dbipsh_->get_statistics()->LogSelectedRecipe(
        cur_rargpsh_->hashCode(), 0);
    // current_dbipsh_->get_statistics()->LogRecipeMemory(cur_rvalpsh);
    current_dbipsh_->get_statistics()->GetDigest(
        cur_rargpsh_->graphHashCode(), 0, 0, cur_rargpsh_->hashCode(), false);
  } else {
    std::shared_ptr<RecipeArgumentSpec> rargpsh_graph =
        std::make_shared<RecipeArgumentSpec>(input_refs_, graph_key_, op_strs_);
    current_dbipsh_ = DynamicBucketInfoMap::get_instance().get(rargpsh_graph);
    HABANA_ASSERT(
        (current_dbipsh_ != nullptr),
        "Dynamic bucketinfo got NULL in static cache hit");
    current_dbipsh_->SetLastUsedStepForBucket(
        0, current_dbipsh_->get_statistics()->GetCurrentStep());

    current_dbipsh_->get_statistics()->LogSelectedRecipe(
        cur_rargpsh_->hashCode(), 0);
    current_dbipsh_->get_statistics()->LogShapes(jit_ir_graph_, input_tshapes);

    auto t_ns_base{current_dbipsh_->GetTimeBase(0)};
    auto t_ns{current_dbipsh_->GetTime(0)};

    current_dbipsh_->get_statistics()->LogLaunchBase(t_ns_base, 0);
    current_dbipsh_->get_statistics()->LogLaunch(t_ns, 0);
    current_dbipsh_->get_statistics()->LogLaunchPerf(t_ns_base, t_ns, 0);
    if (t_ns && t_ns_base) {
      habana_helpers::DynamicBucketInfo::update_improvement_map(
          cur_rargpsh_->hashCode(), (t_ns < t_ns_base));
    }
    current_dbipsh_->get_statistics()->GetDigest(
        cur_rargpsh_->graphHashCode(), 0, 0, cur_rargpsh_->hashCode(), true);
  }

  current_dbipsh_->get_statistics()->DumpAndNextStep();
}

void HabanaLaunchOpPT::UpdatePatchingInformation(
    RecipeValueSpec& rv,
    bool is_graph_empty,
    bool is_ds_patching_update,
    std::optional<
        std::reference_wrapper<const std::unordered_map<int64_t, at::Tensor>>>
        local_tidx_to_tensor_map,
    const std::unordered_map<synTensor, synTensor>& synapse_orig_to_new_handle,
    const bool is_shape_agnostic_graph) {
  intermediate_tensors_ptr_sh_ = std::make_shared<VecOfIValPtrSh>();

  // The aten_output_num is the total number of outputs
  size_t aten_output_num = rv.get_aten_output_num();
  aten_outputs_.resize(aten_output_num);
  if (!is_ds_patching_update) {
    rv.update_patching_table(
        input_refs_,
        intermediate_tensors_ptr_sh_,
        dma_inputs_,
        aten_outputs_,
        map_shape_.m_actual_shapes,
        local_tidx_to_tensor_map,
        allocated_outputs_,
        out_shapes_,
        synapse_orig_to_new_handle,
        is_shape_agnostic_graph);
  } else {
    rv.update_patching_table(
        input_refs_,
        intermediate_tensors_ptr_sh_,
        dma_inputs_,
        aten_outputs_,
        map_shape_.m_actual_shapes,
        local_tidx_to_tensor_map,
        allocated_outputs_);
  }
  if (!dry_run_) {
    if (!is_graph_empty) {
      rv.patch_launch_info(syn_launch_info_, external_tensor_info_indexes_);
    } else {
      PT_BRIDGE_DEBUG("Skipping patch_launch_info for empty recipe");
    }
  }
  PT_BRIDGE_DEBUG(rv);
}

void HabanaLaunchOpPT::run(
    torch::jit::Stack& stack,
    std::shared_ptr<habana::RecipeArgumentSpec> cached_rarg_psh,
    std::optional<std::vector<at::Tensor>> allocated_outputs,
    std::optional<std::vector<std::vector<int64_t>>> output_shapes,
    bool dry_run,
    HabanaLaunchOpPipeline::PipelineCallBase& pipeline_execution) {
  PT_BRIDGE_BEGIN;

  if (enable_shape_agnostic_caching_) {
    HABANA_ASSERT(output_shapes.has_value());
    out_shapes_ = std::move(*output_shapes);
    HABANA_ASSERT(
        out_shapes_.size() == jit_ir_graph_->outputs().size(),
        "number of output shapes for patching ",
        out_shapes_.size(),
        " is not equal to #outputs in jit graph ",
        jit_ir_graph_->outputs().size());
  }

  static int idx{1};
  ProcessInputStack(stack);
  allocated_outputs_ = std::move(allocated_outputs);

  dry_run_ = dry_run;
  auto& device = HPUDeviceContext::get_device();

  const auto eager_mode =
      (execution_mode_ == habana_helpers::HabanaFrontendTypes::EAGER);
  const auto compile_mode =
      (execution_mode_ == habana_helpers::HabanaFrontendTypes::COMPILE);

  if (!compile_mode) {
    graph_key_with_perm_ = graph_key_;
    graph_symint_hash_ = habana::ComputeSymSizeHashCode(input_refs_);
    graph_key_with_perm_ =
        at::hash_combine(graph_key_with_perm_, graph_symint_hash_);
    graph_perm_hash_ = habana::ComputePermutationHashCode(input_refs_);
    graph_key_with_perm_ =
        at::hash_combine(graph_key_with_perm_, graph_perm_hash_);
  }

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
      graph_key_with_perm_,
      "is dynamic : ",
      refine_ds_enabled_);

  if (enable_optim_output_sif_) {
    DumpSymbolValueMap(in_symbol_value_map_);
  }
  idx += 1;
  if (enable_caching_ || IS_BRIDGE_DEBUG_ENABLED) {
    if (maybe_static_recipe_ &&
        ((cached_rarg_psh.get() == nullptr) ||
         (cached_rarg_psh->graphWithPermuteHashCode() != graph_perm_hash_))) {
      cur_rargpsh_ = std::make_shared<RecipeArgumentSpec>(
          false,
          input_refs_,
          jit_ir_graph_,
          graph_key_,
          op_strs_,
          graph_symint_hash_,
          graph_perm_hash_);
      auto context = habana_lazy::get_device_lazy_execution_context();

      if ((context->getCapturing() &&
           (GET_ENV_FLAG_NEW(PT_HPU_DISABLE_HPUGRAPH_REPLAY_HASHCHECK)))) {
        context->saveRecipeArgSpec(cur_rargpsh_);
      }
    } else {
      cur_rargpsh_ = cached_rarg_psh;
    }
  }

  auto is_enable_4stage_pipeline = enable_4stage_pipeline_;

  // eager and graph recipe caching :: begin
  if (enable_caching_ && maybe_static_recipe_) {
    HABANA_ASSERT(
        enable_graph_caching_ || enable_eager_caching_,
        " something went wrong! either eager or graph recipe caching should be enabled");
    PT_BRIDGE_DEBUG("Getting cached recipe : ", cur_rargpsh_->hashCode());
    auto recipe_holder = GetCachedRecipe(cur_rargpsh_);

    if (ABSL_PREDICT_TRUE(recipe_holder)) {
      recipe_launcher_ = recipe_holder->rl_;
      auto& rvs = *recipe_holder->rvs_;
      emitCacheEvent(
          habana_helpers::EventDispatcher::Topic::CACHE_HIT,
          std::to_string(cur_rargpsh_->hashCode()));

      rvs.update_hit_count();

      PT_BRIDGE_DEBUG(
          id_str_,
          ": ",
          "HabanaOp recipe cache hit :: key ",
          cur_rargpsh_->hashCode(),
          "\n",
          rvs.header_str(),
          "\n",
          rvs.digest_str());
      PT_IRGRAPH_DEBUG("HabanaOp recipe cache hit :: static shapes");
      PT_TEST_DEBUG("HabanaOp recipe cache hit :: static path");

      UpdatePatchingInformation(rvs, !recipe_launcher_->recipe_);

      // currently only eager backend supports pipelining
      // can be merged once non-eager backends support pipelining
      if (enable_graph_caching_ && !compile_mode) {
        ExecuteSynapseCache(graph_key_with_perm_);
      } else {
        PT_LAZY_EAGER_DEBUG(
            "[LAZY EAGER MT] Enqueue new task to the Compile and Execute Thread");
        if (!is_enable_4stage_pipeline) {
          ExecuteSynapseCache(graph_key_with_perm_);
        } else {
          execution_control_.cached_task(graph_key_with_perm_);
          pipeline_execution();
        }
      }
      PT_BRIDGE_END;
      return;
    } else {
      emitCacheEvent(
          habana_helpers::EventDispatcher::Topic::CACHE_MISS,
          std::to_string(cur_rargpsh_->hashCode()));
      PT_BRIDGE_DEBUG(
          id_str_,
          ": ",
          "HabanaOp recipe cache miss :: key ",
          cur_rargpsh_->hashCode());
      PT_IRGRAPH_DEBUG("HabanaOp recipe cache miss :: static shapes");
      PT_TEST_DEBUG("HabanaOp recipe cache miss :: static path");
    }
  }
  // eager and graph recipe caching :: end

  CreateValueToIvalueMapForInputs();
  ProcessGraphForConstantTensors(*jit_ir_graph_, value_to_ivalue_);

  if (enable_shape_agnostic_caching_) {
    ValidateInputsAndOutputsAndDisableSA(input_refs_);
  }

  // shape agnostic caching :: begin
  if (enable_shape_agnostic_caching_ &&
      jit_graph_and_meta_data_->get_is_shape_agnostic_supported()) {
    auto rvs = jit_graph_and_meta_data_->get_shape_agnostic_recipe();
    if (rvs == nullptr) {
      PT_EAGER_DEBUG("[SHAPE AGNOSTIC] shape agnostic cache miss (begin)");
      HABANA_ASSERT(
          eager_mode == true,
          "eager_mode is expected true for supporting shape agnostic graph");
      is_shape_agnostic_supported_ = true;
      constexpr bool dry_run__ = false;
      auto syn_graph =
          std::make_shared<synapse_helpers::graph>(habana_helpers::create_graph(
              device.id(), GetSynapseGraphName(), dry_run__, eager_mode));

      constexpr bool is_shape_agnostic_graph = true;
      syn_graph->set_shape_agnostic_graph(is_shape_agnostic_graph);
      BuildSynapseGraph(syn_graph, jit_graph_and_meta_data_->syn_build_cache_);
      syn_graph_ptr_->set_num_of_inter_tensors(intermediate_syn_tensors_count_);

      if (syn_graph_ptr_->is_empty()) {
        PT_LAZY_EAGER_DEBUG(
            "Empty synapse graph. Nothing to duplicate. will update outputs directly.");
        UpdateOutputs();
        return;
      }

      auto original_syn_graph = std::move(*syn_graph_ptr_);
      auto [duplicate_graph, tensorsMap, nodesMap] =
          synapse_helpers::graph::duplicate(original_syn_graph);
      syn_graph_ptr_ =
          std::make_shared<synapse_helpers::graph>(std::move(duplicate_graph));
      MaybePrintDuplicateGraphInformation(
          *syn_graph_ptr_, tensorsMap, nodesMap, false);

      std::vector<std::pair<synTensor, std::vector<int64_t>>>
          duplicate_tensors_shape_map;
      ConstructDuplicateShapeMap(tensorsMap, duplicate_tensors_shape_map);

      if (syn_graph_ptr_->get_num_of_shape_tensors() > 0) {
        jit_graph_and_meta_data_->set_is_shape_agnostic_supported(false);
        PT_EAGER_DEBUG(
            "[SHAPE AGNOSTIC] Shape agnostic not supported for op ",
            name_,
            ", with intermediate shape tensors : ",
            syn_graph_ptr_->get_num_of_shape_tensors());
      }

      if (jit_graph_and_meta_data_->get_is_shape_agnostic_supported()) {
        bool compound_ops_flag =
            ((habana_kernels_.size() - meta_attribute_nodes_count_) !=
             syn_graph_ptr_->get_num_of_nodes());

        PT_EAGER_DEBUG(
            "[SHAPE AGNOSTIC] Number of kernels: ",
            habana_kernels_.size(),
            ", number of JIT IR nodes with meta attribute : ",
            meta_attribute_nodes_count_,
            ", number of synapse nodes: ",
            syn_graph_ptr_->get_num_of_nodes(),
            ", flag compound op(s): ",
            compound_ops_flag);

        bool non_persistent_tensors_flag =
            ((syn_graph_ptr_->get_num_of_tensors() -
              syn_graph_ptr_->get_num_of_const_tensors()) !=
             (pt_to_synapse_tensors_.size() + implicit_syn_tensors_count_));

        PT_EAGER_DEBUG(
            "[SHAPE AGNOSTIC] Total num of syn tensors: ",
            syn_graph_ptr_->get_num_of_tensors(),
            ", num of const tensors: ",
            syn_graph_ptr_->get_num_of_const_tensors(),
            ", num of persistent tensors: ",
            pt_to_synapse_tensors_.size(),
            ", num of implicit tensors: ",
            implicit_syn_tensors_count_,
            ", flag non persistent tensor(s): ",
            non_persistent_tensors_flag);

        if (compound_ops_flag || non_persistent_tensors_flag) {
          bool syn_infer_shapes = true;
          if (get_require_h2d_st()) {
            jit_graph_and_meta_data_->set_is_shape_agnostic_supported(false);
            PT_EAGER_DEBUG(
                "[SHAPE AGNOSTIC] Shape agnostic not supported for op ",
                name_,
                ", syanpse shape inference expects shape or h2d tensor(s) !");
            syn_infer_shapes = false;
          }
          // ToDO: Remove try run hybrid sif logic once shape tensor(s)
          // can be queried using shared layer.
          // Try run hybrid sif with dynamic shapes flag to detect
          // if shape tensor(s) required for synapse shape inference
          try {
            habana::ShapeInference::ResetSifTensorId();
            std::unordered_map<int64_t, at::Tensor> tmp_map;
            constexpr bool dynamic_shapes_true = true;
            if (RunHybridSif<dynamic_shapes_true>(tmp_map)) {
              jit_graph_and_meta_data_->set_is_shape_agnostic_supported(false);
              PT_EAGER_DEBUG(
                  "[SHAPE AGNOSTIC] Shape agnostic not supported for op ",
                  name_,
                  ", syanpse shape inference expects shape tensor(s) !");
              syn_infer_shapes = false;
            }
          } catch (std::exception& e) {
            // RunHybridSif<true> can return TORCH CHECK from
            // AllocateAndAddSynapseNode if ComputeOutputShape
            // is not supported or failed during early validation
            PT_EAGER_DEBUG(
                "[SHAPE AGNOSTIC] Shape agnostic not supported for op ",
                name_,
                ", RunHybridSif with dynamic shapes failed ! ",
                "what(): ",
                e.what());
            syn_infer_shapes = false;
          }

          // Try infer shapes using synapse shape inference if possible
          if (syn_infer_shapes) {
            if (syn_graph_ptr_->inferShapes()) {
              jit_graph_and_meta_data_->set_is_synapse_shape_inf_required(true);
              PT_EAGER_DEBUG(
                  "[SHAPE AGNOSTIC] syanpse shape inference is required");
            } else {
              jit_graph_and_meta_data_->set_is_shape_agnostic_supported(false);
              PT_EAGER_DEBUG(
                  "[SHAPE AGNOSTIC] Shape agnostic not supported for op ",
                  name_,
                  ", Cache miss synapse shape inference failed !");
              // Reset duplicate tensors shape to correct shapes
              ResetDuplicateTensorShapes(
                  syn_graph_ptr_, duplicate_tensors_shape_map);
            }
          }
        }
      }
      PT_EAGER_DEBUG(
          "[SHAPE AGNOSTIC] Shape agnostic SIF tinfo map size : ",
          sif_tidx_to_tinfo_map_.size(),
          " intermediate tensors size : ",
          intermediate_syn_tensors_count_);

      // TODO do we need sync ????
      pipeline_execution.compile_sync();
      auto rvs = CompileSynapseGraphAndPatchTable();
      // TODO turn on this condition
      // if (jit_graph_and_meta_data_->get_is_shape_agnostic_supported())
      {
        rvs->shape_agnostic_synapse_graph_ =
            std::make_unique<synapse_helpers::graph>(
                std::move(original_syn_graph));
        get_jit_graph_and_meta_data()->set_shape_agnostic_recipe(rvs);
      }

      PT_LAZY_EAGER_DEBUG(
          "[LAZY EAGER MT] Enqueue new task to the Compile and Execute Thread");
      execution_control_.no_compile();
      // In case of SAG cache miss we have to wait till a compile thread sets
      // permutation for outputs (In case of SAG cache hit, that info is taken
      // from SAG recipe (from dtensorinfos))
      pipeline_execution();
      return;
    } else {
      PT_EAGER_DEBUG("[SHAPE AGNOSTIC] shape agnostic cache hit (begin)");
      is_shape_agnostic_supported_ = true;

      auto [duplicate_graph, tensorsMap, nodesMap] =
          synapse_helpers::graph::duplicate(
              *rvs->shape_agnostic_synapse_graph_);
      syn_graph_ptr_ =
          std::make_shared<synapse_helpers::graph>(std::move(duplicate_graph));
      MaybePrintDuplicateGraphInformation(
          *syn_graph_ptr_, tensorsMap, nodesMap, true);

      RecipeValueSpec& rv = *rvs;
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

      std::unordered_map<synNodeId, synNodeId>
          synapse_nodes_orig_to_new_handle{};
      for (size_t i = 0; i < nodesMap.size(); i++) {
        synapse_nodes_orig_to_new_handle.insert(
            {nodesMap.at(i).origHandle, nodesMap.at(i).newHandle});
      }

      for (const auto& out_shape : out_shapes_) {
        PT_EAGER_DEBUG("[SHAPE AGNOSTIC] output shape - ", out_shape);
      }

      /*
       * Hybrid SIF is used for shape inference for intermediate tensors
       * and node parameters inference only if param agnostic is supported
       * Inputs shape is retrieved from input refs.
       * Ouptut shape info is passed in the jit ir graph meta data.
       */
      std::unordered_map<int64_t, at::Tensor> local_tidx_to_tensor_map{};
      auto node_params_vec_ptr =
          jit_graph_and_meta_data_->get_is_param_agnostic_supported()
          ? std::make_shared<std::vector<InferNodeParams>>()
          : nullptr;
      if (syn_graph_ptr_->get_num_of_inter_tensors() > 0) {
        habana::ShapeInference::ResetSifTensorId();
        constexpr bool dynamic_shapes_false = false;
        RunHybridSif<dynamic_shapes_false>(
            local_tidx_to_tensor_map, node_params_vec_ptr);
      }

      constexpr bool is_shape_agnostic_graph = true;
      UpdatePatchingInformation(
          rv,
          syn_graph_ptr_->is_empty(),
          false,
          local_tidx_to_tensor_map,
          synapse_orig_to_new_handle,
          is_shape_agnostic_graph);

      // Run Synapse Shape inference if required
      if (jit_graph_and_meta_data_->get_is_synapse_shape_inf_required()) {
        HABANA_ASSERT(
            syn_graph_ptr_->inferShapes() == true,
            "[SHAPE AGNOSTIC] Cache hit Synapse shape inference failed !");
      }

      // Get updated nodes params for if param agnostic is supported
      // and for single node graph for which Hybrid SIF did not run earlier
      // ToDO: Check and optimize Hybrid SIF for getting only node params
      if (node_params_vec_ptr && local_tidx_to_tensor_map.empty()) {
        PT_EAGER_DEBUG("[SHAPE AGNOSTIC] Calling Hybrid SIF for node params");
        constexpr bool dynamic_shapes_false = false;
        RunHybridSif<dynamic_shapes_false>(
            local_tidx_to_tensor_map, node_params_vec_ptr);
      }

      // Update node params only if available and param agnostic is supported
      if (node_params_vec_ptr) {
        rv.update_node_params(
            synapse_nodes_orig_to_new_handle,
            syn_graph_ptr_->get_syn_node_id_vec(),
            syn_graph_ptr_->get_graph_handle(),
            node_params_vec_ptr);
      }

      recipe_launcher_ = std::make_unique<RecipeLauncher>(rv);

      // Once SAG supports control edges, we'll have to double-check if we need
      // additional control edge processing here

      PT_LAZY_EAGER_DEBUG(
          "[LAZY EAGER MT] Enqueue new task to the Compile and Execute Thread");
      execution_control_.sag_cache_hit();
      pipeline_execution();
    }
    PT_BRIDGE_END;
    return;
  }
  // shape agnostic caching :: end
  if ((!eager_mode &&
       HabanaLaunchOpUtils::ref_input_shape_map().count(graph_key_with_perm_) &&
       refine_ds_enabled_) ||
      enable_user_dynamic_ranges_) {
    PT_DYNAMIC_SHAPE_DEBUG(
        "JIT IR graph_hash_code : ",
        graph_key_,
        ", hash_code with data layout : ",
        graph_key_with_perm_,
        "\nStarting dynamic shape flow");

    ProcessHabanaFusedOpWithDS(pipeline_execution);
    PT_BRIDGE_END;
    return;
  }

  // Remember the input shapes for creating dynamic bucket info structure later.
  // Note that this needs to be done before execution of graph, otherwise
  // input_refs_ will get overwritten by outputs and we will create bucket
  // with incorrect shapes.

  if (!eager_mode &&
      (habana_helpers::GetRefineDynamicShapeStatus() || refine_ds_enabled_)) {
    CreateStaticCompilationDBI(graph_key_with_perm_);
  }

  constexpr bool dry_run__ = false;
  const auto use_eager_compiler =
      eager_mode && jit_graph_and_meta_data_->get_is_eager_compiler_supported();
  auto syn_graph =
      std::make_shared<synapse_helpers::graph>(habana_helpers::create_graph(
          device.id(), GetSynapseGraphName(), dry_run__, use_eager_compiler));
  map_shape_.m_pass = ShapeInfo::InferencePass::INVALID;
  BuildSynapseGraph(syn_graph, jit_graph_and_meta_data_->syn_build_cache_);

  if ((eager_mode || compile_mode) &&
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

    if (!is_permute_data_cached || enable_caching_) {
      // TODO may be we don't need sync with compile thread
      pipeline_execution.compile_sync();
      execution_control_.no_compile();
      CompileSynapseGraphAndPatchTable();
    }
    pipeline_execution();
  } else {
    CompileSynapseGraphAndPatchTable();
    ExecuteSynapseGraph();
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

    old_stack = pt_stack_;
    old_pt_stack_sh = pt_stack_sh_;
    pt_stack_sh_.clear();

    new_stack = CreateStack(*pt_stack_, graph_input_info.min_input_tshapes);
    SetH2DMinMaxData(
        *old_stack,
        graph_input_info.min_input_tshapes,
        ShapeInfo::InferencePass::MIN_SHAPE);
    pt_stack_ = &new_stack;

    for (size_t j{0}; j < new_stack.size(); j++) {
      IValPtrShared ivpsh = std::make_shared<IVal>(new_stack[j]);
      pt_stack_sh_.push_back(ivpsh);
    }
    map_shape_.m_pass = ShapeInfo::InferencePass::MIN_SHAPE;
    std::string error_str;

    try {
      run_pass();
    } catch (std::exception& e) {
      error_str = e.what();
      PT_DYNAMIC_SHAPE_DEBUG(
          "Exception occured in Pass = ",
          map_shape_.m_pass,
          " - Details :\n",
          error_str);

      pt_stack_sh_.clear();
      pt_stack_ = old_stack;
      pt_stack_sh_ = old_pt_stack_sh;
      throw;
    }

    pt_stack_sh_.clear();
    pt_stack_ = old_stack;
    pt_stack_sh_ = old_pt_stack_sh;

    PT_DYNAMIC_SHAPE_DEBUG(
        "Pass = ",
        map_shape_.m_pass,
        " completed. Shapes\n",
        map_shape_.m_min_shapes);
  }

  // Max shape inference
  {
    torch::jit::Stack new_stack;
    torch::jit::Stack* old_stack = nullptr;
    VecOfIValPtrSh old_pt_stack_sh;

    old_stack = pt_stack_;
    old_pt_stack_sh = pt_stack_sh_;
    pt_stack_sh_.clear();

    new_stack = CreateStack(*pt_stack_, graph_input_info.max_input_tshapes);
    SetH2DMinMaxData(
        *old_stack,
        graph_input_info.max_input_tshapes,
        ShapeInfo::InferencePass::MAX_SHAPE);
    pt_stack_ = &new_stack;

    for (size_t j{0}; j < new_stack.size(); j++) {
      IValPtrShared ivpsh = std::make_shared<IVal>(new_stack[j]);
      pt_stack_sh_.push_back(ivpsh);
    }
    map_shape_.m_pass = ShapeInfo::InferencePass::MAX_SHAPE;
    std::string error_str;

    try {
      run_pass();
    } catch (std::exception& e) {
      error_str = e.what();
      PT_DYNAMIC_SHAPE_DEBUG(
          "Exception occured in Pass = ",
          map_shape_.m_pass,
          " - Details :\n",
          error_str);

      pt_stack_sh_.clear();
      pt_stack_ = old_stack;
      pt_stack_sh_ = old_pt_stack_sh;
      throw;
    }

    pt_stack_sh_.clear();
    pt_stack_ = old_stack;
    pt_stack_sh_ = old_pt_stack_sh;

    PT_DYNAMIC_SHAPE_DEBUG(
        "Pass = ",
        map_shape_.m_pass,
        " completed. Shapes\n",
        map_shape_.m_max_shapes);
  }

  auto& device = HPUDeviceContext::get_device();
  std::string graphName{GetSynapseGraphName()};

  auto syn_graph = std::make_shared<synapse_helpers::graph>(
      synapse_helpers::graph::create_for_refinement(device, name_));

  std::shared_ptr<synapse_helpers::graph::recipe_handle> recipe;
  auto rvs = std::make_shared<RecipeValueSpec>(jit_ir_graph_);
  // Compile the graph
  {
    CreateValueToIvalueMapForInputs();

    syn_graph->set_dynamic_graph(true);

    std::string error_str;
    try {
      cur_ds_token_ = new_bucket.getToken();
      cur_rargpsh_ = std::make_shared<RecipeArgumentSpec>(
          input_refs_, graph_key_, op_strs_, cur_ds_token_);
      new_recipe_key = cur_rargpsh_->hashCode();

      map_shape_.m_pass = ShapeInfo::InferencePass::INVALID;
      SynBuildCache cache;
      BuildSynapseGraph(syn_graph, cache);
      recipe = CompileSynapseGraph();
      ConstructPatchingTableAndAtenOutputs(*rvs, recipe);
      UpdateSynapsePermutations(*rvs, *recipe);
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
  rvs->dynamic_graph = syn_graph->is_dynamic_graph();
  rvs->set_op_strs(cur_rargpsh_->get_op_strs());
  recipe_launcher_ = std::make_unique<RecipeLauncher>(*rvs, recipe);
  auto recipe_holder = std::make_shared<RecipeHolder>(recipe_launcher_, rvs);
  HPUDeviceContext::recipe_cache().add(cur_rargpsh_, recipe_holder);
  DynamicBucketInfoMap::get_instance().add(cur_rargpsh_, current_dbipsh_);

  new_recipe_key = cur_rargpsh_->hashCode();
  // Add the recipe to the corresponding bucket
  new_bucket.SetSynapseRecipePtr(rvs);

  PT_DYNAMIC_SHAPE_DEBUG(
      "HabanaOp recipe cache :: adding new recipe to cache ::",
      rvs->header_str(),
      "\n",
      rvs->digest_str(),
      "\n",
      "--------------------");

  ClearMembers();
  ClearStatics();

  PT_BRIDGE_END;
}

void HabanaLaunchOpPT::run_pass() {
  PT_BRIDGE_BEGIN;
  auto& device = HPUDeviceContext::get_device();

  // Run the compile and execute method to infer the shapes
  auto syn_graph = std::make_shared<synapse_helpers::graph>(
      habana_helpers::create_graph(device.id(), GetSynapseGraphName(), true));
  syn_graph->set_dynamic_graph(true);
  syn_graph->set_optim_output_sif_enabled(enable_optim_output_sif_);
  SynBuildCache cache;

  if (enable_optim_output_sif_ &&
      habana::ShapeInference::GetCurrentPass() ==
          habana::ShapeInfo::InferencePass::OUTPUT_SHAPE) {
    if (is_symval_changed_from_prev_) {
      BuildSynapseGraphLite(syn_graph, cache);
    } else {
      PT_DYNAMIC_SHAPE_DEBUG(
          "Skipping OUTPUT_PASS SIF as symbol values have not changed from previous run!!");
    }
  } else {
    CreateValueToIvalueMapForInputs();
    habana_helpers::createGraphInputStackIndexMap(
        jit_ir_graph_, org_stack_index_map);
    BuildSynapseGraph(syn_graph, cache, true);
  }

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
  map_shape_.m_pass = pass;
  if ((pass == ShapeInfo::InferencePass::MIN_SHAPE) ||
      (pass == ShapeInfo::InferencePass::MAX_SHAPE)) {
    old_stack = pt_stack_;
    old_pt_stack_sh = pt_stack_sh_;
    pt_stack_sh_.clear();
    try {
      if (pass == ShapeInfo::InferencePass::MIN_SHAPE) {
        new_stack = CreateStack(*pt_stack_, graph_input_info.min_input_tshapes);
        SetH2DMinMaxData(
            *old_stack,
            graph_input_info.min_input_tshapes,
            ShapeInfo::InferencePass::MIN_SHAPE);
        SetH2DMinMaxData(
            new_stack,
            graph_input_info.min_input_tshapes,
            ShapeInfo::InferencePass::MIN_SHAPE);
      } else {
        new_stack = CreateStack(*pt_stack_, graph_input_info.max_input_tshapes);
        SetH2DMinMaxData(
            *old_stack,
            graph_input_info.max_input_tshapes,
            ShapeInfo::InferencePass::MAX_SHAPE);
        SetH2DMinMaxData(
            new_stack,
            graph_input_info.max_input_tshapes,
            ShapeInfo::InferencePass::MAX_SHAPE);
      }
    } catch (std::exception& e) {
      std::string error = e.what();
      std::string error_str = error.substr(0, error.find("\n"));
      PT_DYNAMIC_SHAPE_DEBUG(
          "Exception occured in CreateStack for Pass = ", pass);
      PT_DYNAMIC_SHAPE_DEBUG("Exception Details : ", error_str);
      RevertH2DMinMaxData();
      if (old_stack) {
        pt_stack_sh_.clear();
        pt_stack_ = old_stack;
        pt_stack_sh_ = old_pt_stack_sh;
      }
      throw PassException(pass, error_str);
    }

    pt_stack_ = &new_stack;

    size_t j = new_stack.size() - num_inputs_;
    for (; j < new_stack.size(); j++) {
      IValPtrShared ivpsh = std::make_shared<IVal>(new_stack[j]);
      pt_stack_sh_.push_back(ivpsh);
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
    pt_stack_sh_.clear();
    pt_stack_ = old_stack;
    pt_stack_sh_ = old_pt_stack_sh;
  }

  if (updatemax_graph_) {
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

  if (graph_input_info.current_bucket_id == 0) {
    // current_bucket_id = 0 having ranges means this is user min max case
    // if there is fallback in such case, set next policy to CURRENT for both
    // and recalculate ranges
    graph_input_info.min_policy = habana_helpers::DynamicDimsPolicy::CURRENT;
    graph_input_info.max_policy = habana_helpers::DynamicDimsPolicy::CURRENT;
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

  // In case the fallback happened for mark_dynamic, reset both min and max
  // data, because the policy was changed to CURRENT for both, an exception
  // in MAX although genuine will cause exit.
  if (graph_input_info.current_bucket_id == 0) {
    graph_input_info.min_input_tshapes.clear();
    graph_input_info.min_input_tshapes.insert(
        fallback_ranges.min_shapes.begin(), fallback_ranges.min_shapes.end());
    graph_input_info.max_input_tshapes.clear();
    graph_input_info.max_input_tshapes.insert(
        fallback_ranges.max_shapes.begin(), fallback_ranges.max_shapes.end());
  }

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
    DynamicShapeInfo& graph_input_info,
    HabanaLaunchOpPipeline::PipelineCallBase& pipeline_execution) {
  auto& device = HPUDeviceContext::get_device();
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
    cur_rargpsh_ = std::make_shared<RecipeArgumentSpec>(
        input_refs_, graph_key_, op_strs_, new_ds_token);
    current_dbipsh_->SetRecipeKeyForBucket(
        current_bucket_id_, cur_rargpsh_->hashCode());
    DynamicBucketInfoMap::get_instance().add(cur_rargpsh_, current_dbipsh_);
  }

  if (ranges.empty()) {
    if (graph_input_info.max_policy ==
        habana_helpers::DynamicDimsPolicy::CURRENT) {
      last_compilation_pass = habana_helpers::CompilationPass::DYNAMIC_CURRENT;
    }
  } else {
    last_compilation_pass = habana_helpers::CompilationPass::DYNAMIC_MAX;
  }
  map_shape_.m_pass = ShapeInfo::InferencePass::INVALID;
  auto syn_graph = std::make_shared<synapse_helpers::graph>(
      habana_helpers::create_graph(device.id(), GetSynapseGraphName()));
  syn_graph->set_dynamic_graph(is_dynamic_graph);
  syn_graph->set_optim_output_sif_enabled(enable_optim_output_sif_);
  EvictSynapseRecipe(graph_input_info.current_bucket_id);
  SynBuildCache cache;
  BuildSynapseGraph(syn_graph, cache);

  if (enable_4stage_pipeline_) {
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
    current_dbipsh_->get_statistics()->LogUsedBucket(
        graph_input_info.current_bucket_id,
        jit_ir_graph_,
        ranges,
        // setting refine candidate false here since for
        // refinement pipelining is disabled
        false);
    current_dbipsh_->get_statistics()->LogSelectedRecipe(
        current_dbipsh_->GetRecipeKeyForBucket(
            graph_input_info.current_bucket_id),
        0);
    current_dbipsh_->get_statistics()->GetDigest(
        cur_rargpsh_->graphHashCode(),
        current_bucket_id_,
        cur_ds_token_,
        cur_rargpsh_->hashCode(),
        false);
    current_dbipsh_->get_statistics()->DumpAndNextStep();
    bool is_permute_data_cached = jit_graph_and_meta_data_->is_permute_set();
    if (is_permute_data_cached) {
      ApplyOutputPermutationsFromCache();
    } else {
      permutation_info_saver_ =
          std::make_unique<PermutationInfoSaver>(jit_graph_and_meta_data_);
    }
    PT_DYNAMIC_SHAPE_DEBUG("Cache miss pipeline flow");
    pipeline_execution.compile_sync();
    execution_control_.no_compile();
    CompileSynapseGraphAndPatchTable();
    pipeline_execution();
  } else {
    CompileSynapseGraphAndPatchTable();
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
          (current_dbipsh_->GetMFUBucket() ==
           graph_input_info.current_bucket_id);
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
      current_dbipsh_->get_statistics()->LogRecipeMemory(
          *recipe_launcher_->recipe_);
    }
    current_dbipsh_->get_statistics()->GetDigest(
        cur_rargpsh_->graphHashCode(),
        current_bucket_id_,
        cur_ds_token_,
        cur_rargpsh_->hashCode(),
        false);
    current_dbipsh_->get_statistics()->DumpAndNextStep();
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
