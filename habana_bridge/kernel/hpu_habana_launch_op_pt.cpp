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

#include "habana_bridge/kernel/ds_graph_recompile.h"
#include "habana_bridge/kernel/hpu_shape_inference.h"
#include "habana_bridge/kernel/refinement_engine.h"
#include "habana_bridge/passes/hpu_habana_persistence_marker_pass.h"

#include "habana_helpers/graph.h"
#include "habana_helpers/logging.h"
#include "habana_helpers/misc_utils.h"
#include "habana_helpers/tensor_utils.h"
#include "habana_helpers/unused_macro.h"

#include "habana_kernels/hccl_kernels.h"
#include "habana_kernels/kernel_utils.h"
#include "habana_kernels/lazy_kernels_declarations.h"
#include "habana_kernels/random_gen_kernels.h"
#include "habana_kernels/unary_kernels.h"

#include "habana_lazy/aten_lazy_bridge.h"
#include "habana_lazy/hlexec.h"
#include "habana_lazy/hpu_lazy_tensors.h"

#include "hpu_ops/hpu_op_helper.h"

#include "pytorch_helpers/util/jitgraph_utils.h"
#include "synapse_helpers/env_flags.h"

using namespace torch::jit;
using namespace jitgraph_utils;

namespace habana {

// static initializations
const std::unordered_set<std::string> HabanaMetaOpList::meta_ops = {
    // Add aten string here for ops to support
    // e.g  :: "aten::view"
    "aten::size",
    "prim::dtype"};

std::unordered_set<std::string> HabanaLaunchOpPT::watchlist_ = {};
//--------------------------------------

bool dropCachedRecipe_LRU(size_t& recipe_count) {
  bool dropped{false};
  dropped = RecipeCacheLRU::get_cache().drop_lru(recipe_count);
  return dropped;
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
    std::shared_ptr<habana_lazy::OptimizedJITGraphAndMetaData>
        optimized_jit_graph_and_meta_data)
    : name(optimized_jit_graph_and_meta_data->GetOpName()),
      graph_index(optimized_jit_graph_and_meta_data->GetGraphIndex()),
      jit_ir_graph(optimized_jit_graph_and_meta_data->get_cached_graph()),
      debug(optimized_jit_graph_and_meta_data->GetDbgFlag()) {
  refine_ds_enabled_ = GET_ENV_FLAG_NEW(PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES);
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
  if (!HPUDeviceAllocator::drop_cached_recipe_cb) {
    HPUDeviceAllocator::drop_cached_recipe_cb = dropCachedRecipe_LRU;
  }

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

  enable_caching_ = GET_ENV_FLAG_NEW(PT_HPU_PGM_ENABLE_CACHE);

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
  if (node->output(0)->type() == torch::ListType::ofTensors() &&
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
      output_metadata.emplace_back(md);
    }
  } else {
    for (auto value_out : node_outs) {
      OutputMetaData md(*value_out);
      md.persistent = nodeOutputPersistencePerValue(node, value_out);
      if (md.persistent) {
        md.external = IsValueExternal(value_out);
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
    at::Tensor& pt_tensor) {
  PT_BRIDGE_TRACE;
  auto impl = habana_lazy::GetHbInternalTensorImpl(pt_tensor);

  if (impl && impl->isShapeTensor()) {
    void* host_ptr = impl->get_compile_host_ptr();
    auto& syn_tensor = habana_op->AllocateSynapseInput(
        *syn_graph_ptr, pt_tensor, true, impl->getTensorType(), host_ptr);
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
    auto& syn_tensor = AllocateSynapseTensor(habana_op, pt_tensor);
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
        syn_tensor.tensor_type());

    auto impl = habana_lazy::GetHbInternalTensorImpl(pt_tensor);
    if (impl) {
      ti->set_host_ptr(impl->get_host_ptr());
    }
    tiv.push_back(ti);
    ivalue_to_tensor_info_map[ivalue] = ti;

    if (enable_caching_) {
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

    if (enable_caching_) {
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
    torch::jit::Node* node,
    torch::jit::Stack& stack) {
  auto node_ins = node->inputs();
  int input_idx = 0;
  auto node_qual_str = node->kind().toQualString();
  for (const auto value_in : node_ins) {
    auto value_exists = value_to_ivalue.find(value_in);
    HABANA_ASSERT(value_exists != std::end(value_to_ivalue));
    auto ivalue = value_exists->second;
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
    // input_idx is simply the index of 1st non-tensor input argument, which is
    // the 1st time we come into else part. Since we want to generate the
    // seed_tensor only once that is why the check on 1st non-tensor input
    // argument.
    else if (
        (!strcmp("hpu::randperm_out", node_qual_str) && 0 == input_idx) ||
        (!strcmp("hpu::randperm_out_ds", node_qual_str) && 1 == input_idx)) {
      // Create the seed tensor
      // TODO : check for the generator when the generator could be passed
      // as an IValues
      at::Tensor seed_tensor =
          RandpermOperator::GenerateAndCopySeedToHPU(stack, true);

      auto& syn_tensor =
          habana_op->AllocateSynapseInput(*syn_graph_ptr, seed_tensor, true);

      std::ostringstream oss;
      oss << "%dma_input" << '_' << dma_input_idx;
      dma_input_idx++;
      std::string irn{oss.str()};
      PtTensorInfoShared ti = std::make_shared<PtTensorInfo>(
          seed_tensor,
          syn_tensor.name(),
          irn,
          watch_tensor_flag_,
          syn_tensor.id(),
          DATA_TENSOR,
          habana_op->getDMAInputGeneratorType());
      ivalue_to_tensor_info_map[value_to_ivalue[value_in]] = ti;
      auto dma_tensor_idx = aten_dma_inputs.size();
      ti->set_dma_tensor_idx(dma_tensor_idx);
      dma_input_tensorinfos.emplace_back(ti);
      // Saving as persistent intermediate tensor
      aten_dma_inputs.push_back(seed_tensor);
      input_idx++;
    }
  } // for (const auto value_in : node_ins)
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
      out_syntensor.tensor_type());
  ti->set_external(out_syntensor.is_external());
  ivalue_to_tensor_info_map[ivpsh] = ti;
  void* buffp = ti->get_buffer_start();

  if (false == isInGraphOutputs(vp)) {
    if (ti->is_ZST() == false && buff_to_input_ivpsh_map.count(buffp)) {
      // Case 1.A: intermediate persistent tensor which an alias of an input
      PT_BRIDGE_DEBUG("Adding to duplicate_input_tivs ", ti);
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
    if (!enable_caching_) {
      // Case 2.A: graph output tensor
      output_tensorinfos.emplace_back(ti);
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
    OutputShapeInfRetType& outputs) {
  auto output_nodes = node->outputs();
  auto habana_kernel_meta_data = habana_op->GetKernelMetaData();

  bool shape_inf_flag = enable_fast_shape_inf_ &&
      syn_graph_ptr->is_dynamic_graph() &&
      !(m_map_shape.m_pass == ShapeInfo::InferencePass::MIN_SHAPE ||
        m_map_shape.m_pass == ShapeInfo::InferencePass::MAX_SHAPE);

  if (shape_inf_flag && !outputs.empty()) {
    HABANA_ASSERT(
        habana_op->GetSynOutputs().size() == outputs.GetOutputTensor().size(),
        "For node ",
        node->kind().toQualString(),
        "GetSynOutputs().size()=",
        habana_op->GetSynOutputs().size(),
        ", whereas GetOutputTensor().size()=",
        outputs.GetOutputTensor().size());
  }

  if (node->output(0)->type() == torch::ListType::ofTensors() &&
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
        if (shape_inf_flag) {
          if (!outputs.empty()) {
            auto output = outputs.GetOutputTensor().at(output_tensor_idx);
            auto output_sif_tidx{std::get<0>(output)};
            PT_TEST_DEBUG_TH(
                "Adding output tensor to sif_tidx_to_tinfo_map : ",
                output_sif_tidx,
                " -> ",
                *ti);
            sif_tidx_to_tinfo_map.insert({output_sif_tidx, ti});
          } else {
            // outputs is empty
            PT_TEST_DEBUG_TH(
                "Adding output tensor to sif_tidx_to_tinfo_map : ",
                cur_sif_tidx,
                " -> ",
                *ti);
            sif_tidx_to_tinfo_map.insert({cur_sif_tidx, ti});
          }
        }
      }

      SharedSynTensorOrRefListPtr tensorList =
          std::make_shared<SynTensorOrRefList>();
      tensorList->emplace_back(tensor_or_ref(out_tensor_syn));
      pt_to_synapse_tensors.emplace(
          value_to_ivalue[output_nodes[output_nodes_idx]], tensorList);

      // Validate external flag was set correctly
      const auto& value = output_nodes.at(output_tensor_idx);
      bool required_external = persistence_marker_pass_data_ptr_.get()
          ? persistence_marker_pass_data_ptr_->IsExternalNode(value)
          : false;
      if (required_external) {
        HABANA_ASSERT(
            out_tensor_syn.is_external() == required_external,
            "Output ",
            output_tensor_idx,
            " of node ",
            node->kind().toQualString(),
            " is not external");
      }

      output_nodes_idx++;
    }
    output_tensor_idx++;
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
    torch::jit::Node* node,
    std::vector<size_t>& intermediate_shape_tensors,
    std::vector<size_t>& inputs_shape_tensors) {
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
          PT_TEST_DEBUG_TH("auto_gen path: intermediate shape tensor : ", *ti);
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
          PT_TEST_DEBUG_TH(
              "manual path: adding intermediate shape tensor for index = ",
              intermediate_shape_tensors.back(),
              " : ",
              *ti);
        } else {
          inputs_shape_tensors.emplace_back(shape_tensor_tinfos.size());
          PT_TEST_DEBUG_TH(
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
        habana_op, node, intermediate_shape_tensors, inputs_shape_tensors);
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

    PtTensorInfoShared ti = std::make_shared<PtTensorInfo>(
        value_to_ivalue[value_in],
        meta_syn_tensors.back().name(),
        value_in,
        watch_tensor_flag_,
        meta_syn_tensors.back().id(),
        meta_syn_tensors.back().tensor_type());
    ivalue_to_tensor_info_map[value_to_ivalue[value_in]] = ti;
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
        syn_tensor.tensor_type());
    ti->set_restrided(true);

    value_to_ivalue.erase(value_in);

    if (enable_caching_) {
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
        auto ivptrsh_updated = jit_graph_and_meta_data->get_prim_nodes_ival(
            prim_nodes_ival_counter);
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

void HabanaLaunchOpPT::validateOutputShapeDynamic(
    const HabanaOperatorPtr& HabanaKernel,
    const OutputShapeInfRetType& output_shape_handle,
    const std::string& opname) {
  auto lowering_kernels = HabanaKernel->GetKernels();
  auto output_shape_kernels = output_shape_handle.GetKernels();
  HABANA_ASSERT(
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

void HabanaLaunchOpPT::BuildSynapseGraph(
    synapse_helpers::graph& syn_graph,
    bool is_shape_inference) {
  PT_BRIDGE_BEGIN;
  // figure out the right device id
  auto& device = synapse_helpers::HPURegistrar::get_device();
  synDeviceId device_id = device.id();

  synapse_helpers::detail::tensor_name_generator::reset();

  syn_graph_ptr = &syn_graph;

  if (refine_ds_enabled_) {
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
    if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) != 0) {
      persistence_marker_pass_data_ptr_ =
          std::move(PersistenceMarkerPass(this).VisitGraph(jit_ir_graph));
    }
  }

  habana::ShapeInference::ResetSifTensorId();

  size_t outputs_metadata_index = 0;
  // Collect inputs shape tensors accross all nodes
  std::vector<size_t> inputs_shape_tensors_vec;
  for (auto* node : graph_nodes) {
    std::vector<IdxTensorTup> intermediate_shape_tensor_cs;
    std::vector<size_t> intermediate_shape_tensors;
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

    PT_BRIDGE_DEBUG("Going to add ", *node);

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
    GetSynapseInputs(HabanaKernel, node, input_stack);

    // setup the config params for the kernels
    if (is_jit_cached_graph_info_available == false) {
      auto outputs_metadata = nodeOutputMetaData(node);
      jit_graph_and_meta_data->set_outputs_metadata(outputs_metadata);
    }
    PT_BRIDGE_DEBUG(DumpNodeInputs(node));
    OutputMetaDataVector& outputs_metadata =
        jit_graph_and_meta_data->get_outputs_metadata(outputs_metadata_index);
    outputs_metadata_index++;

    habana::OutputShapeInfRetType kernel_output_cs(true);
    if ((outputs_metadata.size() == 1) && (!is_shape_inference) &&
        (outputs_metadata.at(0).persistent == true) &&
        (std::string(opname).find("strided_insert") != std::string::npos)) {
      ProcessStridedInsertAtOutput(
          node, HabanaKernel, input_stack, syn_graph, outputs_metadata);
    } else {
      HabanaKernel->AllocateAndAddSynapseNode(
          syn_graph, input_stack, outputs_metadata);
      if ((enable_fast_shape_inf_ ||
           GET_ENV_FLAG_NEW(PT_HPU_VALIDATE_COMPUTE_SHAPE)) &&
          (syn_graph.is_dynamic_graph()
               ? !(m_map_shape.m_pass == ShapeInfo::InferencePass::MIN_SHAPE ||
                   m_map_shape.m_pass == ShapeInfo::InferencePass::MAX_SHAPE)
               : true)) {
        PT_TEST_DEBUG_TH(
            "Current sif tensor id = ",
            habana::ShapeInference::GetSifTensorId());
        HabanaOperatorPtr csHabanaKernel =
            KernelRegistry().get(device_id, op, getNodeScalarType(node));
        kernel_output_cs = csHabanaKernel->ComputeOutputShape(input_stack);
        if (!kernel_output_cs.empty()) {
          // Output shape info based flow
          PT_TEST_DEBUG_TH(
              "After ComputeOutputShape for ",
              node_qual_str,
              ": sif tensor id = ",
              habana::ShapeInference::GetSifTensorId());
          if (enable_fast_shape_inf_ && !is_shape_inference &&
              syn_graph.is_dynamic_graph()) {
            ProcessShapeTensorsCS(
                kernel_output_cs, intermediate_shape_tensor_cs);
          }
          validateOutputShape(
              HabanaKernel, kernel_output_cs, syn_graph, opname);
        }
      }
    }

    jit_to_synapse_node_idx_map.emplace(
        node, syn_graph_ptr->get_node_indices());
    syn_graph_ptr->clear_node_indices();

    if (refine_ds_enabled_ && (!is_shape_inference)) {
      ProcessSynapseShapeTensors(
          HabanaKernel,
          node,
          intermediate_shape_tensors,
          inputs_shape_tensors_vec);
      if (enable_fast_shape_inf_ && syn_graph.is_dynamic_graph() &&
          !kernel_output_cs.empty()) {
        size_t index = 0;
        HABANA_ASSERT(
            intermediate_shape_tensors.size() ==
                intermediate_shape_tensor_cs.size(),
            "intermediate_shape_tensors.size=",
            intermediate_shape_tensors.size(),
            " not matching with intermediate_shape_tensor_cs.size=",
            intermediate_shape_tensor_cs.size());
        for (auto& idx : intermediate_shape_tensors) {
          auto tensor_idx = std::get<0>(intermediate_shape_tensor_cs[index++]);
          PT_TEST_DEBUG_TH(
              "Adding entry for intermediate tensor to sif_tidx_to_tinfo_map : ",
              tensor_idx,
              " -> ",
              *shape_tensor_tinfos[idx]);
          sif_tidx_to_tinfo_map.insert({tensor_idx, shape_tensor_tinfos[idx]});
        }
      }
    }

    // Get the output tensors created back from the kernel and do the
    // subsequent processing.
    // We set type so that the created tensor is propagated throughout graph
    auto cur_sif_tidx =
        ProcessSynapseOutputs(HabanaKernel, node, kernel_output_cs);

    // HybridSif specific
    if (refine_ds_enabled_ && enable_fast_shape_inf_ &&
        syn_graph.is_dynamic_graph() && !is_shape_inference &&
        kernel_output_cs.empty()) {
      // Increment the sif tensor id
      auto output_count = get_output_tensors_count(HabanaKernel, syn_graph);
      habana::ShapeInference::IncrementSifTensorId(output_count);
      PT_TEST_DEBUG_TH(
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
              opname,
              " as persistent intermediate");
        }
      }
    }

    if (!is_shape_inference && habana_lazy::IsCollective(node->kind())) {
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
  if (refine_ds_enabled_ && enable_fast_shape_inf_ &&
      syn_graph.is_dynamic_graph() && !is_shape_inference) {
    for (size_t i = 0; i < jit_ir_graph->inputs().size(); ++i) {
      auto input = jit_ir_graph->inputs().at(i);
      HABANA_ASSERT(value_to_ivalue.count(input));
      auto input_ivalue = value_to_ivalue[input];
      HABANA_ASSERT(ivalue_to_tensor_info_map.count(input_ivalue));
      auto tensor_idx = habana::ShapeInference::ReadAndIncrementSifTensorId();
      PT_TEST_DEBUG_TH(
          "Adding entry for input tensor to sif_tidx_to_tinfo_map : ",
          tensor_idx,
          " -> ",
          *ivalue_to_tensor_info_map[input_ivalue]);
      sif_tidx_to_tinfo_map.insert(
          {tensor_idx, ivalue_to_tensor_info_map[input_ivalue]});
    }

    // Generate patching info for inputs shape tensors during fast sif
    for (auto const& idx : inputs_shape_tensors_vec) {
      auto tensor_idx = habana::ShapeInference::ReadAndIncrementSifTensorId();
      PT_TEST_DEBUG_TH(
          "Adding entry for input shape tensor to sif_tidx_to_tinfo_map : ",
          tensor_idx,
          " -> ",
          *shape_tensor_tinfos[idx]);
      sif_tidx_to_tinfo_map.insert({tensor_idx, shape_tensor_tinfos[idx]});
    }
  }

  // allow permutation only for output tensors
  if (GET_ENV_FLAG_NEW(PT_HPU_ENABLE_SYNAPSE_LAYOUT_HANDLING) &&
      GET_ENV_FLAG_NEW(PT_HPU_ENABLE_SYNAPSE_OUTPUT_PERMUTE) &&
      !GET_ENV_FLAG_NEW(PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES)) {
    for (auto ti : output_tensorinfo_map) {
      auto ival = ti.first;
      auto iter = pt_to_synapse_tensors.find(ival);
      HABANA_ASSERT(pt_to_synapse_tensors.count(ival));
      if (iter != pt_to_synapse_tensors.end()) {
        auto syn_vec = (iter->second);
        auto& out_syntensor = (*syn_vec)[0];
        if (iter->second->size() != 1) {
          PT_BRIDGE_DEBUG(
              "Not setting synapse allow permutation on tensor: ",
              out_syntensor.ref().id(),
              " because the PT tensor is mapped to multiple synapse tensors");
          continue;
        }
        auto rank = out_syntensor.ref().pt_shape().size();
        if (rank != 5 && rank != 4) {
          PT_BRIDGE_DEBUG(
              "Not setting synapse allow permutation on tensor: ",
              out_syntensor.ref().id(),
              " because the PT tensor rank is not 4 or 5. other ranks are not supported. current rank: ",
              rank);
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
        auto syn_vec = (iter->second);
        auto& out_syntensor = (*syn_vec)[0];
        if (iter->second->size() != 1) {
          PT_BRIDGE_DEBUG(
              "Not setting synapse allow permutation on tensor: ",
              out_syntensor.ref().id(),
              " because the PT tensor is mapped to multiple synapse tensors");
          continue;
        }
        auto rank = out_syntensor.ref().shape().rank().value;
        if (rank != 5 && rank != 4) {
          PT_BRIDGE_DEBUG(
              "Not setting synapse allow permutation on tensor: ",
              out_syntensor.ref().id(),
              " because the PT tensor rank is not 4 or 5. other ranks are not supported. current rank: ",
              rank);
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
    auto input = input_refs[i];
    if (input.isTensor()) {
      at::Tensor pt_tensor = input.toTensor();
      habana_helpers::TensorShape shape(
          pt_tensor.sizes(), pt_tensor.scalar_type());
      auto impl = habana_lazy::GetHbInternalTensorImpl(pt_tensor);
      HABANA_ASSERT(impl);
      shape.set_tensor_type(impl->getTensorType());
      shape_map[i] = shape;
    }
  }
}

void HabanaLaunchOpPT::CreateValueToIvalueMapForInputs() {
  PT_BRIDGE_BEGIN;
  for (size_t j = 0; j < pt_stack_sh.size(); j++) {
    auto value_input = jit_ir_graph->inputs().at(j);
    auto ivpsh = pt_stack_sh[j];
    value_to_ivalue[value_input] = ivpsh;
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
  PT_BRIDGE_BEGIN;
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
            static_cast<synStreamHandle>(
                syn_device.get_compute_stream(hpu_stream)));
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
  PT_BRIDGE_END;
}

void HabanaLaunchOpPT::EvictSynapseRecipe(size_t& dsi_bucket_id) {
  size_t num_recipes = 1;
  bool dropped{true};
  // Keep evicting recipes until the memory usage goes below threshold
  while (dropped && habana::IsHostMemoryThresholdReached()) {
    dropped = dropCachedRecipe_LRU(num_recipes);
    if (dropped) {
      auto dropped_arg = RecipeCacheLRU::get_cache().dropped_recipe.first;
      auto dropped_val = RecipeCacheLRU::get_cache().dropped_recipe.second;
      auto dropped_dbi = DynamicBucketInfoMap::get_instance().get(dropped_arg);
      auto dropped_bid = dropped_dbi->EvictBucket(dropped_val);
      if ((dropped_dbi == current_dbipsh_) &&
          dropped_bid < current_bucket_id_) {
        current_bucket_id_ -= 1;
        dsi_bucket_id = current_bucket_id_;
      }
    }
  }
}

void HabanaLaunchOpPT::ProcessHabanaFusedOpWithDS() {
  PT_BRIDGE_BEGIN;

  RecipeCacheLRU::SetHostMemoryThreshold();

  std::shared_ptr<RecipeArgumentSpec> rargpsh_graph =
      std::make_shared<RecipeArgumentSpec>(input_refs, graph_key, op_strs);
  PT_DYNAMIC_SHAPE_DEBUG(
      "====================\n",
      "Processing with dynamic shape enabled\n",
      "JIT IR graph_hash_code : ",
      rargpsh_graph->graphHashCode());

  current_dbipsh_ = DynamicBucketInfoMap::get_instance().get(rargpsh_graph);
  if (nullptr == current_dbipsh_) {
    PT_DYNAMIC_SHAPE_DEBUG("Creating new DynamicBucketInfo");
    current_dbipsh_ = std::make_shared<habana_helpers::DynamicBucketInfo>(
        rargpsh_graph->graphHashCode());
    DynamicBucketInfoMap::get_instance().add(rargpsh_graph, current_dbipsh_);
    current_dbipsh_->create_statistics(
        habana_helpers::CompilationStatistics::Create(
            GetSynapseGraphName(), current_dbipsh_->getCount()));
  }

  DynamicShapeInfo graph_input_info;
  CreateDynamicBucketInputShapes(graph_input_info.act_input_tshapes);
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

  cur_rargpsh = std::make_shared<RecipeArgumentSpec>(
      input_refs, graph_key, op_strs, cur_ds_token_);
  DynamicBucketInfoMap::get_instance().add(cur_rargpsh, current_dbipsh_);

  // Check for cached recipe
  if (enable_caching_) {
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
        PT_TEST_DEBUG(
            "Graph: ",
            name,
            '_',
            graph_index,
            ", graph_key: ",
            rargpsh_graph->graphHashCode(),
            ", recipe cache hit, recipe_key: ",
            cur_rargpsh->hashCode());
        PT_DYNAMIC_SHAPE_DEBUG("Running output shape inference pass");
        if (enable_fast_shape_inf_) {
          PT_DYNAMIC_SHAPE_DEBUG("HybridSif_BEGIN");

          habana::ShapeInference::ResetSifTensorId();
          RunHybridSif(tidx_to_tensor_map);
          PT_DYNAMIC_SHAPE_DEBUG("HybridSif_END");
        } else {
          PT_DYNAMIC_SHAPE_DEBUG("OutputSif_BEGIN");
          try_run_shape_inference(
              ShapeInfo::InferencePass::OUTPUT_SHAPE, graph_input_info);
          PT_DYNAMIC_SHAPE_DEBUG("OutputSif_END");
        }
        current_dbipsh_->get_statistics()->LogUsedBucket(
            current_bucket_id_, jit_ir_graph, ranges, 0);
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
          tidx_to_tensor_map);

      if (enable_tensor_dump_) {
        DumpTensors_pre(rv);
      }

      {
        std::lock_guard<std::mutex> lg(current_dbipsh_->get_refine_mutex());
        // Initiate recipe execution time collection
        InitiateSynlaunchTimeCapture(rv);
      }

      rv.launch(
          hpu_stream, input_refs, intermediate_tensors_ptr, dma_inputs_ptr);
      if (enable_tensor_dump_) {
        DumpTensors(rv);
      }

      // Update the stack from the recipe itself
      UpdateOutputs(rv);
      ReturnCachedRecipe(rv);
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

      current_dbipsh_->get_statistics()->DumpAndNextStep();
      ClearMembers();
      ClearStatics();

      RefinementEngine::GetEngine().AddGraphKey(rargpsh_graph->graphHashCode());
      PT_BRIDGE_END;
      return;
    } else {
      PT_DYNAMIC_SHAPE_DEBUG(
          "HabanaOp recipe cache miss :: key ", cur_rargpsh->hashCode());
      PT_IRGRAPH_DEBUG("HabanaOp recipe cache miss :: dynamic shapes");
    }
  }

  CompileAndRunDynamicGraph(graph_input_info);
  habana_helpers::DynamicBucketInfo::inc_original_recipe_count();

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
    std::string map_name) {
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
  if (GET_ENV_FLAG_NEW(PT_HPU_ENABLE_SYNAPSE_LAYOUT_HANDLING)) {
    auto impl = habana_lazy::GetHbInternalTensorImpl(parent_tensor);
    if (impl) {
      if (!ti.get_allow_permutation()) {
        impl->SetMemoryPermutation({});
        PT_BRIDGE_DEBUG(
            "Resetting tensor ",
            ti.get_tensor_id(),
            " permutation because it is not allowed permutation (cache hit flow)");
      } else {
        PT_BRIDGE_DEBUG(
            "Setting tensor ",
            ti.get_tensor_id(),
            " permutation from the TensorInfo cache record: ",
            VecToString(ti.getHbInternalPermute()),
            " old permutation was: ",
            impl->GetMemoryPermutation());
        impl->SetMemoryPermutation(ti.getHbInternalPermute());
      }
    }
    pt_outdup = parent_tensor;
  } else {
    pt_outdup =
        at::as_strided(parent_tensor, pt_sizes, pt_strides, pt_opt_offset);
  }

  ti.patch(pt_outdup);

  IValPtrShared ivpsh = std::make_shared<IVal>(pt_outdup);
  aten_outputs->at(output_idx) = ivpsh;
}

void HabanaLaunchOpPT::ReturnCachedRecipe(RecipeValueSpec& rv) {
  PT_BRIDGE_BEGIN;
  rv.set_use_flag(false);
  PT_BRIDGE_END;
}

void HabanaLaunchOpPT::run(torch::jit::Stack& input_st) {
  PT_BRIDGE_BEGIN;
  static int idx{1};
  ProcessInputStack(input_st);

  iteration_count_++;
  auto& device = synapse_helpers::HPURegistrar::get_device();

  PT_BRIDGE_DEBUG(
      "Lowering:\n",
      "JIT_IR_Graph_BEGIN\n",
      "Graph ",
      idx,
      '\n',
      jit_ir_graph->toString(),
      "JIT_IR_Graph_END\n");
  idx += 1;

  // Handle everything related to graph when dynamic flag is set.
  if (refine_ds_enabled_) {
    jit_graph_and_meta_data->clear_cached_graph_info();
    jit_graph_and_meta_data->set_jit_cached_graph_info_available_flag(
        false); // Disable Optimized Lowering based on Cached precalculated
                // graph information.
    ProcessHabanaFusedOpWithDS();
    return;
  }

  if (enable_caching_ || IS_BRIDGE_DEBUG_ENABLED || refine_ds_enabled_) {
    cur_rargpsh = std::make_shared<RecipeArgumentSpec>(
        false, input_refs, jit_ir_graph, graph_key, op_strs);
  }

  // caching :: begin
  if (enable_caching_) {
    cur_rvalpsh = GetCachedRecipe(cur_rargpsh);

    if (ABSL_PREDICT_TRUE(cur_rvalpsh)) {
      RecipeValueSpec& rv = *cur_rvalpsh;
      rv.update_hit_count();

      PT_BRIDGE_DEBUG(
          id_str,
          ": ",
          "HabanaOp recipe cache hit :: key ",
          cur_rargpsh->hashCode(),
          "\n",
          rv.header_str(),
          "\n",
          rv.digest_str());
      PT_IRGRAPH_DEBUG("HabanaOp recipe cache hit :: static shapes");

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
      rv.launch(
          hpu_stream, input_refs, intermediate_tensors_ptr, dma_inputs_ptr);

      if (enable_tensor_dump_) {
        DumpTensors(rv);
      }

      // Update the stack from the recipe itself
      UpdateOutputs(rv);
      ReturnCachedRecipe(rv);

      ClearStatics();
      PT_BRIDGE_END;
      return;
    } else {
      PT_BRIDGE_DEBUG(
          id_str,
          ": ",
          "HabanaOp recipe cache miss :: key ",
          cur_rargpsh->hashCode());
      PT_IRGRAPH_DEBUG("HabanaOp recipe cache miss :: static shapes");
    }
  }
  // caching :: end

  bool is_jit_cached_graph_info_available =
      jit_graph_and_meta_data->get_jit_cached_graph_info_available_flag();
  if (is_jit_cached_graph_info_available == false) {
    jit_graph_and_meta_data->clear_cached_graph_info();
  }

  CreateValueToIvalueMapForInputs();

  auto syn_graph =
      habana_helpers::create_graph(device.id(), GetSynapseGraphName());
  BuildSynapseGraph(syn_graph);
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

  PT_BRIDGE_END;
}

void HabanaLaunchOpPT::CompileGraphWithRange(
    torch::jit::Stack& input_st,
    habana_helpers::ResultShapes& input_ranges,
    habana_helpers::Bucket& new_bucket,
    size_t& new_recipe_key,
    std::shared_ptr<habana_helpers::CompilationStatistics> statpsh) {
  PT_BRIDGE_BEGIN;
  ProcessInputStack(input_st);

  PT_DYNAMIC_SHAPE_DEBUG(
      "Input range for new bucket:\n",
      "Min\n",
      input_ranges.min_shapes,
      "Max\n",
      input_ranges.max_shapes,
      "--------------------");

  CreateValueToIvalueMapForInputs();

  DynamicShapeInfo graph_input_info;
  CreateDynamicBucketInputShapes(graph_input_info.act_input_tshapes);

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

  auto& device = synapse_helpers::HPURegistrar::get_device();
  std::string graphName{GetSynapseGraphName()};

  auto create_graph_for_refinement{[&]() -> synapse_helpers::graph {
    auto graph_or_error =
        synapse_helpers::graph::create_for_refinement(device, name);

    if (absl::holds_alternative<synapse_helpers::synapse_error>(
            graph_or_error)) {
      auto error = absl::get<synapse_helpers::synapse_error>(graph_or_error);
      TORCH_CHECK(error.status, error.error);
    }
    return absl::get<synapse_helpers::graph>(std::move(graph_or_error));
  }};

  auto syn_graph = create_graph_for_refinement();

  // Compile the graph
  uint64_t current_step{statpsh->GetCurrentStep()};
  {
    CreateValueToIvalueMapForInputs();

    syn_graph.set_dynamic_graph(true);

    std::string error_str;
    try {
      cur_ds_token_ = new_bucket.getToken();
      cur_rargpsh = std::make_shared<RecipeArgumentSpec>(
          input_refs, graph_key, op_strs, cur_ds_token_);
      new_recipe_key = cur_rargpsh->hashCode();
      statpsh->LogRefineCompilation(
          input_ranges,
          jit_ir_graph,
          new_recipe_key,
          new_bucket.GetIndex(),
          current_step);

      m_map_shape.m_pass = ShapeInfo::InferencePass::INVALID;
      BuildSynapseGraph(syn_graph);
      CompileSynapseGraph();
      ConstructPatchingTable();
    } catch (std::exception& e) {
      error_str = e.what();
      PT_DYNAMIC_SHAPE_DEBUG(
          "Exception occured in compilation - Details :\n", error_str);

      std::string result_str{"FAIL"};
      statpsh->LogRefineResult(result_str, current_step);

      throw;
    }
  }
  PT_DYNAMIC_SHAPE_DEBUG("Compilation completed");

  // Add the <key,value> pair to the map
  cur_rvalpsh->dynamic_graph = syn_graph.is_dynamic_graph();
  cur_rvalpsh->set_op_strs(cur_rargpsh->get_op_strs());
  RecipeCacheLRU::get_cache().add(cur_rargpsh, cur_rvalpsh);

  new_recipe_key = cur_rargpsh->hashCode();
  // Add the recipe to the corresponding bucket
  new_bucket.SetSynapseRecipePtr(cur_rvalpsh);

  std::string result_str{"OK"};
  statpsh->LogRefineResult(result_str, current_step);

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
  auto& device = synapse_helpers::HPURegistrar::get_device();

  //
  // Run the compile and execute method to infer the shapes
  auto syn_graph =
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
  PT_DYNAMIC_SHAPE_DEBUG("Handling the exception .. ");
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
        PT_DYNAMIC_SHAPE_FATAL("Unhandled exception exiting .. ");
        throw std::runtime_error("Exception was not handled ..");
      }
      graph_input_info.min_policy = habana_helpers::DynamicDimsPolicy::CURRENT;
      graph_input_info.max_policy = habana_helpers::DynamicDimsPolicy::CURRENT;
      break;
    }
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
      auto syn_graph =
          habana_helpers::create_graph(device.id(), GetSynapseGraphName());
      syn_graph.set_dynamic_graph(is_dynamic_graph);
      BuildSynapseGraph(syn_graph);
      CompileSynapseGraph();
      ConstructPatchingTable();
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
    if (last_compilation_pass != habana_helpers::CompilationPass::STATIC) {
      current_dbipsh_->get_statistics()->LogUsedBucket(
          graph_input_info.current_bucket_id, jit_ir_graph, ranges, 0);
    }
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

    if (node_str.find("strided_insert") == std::string::npos) {
      break;
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

} // namespace habana
