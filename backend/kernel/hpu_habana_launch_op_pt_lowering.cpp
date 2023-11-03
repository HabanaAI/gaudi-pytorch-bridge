/*******************************************************************************
 * Copyright (C) 2021-2023 Habana Labs, Ltd. an Intel Company
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
#include <cstdint>
#include "backend/backend_meta.h"
#include "backend/habana_device/hpu_cached_devices.h"
#include "backend/helpers/runtime_config.h"
#include "backend/kernel/hpu_habana_launch_op_pt.h"
#include "backend/synapse_helpers/env_flags.h"
#include "backend/synapse_helpers/tcmalloc_helper.h"
#include "habana_helpers/logging.h"
#include "habana_kernels/hccl_kernels.h"
#include "hpu_habana_launch_op_pt.h"

void habana::HabanaLaunchOpPT::CopyInputStack(torch::jit::Stack& input_st) {
  // Keep a handle to the stack for future use
  pt_stack = &input_st;
  input_tms.reserve(pt_stack->size());
  pt_stack_sh.reserve(pt_stack->size());

  for (size_t i = 0; i < pt_stack->size(); i++) {
    if (pt_stack->at(i).isTensor()) {
      auto& t{pt_stack->at(i).toTensor()};
      input_tms.emplace_back(
          t.sizes().vec(), t.strides().vec(), t.suggest_memory_format());
    } else {
      input_tms.emplace_back(
          std::vector<int64_t>(),
          std::vector<int64_t>(),
          c10::MemoryFormat::Preserve);
    }

    IValPtrShared ivpsh = std::make_shared<IVal>(pt_stack->at(i));
    pt_stack_sh.emplace_back(ivpsh);
    if (ivpsh->isTensor() || ivpsh->isTensorList()) {
      num_tensor_inputs++;
    }
  }
}

void habana::HabanaLaunchOpPT::ClearMembers(bool is_shape_inference) {
  if (is_shape_inference == false) {
    pt_stack = nullptr;
    pt_stack_sh.clear();
    num_tensor_inputs = 0;
  }

  jit_graph_and_meta_data_->clear_cached_graph_info();
  prim_nodes_ival_counter = 0;
  restride_node_swap_counter = 0;
  restride_node_out_val_counter = 0;

  value_to_ivalue.clear();
  syn_graph_ptr_ = nullptr;
  cur_rvalpsh = nullptr;
  intermediate_tensors_ptr_sh_ = nullptr;
  aten_outputs_ptr_sh_ = nullptr;

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

  output_tensorinfo_map.clear();

  pt_to_synapse_tensors.clear();
  meta_syn_tensors.clear();
  buff_to_input_ivpsh_map.clear();
  buff_to_intermediate_ivpsh_map.clear();
  buff_to_output_ivpsh_map.clear();
  buff_to_syn_tensor_map.clear();

  jit_to_synapse_node_idx_map.clear();
  collective_kernels_info.clear();
  syn_launch_info_.clear();
  external_tensor_info_indexes_.clear();
  dma_inputs_.clear();
}

void habana::HabanaLaunchOpPT::ClearStatics(bool is_shape_inference) {
  if (is_shape_inference == false) {
    habana::ShapeInference::Reset();
  }
}

void habana::HabanaLaunchOpPT::ApplyOutputPermutationsFromCache() {
  for (auto& el : jit_graph_and_meta_data_->get_permute()) {
    auto oit =
        value_to_ivalue.find(jit_ir_graph_->outputs().at(el.output_index));
    HABANA_ASSERT(oit != value_to_ivalue.end());
    auto& pt_tensor = oit->second;
    HABANA_ASSERT(pt_tensor->isTensor());
    habana_helpers::set_tensor_memory_permutations(
        pt_tensor->toTensor(), el.permutation);
  }
}

/**
 * Queries the synapse recipe output permutations and sets it to the BE tensors
 * so that the permutation is taken into account when copying the tensor back to
 * the host or passing it to the next graph.
 */
void habana::HabanaLaunchOpPT::UpdateSynapsePermutations() {
  PT_LAZY_TRACE;

  if (syn_graph_ptr_->is_empty()) {
    PT_BRIDGE_DEBUG("Empty synapse graph. Skip UpdateSynapsePermutations.");
    return;
  }
  auto tinfos = cur_rvalpsh->dtensorinfos;
  if (!tinfos) {
    PT_BRIDGE_DEBUG("empty cur_rvalpsh->dtensorinfos, nothing to update");
    return;
  }

  auto permutation_info_saver = std::move(permutation_info_saver_);

  std::function<void(
      const at::Tensor&, synapse_helpers::layouts::MemoryPermutation, uint64_t)>
      set_memory_permutations =
          [](const at::Tensor& tensor,
             synapse_helpers::layouts::MemoryPermutation permutation,
             uint64_t) {
            habana_helpers::set_tensor_memory_permutations(tensor, permutation);
          };

  if (jit_graph_and_meta_data_->is_permute_set()) {
    set_memory_permutations = [](const at::Tensor&,
                                 synapse_helpers::layouts::MemoryPermutation,
                                 uint64_t) {};
  }

  if (permutation_info_saver) {
    set_memory_permutations =
        [&](const at::Tensor& tensor,
            synapse_helpers::layouts::MemoryPermutation permutation,
            uint64_t output_index) {
          permutation_info_saver->add_permutation(output_index, permutation);
          habana_helpers::set_tensor_memory_permutations(tensor, permutation);
        };
  }

  // creating an opposite map to be able to find the tensors to update
  std::unordered_map<uint64_t, IValPtrShared> synapse_to_pt_tensor;
  for (auto& el : pt_to_synapse_tensors) {
    for (synapse_helpers::tensor& tensor : *el.second) {
      synapse_to_pt_tensor.insert({tensor.id(), el.first});
    }
  }
  if (GET_ENV_FLAG_NEW(PT_HPU_ENABLE_SYNAPSE_OUTPUT_PERMUTE)) {
    std::map<uint64_t, uint64_t> persistent_to_tensor_id;
    std::vector<synRetrievedLaunchTensorInfoExt> tensor_info_vec;
    // creating a map of tensor id to tinfo
    // preparing the tensors to query their permutation
    std::map<uint64_t, PtTensorInfoShared> tinfo_map;
    for (size_t i = 0; i < tinfos->size(); ++i) {
      auto& info = (*tinfos)[i];
      if (info->is_output() && !info->is_ZST()) {
        // HABANA_ASSERT(tinfo_map.count(info->get_tensor_id() == 0));
        tinfo_map[info->get_tensor_id()] = info;
        if (info->get_allow_permutation()) {
          synRetrievedLaunchTensorInfoExt record = {};
          record.tensorId = cur_rvalpsh->tensor_ids[i];
          PT_BRIDGE_DEBUG(
              "preparing to query tensor: ",
              info->get_tensor_id(),
              " persistent tensor id: ",
              record.tensorId);
          persistent_to_tensor_id[record.tensorId] = info->get_tensor_id();
          tensor_info_vec.push_back(record);
        }
      }
    }
    // querying synapse output tensors permutations:
    synapse_helpers::graph::query_recipe_tensor_info(
        cur_rvalpsh->recipe, tensor_info_vec);

    // updating the BE tensor and the cache record with the permutation
    for (auto& info : tensor_info_vec) {
      HABANA_ASSERT(persistent_to_tensor_id.count(info.tensorId));
      auto tensor_id = persistent_to_tensor_id[info.tensorId];
      if (info.tensorType == TENSOR_TYPE_INVALID) {
        PT_BRIDGE_DEBUG(
            "Synapse returned a TENSOR_TYPE_INVALID when querying the persistent tensors for permutations, in tensor: ",
            tensor_id,
            " . It means that the synapse tensor is not in the recipe, probably not attached to a node");
        continue;
      }
      std::vector<uint8_t> permute_vec(
          info.tensorPermutation, info.tensorPermutation + info.tensorDims);
      // if this is an identity permutation we set empty permute
      bool is_identity_perm = true;
      for (size_t i = 0; i < permute_vec.size() - 1; ++i) {
        if (permute_vec[i] + 1 != permute_vec[i + 1]) {
          PT_BRIDGE_DEBUG("Detected a real permutation (not identity)");
          is_identity_perm = false;
          break;
        }
      }
      auto permute_or_empty =
          is_identity_perm ? std::vector<uint8_t>() : permute_vec;
      PT_BRIDGE_DEBUG(
          "Synapse returned persistent tensorId=",
          info.tensorId,
          " which is bridge tensor id: ",
          tensor_id,
          "; info.tensorPermutation = {",
          VecToString(permute_vec),
          "}\n");

      auto iter = synapse_to_pt_tensor.find(tensor_id);
      TORCH_CHECK(
          iter != synapse_to_pt_tensor.end(),
          "Failed to find PT tensor to update permutation");
      TORCH_CHECK(
          iter->second->isTensor(),
          "Update permutation on non-tensor output is not supported");

      // update the dttensorinfo record to update the cache
      HABANA_ASSERT(tinfo_map.count(tensor_id));
      auto info_record = tinfo_map[tensor_id];
      if (!info_record->getHbInternalPermute().empty() &&
          permute_vec != info_record->getHbInternalPermute()) {
        PT_BRIDGE_DEBUG(
            "While trying to update PT tensor permutation, found that the PT tensor already has a permutation -  id: ",
            tensor_id,
            " persistent_id: ",
            info.tensorId,
            " existing permutation:",
            VecToString(info_record->getHbInternalPermute()),
            " new permutation: ",
            VecToString(permute_vec));
      }
      info_record->setHbInternalPermute(permute_or_empty);

      // updating the permute on the internal hb lazy tensor
      set_memory_permutations(
          iter->second->toTensor(),
          permute_or_empty,
          info_record->get_output_index());
    }
  }
  // clear the permutation of all the dtensorinfo that are not allowed
  // permutation. for example, weights tenor that serves as graph input and
  // ouput, when the allow permutation is disabled then synapse returns it dense
  // NCHW even if the input was permuted.
  for (size_t i = 0; i < tinfos->size(); ++i) {
    auto& info = (*tinfos)[i];
    if (!info->get_allow_permutation() && info->is_output()) {
      auto iter = synapse_to_pt_tensor.find(info->get_tensor_id());
      TORCH_CHECK(
          iter != synapse_to_pt_tensor.end(),
          "Failed to find PT tensor to update permutation");
      // updating the permute on the internal hb lazy tensor
      if (iter->second->isTensor()) {
        PT_BRIDGE_DEBUG(
            "Resetting tensor ",
            info->get_tensor_id(),
            " permutation because it is not allowed permutation");
        set_memory_permutations(
            iter->second->toTensor(), {}, info->get_output_index());
      }
      if (!info->getHbInternalPermute().empty()) {
        PT_BRIDGE_DEBUG(
            "While trying to reset PT tensor permutation, found that the PT tensor already has a permutation -  id: ",
            info->get_tensor_id(),
            " existing permutation:",
            VecToString(info->getHbInternalPermute()));
      }
      info->setHbInternalPermute({});
    }
  }
}

// Based on:
// synapse/tests/gaudi_tests/gaudi_test_infra.cpp
static void getTensorSectionId(
    const synRecipeHandle& recipeHandle,
    const synTensor& tensor,
    synSectionId& sectionId,
    bool& isInput) {
  synStatus status;
  uint32_t numOfTensors = 0;
  status = synTensorRetrieveLaunchAmount(recipeHandle, &numOfTensors);
  HABANA_ASSERT(
      status == synStatus::synSuccess, Logger::synStatusToStr(status));
  uint64_t ids[numOfTensors];
  status = synTensorRetrieveLaunchIds(recipeHandle, ids, numOfTensors);
  HABANA_ASSERT(
      status == synStatus::synSuccess, Logger::synStatusToStr(status));
  synRetrievedLaunchTensorInfo tensorInfos[numOfTensors];
  for (unsigned i = 0; i < numOfTensors; i++) {
    tensorInfos[i].tensorId = ids[i];
  }
  status =
      synTensorRetrieveLaunchInfoById(recipeHandle, numOfTensors, tensorInfos);
  HABANA_ASSERT(
      status == synStatus::synSuccess, Logger::synStatusToStr(status));

  // get tensor name
  char tensorName[ENQUEUE_TENSOR_NAME_MAX_SIZE];
  status = synTensorGetName(tensor, ENQUEUE_TENSOR_NAME_MAX_SIZE, tensorName);
  HABANA_ASSERT(
      status == synStatus::synSuccess, Logger::synStatusToStr(status));

  // search for tensor according to tensor name and set it's sectionId
  for (unsigned tensorIdx = 0; tensorIdx < numOfTensors; tensorIdx++) {
    if (strcmp(tensorInfos[tensorIdx].tensorName, tensorName) == 0) {
      sectionId = tensorInfos[tensorIdx].tensorSectionId;
      isInput = (tensorInfos[tensorIdx].isInput != 0);
      return;
    }
  }
  sectionId = INVALID_SECTION_ID;
  isInput = true;
}

void habana::HabanaLaunchOpPT::HandleRecipeWithNewChecksum(
    std::shared_ptr<c10::IValue> _src,
    size_t _section_size,
    size_t _checksum,
    size_t _key,
    char* _section_data_ptr,
    size_t _old_size,
    int _device_id) {
  auto& _tensor = _src->toTensor();
  auto tmeta{get_tensor_extra_meta(_tensor)};
  auto const_id = tmeta->get_const_id();
  // reallocation is required if old_size is not same as section size
  // or if old_size is same as section_size but checksum is new
  bool anyChecksumExists =
      (m_const_checksum_map.find(const_id) != m_const_checksum_map.end());
  bool reallocation_required =
      (anyChecksumExists or (_old_size != _section_size));
  if (reallocation_required) {
    PT_BRIDGE_DEBUG(
        "Needed reallocation (bridge) old_size ",
        _old_size,
        " != ",
        _section_size,
        " Checksum: ",
        _checksum);
    tmeta->set_nbytes_inference(_old_size);
    at::DataPtr data = _tensor.storage().allocator()->allocate(_section_size);
    auto old_data_ptr = _tensor.storage().set_data_ptr(std::move(data));
    _tensor.storage().set_nbytes(_section_size);
    ivalue_to_tensor_info_map[_src]->set_buffer(
        (void*)(_tensor.storage().data_ptr().get()));
    if (anyChecksumExists) {
      StorePrevDataPtr(
          const_id,
          std::move(old_data_ptr),
          m_const_checksum_map[const_id].first);
    }
  }
  InsertConstantChecksum(const_id, _checksum);
  PushConstantChecksumInfo(const_id, _checksum, _key, _section_size);
  auto& device = HPURegistrar::get_device(_device_id);
  std::atomic<bool> copyDone{false};
  device.copy_data_to_device(
      _section_data_ptr,
      reinterpret_cast<synapse_helpers::device_ptr>(_tensor.data_ptr()),
      reinterpret_cast<synapse_helpers::device_ptr>(
          _tensor.storage().data_ptr().get()),
      _section_size,
      [&copyDone]() { copyDone = true; },
      false,
      true);
  // wait for copy completion
  while (!copyDone) {
    std::this_thread::yield();
  }
}

void habana::HabanaLaunchOpPT::HandleRecipeWithExistingChecksumInCache(
    int _const_id,
    size_t _checksum,
    size_t _key,
    at::Tensor& _tensor) {
  AddRecipeForSharedChecksum(_const_id, _checksum, _key);
  GetConstPtrForRecipe(_const_id, _key, _tensor);
  InsertConstantChecksum(_const_id, _checksum);
  PT_BRIDGE_DEBUG(
      "Tensor with const_id: ",
      _const_id,
      " has moved data pointer for the data corresponding to checksum: ",
      _checksum,
      " for cache miss on key ",
      _key);
}

void habana::HabanaLaunchOpPT::HandleRecipeWithChecksumOnDevice(
    int _const_id,
    size_t _checksum,
    size_t _key) {
  AddRecipeForSharedChecksum(_const_id, _checksum, _key);
  PT_BRIDGE_DEBUG(
      "Constant tensor already exists on the device, avoiding re-copy to device, checksum: ",
      _checksum,
      " const_id: ",
      _const_id,
      " recipe key: ",
      _key);
}

void habana::HabanaLaunchOpPT::PostCompilationStepForConstTensors(
    RecipeValueSpec& rv) {
  std::vector<synSectionId> constSectionIds;
  for (auto iter = pt_to_synapse_tensors.begin();
       iter != pt_to_synapse_tensors.end();
       ++iter) {
    auto& src = iter->first->toTensor();
    PT_BRIDGE_DEBUG("tensor storage:: ", src.has_storage());
    if (src.has_storage()) {
      auto tmeta{get_tensor_extra_meta(src)};
      auto smeta{habana::get_storage_extra_meta(src)};
      synapse_helpers::layouts::MemoryPermutation permutation = {};
      auto allow = false;
      if (smeta) {
        permutation = smeta->get_memory_permutation();
        allow = smeta->get_dont_allow_permutation();
      }
      PT_BRIDGE_DEBUG("tensor is_const_tensor:  ", tmeta->is_const_tensor());
      for (synapse_helpers::tensor& tensor : *(iter->second)) {
        if (tmeta->is_const_tensor()) {
          PT_BRIDGE_DEBUG(
              "const tensor name:: ",
              tensor.name(),
              " const id: ",
              tmeta->get_const_id());
          // habana_helpers::handle_const_section_tensor(src, tensor);
          // remove the const marking to avoid copy more than once
          TensorExtraMeta::set_const_tensor(src, false);
          PT_BRIDGE_DEBUG(
              "tensor is_const_tensor:  ", tmeta->is_const_tensor());
          uint64_t section_size = 0, section_data = 0;
          synSectionId tensorSectionId;
          bool isInput;
          getTensorSectionId(
              rv.recipe->syn_recipe_handle_,
              tensor.get(),
              tensorSectionId,
              isInput);
          if (!isInput) {
            PT_BRIDGE_DEBUG("non-input tensor section ID:  ", tensorSectionId);
            continue;
          }
          HABANA_ASSERT(tensorSectionId != INVALID_SECTION_ID);
          PT_BRIDGE_DEBUG("tensor section ID:  ", tensorSectionId);
          synStatus status;
          status = synRecipeSectionGetProp(
              rv.recipe->syn_recipe_handle_,
              tensorSectionId,
              SECTION_SIZE,
              &section_size);
          HABANA_ASSERT(
              status == synStatus::synSuccess, Logger::synStatusToStr(status));
          PT_BRIDGE_DEBUG(
              "section_size:: ",
              section_size,
              " , size (bridge) :: ",
              tensor.get_host_ptr_size());
          if (section_size) {
            auto device_id = tensor.device_id();
            HABANA_ASSERT(
                status == synStatus::synSuccess,
                Logger::synStatusToStr(status));
            status = synRecipeSectionGetProp(
                rv.recipe->syn_recipe_handle_,
                tensorSectionId,
                SECTION_DATA,
                &section_data);
            char* section_data_ptr = (char*)section_data;
            HABANA_ASSERT(
                status == synStatus::synSuccess,
                Logger::synStatusToStr(status));
            auto checksum = GetDataChecksum(section_data_ptr, section_size);
            HABANA_ASSERT(
                tmeta->has_valid_const_id(),
                "Constant tensor can not have constant id as -1");
            [[maybe_unused]] auto& device = HPURegistrar::get_device(device_id);
            status = synHostMap(device_id, section_size, section_data_ptr);
            HABANA_ASSERT(
                status == synStatus::synSuccess,
                Logger::synStatusToStr(status));
            auto const_id = tmeta->get_const_id();
            auto checksum_found = DoesCheckSumExist(const_id, checksum);
            PT_BRIDGE_DEBUG(
                "const_id: ",
                const_id,
                " checksum_found: ",
                checksum_found,
                " checksum: ",
                checksum);
            if (!checksum_found) {
              auto old_size = tensor.get_host_ptr_size();
              HandleRecipeWithNewChecksum(
                  iter->first,
                  section_size,
                  checksum,
                  cur_rargpsh->hashCode(),
                  section_data_ptr,
                  old_size,
                  device_id);
              if (smeta) {
                auto new_extra_smeta{habana::get_storage_extra_meta(src)};
                new_extra_smeta->set_memory_permutation(permutation);
                new_extra_smeta->set_dont_allow_permutation(allow);
              }
            } else if (m_const_checksum_map[const_id].first == checksum) {
              HandleRecipeWithChecksumOnDevice(
                  const_id, checksum, cur_rargpsh->hashCode());
            } else {
              HandleRecipeWithExistingChecksumInCache(
                  const_id, checksum, cur_rargpsh->hashCode(), src);
            }

            constSectionIds.emplace_back(tensorSectionId);
            status = synHostUnmap(device_id, section_data_ptr);
            delete[] section_data_ptr;
            HABANA_ASSERT(
                status == synStatus::synSuccess,
                Logger::synStatusToStr(status));
          }
        }
      }
    }
  }
  synRecipeSectionHostBuffersClear(
      rv.recipe->syn_recipe_handle_,
      constSectionIds.data(),
      constSectionIds.size());
  constSectionIds.clear();
  // Call TcMalloc extension to release memory
  synapse_helpers::ReleaseFreeMemory();
}

void habana::HabanaLaunchOpPT::CompileSynapseGraph(bool allocate_rval) {
  TORCH_CHECK(syn_graph_ptr_, "Synapse graph pointer is null");

  if (syn_graph_ptr_->is_empty()) {
    PT_BRIDGE_DEBUG("Empty synapse graph. Nothing to compile.");
    // No need to allocate for lazy eager shape agnostic cache hit scenario
    if (allocate_rval) {
      cur_rvalpsh = std::make_shared<RecipeValueSpec>(nullptr, jit_ir_graph_);
    } else {
      cur_rvalpsh->recipe = nullptr;
      cur_rvalpsh->jit_graph_ = jit_ir_graph_;
    }
    return;
  }

  std::chrono::steady_clock::time_point t_start;
  t_start = std::chrono::steady_clock::now();
  auto cur_recipe = syn_graph_ptr_->compile();
  auto t_compile = std::chrono::steady_clock::now() - t_start;
  t_compile_ns =
      std::chrono::duration_cast<std::chrono::nanoseconds>(t_compile).count();

  RecipeValueSpec::increment_compile_count();
  // No need to allocate for lazy eager shape agnostic cache hit scenario
  if (allocate_rval) {
    cur_rvalpsh = std::make_shared<RecipeValueSpec>(cur_recipe, jit_ir_graph_);
  } else {
    // SAG cache hit case - to avoid race condition with execute thread
    hpu_op_recipe_ = cur_recipe;
  }
  PT_EAGER_DEBUG(
      "[SHAPE AGNOSTIC] cur recipe syn recipe handle : ",
      cur_recipe->syn_recipe_handle_);
  RecipeValueSpec& rv = *cur_rvalpsh;

  if (habana_helpers::IsInferenceMode()) {
    HabanaLaunchOpPT::PostCompilationStepForConstTensors(rv);
  }

  if (allocate_rval) {
    // Get workspace size of the compiled recipe
    rv.workspace_size =
        synapse_helpers::graph::query_workspace_size(*cur_recipe);
  } else {
    // SAG cache hit case - to avoid race condition with execute thread
    hpu_op_workspace_size_ =
        synapse_helpers::graph::query_workspace_size(*cur_recipe);
  }
}

void habana::HabanaLaunchOpPT::ConstructPatchingTableAndAtenOutputs() {
  TORCH_CHECK(syn_graph_ptr_, "Synapse graph pointer is null");
  TORCH_CHECK(cur_rvalpsh, "Recipe pointer is null");
  RecipeValueSpec& rv = *cur_rvalpsh;
  if (syn_graph_ptr_->is_empty() && collective_kernels_info.empty()) {
    PT_BRIDGE_DEBUG(
        "Empty synapse graph. No need to construct the patching table.");
    return;
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

  rv.collective_kernels_info = collective_kernels_info;

  // tinfos for outputs are populated during compile
  // need to be reordered only when the tensor handles are released
  if (!enable_caching_ && !enable_shape_agnostic_caching_) {
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

    size_t output_idx{0};
    for (auto output : jit_ir_graph_->outputs()) {
      auto oit = value_to_ivalue.find(output);
      TORCH_CHECK(
          oit != value_to_ivalue.end(),
          "value_to_ivalue does not have an entry for %",
          output->debugName());
      IValPtrShared ivpsh = oit->second;
      if (output_tensorinfo_map.count(ivpsh)) {
        auto it = output_tensorinfo_map.find(ivpsh);
        it->second->set_output_index(output_idx);
        output_tensorinfos.push_back(it->second);
        output_tensorinfo_map.erase(ivpsh);
      }
      aten_outputs_ptr_sh_->push_back(ivpsh);
      output_idx++;
    }
    TORCH_CHECK(
        output_tensorinfo_map.empty(),
        "output_tensorinfo_map still contains ",
        output_tensorinfo_map.size(),
        " tensors.");

    rv.dtensorinfos->insert(
        rv.dtensorinfos->end(),
        output_tensorinfos.begin(),
        output_tensorinfos.end());

    rv.num_outputs = output_tensorinfos.size();
    rv.num_tinfos = rv.dtensorinfos->size();
  } else {
    // TODO :
    //   preclude any interim tinfo from adding to output_tensorinfo_map
    OrderOutputTinfos(rv);
  }

  for (auto& ti : *rv.dtensorinfos) {
    if (!ti->is_duplicate()) {
      rv.ntensorbytes += ti->get_size();
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

  rv.populate_syn_tensor_ids();

  if (enable_caching_ || IS_BRIDGE_DEBUG_ENABLED ||
      (refine_ds_enabled_ && current_dbipsh_)) {
    TORCH_CHECK(cur_rargpsh != nullptr, "Encountered null cur_rargpsh");
    rv.set_key(cur_rargpsh->hashCode());
    rv.set_graph_key(graph_key_);
    rv.set_graph_name(GetSynapseGraphName());
    rv.set_op_strs(cur_rargpsh->get_op_strs());
  } else if (enable_shape_agnostic_caching_) {
    rv.set_graph_key(graph_key_);
    rv.set_graph_name(GetSynapseGraphName());
  }
  rv.sif_tidx_to_tinfo_map = sif_tidx_to_tinfo_map;

  if (rv.recipe) {
    rv.patch_launch_info(syn_launch_info_, external_tensor_info_indexes_);
  } else {
    PT_BRIDGE_DEBUG("Skipping patch_launch_info for empty recipe");
  }
}

void habana::HabanaLaunchOpPT::StoreCompiledInformation() {
  TORCH_CHECK(syn_graph_ptr_, "Synapse graph pointer is null");
  TORCH_CHECK(cur_rvalpsh, "Recipe pointer is null");
  RecipeValueSpec& rv = *cur_rvalpsh;
  if (syn_graph_ptr_->is_empty() && rv.collective_kernels_info.empty()) {
    return;
  }

  if (refine_ds_enabled_ && current_dbipsh_) {
    // Initiate recipe execution time collection
    if (GET_ENV_FLAG_NEW(PT_ENABLE_SYNLAUNCH_TIME_CAPTURE)) {
      InitiateSynlaunchTimeCapture(rv);
    }

    // Add the jit_ir_graph to current_dbipsh_
    current_dbipsh_->SetJitIRGraphPtr(jit_ir_graph_);
    if (GET_ENV_FLAG_NEW(PT_ENABLE_SYNLAUNCH_TIME_CAPTURE)) {
      current_dbipsh_->UpdateCompileTime(t_compile_ns, current_bucket_id_);
    }
  }

  // Save cache before calling launch to unblock other ranks who may wait on
  // this cache entry to be flushed to disk
  if (enable_caching_) {
    // Add the <key,value> pair to the map
    if (refine_ds_enabled_ && current_dbipsh_) {
      rv.dynamic_graph = syn_graph_ptr_->is_dynamic_graph();
      // Add the recipe to the corresponding bucket
      current_dbipsh_->SetSynapseRecipePtr(current_bucket_id_, cur_rvalpsh);
    }
    RecipeCacheLRU::get_cache().add(cur_rargpsh, cur_rvalpsh);
    PT_BRIDGE_DEBUG(
        "HabanaOp recipe cache :: adding new recipe to cache :: ", rv.key);
  }

  rv.update_hit_count();
}

void habana::HabanaLaunchOpPT::ExecuteSynapseGraph() {
  TORCH_CHECK(syn_graph_ptr_, "Synapse graph pointer is null");
  TORCH_CHECK(cur_rvalpsh, "Recipe pointer is null");
  RecipeValueSpec& rv = *cur_rvalpsh;
  if (syn_graph_ptr_->is_empty() && rv.collective_kernels_info.empty()) {
    PT_BRIDGE_DEBUG("Empty synapse graph. Will update outputs directly.");
    UpdateOutputs();
    return;
  }

  [[maybe_unused]] auto& device = HPURegistrar::get_device();

  PT_BRIDGE_DEBUG(
      "HabanaOp recipe cache :: launching new recipe", rv.header_str());

  std::shared_ptr<VecOfIValPtrSh> intermediate_tensors_ptr = nullptr;

  if (get_intermediate_tensors_ptrsh()) {
    intermediate_tensors_ptr = get_intermediate_tensors_ptrsh();
  } else {
    intermediate_tensors_ptr = std::make_shared<VecOfIValPtrSh>();
    for (auto& tensor : aten_intermediates) {
      IValPtrShared ivpsh = std::make_shared<IVal>(tensor);
      intermediate_tensors_ptr->push_back(ivpsh);
    }
  }

  if (!dry_run_) {
    rv.launch(
        hpu_stream_,
        input_refs,
        intermediate_tensors_ptr,
        *aten_outputs_ptr_sh_,
        syn_launch_info_,
        external_tensor_info_indexes_);
  }

  if (!rv.collective_kernels_info.empty()) {
    for (auto kernel_info : rv.collective_kernels_info) {
      CollectiveOperator* collective =
          dynamic_cast<CollectiveOperator*>(kernel_info->kernel.get());
      HABANA_ASSERT(collective);
      collective->clear_all_pt_and_syn_tensors();
    }
  }

  if (enable_graph_caching_ && refine_ds_enabled_ && current_dbipsh_) {
    PT_DYNAMIC_SHAPE_DEBUG(
        current_dbipsh_->digest_str(),
        current_dbipsh_->history_str(),
        "Recipe Header::",
        rv.header_str(),
        "\n",
        rv.digest_str(),
        "\n",
        "--------------------");
  } else {
    PT_BRIDGE_DEBUG(rv.digest_str());
  }

  if (!jit_graph_and_meta_data_->get_is_pipeline_supported()) {
    UpdateRecipeOutputs();
  }
}

void habana::HabanaLaunchOpPT::FlattenAndLinkInputTIVs(RecipeValueSpec& rv) {
  // dtensorinfos maintain the flattened tinfo list
  rv.dtensorinfos = std::make_shared<std::vector<PtTensorInfoShared>>(
      std::vector<PtTensorInfoShared>());

  std::unordered_map<void*, size_t> buff_to_inputtividx_map;
  for (auto& tiv : input_tivs) {
    if (absl::holds_alternative<PtTensorInfoShared>(tiv)) {
      const auto ti = absl::get<PtTensorInfoShared>(tiv);
      rv.dtensorinfos->push_back(ti);
      if (enable_caching_ || enable_shape_agnostic_caching_) {
        void* buffp = ti->get_buffer_start();
        buff_to_inputtividx_map.emplace(buffp, rv.dtensorinfos->size() - 1);
      }
    } else if (absl::holds_alternative<std::vector<PtTensorInfoShared>>(tiv)) {
      for (const auto& ti : absl::get<std::vector<PtTensorInfoShared>>(tiv)) {
        rv.dtensorinfos->push_back(ti);
        if (enable_caching_ || enable_shape_agnostic_caching_) {
          void* buffp = ti->get_buffer_start();
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
    if (absl::holds_alternative<PtTensorInfoShared>(tiv)) {
      auto ti = absl::get<PtTensorInfoShared>(tiv);
      if (enable_caching_ || enable_shape_agnostic_caching_) {
        void* buffp = ti->get_buffer_start();
        auto it_parent = buff_to_inputtividx_map.find(buffp);

        std::ostringstream err;
        err << *ti;

        TORCH_CHECK(
            buff_to_inputtividx_map.end() != it_parent,
            "parent tinfo is missing for input duplicate ",
            err.str());

        ti->set_duplicate_flag(true);
        size_t parent_idx = it_parent->second;
        TORCH_CHECK(
            parent_idx < num_inputs,
            "out of bound parent index : ",
            parent_idx,
            " for ",
            ti->get_syn_name());
        ti->set_parent_index(parent_idx);
        PT_BRIDGE_DEBUG(
            "FlattenAndLinkInputTIVs: Input duplicate: parent idx ",
            parent_idx,
            " parent buffer ptr ",
            rv.dtensorinfos->at(parent_idx)->get_buffer(),
            " duplicate_tiv buffer ptr ",
            ti->get_buffer());
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

void habana::HabanaLaunchOpPT::OrderInputs() {
  if (enable_caching_ || enable_shape_agnostic_caching_) {
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

void habana::HabanaLaunchOpPT::OrderOutputTinfos(RecipeValueSpec& rv) {
  bool has_empty_name = false;

  std::unordered_map<void*, size_t> buff_to_outputtinfoidx_map;
  // push the actual output tinfos
  size_t output_idx{0};
  for (auto output : jit_ir_graph_->outputs()) {
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
        it->second->set_output_index(output_idx);
        output_tensorinfos.push_back(it->second);
        if (it->second->get_syn_name().empty()) {
          has_empty_name = true;
        }
        output_tensorinfo_map.erase(ivpsh);
      } else if (duplicate_input_to_outtinfo_map.count(ivpsh)) {
        auto it_dup = duplicate_input_to_outtinfo_map.find(ivpsh);
        it_dup->second->set_output_index(output_idx);
      } else if (duplicate_intermediate_to_outtinfo_map.count(ivpsh)) {
        auto it_dup = duplicate_intermediate_to_outtinfo_map.find(ivpsh);
        it_dup->second->set_output_index(output_idx);
      } else if (duplicate_output_to_outtinfo_map.count(ivpsh)) {
        auto it_dup = duplicate_output_to_outtinfo_map.find(ivpsh);
        it_dup->second->set_output_index(output_idx);
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
    aten_outputs_ptr_sh_->push_back(ivpsh);
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
      void* buffp = ti->get_buffer_start();
      // Duplicate analysis for the persistent intermediates
      if (buff_to_interim_tividx_map.count(buffp)) {
        ti->set_duplicate_flag(true);
        ti->set_parent_index(buff_to_interim_tividx_map[buffp]);
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
    void* buffp = ti->get_buffer_start();

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
    void* buffp = ti->get_buffer_start();
    auto it_parent = buff_to_outputtinfoidx_map.find(buffp);

    std::ostringstream err;
    err << *ti;

    TORCH_CHECK(
        buff_to_outputtinfoidx_map.end() != it_parent,
        "parent tinfo is missing for output duplicate ",
        err.str());

    ti->set_duplicate_flag(true);
    size_t parent_idx = it_parent->second;
    TORCH_CHECK(
        parent_idx >= outputs_start && parent_idx < outputs_end,
        "for output duplicate ",
        ti->get_syn_name(),
        "parent index should be within [",
        outputs_start,
        ',',
        outputs_end,
        ')');
    ti->set_parent_index(parent_idx);
    rv.dtensorinfos->push_back(ti);
    nduplicates++;
  }
  rv.num_outduplicates = nduplicates;

  // Create the input tensor tiv to idx map, this is required
  // to match the input to output duplicates against their parent idx.
  std::unordered_map<void*, size_t> buff_to_inputtividx_map;
  size_t in_idx = 0;
  for (auto& tiv : input_tivs) {
    if (absl::holds_alternative<PtTensorInfoShared>(tiv)) {
      const auto ti = absl::get<PtTensorInfoShared>(tiv);
      void* buffp = ti->get_buffer_start();
      buff_to_inputtividx_map.emplace(buffp, in_idx++);
    } else if (absl::holds_alternative<std::vector<PtTensorInfoShared>>(tiv)) {
      for (const auto& ti : absl::get<std::vector<PtTensorInfoShared>>(tiv)) {
        void* buffp = ti->get_buffer_start();
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
    void* buffp = ti->get_buffer_start();
    auto it_parent = buff_to_inputtividx_map.find(buffp);

    std::ostringstream err;
    err << *ti;

    TORCH_CHECK(
        buff_to_inputtividx_map.end() != it_parent,
        "parent tinfo is missing for input_to_out duplicate ",
        err.str());

    ti->set_duplicate_flag(true);
    auto parent_idx = it_parent->second;
    TORCH_CHECK(
        parent_idx < rv.num_inputs,
        "for in_to_out duplicate ",
        ti->get_syn_name(),
        "parent index ",
        parent_idx,
        " should be within [",
        0,
        ',',
        rv.num_inputs,
        ')');
    ti->set_parent_index(parent_idx);
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
    void* buffp = ti->get_buffer_start();
    auto it_parent = buff_to_interim_tividx_map.find(buffp);

    std::ostringstream err;
    err << *ti;

    TORCH_CHECK(
        buff_to_interim_tividx_map.end() != it_parent,
        "parent tinfo is missing for interim_to_out duplicate ",
        err.str());

    ti->set_duplicate_flag(true);
    auto parent_idx = it_parent->second;
    TORCH_CHECK(
        (parent_idx >= intermediates_start && parent_idx < intermediates_end),
        "for interim to out duplicate ",
        ti->get_syn_name(),
        "parent index ",
        parent_idx,
        " should be within [",
        intermediates_start,
        ',',
        intermediates_end,
        ')');
    ti->set_parent_index(parent_idx);
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
    void* buffp = ti->get_buffer_start();
    auto it_parent = buff_to_outputtinfoidx_map.find(buffp);

    std::ostringstream err;
    err << *ti;

    TORCH_CHECK(
        buff_to_outputtinfoidx_map.end() != it_parent,
        "parent tinfo is missing for output_to_out duplicate ",
        err.str());

    ti->set_duplicate_flag(true);
    auto parent_idx = it_parent->second;
    TORCH_CHECK(
        parent_idx >= outputs_start && parent_idx < outputs_end,
        parent_idx < rv.num_inputs,
        "for out_to_out duplicate ",
        ti->get_syn_name(),
        "parent index ",
        parent_idx,
        " should be within [",
        outputs_start,
        ',',
        outputs_end,
        ')');
    ti->set_parent_index(parent_idx);
    rv.dtensorinfos->push_back(ti);
    nduplicates++;
  }
  rv.num_output_to_outduplicates = nduplicates;

  rv.num_tinfos = rv.dtensorinfos->size();
}

void habana::HabanaLaunchOpPT::RestoreInputTensorMetadata() {
  for (size_t i{0}; i < pt_stack->size(); i++) {
    if (pt_stack->at(i).isTensor()) {
      auto& input_tensor = pt_stack->at(i).toTensor();
      input_tensor.unsafeGetTensorImpl()->set_sizes_and_strides(
          input_tms.at(i).sizes, input_tms.at(i).strides);
      input_tensor.unsafeGetTensorImpl()->empty_tensor_restride(
          input_tms.at(i).mf);
    }
  }
}

void habana::HabanaLaunchOpPT::UpdateOutputs() {
  PT_BRIDGE_BEGIN;
  // Restore the metadata of the inputs
  RestoreInputTensorMetadata();

  // Update the stack from the JIT IR outputs
  torch::jit::drop(*pt_stack, num_inputs);
  for (auto output : jit_ir_graph_->outputs()) {
    auto oit = value_to_ivalue.find(output);
    TORCH_CHECK(
        oit != value_to_ivalue.end(),
        "value_to_ivalue does not have an entry for %",
        output->debugName());
    IValPtrShared ivpsh = oit->second;
    pt_stack->insert(pt_stack->end(), *ivpsh);
  }

  jit_graph_and_meta_data_->set_syn_graph_empty_flag(true);
  PT_BRIDGE_DEBUG(
      "Empty synapse recipe. The corresponding JIT IR should not cached");
  PT_BRIDGE_END;
}

void habana::HabanaLaunchOpPT::UpdateRecipeOutputs() {
  // Restore the metadata of the inputs
  RestoreInputTensorMetadata();

  // Update the stack from the recipe itself
  torch::jit::drop(*pt_stack, num_inputs);
  for (const auto& ivpsh : *aten_outputs_ptr_sh_) {
    pt_stack->insert(pt_stack->end(), *ivpsh);
  }
}

void habana::HabanaLaunchOpPT::ProcessInputStack(torch::jit::Stack& input_st) {
  num_inputs = jit_ir_graph_->inputs().size();
  PT_EAGER_DEBUG("[SHAPE AGNOSTIC] #graph_inputs : ", num_inputs);
  TORCH_CHECK(
      num_inputs == input_st.size(),
      "Input stack size=",
      input_st.size(),
      " is not matching with #graph_inputs=",
      num_inputs);

  num_tensor_inputs = 0;
  input_refs = torch::jit::last(input_st, num_inputs);

  // All tensors should be on Habana, we should assert otherwise
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
      is_all_hpu == true, " Habana Fusion needs all tensors to be in HPU");

  // Set the habana operators to capture data
  habana::ShapeInference::Capture(&m_map_shape);

  CopyInputStack(input_st);
}
