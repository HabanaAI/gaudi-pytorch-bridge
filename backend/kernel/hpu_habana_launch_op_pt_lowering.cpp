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
#include <cstdint>
#include "backend/backend_meta.h"
#include "backend/habana_device/hpu_cached_devices.h"
#include "backend/helpers/runtime_config.h"
#include "backend/kernel/constant_information.h"
#include "backend/kernel/hpu_habana_launch_op_pt.h"
#include "backend/synapse_helpers/env_flags.h"
#include "backend/synapse_helpers/tcmalloc_helper.h"
#include "habana_helpers/logging.h"
#include "habana_kernels/hccl_kernels.h"
#include "hpu_habana_launch_op_pt.h"

void habana::HabanaLaunchOpPT::CopyInputStack(torch::jit::Stack& input_st) {
  // Keep a handle to the stack for future use
  pt_stack_ = &input_st;
  input_tms_.reserve(pt_stack_->size());
  pt_stack_sh_.reserve(pt_stack_->size());

  for (size_t i = 0; i < pt_stack_->size(); i++) {
    if (pt_stack_->at(i).isTensor()) {
      auto& t{pt_stack_->at(i).toTensor()};
      input_tms_.emplace_back(
          t.sizes().vec(), t.strides().vec(), t.suggest_memory_format());
    } else {
      input_tms_.emplace_back(
          std::vector<int64_t>(),
          std::vector<int64_t>(),
          c10::MemoryFormat::Preserve);
    }

    IValPtrShared ivpsh = std::make_shared<IVal>(pt_stack_->at(i));
    pt_stack_sh_.emplace_back(ivpsh);
    if (ivpsh->isTensor() || ivpsh->isTensorList()) {
      num_tensor_inputs_++;
    }
  }
}

void habana::HabanaLaunchOpPT::ClearMembers(bool is_shape_inference) {
  if (is_shape_inference == false) {
    pt_stack_ = nullptr;
    pt_stack_sh_.clear();
    num_tensor_inputs_ = 0;
    recipe_launcher_ = nullptr;
  }

  prim_nodes_ival_counter_ = 0;
  restride_node_swap_counter_ = 0;
  restride_node_out_val_counter_ = 0;

  value_to_ivalue_.clear();
  syn_graph_ptr_ = nullptr;
  intermediate_tensors_ptr_sh_ = nullptr;
  aten_outputs_.clear();

  habana_kernels_.clear();

  input_tivs_.clear();
  duplicate_input_tivs_.clear();
  input_tiv_map_.clear();

  intermediate_tinfos_.clear();
  dma_input_tensorinfos_.clear();
  shape_tensor_tinfos_.clear();
  output_tensorinfos_.clear();
  duplicate_outtinfos_.clear();
  duplicate_input_to_outtinfo_map_.clear();
  duplicate_intermediate_to_outtinfo_map_.clear();

  aten_intermediates_.clear();

  output_tensorinfo_map_.clear();

  pt_to_synapse_tensors_.clear();
  meta_syn_tensors_.clear();
  buff_to_input_ivpsh_map_.clear();
  buff_to_intermediate_ivpsh_map_.clear();
  buff_to_output_ivpsh_map_.clear();
  buff_to_syn_tensor_map_.clear();

  jit_to_synapse_node_idx_map_.clear();
  collective_kernels_info_.Clear();
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
        value_to_ivalue_.find(jit_ir_graph_->outputs().at(el.output_index));
    HABANA_ASSERT(oit != value_to_ivalue_.end());
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
void habana::HabanaLaunchOpPT::UpdateSynapsePermutations(
    RecipeValueSpec& rvs,
    const synapse_helpers::graph::recipe_handle& recipe) {
  PT_LAZY_TRACE;

  if (syn_graph_ptr_->is_empty()) {
    PT_BRIDGE_DEBUG("Empty synapse graph. Skip UpdateSynapsePermutations.");
    return;
  }

  auto& tinfos = rvs.dtensorinfos;
  if (tinfos.size() == 0) {
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
  for (auto& el : pt_to_synapse_tensors_) {
    for (synapse_helpers::tensor& tensor : *el.second) {
      synapse_to_pt_tensor.insert({tensor.id(), el.first});
    }
  }
  if (GET_ENV_FLAG_NEW(PT_HPU_ENABLE_SYNAPSE_OUTPUT_PERMUTE)) {
    std::map<uint64_t, uint64_t> persistent_to_tensor_id;
    std::vector<synRetrievedLaunchTensorInfo> tensor_info_vec;
    // creating a map of tensor id to tinfo
    // preparing the tensors to query their permutation
    std::map<uint64_t, PtTensorInfoShared> tinfo_map;
    for (size_t i = 0; i < tinfos.size(); ++i) {
      auto& info = tinfos[i];
      if (info->is_output() && !info->is_ZST()) {
        // HABANA_ASSERT(tinfo_map.count(info->get_tensor_id() == 0));
        tinfo_map[info->get_tensor_id()] = info;
        if (info->get_allow_permutation()) {
          synRetrievedLaunchTensorInfo record = {};
          record.tensorId = rvs.tensor_ids_[i];
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
    synapse_helpers::graph::query_recipe_tensor_info(recipe, tensor_info_vec);

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
  for (size_t i = 0; i < tinfos.size(); ++i) {
    auto& info = tinfos[i];
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

static synRetrievedLaunchTensorInfo* getRecipeTensorInfos(
    const synRecipeHandle& recipeHandle,
    uint32_t numOfTensors) {
  synStatus status;
  uint64_t ids[numOfTensors];
  status = synTensorRetrieveLaunchIds(recipeHandle, ids, numOfTensors);
  HABANA_ASSERT(
      status == synStatus::synSuccess, Logger::synStatusToStr(status));
  auto tensorInfos = new synRetrievedLaunchTensorInfo[numOfTensors];
  for (unsigned i = 0; i < numOfTensors; i++) {
    tensorInfos[i].tensorId = ids[i];
  }
  status =
      synTensorRetrieveLaunchInfoById(recipeHandle, numOfTensors, tensorInfos);
  HABANA_ASSERT(
      status == synStatus::synSuccess, Logger::synStatusToStr(status));
  return tensorInfos;
}

// Based on:
// synapse/tests/gaudi_tests/gaudi_test_infra.cpp
static void getTensorSectionId(
    const synTensor& tensor,
    synSectionId& sectionId,
    synRetrievedLaunchTensorInfo* tensorInfos,
    uint32_t numOfTensors,
    bool& isInput) {
  synStatus status;
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

void habana::HabanaLaunchOpPT::HandleChecksum(
    at::Tensor& tensor,
    size_t data_size,
    bool checksum_found,
    ConstantInformation::checksum_t checksum,
    ConstantInformation::key_t key,
    char* data_ptr,
    size_t old_size,
    int device_id) {
  auto tmeta{habana::get_tensor_extra_meta(tensor)};
  ConstantInformation::id_t const_id{tmeta->get_const_id()};
  auto& constant_information = ConstantInformationValue();
  if (!checksum_found) {
    HandleTensorWithNewChecksum(
        tensor,
        data_size,
        checksum,
        ConstantInformation::key_t{key},
        (char*)data_ptr,
        old_size,
        device_id);
  } else if (constant_information.GetChecksumForId(const_id) == checksum) {
    HandleTensorWithChecksumOnDevice(
        const_id,
        checksum,
        ConstantInformation::key_t{cur_rargpsh_->hashCode()});
  } else {
    HandleTensorWithExistingChecksumInCache(
        const_id,
        checksum,
        ConstantInformation::key_t{cur_rargpsh_->hashCode()},
        tensor);
  }
}

void habana::HabanaLaunchOpPT::DeserializeConstSection(
    at::Tensor& tensor,
    const size_t key) {
  if (!IS_ENV_FLAG_DEFINED_NEW(PT_HPU_RECIPE_CACHE_CONFIG)) {
    return;
  }
  auto tmeta{habana::get_tensor_extra_meta(tensor)};
  TensorExtraMeta::set_const_tensor(tensor, true);
  auto const_serializer = tmeta->get_const_section_data_serializer();
  auto file_path =
      const_serializer->getSerializedRecipeFullPath(tmeta->get_const_id(), key);
  std::uintmax_t file_size = fs::file_size(file_path);
  TensorExtraMeta::set_const_tensor(tensor, false);
  if (file_size == 0) {
    HandleTensorWithZeroSize(tensor, ConstantInformation::key_t{key});
    return;
  }
  auto& device = habana::HPUDeviceContext::get_device();
  auto device_id = device.id();
  void* data{};
  auto status = device.get_host_memory().malloc(&data, file_size);
  HABANA_ASSERT(status == synSuccess, Logger::synStatusToStr(status));
  const_serializer->deserializePerRecipe(
      data, file_size, tmeta->get_const_id(), key);
  ConstantInformation::checksum_t checksum{GetDataChecksum(data, file_size)};
  auto old_size = tmeta->get_host_size();
  ConstantInformation::id_t const_id{tmeta->get_const_id()};
  auto& constant_information = ConstantInformationValue();
  auto checksum_found =
      constant_information.DoesCheckSumExist(const_id, checksum);
  HandleChecksum(
      tensor,
      file_size,
      checksum_found,
      checksum,
      ConstantInformation::key_t{key},
      (char*)data,
      old_size,
      device_id);
  device.get_host_memory().free(data);
}

void habana::HabanaLaunchOpPT::SerializeConstSection(
    at::Tensor& tensor,
    size_t section_size,
    char* section_data_ptr,
    const size_t key) {
  if (!IS_ENV_FLAG_DEFINED_NEW(PT_HPU_RECIPE_CACHE_CONFIG)) {
    return;
  }
  auto tmeta{habana::get_tensor_extra_meta(tensor)};
  tmeta->get_const_section_data_serializer()->serializePerRecipe(
      section_data_ptr, section_size, tmeta->get_const_id(), key);
}

void habana::HabanaLaunchOpPT::HandleTensorWithZeroSize(
    at::Tensor& tensor,
    ConstantInformation::key_t key) {
  auto tmeta{get_tensor_extra_meta(tensor)};
  auto old_size = tmeta->get_host_size();
  ConstantInformation::id_t const_id{tmeta->get_const_id()};
  auto& constant_information = ConstantInformationValue();
  auto checksum_if_exists = constant_information.GetChecksumForId(const_id);
  tmeta->set_nbytes_inference(old_size);
  ConstantInformation::checksum_t checksum{0};
  constant_information.Insert(const_id, checksum);
  constant_information.PushInfo(const_id, checksum, key, 0 /*_section_size*/);
  at::DataPtr data = tensor.storage().allocator()->allocate(0);
  auto old_data_ptr = tensor.storage().set_data_ptr(std::move(data));
  tensor.storage().set_nbytes(0);
  if (checksum_if_exists.has_value() and
      checksum_if_exists.value() != checksum) {
    PT_BRIDGE_DEBUG(
        "For const_id: ",
        const_id,
        " Checksum has valid value for another recipe");
    constant_information.StorePrevDataPtr(
        const_id, std::move(old_data_ptr), checksum_if_exists.value());
  }
}

void habana::HabanaLaunchOpPT::UpdateTensorInfoMap(
    std::shared_ptr<c10::IValue> src,
    void* ptr) {
  ivalue_to_tensor_info_map_[src]->set_buffer(ptr);
}

habana::HabanaLaunchOpPT::permuteInfo habana::HabanaLaunchOpPT::GetPermuteInfo(
    StorageExtraMeta* _smeta) {
  synapse_helpers::layouts::MemoryPermutation permutation = {};
  bool allow = false;
  if (_smeta) {
    permutation = _smeta->get_memory_permutation();
    allow = _smeta->get_dont_allow_permutation();
  }
  return make_pair(permutation, allow);
}

void habana::HabanaLaunchOpPT::SetPermuteInfo(
    StorageExtraMeta* _new_smeta,
    StorageExtraMeta* _smeta,
    habana::HabanaLaunchOpPT::permuteInfo _info) {
  if (_smeta) {
    _new_smeta->set_memory_permutation(_info.first);
    _new_smeta->set_dont_allow_permutation(_info.second);
  }
}

void habana::HabanaLaunchOpPT::HandleTensorWithNewChecksum(
    at::Tensor& tensor,
    size_t section_size,
    ConstantInformation::checksum_t checksum,
    ConstantInformation::key_t key,
    char* section_data_ptr,
    size_t old_size,
    int) {
  auto tmeta{get_tensor_extra_meta(tensor)};
  ConstantInformation::id_t const_id{tmeta->get_const_id()};
  // reallocation is required if old_size is not same as section size
  // or if old_size is same as section_size but checksum is new
  auto& constant_information = ConstantInformationValue();
  ConstantInformation::checksum_t host_checksum{tmeta->get_host_checksum()};
  auto checksum_if_exists = constant_information.GetChecksumForId(const_id);

  if (checksum_if_exists.has_value() or (checksum != host_checksum)) {
    // Reallocation is required
    auto smeta{habana::get_storage_extra_meta(tensor)};
    auto info = GetPermuteInfo(smeta);
    tmeta->set_nbytes_inference(old_size);
    at::DataPtr data = tensor.storage().allocator()->allocate(section_size);
    PT_BRIDGE_DEBUG(
        "Needed reallocation (bridge) old_size ",
        old_size,
        " != ",
        section_size,
        " Checksum: ",
        checksum,
        " Allocated data_ptr: ",
        data.get());
    auto old_data_ptr = tensor.storage().set_data_ptr(std::move(data));
    tensor.storage().set_nbytes(section_size);
    if (checksum_if_exists.has_value()) {
      constant_information.StorePrevDataPtr(
          const_id, std::move(old_data_ptr), checksum_if_exists.value());
    }
    auto new_extra_smeta{habana::get_storage_extra_meta(tensor)};
    SetPermuteInfo(new_extra_smeta, smeta, info);
  }
  constant_information.Insert(const_id, checksum);
  constant_information.PushInfo(const_id, checksum, key, section_size);
  if (checksum == host_checksum and !checksum_if_exists.has_value()) {
    // If synapse has not modified the tensor data (old size equal section size)
    // And no other recipe has a checksum before this then no need to copy the
    // new data
    return;
  }
  std::atomic<bool> copyDone{false};
  HPUDeviceContext::copy_data_to_device(
      section_data_ptr,
      reinterpret_cast<synapse_helpers::device_ptr>(tensor.data_ptr()),
      reinterpret_cast<synapse_helpers::device_ptr>(
          tensor.storage().data_ptr().get()),
      section_size,
      [&copyDone]() { copyDone = true; },
      false,
      true);
  // wait for copy completion
  while (!copyDone) {
    std::this_thread::yield();
  }
}

void habana::HabanaLaunchOpPT::HandleTensorWithExistingChecksumInCache(
    ConstantInformation::id_t const_id,
    ConstantInformation::checksum_t checksum,
    ConstantInformation::key_t key,
    at::Tensor& tensor) {
  auto& constant_information = ConstantInformationValue();
  constant_information.AddRecipe(const_id, checksum, key);
  constant_information.GetConstPtrForRecipe(const_id, key, tensor);
  constant_information.Insert(const_id, checksum);
  PT_BRIDGE_DEBUG(
      "Tensor with const_id: ",
      const_id,
      " has moved data pointer for the data corresponding to checksum: ",
      checksum,
      " for cache miss on key ",
      key);
}

void habana::HabanaLaunchOpPT::HandleTensorWithChecksumOnDevice(
    habana::ConstantInformation::id_t const_id,
    habana::ConstantInformation::checksum_t checksum,
    habana::ConstantInformation::key_t key) {
  ConstantInformationValue().AddRecipe(const_id, checksum, key);
  PT_BRIDGE_DEBUG(
      "Constant tensor already exists on the device, avoiding re-copy to device, checksum: ",
      checksum,
      " const_id: ",
      const_id,
      " recipe key: ",
      key);
}

void habana::HabanaLaunchOpPT::PostCompilationStepForConstTensors(
    synapse_helpers::graph::recipe_handle& recipe) {
  std::vector<synSectionId> constSectionIds;
  uint32_t numOfTensors = 0;
  synStatus status =
      synTensorRetrieveLaunchAmount(recipe.syn_recipe_handle_, &numOfTensors);
  HABANA_ASSERT(
      status == synStatus::synSuccess, Logger::synStatusToStr(status));
  auto tensorInfos =
      getRecipeTensorInfos(recipe.syn_recipe_handle_, numOfTensors);
  for (auto iter = pt_to_synapse_tensors_.begin();
       iter != pt_to_synapse_tensors_.end();
       ++iter) {
    auto& src = iter->first->toTensor();
    // PT_BRIDGE_DEBUG("tensor ", src, " has_storage: ", src.has_storage());
    if (src.has_storage()) {
      auto tmeta{get_tensor_extra_meta(src)};
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
          uint64_t section_size = 0, section_data = 0;
          synSectionId tensorSectionId;
          bool isInput;
          getTensorSectionId(
              tensor.get(),
              tensorSectionId,
              tensorInfos,
              numOfTensors,
              isInput);
          if (!isInput) {
            PT_BRIDGE_DEBUG("non-input tensor section ID:  ", tensorSectionId);
            continue;
          }
          HABANA_ASSERT(tensorSectionId != INVALID_SECTION_ID);
          synStatus status;
          status = synRecipeSectionGetProp(
              recipe.syn_recipe_handle_,
              tensorSectionId,
              SECTION_SIZE,
              &section_size);
          HABANA_ASSERT(
              status == synStatus::synSuccess, Logger::synStatusToStr(status));
          PT_BRIDGE_DEBUG(
              "section ID: ",
              tensorSectionId,
              " section_size:: ",
              section_size,
              " , size (bridge) :: ",
              tensor.get_host_ptr_size());
          if (section_size) {
            auto device_id = tensor.device_id();
            HABANA_ASSERT(
                status == synStatus::synSuccess,
                Logger::synStatusToStr(status));
            status = synRecipeSectionGetProp(
                recipe.syn_recipe_handle_,
                tensorSectionId,
                SECTION_DATA,
                &section_data);
            char* section_data_ptr = (char*)section_data;
            HABANA_ASSERT(
                status == synStatus::synSuccess,
                Logger::synStatusToStr(status));
            ConstantInformation::checksum_t checksum{
                GetDataChecksum(section_data_ptr, section_size)};
            HABANA_ASSERT(
                tmeta->has_valid_const_id(),
                "Constant tensor can not have constant id as -1");
            [[maybe_unused]] auto& device =
                HPUDeviceContext::get_device(device_id);
            status = synHostMap(device_id, section_size, section_data_ptr);
            HABANA_ASSERT(
                status == synStatus::synSuccess,
                Logger::synStatusToStr(status));
            ConstantInformation::id_t const_id{tmeta->get_const_id()};
            auto& constant_information = ConstantInformationValue();
            auto checksum_found =
                constant_information.DoesCheckSumExist(const_id, checksum);
            PT_BRIDGE_DEBUG(
                "const_id: ",
                const_id,
                " checksum_found: ",
                checksum_found,
                " checksum: ",
                checksum);
              auto old_size = tensor.get_host_ptr_size();
              HandleChecksum(
                  iter->first->toTensor(),
                  section_size,
                  checksum_found,
                  checksum,
                  ConstantInformation::key_t{cur_rargpsh_->hashCode()},
                  section_data_ptr,
                  old_size,
                  device_id);
              UpdateTensorInfoMap(
                  iter->first,
                  (void*)(iter->first->toTensor().storage().data_ptr().get()));

              constSectionIds.emplace_back(tensorSectionId);
              SerializeConstSection(
                  src,
                  section_size,
                  section_data_ptr,
                  cur_rargpsh_->hashCode());
              status = synHostUnmap(device_id, section_data_ptr);
              delete[] section_data_ptr;
              HABANA_ASSERT(
                  status == synStatus::synSuccess,
                  Logger::synStatusToStr(status));
          } else {
            HandleTensorWithZeroSize(
                iter->first->toTensor(),
                ConstantInformation::key_t{cur_rargpsh_->hashCode()});
            SerializeConstSection(
                src, section_size, nullptr, cur_rargpsh_->hashCode());
            UpdateTensorInfoMap(
                iter->first,
                (void*)(iter->first->toTensor().storage().data_ptr().get()));
          }
        }
      }
    }
  }

  if (constSectionIds.size()) {
    synRecipeSectionHostBuffersClear(
        recipe.syn_recipe_handle_,
        constSectionIds.data(),
        constSectionIds.size());
    constSectionIds.clear();
  }

  delete[] tensorInfos;
  // Call TcMalloc extension to release memory
  synapse_helpers::ReleaseFreeMemory();
}

std::shared_ptr<synapse_helpers::graph::recipe_handle> habana::
    HabanaLaunchOpPT::CompileSynapseGraph() {
  TORCH_CHECK(syn_graph_ptr_, "Synapse graph pointer is null");

  if (syn_graph_ptr_->is_empty()) {
    PT_BRIDGE_DEBUG("Empty synapse graph. Nothing to compile.");
    // No need to allocate for lazy eager shape agnostic cache hit scenario
    return nullptr;
  }

  std::chrono::steady_clock::time_point t_start;
  t_start = std::chrono::steady_clock::now();
  auto recipe = syn_graph_ptr_->compile();
  auto t_compile = std::chrono::steady_clock::now() - t_start;
  t_compile_ns_ =
      std::chrono::duration_cast<std::chrono::nanoseconds>(t_compile).count();

  RecipeValueSpec::increment_compile_count();

  PT_EAGER_DEBUG(
      "[SHAPE AGNOSTIC] cur recipe syn recipe handle : ",
      recipe->syn_recipe_handle_);

  if (habana_helpers::IsInferenceMode()) {
    HabanaLaunchOpPT::PostCompilationStepForConstTensors(*recipe);
  }
  return recipe;
}

void habana::HabanaLaunchOpPT::ConstructPatchingTableAndAtenOutputs(
    RecipeValueSpec& rv,
    const std::shared_ptr<synapse_helpers::graph::recipe_handle>& recipe) {
  TORCH_CHECK(syn_graph_ptr_, "Synapse graph pointer is null");
  if (syn_graph_ptr_->is_empty() && collective_kernels_info_.Empty()) {
    PT_BRIDGE_DEBUG(
        "Empty synapse graph. No need to construct the patching table.");
    return;
  }

  // input_tivs_ need to be reordered for patching
  OrderInputs();

  // rv.num_inputs and rv.num_induplicates will be set by
  // FlattenAndLinkInputTIVs
  FlattenAndLinkInputTIVs(rv);

  if (!dma_input_tensorinfos_.empty()) {
    rv.num_dma_inputs = dma_input_tensorinfos_.size();
    rv.dtensorinfos.insert(
        rv.dtensorinfos.end(),
        dma_input_tensorinfos_.begin(),
        dma_input_tensorinfos_.end());
  }

  if (!shape_tensor_tinfos_.empty()) {
    rv.num_shape_tensors = shape_tensor_tinfos_.size();
    rv.dtensorinfos.insert(
        rv.dtensorinfos.end(),
        shape_tensor_tinfos_.begin(),
        shape_tensor_tinfos_.end());
  }

  *rv.collective_kernels_info = std::move(collective_kernels_info_);

  // tinfos for outputs are populated during compile
  // need to be reordered only when the tensor handles are released
  if (!enable_caching_ && !enable_shape_agnostic_caching_) {
    if (!intermediate_tinfos_.empty()) {
      rv.num_intermediates = intermediate_tinfos_.size();
      rv.dtensorinfos.insert(
          rv.dtensorinfos.end(),
          intermediate_tinfos_.begin(),
          intermediate_tinfos_.end());
    }

    // At this point, tinfos for inputs, input duplicates and intermediates
    // are populated
    TORCH_CHECK(
        (rv.num_inputs + rv.num_induplicates + rv.num_dma_inputs +
             rv.num_shape_tensors + rv.num_intermediates ==
         rv.dtensorinfos.size()),
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
        rv.dtensorinfos.size());

    size_t output_idx{0};
    for (auto output : jit_ir_graph_->outputs()) {
      auto oit = value_to_ivalue_.find(output);
      TORCH_CHECK(
          oit != value_to_ivalue_.end(),
          "value_to_ivalue_ does not have an entry for %",
          output->debugName());
      IValPtrShared ivpsh = oit->second;
      if (output_tensorinfo_map_.count(ivpsh)) {
        auto it = output_tensorinfo_map_.find(ivpsh);
        it->second->set_output_index(output_idx);
        output_tensorinfos_.push_back(it->second);
        output_tensorinfo_map_.erase(ivpsh);
      }
      aten_outputs_.push_back(ivpsh);
      output_idx++;
    }
    TORCH_CHECK(
        output_tensorinfo_map_.empty(),
        "output_tensorinfo_map_ still contains ",
        output_tensorinfo_map_.size(),
        " tensors.");

    rv.dtensorinfos.insert(
        rv.dtensorinfos.end(),
        output_tensorinfos_.begin(),
        output_tensorinfos_.end());

    rv.num_outputs = output_tensorinfos_.size();
  } else {
    // TODO :
    //   preclude any interim tinfo from adding to output_tensorinfo_map_
    OrderOutputTinfos(rv);
  }

  // At this point, tinfos for inputs, input duplicates, dma_inputs,
  // intermediates, outputs and output duplicates are populated
  auto total_tinfos = rv.num_inputs + rv.num_induplicates + rv.num_dma_inputs +
      rv.num_shape_tensors + rv.num_intermediates + rv.num_outputs +
      rv.num_outduplicates + rv.num_input_to_outduplicates +
      rv.num_intermediate_to_outduplicates + rv.num_output_to_outduplicates;

  TORCH_CHECK(
      total_tinfos == rv.dtensorinfos.size(),
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
      rv.dtensorinfos.size());

  if (enable_caching_ || IS_BRIDGE_DEBUG_ENABLED ||
      (refine_ds_enabled_ && current_dbipsh_)) {
    TORCH_CHECK(cur_rargpsh_ != nullptr, "Encountered null cur_rargpsh");
    rv.set_key(cur_rargpsh_->hashCode());
    rv.set_graph_key(graph_key_);
    rv.set_graph_name(GetSynapseGraphName());
    rv.set_op_strs(cur_rargpsh_->get_op_strs());
  } else if (enable_shape_agnostic_caching_) {
    rv.set_graph_key(graph_key_);
    rv.set_graph_name(GetSynapseGraphName());
  }
  rv.sif_tidx_to_tinfo_map = sif_tidx_to_tinfo_map_;
  rv.enable_optim_output_sif_ = enable_optim_output_sif_;
  if (enable_optim_output_sif_) {
    rv.st_to_tensor_idx_map = ShapeInference::GetTensorMapping();
    rv.dynamic_nodes_with_backend_STs =
        std::move(dynamic_nodes_with_backend_STs);
    rv.ds_sifinfo_map[sym_expr_hash_] = std::move(ds_sif_info_);
  }
  rv.disabled_jit_ir_ops_ = HabanaLaunchOpUtils::disabled_jit_ir_ops();

  if (recipe) {
    rv.populate_syn_tensor_ids(*recipe);
    rv.patch_launch_info(syn_launch_info_, external_tensor_info_indexes_);
  } else {
    PT_BRIDGE_DEBUG(
        "Skipping patch_launch_info and retrieving tensor ids for empty recipe");
  }
}

void habana::HabanaLaunchOpPT::StoreCompiledInformation(
    std::shared_ptr<RecipeValueSpec>& rvs) {
  TORCH_CHECK(syn_graph_ptr_, "Synapse graph pointer is null");
  TORCH_CHECK(recipe_launcher_, "Recipe pointer is null");
  if (syn_graph_ptr_->is_empty() && rvs->collective_kernels_info->Empty()) {
    return;
  }

  if (refine_ds_enabled_ && current_dbipsh_) {
    // Initiate recipe execution time collection
    if (GET_ENV_FLAG_NEW(PT_ENABLE_SYNLAUNCH_TIME_CAPTURE)) {
      InitiateSynlaunchTimeCapture(*recipe_launcher_);
    }

    // Add the jit_ir_graph to current_dbipsh_
    current_dbipsh_->SetJitIRGraphPtr(jit_ir_graph_);
    if (GET_ENV_FLAG_NEW(PT_ENABLE_SYNLAUNCH_TIME_CAPTURE)) {
      current_dbipsh_->UpdateCompileTime(t_compile_ns_, current_bucket_id_);
    }
  }

  // Save cache before calling launch to unblock other ranks who may wait on
  // this cache entry to be flushed to disk
  if (enable_caching_) {
    // Add the <key,value> pair to the map
    if (refine_ds_enabled_ && current_dbipsh_) {
      rvs->dynamic_graph = syn_graph_ptr_->is_dynamic_graph();
      // Add the recipe to the corresponding bucket
      current_dbipsh_->SetSynapseRecipePtr(current_bucket_id_, rvs);
    }
    auto rh = std::make_shared<RecipeHolder>(recipe_launcher_, rvs);
    HPUDeviceContext::recipe_cache().add(cur_rargpsh_, rh);
    PT_BRIDGE_DEBUG(
        "HabanaOp recipe cache :: adding new recipe to cache :: ", rvs->key);
  }

  rvs->update_hit_count();
}

void habana::HabanaLaunchOpPT::ExecuteSynapseGraph() {
  TORCH_CHECK(syn_graph_ptr_, "Synapse graph pointer is null");
  TORCH_CHECK(recipe_launcher_, "Recipe pointer is null");
  if (syn_graph_ptr_->is_empty() &&
      recipe_launcher_->collective_kernels_info_->Empty()) {
    PT_BRIDGE_DEBUG("Empty synapse graph. Will update outputs directly.");
    UpdateOutputs();
    return;
  }

  [[maybe_unused]] auto& device = HPUDeviceContext::get_device();

  PT_BRIDGE_DEBUG("HabanaOp recipe cache :: launching new recipe");

  std::shared_ptr<VecOfIValPtrSh> intermediate_tensors_ptr = nullptr;

  if (get_intermediate_tensors_ptrsh()) {
    intermediate_tensors_ptr = get_intermediate_tensors_ptrsh();
  } else {
    intermediate_tensors_ptr = std::make_shared<VecOfIValPtrSh>();
    for (auto& tensor : aten_intermediates_) {
      IValPtrShared ivpsh = std::make_shared<IVal>(tensor);
      intermediate_tensors_ptr->push_back(ivpsh);
    }
  }

  if (!dry_run_) {
    recipe_launcher_->Launch(
        hpu_stream_,
        input_refs_,
        intermediate_tensors_ptr,
        aten_outputs_,
        syn_launch_info_,
        external_tensor_info_indexes_);
  }

  // TODO: must be run only one time
  recipe_launcher_->collective_kernels_info_->ClearAllPtAndSynTensors();

  if (enable_graph_caching_ && refine_ds_enabled_ && current_dbipsh_) {
    PT_DYNAMIC_SHAPE_DEBUG(
        current_dbipsh_->digest_str(),
        current_dbipsh_->history_str(),
        "--------------------");
  }

  if (!jit_graph_and_meta_data_->get_is_pipeline_supported()) {
    UpdateRecipeOutputs();
  }
}

void habana::HabanaLaunchOpPT::FlattenAndLinkInputTIVs(RecipeValueSpec& rv) {
  // dtensorinfos maintain the flattened tinfo list
  HABANA_ASSERT(rv.dtensorinfos.empty());

  std::unordered_map<void*, size_t> buff_to_inputtividx_map;
  for (auto& tiv : input_tivs_) {
    if (absl::holds_alternative<PtTensorInfoShared>(tiv)) {
      const auto ti = absl::get<PtTensorInfoShared>(tiv);
      rv.dtensorinfos.push_back(ti);
      if (enable_caching_ || enable_shape_agnostic_caching_) {
        void* buffp = ti->get_buffer_start();
        buff_to_inputtividx_map.emplace(buffp, rv.dtensorinfos.size() - 1);
      }
    } else if (absl::holds_alternative<std::vector<PtTensorInfoShared>>(tiv)) {
      for (const auto& ti : absl::get<std::vector<PtTensorInfoShared>>(tiv)) {
        rv.dtensorinfos.push_back(ti);
        if (enable_caching_ || enable_shape_agnostic_caching_) {
          void* buffp = ti->get_buffer_start();
          buff_to_inputtividx_map.emplace(buffp, rv.dtensorinfos.size() - 1);
        }
      }
    } else {
      TORCH_CHECK(false, "Error condition for input tiv");
    }
  }
  // At this point inputs tinfos are populated
  rv.num_inputs = rv.dtensorinfos.size();

  // Link the input tivs with the duplicate
  size_t nduplicates{0};
  for (auto& tiv : duplicate_input_tivs_) {
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
            parent_idx < num_inputs_,
            "out of bound parent index : ",
            parent_idx,
            " for ",
            ti->get_syn_name());
        ti->set_parent_index(parent_idx);
        PT_BRIDGE_DEBUG(
            "FlattenAndLinkInputTIVs: Input duplicate: parent idx ",
            parent_idx,
            " parent buffer ptr ",
            rv.dtensorinfos.at(parent_idx)->get_buffer(),
            " duplicate_tiv buffer ptr ",
            ti->get_buffer());
      }
      rv.dtensorinfos.push_back(ti);
      nduplicates++;
    } else {
      TORCH_CHECK(false, "duplicate tiv must be a tensor");
    }
  }
  TORCH_CHECK(
      nduplicates == duplicate_input_tivs_.size(),
      "#duplicate_input_tivs_ ",
      duplicate_input_tivs_.size(),
      " is not matching with num_induplicates ",
      nduplicates);

  rv.num_induplicates = nduplicates;

  // At this point inputs and duplicate tinfos are populated
  TORCH_CHECK(
      (rv.num_inputs + rv.num_induplicates == rv.dtensorinfos.size()),
      "num_inputs ",
      rv.num_inputs,
      "num_induplicates ",
      rv.num_induplicates,
      " are not adding up to #dtensorinfos ",
      rv.dtensorinfos.size());
}

void habana::HabanaLaunchOpPT::OrderInputs() {
  if (enable_caching_ || enable_shape_agnostic_caching_) {
    // Order the input_tivs_ according to the order of suggraph inputs
    size_t i = pt_stack_sh_.size() - num_inputs_;
    for (; i < pt_stack_sh_.size(); i++) {
      IValPtrShared ivpsh = pt_stack_sh_.at(i);
      if (ivpsh->isTensor() || ivpsh->isTensorList()) {
        auto it = input_tiv_map_.find(ivpsh);
        if (it != input_tiv_map_.end()) {
          input_tivs_.push_back(it->second);
        } else {
          TORCH_CHECK(false, "synapse tensor not found for input index", i);
        }
      }
    }
    TORCH_CHECK(
        input_tivs_.size() == num_tensor_inputs_,
        "number of input tensors ",
        num_tensor_inputs_,
        " mismatch with #input_tivs_ ",
        input_tivs_.size());
  }
}

void habana::HabanaLaunchOpPT::OrderOutputTinfos(RecipeValueSpec& rv) {
  bool has_empty_name = false;

  std::unordered_map<void*, size_t> buff_to_outputtinfoidx_map;
  // push the actual output tinfos
  size_t output_idx{0};
  for (auto output : jit_ir_graph_->outputs()) {
    auto oit = value_to_ivalue_.find(output);
    TORCH_CHECK(
        oit != value_to_ivalue_.end(),
        "value_to_ivalue_ does not have an entry for %",
        output->debugName());

    IValPtrShared ivpsh = oit->second;
    TORCH_CHECK(nullptr != ivpsh, "IValPtrShared for subgraph output is null");

    // Checking where we can find the outputs
    {
      if (output_tensorinfo_map_.count(ivpsh)) {
        auto it = output_tensorinfo_map_.find(ivpsh);
        it->second->set_output_index(output_idx);
        output_tensorinfos_.push_back(it->second);
        if (it->second->get_syn_name().empty()) {
          has_empty_name = true;
        }
        output_tensorinfo_map_.erase(ivpsh);
      } else if (duplicate_input_to_outtinfo_map_.count(ivpsh)) {
        auto it_dup = duplicate_input_to_outtinfo_map_.find(ivpsh);
        it_dup->second->set_output_index(output_idx);
      } else if (duplicate_intermediate_to_outtinfo_map_.count(ivpsh)) {
        auto it_dup = duplicate_intermediate_to_outtinfo_map_.find(ivpsh);
        it_dup->second->set_output_index(output_idx);
      } else if (duplicate_output_to_outtinfo_map_.count(ivpsh)) {
        auto it_dup = duplicate_output_to_outtinfo_map_.find(ivpsh);
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
    aten_outputs_.push_back(ivpsh);
    output_idx++;
  }

  TORCH_CHECK(!has_empty_name, "empty tensor name");

  size_t intermediates_start = rv.num_inputs + rv.num_induplicates +
      rv.num_dma_inputs + rv.num_shape_tensors;

  TORCH_CHECK(
      output_tensorinfo_map_.empty(),
      "output_tensorinfo_map_ still contains ",
      output_tensorinfo_map_.size(),
      " tensors.");

  // Add the intermediates to rv.dtensorinfos
  std::unordered_map<void*, size_t> buff_to_interim_tividx_map;
  size_t interim_tinfo_idx{intermediates_start};
  if (!intermediate_tinfos_.empty()) {
    rv.num_intermediates = intermediate_tinfos_.size();
    for (auto& ti : intermediate_tinfos_) {
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

    rv.dtensorinfos.insert(
        rv.dtensorinfos.end(),
        intermediate_tinfos_.begin(),
        intermediate_tinfos_.end());
  }

  // At this point, within dtensorinfos, tinfos for inputs,
  // input duplicates and intermediates are added.
  size_t intermediates_end = intermediates_start + rv.num_intermediates;
  size_t outputs_start = intermediates_end;

  // Add the outputs to rv.dtensorinfos
  for (auto& ti : output_tensorinfos_) {
    rv.dtensorinfos.push_back(ti);
    void* buffp = ti->get_buffer_start();

    buff_to_outputtinfoidx_map.emplace(buffp, rv.dtensorinfos.size() - 1);
  }
  rv.num_outputs = output_tensorinfos_.size();

  // At this point, within dtensorinfos, tinfos for inputs, input_duplicates,
  // intermediates and outputs are added.
  size_t outputs_end = outputs_start + rv.num_outputs;

  // Link the out tivs with the duplicate.
  // Remember these are not outputs but tensors
  // that go back to the FusedOp from the outputs
  size_t nduplicates{0};
  for (auto& ti : duplicate_outtinfos_) {
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
    rv.dtensorinfos.push_back(ti);
    nduplicates++;
  }
  rv.num_outduplicates = nduplicates;

  // Create the input tensor tiv to idx map, this is required
  // to match the input to output duplicates against their parent idx.
  std::unordered_map<void*, size_t> buff_to_inputtividx_map;
  size_t in_idx = 0;
  for (auto& tiv : input_tivs_) {
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
  for (auto& mi : duplicate_input_to_outtinfo_map_) {
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
    rv.dtensorinfos.push_back(ti);
    nduplicates++;
  }

  rv.num_input_to_outduplicates = nduplicates;

  // Link the out tivs which are duplicate of persistent intermediates
  // These are tensors that are duplicated from a persistent interim and is
  // part of the graph output
  nduplicates = 0;
  for (auto& mi : duplicate_intermediate_to_outtinfo_map_) {
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
    rv.dtensorinfos.push_back(ti);
    nduplicates++;
  }
  rv.num_intermediate_to_outduplicates = nduplicates;

  // Link the out tivs which are duplicate of actual outputs
  // These are tensors that are duplicated from outputs created by individual
  // ops using aten::empty like calls and is part of the graph output
  nduplicates = 0;
  for (auto& mi : duplicate_output_to_outtinfo_map_) {
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
    rv.dtensorinfos.push_back(ti);
    nduplicates++;
  }
  rv.num_output_to_outduplicates = nduplicates;
}

void habana::HabanaLaunchOpPT::RestoreInputTensorMetadata() {
  for (size_t i{0}; i < pt_stack_->size(); i++) {
    if (pt_stack_->at(i).isTensor()) {
      auto& input_tensor = pt_stack_->at(i).toTensor();
      input_tensor.unsafeGetTensorImpl()->set_sizes_and_strides(
          input_tms_.at(i).sizes, input_tms_.at(i).strides);
      input_tensor.unsafeGetTensorImpl()->empty_tensor_restride(
          input_tms_.at(i).mf);
    }
  }
}

void habana::HabanaLaunchOpPT::UpdateOutputs() {
  PT_BRIDGE_BEGIN;
  // Restore the metadata of the inputs
  RestoreInputTensorMetadata();

  // Update the stack from the JIT IR outputs
  torch::jit::drop(*pt_stack_, num_inputs_);
  for (auto output : jit_ir_graph_->outputs()) {
    auto oit = value_to_ivalue_.find(output);
    TORCH_CHECK(
        oit != value_to_ivalue_.end(),
        "value_to_ivalue_ does not have an entry for %",
        output->debugName());
    IValPtrShared ivpsh = oit->second;
    pt_stack_->insert(pt_stack_->end(), *ivpsh);
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
  torch::jit::drop(*pt_stack_, num_inputs_);
  for (const auto& ivpsh : aten_outputs_) {
    pt_stack_->insert(pt_stack_->end(), *ivpsh);
  }
}

void habana::HabanaLaunchOpPT::ProcessInputStack(torch::jit::Stack& input_st) {
  num_inputs_ = jit_ir_graph_->inputs().size();
  PT_EAGER_DEBUG("[SHAPE AGNOSTIC] #graph_inputs : ", num_inputs_);
  TORCH_CHECK(
      num_inputs_ == input_st.size(),
      "Input stack size=",
      input_st.size(),
      " is not matching with #graph_inputs=",
      num_inputs_);

  num_tensor_inputs_ = 0;
  input_refs_ = torch::jit::last(input_st, num_inputs_);

  // All tensors should be on Habana, we should assert otherwise
  bool is_all_hpu = true;
  for (auto& input : input_refs_) {
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
  habana::ShapeInference::Capture(&map_shape_);

  CopyInputStack(input_st);
}
