/**
 * Copyright (c) 2021-2026 Intel Corporation
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

#include "backend/kernel/constant_information.h"
#include <memory>
#include <mutex>
#include <optional>
#include <shared_mutex>
#include "habana_helpers/logging.h"
#include "habana_helpers/python_utils.h"

#include "backend/habana_device/HPUAllocator.h"
#include "backend/habana_device/HPUStream.h"
#include "backend/habana_device/PinnedMemoryAllocator.h"
#include "backend/helpers/get_n_bytes.h"
#include "backend/kernel/hpu_habana_launch_op_pt.h"

namespace habana {

std::shared_ptr<ConstantInformation>& ConstantInformationPtr() {
  static std::shared_ptr<ConstantInformation> constant_checksum_ptr{
      new (ConstantInformation)};
  return constant_checksum_ptr;
}

ConstantInformation& ConstantInformationValue() {
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
  static ConstantInformation& constant_checksum = *ConstantInformationPtr();
  return constant_checksum;
}

void ConstantInformation::Insert(const id_t id, const checksum_t checksum) {
  std::unique_lock lock(checksum_map_mtx_);
  if (auto const_checksum_iterator = const_checksum_map_.find(id);
      const_checksum_iterator == const_checksum_map_.end()) {
    const_checksum_map_.emplace(id, checksum_and_const_infos{checksum, {}});
  } else {
    const_checksum_iterator->second.device_checksum_ = checksum;
  }
}

void ConstantInformation::PushInfo(
    const id_t id,
    const checksum_t checksum,
    const key_t key,
    const uint64_t size) {
  PT_BRIDGE_DEBUG(
      "[PushConstantCheckSumInfo] :: const_id: ",
      id,
      " checksum: ",
      checksum,
      " size: ",
      size,
      " key: ",
      key);
  std::unique_lock lock(checksum_map_mtx_);
  // The call for this function is made only after the check that key exists
  const_checksum_map_.at(id).infos_.emplace_back(
      constInfo_t{checksum, {key}, size, {}});
}

void ConstantInformation::AddRecipe(
    const id_t id,
    const checksum_t checksum,
    const key_t key) {
  // The call for this function is made only after the check that key exists
  std::unique_lock lock(checksum_map_mtx_);
  for (auto& info : const_checksum_map_.at(id).infos_) {
    if (info.checksum_ == checksum) {
      info.recipe_key_.push_back(key);
      return;
    }
  }
  PT_BRIDGE_DEBUG("Const checksum not found when adding recipe.");
}

void ConstantInformation::GetConstPtrForRecipe(
    const id_t id,
    const key_t key,
    at::Tensor& tensor) {
  std::shared_lock lock(checksum_map_mtx_);
  // The call for this function is made only after the check that key exists
  auto& const_checksum = const_checksum_map_.at(id);
  auto current_checksum_on_device = const_checksum.device_checksum_;
  for (auto& info : const_checksum.infos_) {
    for (auto recipe_key : info.recipe_key_) {
      if (recipe_key == key) {
        HABANA_ASSERT(
            info.data_ptr_.has_value(),
            "There is no pointer associated with const_id: ",
            id,
            " for recipe: ",
            key);
        PT_BRIDGE_DEBUG(
            "For tensor with const_id: ",
            id,
            " moving the data pointer to: ",
            info.data_ptr_.value().get())
        auto old_data_ptr =
            tensor.storage().set_data_ptr(std::move(info.data_ptr_.value()));
        tensor.storage().set_nbytes(info.section_size_);
        StorePrevDataPtrImpl(
            id, std::move(old_data_ptr), current_checksum_on_device);
        info.data_ptr_.reset();
        return;
      }
    }
  }
  HABANA_ASSERT(
      false, "Constant information not found in the map for id: ", id);
}

void ConstantInformation::StorePrevDataPtr(
    const id_t id,
    at::DataPtr ptr,
    const checksum_t checksum) {
  std::unique_lock lock(checksum_map_mtx_);
  StorePrevDataPtrImpl(id, std::move(ptr), checksum);
}

void ConstantInformation::StorePrevDataPtrImpl(
    const id_t id,
    at::DataPtr ptr,
    const checksum_t checksum) {
  // The call for this function is made only after the check that key exists
  for (auto& info : const_checksum_map_.at(id).infos_) {
    if (info.checksum_ == checksum) {
      info.data_ptr_ = std::move(ptr);
      PT_BRIDGE_DEBUG(
          "[StorePrevDataPtr] :: const_id: ",
          id,
          " checksum: ",
          info.checksum_,
          " size: ",
          info.section_size_,
          " key: ",
          info.recipe_key_,
          " ptr: ",
          info.data_ptr_.value().get());
      return;
    }
  }
  HABANA_ASSERT(false, "No such checksum found in the map, const_id: ", id);
}

bool ConstantInformation::DoesConstInfoExistForRecipe(const id_t id, key_t key)
    const {
  auto checksum_iterator = const_checksum_map_.find(id);
  if (checksum_iterator == const_checksum_map_.end()) {
    return false;
  }
  // if id exists but recipe not found - that also should throw exception
  for (const auto& info : const_checksum_map_.at(id).infos_) {
    for (const auto& recipe : info.recipe_key_) {
      if (recipe == key) {
        return true;
      }
    }
  }
  return false;
}

ConstantInformation::ConstantChecksums ConstantInformation::
    GetDeviceAndRecipeChecksums(const id_t id, const key_t key) const {
  std::shared_lock lock(checksum_map_mtx_);
  auto checksum_iterator = const_checksum_map_.find(id);
  HABANA_ASSERT(
      checksum_iterator != const_checksum_map_.end(),
      "No checksum exists for id: ",
      id,
      " in the map");
  for (const auto& info : checksum_iterator->second.infos_) {
    for (auto recipe_key : info.recipe_key_) {
      if (recipe_key == key) {
        return {checksum_iterator->second.device_checksum_, info.checksum_};
      }
    }
  }
  HABANA_ASSERT(
      false, "No checksum found for const_id: ", id, " for recipe: ", key);
  return {checksum_iterator->second.device_checksum_, checksum_t{0UL}};
}

bool ConstantInformation::IsCheckSumExistInAnyConstInfo(
    const id_t id,
    const checksum_t checksum) const {
  std::shared_lock lock(checksum_map_mtx_);
  auto const_checksum_iterator = const_checksum_map_.find(id);
  if (const_checksum_iterator == const_checksum_map_.end()) {
    return false;
  }

  return std::any_of(
      const_checksum_iterator->second.infos_.begin(),
      const_checksum_iterator->second.infos_.end(),
      [&checksum](const constInfo_t& info) {
        return info.checksum_ == checksum;
      });
}

std::optional<ConstantInformation::checksum_t> ConstantInformation::
    GetDeviceChecksum(const id_t id) const {
  std::shared_lock lock(checksum_map_mtx_);
  auto const_checksum_iterator = const_checksum_map_.find(id);
  if (const_checksum_iterator == const_checksum_map_.end()) {
    return std::nullopt;
  }

  return const_checksum_iterator->second.device_checksum_;
}

void ConstantInformation::StoreRecipeScaleConstInput(
    key_t recipe_key,
    scaleIndex_t scale_index,
    id_t const_id) {
  std::unique_lock lock(checksum_map_mtx_);
  auto recipe_iterator = recipe_scale_constant_map_.find(recipe_key);
  if (recipe_iterator == recipe_scale_constant_map_.end()) {
    auto recipe_value =
        std::unordered_map<scaleIndex_t, std::unordered_set<id_t>>{
            {scale_index, std::unordered_set<id_t>{const_id}}};
    recipe_scale_constant_map_.emplace(recipe_key, recipe_value);
    return;
  }

  auto recipe_input_iter = recipe_iterator->second.find(scale_index);
  if (recipe_input_iter == recipe_iterator->second.end()) {
    recipe_iterator->second.emplace(
        scale_index, std::unordered_set<id_t>{const_id});
    return;
  }

  auto& const_id_set = recipe_input_iter->second;
  if (const_id_set.find(const_id) == const_id_set.end()) {
    const_id_set.insert(const_id);
    return;
  }

  PT_BRIDGE_DEBUG(
      "Recipe ",
      recipe_key,
      " already contain input index:",
      scale_index,
      " and constant id:",
      const_id);
}

ConstantInformation::id_t ConstantInformation::GetMatchedConstIdForRecipe(
    key_t recipe_key,
    id_t const_id,
    scaleIndex_t scale_index) {
  std::shared_lock lock(checksum_map_mtx_);
  auto recipe_iterator = recipe_scale_constant_map_.find(recipe_key);
  HABANA_ASSERT(recipe_iterator != recipe_scale_constant_map_.end())
  auto recipe_input_iter = recipe_iterator->second.find(scale_index);
  HABANA_ASSERT(recipe_input_iter != recipe_iterator->second.end())
  auto const_id_set = recipe_input_iter->second;
  HABANA_ASSERT(
      !const_id_set.empty(),
      "Const id list is empty for recipe ",
      recipe_key,
      " const_id ",
      const_id);
  return *(const_id_set.begin());
}

void ConstantInformation::CopyMatchedDataPtrForRecipe(
    id_t from_const_id,
    id_t const_id,
    key_t recipe_key,
    at::Tensor& pt_tensor) {
  TensorExtraMeta::prepare_const_tensor(pt_tensor, true);
  auto get_const_info = [&](ConstantInformation::id_t const_id,
                            ConstantInformation::key_t r_key)
      -> const ConstantInformation::constInfo_t& {
    std::shared_lock lock(checksum_map_mtx_);
    auto const_checksum_iterator = const_checksum_map_.find(const_id);
    HABANA_ASSERT(
        const_checksum_iterator != const_checksum_map_.end(),
        "Const id  {} not presented in const_checksum_map_",
        const_id)
    for (auto& info : const_checksum_iterator->second.infos_) {
      for (auto& key : info.recipe_key_) {
        if (key == r_key) {
          return info;
        }
      }
    }
    ConstantInformation::checksum_t checksum{0};
    ConstantInformation::key_t tmp_key{0};
    static ConstantInformation::constInfo_t constInfo{
        checksum, {tmp_key}, 0, {}};
    return constInfo;
  };

  // Read constant info from the compile time constant
  const ConstantInformation::constInfo_t& info =
      get_const_info(from_const_id, recipe_key);
  checksum_t checksum = info.checksum_;
  uint64_t section_size = info.section_size_;
  auto* tmeta{get_tensor_extra_meta(pt_tensor)};
  auto checksum_found = IsCheckSumExistInAnyConstInfo(const_id, checksum);

  if (section_size != 0) {
    if (!checksum_found) {
      /**
       * This part is same as HandleTensorWithNewChecksum.
       * Handling of host checksum is an addition here.
       */
      auto old_size = tmeta->get_host_size();
      ConstantInformation::checksum_t host_checksum{tmeta->get_host_checksum()};
      auto checksum_if_exists = GetDeviceChecksum(const_id);
      if (checksum_if_exists.has_value() or (checksum != host_checksum)) {
        // Reallocation is required
        tmeta->set_nbytes_inference(old_size);
        at::DataPtr data =
            pt_tensor.storage().allocator()->allocate(section_size);
        PT_BRIDGE_DEBUG(
            "Cache hit: Needed reallocation (bridge) old_size ",
            old_size,
            " != ",
            section_size,
            " Checksum: ",
            checksum,
            " Allocated data_ptr: ",
            data.get());
        auto old_data_ptr = pt_tensor.storage().set_data_ptr(std::move(data));
        pt_tensor.storage().set_nbytes(section_size);
        if (checksum_if_exists.has_value()) {
          StorePrevDataPtr(
              const_id, std::move(old_data_ptr), checksum_if_exists.value());
        } else {
          ConstantInformation::checksum_t host_checksum{
              tmeta->get_host_checksum()};
          PushInfo(const_id, host_checksum, recipe_key, old_size);
          StorePrevDataPtr(const_id, std::move(old_data_ptr), host_checksum);
        }
      }
      Insert(const_id, checksum);
      PushInfo(const_id, checksum, recipe_key, section_size);
      if (checksum == host_checksum and !checksum_if_exists.has_value()) {
        // If synapse has not modified the tensor data (old size equal section
        // size) And no other recipe has a checksum before this then no need to
        // copy the new data
        return;
      }
      if (info.data_ptr_.has_value()) {
        std::atomic<bool> copyDone{false};
        habana::HPUDeviceContext::copy_data_within_device(
            reinterpret_cast<synapse_helpers::device_ptr>(
                info.data_ptr_.value().get()),
            reinterpret_cast<synapse_helpers::device_ptr>(pt_tensor.data_ptr()),
            reinterpret_cast<synapse_helpers::device_ptr>(
                info.data_ptr_.value().get()),
            reinterpret_cast<synapse_helpers::device_ptr>(
                pt_tensor.storage().data_ptr().get()),
            section_size,
            [&copyDone]() { copyDone = true; },
            c10::hpu::getCurrentHPUStream());

        // Release GIL if going to wait
        habana_helpers::AutoNoGIL gil_release;
        // wait for copy completion
        while (!copyDone) {
          std::this_thread::yield();
        }
      } else {
        HABANA_ASSERT(
            section_size == tmeta->get_host_size(),
            "Host size and device section size missmatched!!!")
        std::atomic<bool> copyDone{false};
        habana::HPUDeviceContext::copy_data_to_device(
            tmeta->get_host_ptr(),
            reinterpret_cast<synapse_helpers::device_ptr>(pt_tensor.data_ptr()),
            reinterpret_cast<synapse_helpers::device_ptr>(
                pt_tensor.storage().data_ptr().get()),
            habana_helpers::GetNBytes(pt_tensor),
            [&copyDone]() { copyDone = true; },
            false,
            true,
            c10::hpu::getCurrentHPUStream());

        // Release GIL if going to wait
        habana_helpers::AutoNoGIL gil_release_;
        // wait for copy completion
        while (!copyDone) {
          std::this_thread::yield();
        }
      }
    } else if (GetDeviceChecksum(const_id) == checksum) {
      /**
       * This part is same as HandleTensorWithChecksumOnDevice.
       */
      AddRecipe(const_id, checksum, recipe_key);
      PT_BRIDGE_DEBUG(
          "Constant tensor already exists on the device, avoiding re-copy to device, checksum: ",
          checksum,
          " const_id: ",
          const_id,
          " recipe key: ",
          recipe_key);
    } else {
      /**
       * This part is same as HandleTensorWithExistingChecksumInCache.
       */
      AddRecipe(const_id, checksum, recipe_key);
      GetConstPtrForRecipe(const_id, recipe_key, pt_tensor);
      Insert(const_id, checksum);
      PT_BRIDGE_DEBUG(
          "Tensor with const_id: ",
          const_id,
          " has moved data pointer for the data corresponding to checksum: ",
          checksum,
          " for cache miss on key ",
          recipe_key);
    }
  } else {
    /**
     * This part is same as HandleTensorWithZeroSize.
     * Handling of host checksum is an addition here.
     */
    auto* tmeta{get_tensor_extra_meta(pt_tensor)};
    auto old_size = tmeta->get_host_size();
    ConstantInformation::id_t const_id{tmeta->get_const_id()};
    auto checksum_if_exists = GetDeviceChecksum(const_id);
    tmeta->set_nbytes_inference(old_size);
    ConstantInformation::checksum_t checksum{0};
    Insert(const_id, checksum);
    PushInfo(const_id, checksum, recipe_key, 0 /*_section_size*/);
    at::DataPtr data = pt_tensor.storage().allocator()->allocate(0);
    auto old_data_ptr = pt_tensor.storage().set_data_ptr(std::move(data));
    pt_tensor.storage().set_nbytes(0);
    if (checksum_if_exists.has_value() and
        checksum_if_exists.value() != checksum) {
      PT_BRIDGE_DEBUG(
          "For const_id: ",
          const_id,
          " Checksum has valid value for another recipe");
      StorePrevDataPtr(
          const_id, std::move(old_data_ptr), checksum_if_exists.value());
    } else if (!checksum_if_exists.has_value()) {
      ConstantInformation::checksum_t host_checksum{tmeta->get_host_checksum()};
      PushInfo(const_id, host_checksum, recipe_key, old_size);
      StorePrevDataPtr(const_id, std::move(old_data_ptr), host_checksum);
    }
  }
}

bool ConstantInformation::IsNewConstIdForRecipe(
    key_t recipe_key,
    id_t const_id) {
  std::shared_lock lock(checksum_map_mtx_);
  auto const_checksum_iterator = const_checksum_map_.find(const_id);
  if (const_checksum_iterator == const_checksum_map_.end()) {
    return true;
  } else {
    auto& const_checksum = const_checksum_map_.at(const_id);
    for (auto& info : const_checksum.infos_) {
      for (auto key : info.recipe_key_) {
        if (recipe_key == key) {
          return false;
        }
      }
    }
    return true;
  }
}

void ConstantInformation::ClearChecksumInformation() {
  std::unique_lock lock(checksum_map_mtx_);
  const_checksum_map_.clear();
  recipe_scale_constant_map_.clear();
}

bool IsConstantScaleTensor(at::Tensor& tensor) {
  auto* tmeta{habana::get_tensor_extra_meta(tensor)};
  return (tmeta->is_const_tensor() && (tensor.numel() == 1));
}

} // namespace habana
