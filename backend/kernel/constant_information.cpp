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

#include "backend/kernel/constant_information.h"
#include <memory>
#include <mutex>
#include <optional>
#include <shared_mutex>
#include "habana_helpers/logging.h"

namespace habana {

std::shared_ptr<ConstantInformation>& ConstantInformationPtr() {
  static std::shared_ptr<ConstantInformation> constant_checksum_ptr{
      new (ConstantInformation)};
  return constant_checksum_ptr;
}

ConstantInformation& ConstantInformationValue() {
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
  static ConstantInformation& constant_checksum =
      *ConstantInformationPtr().get();
  return constant_checksum;
}

void ConstantInformation::Insert(const id_t id, const checksum_t checksum) {
  std::unique_lock lock(checksum_map_mtx_);
  if (auto const_checksum_iterator = const_checksum_map_.find(id);
      const_checksum_iterator == const_checksum_map_.end()) {
    const_checksum_map_.emplace(id, checksum_and_const_infos{checksum, {}});
  } else {
    const_checksum_iterator->second.checksum_ = checksum;
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
  auto current_checksum_on_device = const_checksum.checksum_;
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

bool ConstantInformation::DoesConstInfoExist(const id_t id, key_t key) const {
  auto checksum_iterator = const_checksum_map_.find(id);
  if (checksum_iterator == const_checksum_map_.end()) {
    return false;
  }
  // if id exists but recipe not found - that also should throw exception
  for (auto& info : const_checksum_map_.at(id).infos_) {
    for (auto& recipe : info.recipe_key_) {
      if (recipe == key) {
        return true;
      }
    }
  }
  return false;
}

ConstantInformation::ConstantChecksums ConstantInformation::
    GetConstCheckSumForRecipe(const id_t id, const key_t key) const {
  std::shared_lock lock(checksum_map_mtx_);
  auto checksum_iterator = const_checksum_map_.find(id);
  HABANA_ASSERT(
      checksum_iterator != const_checksum_map_.end(),
      "No checksum exists for id: ",
      id,
      " in the map");
  for (auto& info : checksum_iterator->second.infos_) {
    for (auto recipe_key : info.recipe_key_) {
      if (recipe_key == key) {
        return {checksum_iterator->second.checksum_, info.checksum_};
      }
    }
  }
  HABANA_ASSERT(
      false, "No checksum found for const_id: ", id, " for recipe: ", key);
  return {checksum_iterator->second.checksum_, checksum_t{0ul}};
}

bool ConstantInformation::DoesCheckSumExist(
    const id_t id,
    const checksum_t checksum) const {
  std::shared_lock lock(checksum_map_mtx_);
  auto const_checksum_iterator = const_checksum_map_.find(id);
  if (const_checksum_iterator == const_checksum_map_.end()) {
    return false;
  }

  for (auto& info : const_checksum_iterator->second.infos_) {
    if (info.checksum_ == checksum) {
      return true;
    }
  }

  return false;
}

std::optional<ConstantInformation::checksum_t> ConstantInformation::
    GetChecksumForId(const id_t id) const {
  std::shared_lock lock(checksum_map_mtx_);
  auto const_checksum_iterator = const_checksum_map_.find(id);
  if (const_checksum_iterator == const_checksum_map_.end()) {
    return std::nullopt;
  }

  return const_checksum_iterator->second.checksum_;
}

void ConstantInformation::ClearChecksumInformation() {
  std::unique_lock lock(checksum_map_mtx_);
  const_checksum_map_.clear();
}
} // namespace habana
