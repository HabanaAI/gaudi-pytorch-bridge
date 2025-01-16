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

#pragma once

#include <ATen/core/TensorBody.h>
#include <absl/base/thread_annotations.h>
#include <absl/container/flat_hash_map.h>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <shared_mutex>
#include "common/strong_type.h"

namespace habana {
/**
 * Stores information about constant sections and it usages between recipes.
 */
struct ConstantInformation {
  ConstantInformation() = default;
  ConstantInformation(const ConstantInformation&) = delete;
  ConstantInformation& operator=(const ConstantInformation&) = delete;
  ConstantInformation(ConstantInformation&&) = delete;
  ConstantInformation& operator=(ConstantInformation&&) = delete;
  ~ConstantInformation() = default;

  using scaleIndex_t = common::StrongType<size_t, struct ValueTag>;
  using id_t = common::StrongType<int, struct IdTag>;
  using checksum_t = common::StrongType<size_t, struct ChecksumTag>;
  using key_t = common::StrongType<size_t, struct KeyTag>;
  using InnerMap = std::unordered_map<scaleIndex_t, std::unordered_set<id_t>>;

  /**
   * Stores checksum for specific constant id.
   *
   * @param id Constant id.
   * @param checksum Constant checksum.
   */
  void Insert(id_t id, checksum_t checksum);

  /**
   * Stores constant information for specific constant id with given checksum
   * and for specific recipe key.
   *
   * @param id Constant id
   * @param checksum Constant checksum
   * @param key Recipe key
   * @param size Constant section size
   */
  void PushInfo(id_t id, checksum_t checksum, key_t key, uint64_t size);

  /**
   * For given constant id and its checksum store another recipe key that uses
   * it.
   *
   * @param id Constant id
   * @param checksum Constant checksum
   * @param key Recipe key
   */
  void AddRecipe(id_t id, checksum_t checksum, key_t key);

  /**
   * Recovers tensor storage for given constant id and given recipe key and sets
   * it for provided tensor.
   *
   * @param id Constant id
   * @param key Recipe key
   * @param[in,out] tensor Tensor to recover storage pointer to
   */
  void GetConstPtrForRecipe(id_t id, key_t key, at::Tensor& tensor);

  /**
   * Stores data ptr for specific constant id with specific checksum.
   *
   * @param id Constant id
   * @param ptr Data pointer
   * @param checksum Constant checksum
   */
  void StorePrevDataPtr(id_t id, at::DataPtr ptr, checksum_t checksum);

  /**
   * Recovered checksums for specific constant id and recipe key.
   */
  struct ConstantChecksums {
    /**
     * Global stored checksum for the constant
     */
    checksum_t device_checksum_;

    /**
     * Constant checksum stored specifically for given recipe
     */
    checksum_t recipe_checksum_;
  };

  /**
   * Recovers checksums stored for given constant id and given recipe key
   * attached to that id.
   *
   * @param id Constant id
   * @param key Recipe key
   */
  ConstantChecksums GetDeviceAndRecipeChecksums(id_t id, key_t key) const;

  /**
   * Checks if given constant id is stored, and has stored specific checksum.
   */
  bool IsCheckSumExistInAnyConstInfo(id_t id, checksum_t checksum) const;

  /**
   * Checks if given constant id is stored, and has stored specific recipe
   * details.
   */
  bool DoesConstInfoExistForRecipe(id_t id, key_t key) const;

  /**
   * Recovers checksum for specific constant id.
   *
   * @param id Checksum id
   *
   * @return Recovered checksum or nullopt in case constant with given id is not
   * stored.
   */
  std::optional<checksum_t> GetDeviceChecksum(id_t id) const;

  /**
   * Removes stored constant section informations.
   */
  void ClearChecksumInformation();

  /**
   * Record all the scale constant inputs of the recipe.
   */
  void StoreRecipeScaleConstInput(
      key_t recipe_key,
      scaleIndex_t scale_idx,
      id_t const_id);

  /**
   * Copy DataPtr from the matched constant id and create constant info.
   */
  void CopyMatchedDataPtrForRecipe(
      id_t from_const_id,
      id_t to_const_id,
      key_t recipe_key,
      at::Tensor& pt_tensor);

  /**
   * Return the matched constant id used at recipe compile time.
   */
  id_t GetMatchedConstIdForRecipe(
      key_t recipe_key,
      id_t const_id,
      scaleIndex_t scale_index);

  /**
   * Check if the constant id is new from recipe compile time.
   */
  bool IsNewConstIdForRecipe(key_t recipe_key, id_t const_id);

 private:
  /**
   * Information about const section and recipes using it.
   */
  struct constInfo_t {
    /**
     * Checksum on the device.
     */
    checksum_t checksum_;

    /**
     * List of keys for recipes sharing the const.
     */
    std::vector<key_t> recipe_key_;

    /**
     * Size of the const section.
     */
    uint64_t section_size_;

    /**
     * Pointer to const section data.
     * If pointer is invalid, nullopt.
     */
    std::optional<at::DataPtr> data_ptr_;
  };

  /**
   * Information on constant section for given constant id.
   */
  struct checksum_and_const_infos {
    /**
     * Specific constant checksum.
     */
    checksum_t device_checksum_;

    /**
     * List of information about const section for different checksums.
     */
    std::vector<constInfo_t> infos_;
  };

  mutable std::shared_mutex checksum_map_mtx_;

  absl::flat_hash_map<id_t, checksum_and_const_infos> const_checksum_map_
      ABSL_GUARDED_BY(checksum_map_mtx_);
  absl::flat_hash_map<key_t, InnerMap> recipe_scale_constant_map_
      ABSL_GUARDED_BY(checksum_map_mtx_);

  /**
   * Stores data ptr for specific constant id with specific checksum.
   *
   * @param id Constant id
   * @param ptr Data pointer
   * @param checksum Constant checksum
   */
  void StorePrevDataPtrImpl(id_t id, at::DataPtr ptr, checksum_t checksum);
};

/**
 * Provides shared_ptr to constant information singleton in case user needs to
 * extend its lifetime. The same object that is served with
 * ConstantInformationValue
 */
std::shared_ptr<struct ConstantInformation>& ConstantInformationPtr();

/**
 * Provides reference to singleton constant information. The same object that is
 * served with ConstantInformationPtr.
 */
ConstantInformation& ConstantInformationValue();

bool IsConstantScaleTensor(at::Tensor& tensor);

} // namespace habana