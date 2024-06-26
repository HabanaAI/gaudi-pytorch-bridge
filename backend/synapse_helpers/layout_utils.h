/*******************************************************************************
 * Copyright (C) 2022-2024 Habana Labs, Ltd. an Intel Company
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
#pragma once
#include <string>
#include <unordered_map>
#include <vector>
#include "backend/synapse_helpers/env_flags.h"
#include "habana_helpers/logging.h"
namespace synapse_helpers {
namespace layouts {

enum class SynapseLayoutFormat {
  WHCN = 0,
  WHDCN = 1,
  SRCK = 2,
  SRQCK = 3,
  DONT_CARE = 4,
  AWHN = 5,
  BCN = 6,
  CN = 7,
  WHN = 8,
  XR = 9,
  AB = 10,
  VN = 11,
  NSB = 12,
  BSN = 13,
  WHC = 14,
  CNT = 15,
  SN = 16,
  SNT = 17,
  WHDC = 18,
  AWHDN = 19,
  WCN = 20,
  CLN = 21,
  INVALID
};

// Memory permutation represents how tensor layout is set in memory
// For Synapse transposes weight we use API: synTensorSetPermutation
// instead of providing strided tensor.
using MemoryPermutation = std::vector<uint8_t>;
static const MemoryPermutation weight_rsck_in_memory = {3, 2, 0, 1};
static const MemoryPermutation weight_qrsck_in_memory = {4, 3, 0, 1, 2};

static constexpr char dont_care[] = "";
static constexpr char pt_default_data_layout[] = "WHCN";
static constexpr char pt_default_3d_data_layout[] = "WHDCN";
static constexpr char pt_default_weight_layout[] = "SRCK";
static constexpr char pt_default_3d_weight_layout[] = "SRQCK";

static const std::unordered_map<const SynapseLayoutFormat, const char*>
    toLayoutStr = {
        {SynapseLayoutFormat::WHCN, pt_default_data_layout},
        {SynapseLayoutFormat::WHDCN, pt_default_3d_data_layout},
        {SynapseLayoutFormat::SRCK, pt_default_weight_layout},
        {SynapseLayoutFormat::SRQCK, pt_default_3d_weight_layout},
        {SynapseLayoutFormat::AWHDN, "AWHDN"},
        {SynapseLayoutFormat::AWHN, "AWHN"},
        {SynapseLayoutFormat::BCN, "BCN"},
        {SynapseLayoutFormat::CN, "CN"},
        {SynapseLayoutFormat::CLN, "CLN"},
        {SynapseLayoutFormat::WHN, "WHN"},
        {SynapseLayoutFormat::XR, "XR"},
        {SynapseLayoutFormat::AB, "AB"},
        {SynapseLayoutFormat::VN, "VN"},
        {SynapseLayoutFormat::NSB, "NSB"},
        {SynapseLayoutFormat::BSN, "BSN"},
        {SynapseLayoutFormat::WHC, "WHC"},
        {SynapseLayoutFormat::CNT, "CNT"},
        {SynapseLayoutFormat::SN, "SN"},
        {SynapseLayoutFormat::SNT, "SNT"},
        {SynapseLayoutFormat::WHDC, "WHDC"},
        {SynapseLayoutFormat::WCN, "WCN"},
        {SynapseLayoutFormat::DONT_CARE, dont_care}};

inline std::vector<const char*> getSynapseLayoutFormat(
    const std::vector<SynapseLayoutFormat>& layout_format) {
  std::vector<const char*> layouts;
  layouts.reserve(layout_format.size());
  for (size_t i = 0; i < layout_format.size(); i++) {
    auto layout = toLayoutStr.find(layout_format[i]);
    HABANA_ASSERT(
        layout != toLayoutStr.end(),
        "Unknown layout in getSynapseLayoutFormat");
    layouts.push_back(layout->second);
  }
  return layouts;
}

enum LayoutIndex {
  // Data layout index for Conv2D input
  _INPUT_N_IDX = 0,
  _INPUT_C_IDX = 1,
  _INPUT_H_IDX = 2,
  _INPUT_W_IDX = 3,

  // Data layout index for Conv3D input
  _INPUT_3D_N_IDX = 0,
  _INPUT_3D_C_IDX = 1,
  _INPUT_3D_D_IDX = 2,
  _INPUT_3D_H_IDX = 3,
  _INPUT_3D_W_IDX = 4,

  // Weight layout index for Conv2D kernel
  _WEIGHT_KERNEL_K_IDX = 0,
  _WEIGHT_KERNEL_C_IDX = 1,
  _WEIGHT_KERNEL_R_IDX = 2,
  _WEIGHT_KERNEL_S_IDX = 3,

  // Weight layout index for Conv3D kernel
  _WEIGHT_KERNEL_3D_K_IDX = 0,
  _WEIGHT_KERNEL_3D_C_IDX = 1,
  _WEIGHT_KERNEL_3D_Q_IDX = 2,
  _WEIGHT_KERNEL_3D_R_IDX = 3,
  _WEIGHT_KERNEL_3D_S_IDX = 4,
};

enum LegacyLayoutIndex {
  // Data layout index for Conv2D input
  __INPUT_N_IDX = 0,
  __INPUT_H_IDX = 1,
  __INPUT_W_IDX = 2,
  __INPUT_C_IDX = 3,

  // Data layout index for Conv3D input
  __INPUT_3D_N_IDX = 0,
  __INPUT_3D_D_IDX = 1,
  __INPUT_3D_H_IDX = 2,
  __INPUT_3D_W_IDX = 3,
  __INPUT_3D_C_IDX = 4,

  // Weight layout index for Conv2D kernel
  __WEIGHT_KERNEL_R_IDX = 0,
  __WEIGHT_KERNEL_S_IDX = 1,
  __WEIGHT_KERNEL_C_IDX = 2,
  __WEIGHT_KERNEL_K_IDX = 3,

  // Weight layout index for Conv3D kernel
  __WEIGHT_KERNEL_3D_Q_IDX = 0,
  __WEIGHT_KERNEL_3D_R_IDX = 1,
  __WEIGHT_KERNEL_3D_S_IDX = 2,
  __WEIGHT_KERNEL_3D_C_IDX = 3,
  __WEIGHT_KERNEL_3D_K_IDX = 4,
};

#define LIST_OF_LAYOUT_IDX                   \
  SET_LAYOUT_IDX_VAR(INPUT_N_IDX)            \
  SET_LAYOUT_IDX_VAR(INPUT_C_IDX)            \
  SET_LAYOUT_IDX_VAR(INPUT_H_IDX)            \
  SET_LAYOUT_IDX_VAR(INPUT_W_IDX)            \
  SET_LAYOUT_IDX_VAR(INPUT_3D_N_IDX)         \
  SET_LAYOUT_IDX_VAR(INPUT_3D_C_IDX)         \
  SET_LAYOUT_IDX_VAR(INPUT_3D_D_IDX)         \
  SET_LAYOUT_IDX_VAR(INPUT_3D_H_IDX)         \
  SET_LAYOUT_IDX_VAR(INPUT_3D_W_IDX)         \
  SET_LAYOUT_IDX_VAR(WEIGHT_KERNEL_K_IDX)    \
  SET_LAYOUT_IDX_VAR(WEIGHT_KERNEL_C_IDX)    \
  SET_LAYOUT_IDX_VAR(WEIGHT_KERNEL_R_IDX)    \
  SET_LAYOUT_IDX_VAR(WEIGHT_KERNEL_S_IDX)    \
  SET_LAYOUT_IDX_VAR(WEIGHT_KERNEL_3D_K_IDX) \
  SET_LAYOUT_IDX_VAR(WEIGHT_KERNEL_3D_C_IDX) \
  SET_LAYOUT_IDX_VAR(WEIGHT_KERNEL_3D_Q_IDX) \
  SET_LAYOUT_IDX_VAR(WEIGHT_KERNEL_3D_R_IDX) \
  SET_LAYOUT_IDX_VAR(WEIGHT_KERNEL_3D_S_IDX)

#define SET_LAYOUT_IDX_VAR(name) \
  const unsigned name = static_cast<unsigned>(_##name);
LIST_OF_LAYOUT_IDX
#undef SET_LAYOUT_IDX_VAR
#undef LIST_OF_LAYOUT_IDX

} // namespace layouts
} // namespace synapse_helpers