#pragma once
#include <string>
#include <unordered_map>
#include <vector>
#include "habana_helpers/logging.h"
#include "synapse_helpers/env_flags.h"
namespace synapse_helpers {
namespace layouts {

enum class SynapseLayoutFormat {
  WHCN = 0,
  WHDCN = 1,
  SRCK = 2,
  SRQCK = 3,
  DONT_CARE = 4,
  INVALID = 5
};

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
        {SynapseLayoutFormat::DONT_CARE, dont_care}};

inline std::vector<const char*> getSynapseLayoutFormat(
    const std::vector<SynapseLayoutFormat>& layout_format) {
  if (!GET_ENV_FLAG_NEW(PT_HPU_ENABLE_SYNAPSE_LAYOUT_HANDLING)) {
    return {};
  }

  std::vector<const char*> layouts;
  layouts.reserve(layout_format.size());
  for (size_t i = 0; i < layout_format.size(); i++) {
    auto layout = toLayoutStr.find(layout_format[i]);
    HABANA_ASSERT(
        layout != toLayoutStr.end(),
        "Unknown layout in getSynapseLayoutFormat");
    layouts[i] = layout->second;
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

#define SET_LAYOUT_IDX_VAR(name)                                      \
  const unsigned name =                                               \
      GET_ENV_FLAG_NEW(PT_HPU_ENABLE_SYNAPSE_LAYOUT_HANDLING) == true \
      ? static_cast<unsigned>(_##name)                                \
      : static_cast<unsigned>(__##name);
LIST_OF_LAYOUT_IDX
#undef SET_LAYOUT_IDX_VAR

} // namespace layouts
} // namespace synapse_helpers