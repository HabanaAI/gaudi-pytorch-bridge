#include "habana_lazy/layout_utils.h"
#include <string>
#include "synapse_helpers/env_flags.h"

namespace habana_lazy {
namespace layouts {

// conv
const char* LayoutUtils::pt_conv_input_layout[] = {
    pt_default_data_layout,
    pt_default_weight_layout,
    dont_care,
    pt_default_data_layout,
    dont_care};
const char* LayoutUtils::pt_conv_output_layout[] = {pt_default_data_layout};
const char* LayoutUtils::pt_conv3d_input_layout[] = {
    pt_default_3d_data_layout,
    pt_default_3d_weight_layout,
    dont_care,
    pt_default_3d_data_layout,
    dont_care};
const char* LayoutUtils::pt_conv3d_output_layout[] = {
    pt_default_3d_data_layout};

// dedw
const char* LayoutUtils::pt_dedw_input_layout[] = {
    pt_default_data_layout,
    pt_default_data_layout,
    dont_care};
const char* LayoutUtils::pt_dedw_output_layout[] = {pt_default_weight_layout};
const char* LayoutUtils::pt_dedw3d_input_layout[] = {
    pt_default_3d_data_layout,
    pt_default_3d_data_layout,
    dont_care};
const char* LayoutUtils::pt_dedw3d_output_layout[] = {
    pt_default_3d_weight_layout};

// dedx
const char* LayoutUtils::pt_dedx_input_layout[] = {
    pt_default_data_layout,
    pt_default_weight_layout,
    dont_care};
const char* LayoutUtils::pt_dedx_output_layout[] = {pt_default_data_layout};
const char* LayoutUtils::pt_dedx3d_input_layout[] = {
    pt_default_3d_data_layout,
    pt_default_3d_weight_layout,
    dont_care};
const char* LayoutUtils::pt_dedx3d_output_layout[] = {
    pt_default_3d_data_layout};

const char** LayoutUtils::getInputLayouts(const std::string& guid) {
  if (!GET_ENV_FLAG_NEW(PT_HPU_ENABLE_SYNAPSE_LAYOUT_HANDLING)) {
    return nullptr;
  }
  if (guid == "spatial_convolution") {
    return pt_conv_input_layout;
  } else if (guid == "spatial_convolution3d") {
    return pt_conv3d_input_layout;
  } else if (guid == "dedw") {
    return pt_dedw_input_layout;
  } else if (guid == "dedw3d") {
    return pt_dedw3d_input_layout;
  } else if (guid == "dedx") {
    return pt_dedx_input_layout;
  } else if (guid == "dedx3d") {
    return pt_dedx3d_input_layout;
  } else {
    return nullptr;
  }
}

const char** LayoutUtils::getOutputLayouts(const std::string& guid) {
  if (!GET_ENV_FLAG_NEW(PT_HPU_ENABLE_SYNAPSE_LAYOUT_HANDLING)) {
    return nullptr;
  }
  if (guid == "spatial_convolution") {
    return pt_conv_output_layout;
  } else if (guid == "spatial_convolution3d") {
    return pt_conv3d_output_layout;
  } else if (guid == "dedw") {
    return pt_dedw_output_layout;
  } else if (guid == "dedw3d") {
    return pt_dedw3d_output_layout;
  } else if (guid == "dedx") {
    return pt_dedx_output_layout;
  } else if (guid == "dedx3d") {
    return pt_dedx3d_output_layout;
  } else {
    return nullptr;
  }
}

} // namespace layouts
} // namespace habana_lazy
